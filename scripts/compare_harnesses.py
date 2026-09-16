#!/usr/bin/env python
"""Did the DROID rig and this harness get the same answer on the same episodes?

    python scripts/compare_harnesses.py \
        runs/pi05_droid/Close-v1/xharness/eval_stdout.log \
        ~/Workspace/droid/data/simulate/2026-09-16T0130_ws-127-0-0-1-8000.json

Sibling of `scripts/check_cross_repo_cell.py`, which asks the same question of two *conda
environments*. This one asks it of two *client stacks*: `eval_main.py` driving a policy
through one of this repo's own adapters, against `droid`'s `policy.py` rollout loop driving
the same policy over the `droid_policy` protocol, into the same MuJoCo scene via
`droid/scripts/sim_server.py --benchmark`.

**The join key is the triple `pipeline.py` already prints**: house, per-house episode
ordinal, and object instance. The rig's side reports the same triple because
`BenchmarkScene.identity()` was written to, which is what makes one comparator able to read
both sides rather than two formats that have to be reconciled by eye.

**Compare the oracle flag, not the at-end one.** `pipeline.py`'s `success=` is the state at
the final frame; this repo's published numbers, and `eval_to_csv.py`'s `oracle_*` columns,
are success at ANY frame. `simulate.py` records both, so the comparison uses the matching
one -- and `--metric at-end` exists for when the native side was scored that way too.

**A stochastic policy makes this measure nothing.** Run the same-harness control first: the
identical cell twice against the same server. openpi replays bitwise only with
`DETERMINISTIC=1` (see scripts/serve_openpi.sh, which explains why); MolmoAct2 samples from
the global torch RNG and cannot. If the control disagrees with itself, a cross-harness
difference is noise and this script's verdict is void rather than failing -- `--control`
takes a second native log and says so for you.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# pipeline.py:1067, e.g.
#   Worker 5 house 222 episode 0 object stand_64ce...f5_1_0_0 completed with success=True
EPISODE_RE = re.compile(
    r"house (?P<house>\d+) episode (?P<episode>\d+) object (?P<object>\S+) "
    r"completed with success=(?P<success>True|False)"
)


def native_outcomes(log: Path) -> dict[tuple[str, str, str], bool]:
    """{(house, episode, object): success} out of a cell's stdout."""
    if not log.exists():
        raise SystemExit(f"{log} does not exist -- was this cell actually run?")
    found: dict[tuple[str, str, str], bool] = {}
    for match in EPISODE_RE.finditer(log.read_text(errors="replace")):
        key = (match["house"], match["episode"], match["object"])
        success = match["success"] == "True"
        if key in found and found[key] != success:
            raise SystemExit(f"{log}: episode {key} reported twice with different outcomes")
        found[key] = success
    if not found:
        raise SystemExit(f"{log}: no per-episode outcome lines found; did the run get past setup?")
    return found


def rig_outcomes(results: Path, metric: str) -> dict[tuple[str, str, str], bool]:
    """The same mapping out of a `simulate.py` results file."""
    if not results.exists():
        raise SystemExit(f"{results} does not exist")
    payload = json.loads(results.read_text())
    field = "success_any" if metric == "oracle" else "success"
    found: dict[tuple[str, str, str], bool] = {}
    for record in payload.get("episodes") or ():
        identity = record.get("benchmark_episode")
        if not identity:
            raise SystemExit(
                f"{results}: episode {record.get('episode')} carries no benchmark_episode, so it "
                "cannot be matched to a native run.\n"
                "Run simulate.py against a sim server started with --benchmark; a sampled scene "
                "has no episode to join on."
            )
        key = (str(identity["house"]), str(identity["episode"]), str(identity["object"]))
        if field not in record:
            raise SystemExit(
                f"{results}: no {field!r} field. A results file written before the sim protocol "
                "gained the oracle flag records only at-end success; re-run, or pass "
                "--metric at-end and score the native side the same way."
            )
        found[key] = bool(record[field])
    if not found:
        raise SystemExit(f"{results}: no episodes")
    return found


def report(native: dict, rig: dict, label_a: str, label_b: str) -> int:
    shared = sorted(set(native) & set(rig))
    only_native = sorted(set(native) - set(rig))
    only_rig = sorted(set(rig) - set(native))

    print(f"  {label_a}: {len(native)} episode(s)")
    print(f"  {label_b}: {len(rig)} episode(s)")
    print(f"  in both:  {len(shared)}")

    if not shared:
        print("\n  Nothing in common. Were the two run against the same benchmark?")
        return 2  # VOID: nothing was compared. Saying DIVERGED here asserts a result.

    disagreements = [key for key in shared if native[key] != rig[key]]
    for key in disagreements:
        house, episode, obj = key
        print(f"    DIFFERS  house {house} episode {episode} {obj}: {label_a}={native[key]} {label_b}={rig[key]}")
    if only_native:
        print(f"\n  {len(only_native)} episode(s) only in {label_a}, e.g. {only_native[:3]}")
    if only_rig:
        print(f"  {len(only_rig)} episode(s) only in {label_b}, e.g. {only_rig[:3]}")

    agreed = len(shared) - len(disagreements)
    coverage = 100.0 * len(shared) / max(len(native), 1)
    print(f"\n  agreed on {agreed}/{len(shared)} shared episode(s) -- {coverage:.1f}% of {label_a}'s cell")
    return 1 if disagreements else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("native_log", type=Path, help="eval_stdout.log from a scripts/eval.py cell")
    parser.add_argument("rig_results", type=Path, help="a data/simulate/*.json from droid's simulate.py")
    parser.add_argument(
        "--control",
        type=Path,
        default=None,
        help="a SECOND native log of the same cell. Checked first: if the policy does not "
        "agree with itself, the cross-harness comparison is void rather than failing.",
    )
    parser.add_argument("--metric", choices=("oracle", "at-end"), default="oracle")
    args = parser.parse_args()

    native = native_outcomes(args.native_log)

    if args.control is not None:
        print("same-harness control:")
        control = native_outcomes(args.control)
        shared = set(native) & set(control)
        flipped = [key for key in sorted(shared) if native[key] != control[key]]
        print(f"  {len(shared)} shared, {len(flipped)} flipped between two native runs")
        if not shared:
            print(
                "\n  The two native logs have no episode in common, so nothing was controlled.\n"
                "  --control wants a second run of the SAME cell. VOID, not passing."
            )
            return 2
        if flipped:
            print(
                "\n  The policy does not reproduce itself, so a cross-harness difference cannot be\n"
                "  attributed to the harness. VOID, not failing. Serve openpi with DETERMINISTIC=1\n"
                "  (both arms, never one) or accept a distribution comparison instead."
            )
            return 2
        print("  the policy reproduces itself; the comparison below is meaningful\n")

    print(f"cross-harness, on the {args.metric} metric:")
    rig = rig_outcomes(args.rig_results, args.metric)
    status = report(native, rig, "native", "rig")
    print()
    if status == 2:
        # print + return, not SystemExit(str): a string argument exits 1, which is the
        # DIVERGED code, and collapsing the two is the distinction this exists to make.
        print("VOID: no episode was compared, so neither harness was judged.")
        return 2
    if status:
        raise SystemExit("DIVERGED: the two harnesses did not agree.")
    shared = len(set(native) & set(rig))
    print(f"OK -- the rig and the native harness agree on all {shared} episode(s) they both ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
