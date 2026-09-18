#!/usr/bin/env python
"""Two harnesses, one episode draw: do their success *distributions* agree?

    python scripts/compare_harness_distributions.py \
        --simulation runs/pi05_droid/Close-v1/20260917_xh/results.csv \
        --deployment ~/Workspace/droid/data/xharness_20260917/rig/pi05_Close-v1/*.json \
        --label pi05_droid/Close-v1

`compare_harnesses.py` answers the sharper question -- did the two agree episode for
episode -- and here it can never issue a verdict. Its `--control` gate is not pedantry:
four runs of the identical 5-episode Close-v1 cell in the *same* harness, minutes apart
with the same benchmark hash, gave 5/5, 3/5, 3/5 and 4/5, with three of the five episodes
flipping. A `molmospaces-bench-v1` EpisodeSpec carries `seed: null`, so the scene is not
reproducible across processes and no amount of policy determinism fixes it.

So this compares what is comparable: two rates, two Wilson intervals, and whether they
overlap. It also prints the episode-level agreement, clearly marked as descriptive --
with a same-harness flip rate that high, an episode-level disagreement count is a
measurement of the benchmark's own noise, not of the harnesses.

The deployment side takes several JSONs because a cell is split by `--seed` across two
scene servers (the server is single-client, so that is the only parallelism there is).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_common import wilson_interval  # noqa: E402


def simulation_side(results_csv: pathlib.Path) -> tuple[int, int, dict]:
    rows = [r for r in csv.DictReader(l for l in open(results_csv) if not l.startswith("#"))]
    overall = [r for r in rows if r["category"] == "OVERALL"]
    if not overall:
        raise SystemExit(f"{results_csv}: no OVERALL row")
    o = overall[0]
    return int(o["oracle_successes"]), int(o["total"]), {}


def deployment_side(paths: list[str]) -> tuple[int, int, dict]:
    """success_any over every episode in every results file, plus the join keys."""
    successes = total = 0
    clamped = ticks = 0
    aborted: list = []
    outcomes: dict[tuple[str, str, str], bool] = {}
    seen_files = 0
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            payload = json.loads(pathlib.Path(path).read_text())
            seen_files += 1
            for record in payload.get("episodes") or ():
                if "success_any" not in record:
                    raise SystemExit(
                        f"{path}: episode {record.get('episode')} has no success_any. "
                        "That file predates the oracle flag; re-run it."
                    )
                steps = int(record.get("steps") or 0)
                if steps == 0:
                    # An episode that never stepped is not a policy failure, and counting
                    # it as one silently depresses the rate. It happened: a whole cell
                    # lost 20 of 50 to "giving up after 10 reconnections" while the server
                    # was answering 200 OK the whole time, and the naive rate read 14.0%
                    # where the episodes that ran gave 23.3%.
                    aborted.append((record.get("seed"), record.get("ended")))
                    continue
                total += 1
                successes += bool(record["success_any"])
                clamped += int(record.get("ticks_clamped_by_limiter") or 0)
                ticks += steps
                ident = record.get("benchmark_episode")
                if ident:
                    key = (str(ident["house"]), str(ident["episode"]), str(ident["object"]))
                    outcomes[key] = bool(record["success_any"])
    if not seen_files:
        raise SystemExit(f"no results files matched: {paths}")
    return (
        successes,
        total,
        {"outcomes": outcomes, "clamped": clamped, "ticks": ticks, "aborted": aborted},
    )


def native_outcomes(log: pathlib.Path) -> dict[tuple[str, str, str], bool]:
    import re

    pattern = re.compile(
        r"house (?P<house>\d+) episode (?P<episode>\d+) object (?P<object>\S+) "
        r"completed with success=(?P<success>True|False)"
    )
    found = {}
    for m in pattern.finditer(log.read_text(errors="replace")):
        found[(m["house"], m["episode"], m["object"])] = m["success"] == "True"
    return found


def band(successes: int, total: int) -> str:
    lo, hi = wilson_interval(successes, total)
    return f"{100.0 * successes / total:5.1f}%  ({successes}/{total})  Wilson95 [{100*lo:5.1f}, {100*hi:5.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--simulation", required=True, help="results.csv from a scripts/eval.py cell")
    ap.add_argument(
        "--deployment", required=True, nargs="+", help="simulate.py results JSON(s); globs fine"
    )
    ap.add_argument("--label", default="")
    ap.add_argument("--leaderboard-pct", type=float, default=None)
    args = ap.parse_args()

    sim_s, sim_n, _ = simulation_side(pathlib.Path(args.simulation))
    dep_s, dep_n, dep_extra = deployment_side(args.deployment)
    dep_outcomes = dep_extra["outcomes"]

    ab = dep_extra["aborted"]
    if ab:
        import collections

        why = collections.Counter(reason for _, reason in ab)
        print(
            f"\n  !! {len(ab)} deployment episode(s) never stepped and are EXCLUDED from the rate."
        )
        for reason, n in why.most_common():
            print(f"     {n:3d} x {reason}")
        print("     An episode that never ran is not a policy failure. Fix the cause and re-run;")
        print("     the rate below is over the episodes that did run, so n is short.")

    print(f"\n{args.label or 'cell'}   metric: success_any (oracle)")
    print(f"  simulation harness   {band(sim_s, sim_n)}")
    print(f"  deployment harness   {band(dep_s, dep_n)}")
    if dep_extra["ticks"]:
        pct = 100.0 * dep_extra["clamped"] / dep_extra["ticks"]
        # The limiter scales a whole action vector uniformly, so a clamped tick keeps its
        # direction and loses speed -- but over 500 ticks that is a different trajectory,
        # not the same one slower. sim.sh already runs --benchmark cells at the policy's
        # nominal 0.2 rad for exactly this reason; anything much above a few percent means
        # the cell is an approximation of the policy rather than the policy.
        flag = "" if pct < 5 else "   <-- above 5%, read this before the rates"
        print(
            f"  deployment limiter   {dep_extra['clamped']}/{dep_extra['ticks']} ticks scaled down ({pct:.1f}%){flag}"
        )

    slo, shi = wilson_interval(sim_s, sim_n)
    dlo, dhi = wilson_interval(dep_s, dep_n)
    overlap = max(slo, dlo) <= min(shi, dhi)
    print(
        f"\n  intervals {'OVERLAP -- consistent' if overlap else 'are DISJOINT -- the harnesses disagree'}"
    )
    if args.leaderboard_pct is not None:
        for name, lo, hi in (("simulation", slo, shi), ("deployment", dlo, dhi)):
            inside = lo * 100 <= args.leaderboard_pct <= hi * 100
            print(
                f"  leaderboard {args.leaderboard_pct:.2f}% is "
                f"{'inside' if inside else 'OUTSIDE'} the {name} interval"
            )

    # Descriptive only -- see the module docstring.
    log = pathlib.Path(args.simulation).parent / "eval_stdout.log"
    if log.exists() and dep_outcomes:
        native = native_outcomes(log)
        shared = sorted(set(native) & set(dep_outcomes))
        if shared:
            agree = sum(native[k] == dep_outcomes[k] for k in shared)
            print(
                f"\n  episode-level (DESCRIPTIVE, not a verdict): agreed on {agree}/{len(shared)}"
                f" shared episode(s)"
            )
            print(
                f"  a same-harness control on this benchmark flips ~3 of 5, so anything"
                f" near {len(shared)*0.6:.0f}/{len(shared)} is indistinguishable from noise"
            )
        else:
            print("\n  no shared episodes to join on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
