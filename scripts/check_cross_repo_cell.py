"""Assert one (policy, task) cell produces identical per-episode outcomes in two environments.

    python scripts/check_cross_repo_cell.py runs/_xrepo/cosmos_edge_mlspaces runs/_xrepo/cosmos_edge_polaris

BENCHMARK.md's environment-parity criteria end with one that scripts/check_env_parity.py cannot
satisfy: "At least one (policy, task) cell run in both repos' environments produces identical
per-episode outcomes given identical seeds. Matching package lists don't prove matching
behaviour; this does."

This is that check. It reads each run's eval_stdout.log, extracts the per-episode outcome lines
that pipeline.py emits, and requires the two sets to be identical -- same episodes evaluated,
same success flag on each. Anything else (a different episode set, a flipped outcome, a
different count) is a behavioural divergence between the two environments, and the interesting
case is precisely when the package sets already agree: then something outside the lock is
leaking in -- driver version, MUJOCO_GL backend, asset version, an env var.

Both runs must use the same seed (scripts/eval.py hardcodes 42), the same benchmark, and the
same --max_episodes, or the comparison is meaningless. This is a smoke test rather than an
acceptance cell, so --max_episodes IS appropriate here, unlike for any number compared to the
leaderboard.

**The policy itself must be deterministic, or this check measures nothing.** Learned policy
servers are not deterministic by default. Cosmos' `action_policy_server_robolab` draws a fresh
random seed per inference call unless launched with `--deterministic-seed` (its `_next_seed()`
returns `self._rng.integers(...)` when `deterministic_seed` is False, and its startup line
reports `deterministic_seed=False`). Run the same-environment control -- the identical cell
twice against the *same* server -- before reading any cross-environment difference as
environmental. If the control also differs, the policy is stochastic and the comparison is
void, not failing.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# pipeline.py:1056, e.g.
#   Worker 1 house 10 episode 1 object tissuepaper_1f28...4_1_0_4 completed with success=True
EPISODE_RE = re.compile(
    r"house (?P<house>\d+) episode (?P<episode>\d+) object (?P<object>\S+) "
    r"completed with success=(?P<success>True|False)"
)


def outcomes(run_dir: Path) -> dict[tuple[str, str, str], bool]:
    """{(house, episode, object): success} for one run directory."""
    log = run_dir / "eval_stdout.log"
    if not log.exists():
        raise SystemExit(f"{log} does not exist -- was this cell actually run?")
    found: dict[tuple[str, str, str], bool] = {}
    for m in EPISODE_RE.finditer(log.read_text(errors="replace")):
        key = (m["house"], m["episode"], m["object"])
        success = m["success"] == "True"
        # A repeated key would make the comparison ambiguous; surface it rather than
        # silently keeping the last one.
        if key in found and found[key] != success:
            raise SystemExit(f"{log}: episode {key} reported twice with different outcomes")
        found[key] = success
    if not found:
        raise SystemExit(f"{log}: no per-episode outcome lines found; did the run get past setup?")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_a", type=Path, help="run directory from environment A")
    ap.add_argument("run_b", type=Path, help="run directory from environment B")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    label_a = args.label_a or args.run_a.name
    label_b = args.label_b or args.run_b.name
    a, b = outcomes(args.run_a), outcomes(args.run_b)

    print(f"{label_a}: {len(a)} episodes, {sum(a.values())} successes")
    print(f"{label_b}: {len(b)} episodes, {sum(b.values())} successes")

    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    flipped = sorted(k for k in set(a) & set(b) if a[k] != b[k])

    for key in only_a:
        print(f"  only in {label_a}: house {key[0]} episode {key[1]} object {key[2]}")
    for key in only_b:
        print(f"  only in {label_b}: house {key[0]} episode {key[1]} object {key[2]}")
    for key in flipped:
        print(
            f"  OUTCOME DIFFERS: house {key[0]} episode {key[1]} object {key[2]}: "
            f"{label_a}={a[key]} {label_b}={b[key]}"
        )

    print()
    # A strict subset on one side is almost always an unfinished or killed run rather than a
    # divergence -- worth saying, because "only in X" for every missing episode otherwise reads
    # like a dramatic disagreement.
    if (only_a and not only_b and not flipped) or (only_b and not only_a and not flipped):
        fewer = label_b if only_a else label_a
        print(
            f"NOTE: {fewer} evaluated a strict subset of the other run's episodes and every "
            f"shared episode agreed. That is the signature of an incomplete run, not a "
            f"divergence -- check whether it finished before trusting this verdict."
        )
    if only_a or only_b or flipped:
        print(
            f"cross-repo cell MISMATCH: {len(only_a)} only-in-{label_a}, "
            f"{len(only_b)} only-in-{label_b}, {len(flipped)} flipped outcomes.\n"
            f"If the package sets already agree (scripts/check_env_parity.py), the cause is "
            f"outside the environment: driver, MUJOCO_GL backend, asset version, or an env var."
        )
        return 1
    print(f"cross-repo cell IDENTICAL: {len(a)} episodes, same outcome on every one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
