#!/usr/bin/env python
"""Compare our results against the frozen leaderboard snapshot.

    python scripts/compare_to_leaderboard.py \\
        --leaderboard reference/leaderboard_snapshot.csv --runs-dir runs

For each (policy, task) cell present in both the leaderboard snapshot and runs/, computes a
Wilson score 95% interval on our success rate and reports PASS if the leaderboard's point
estimate falls inside it. Also pools the 7 Group B tasks into one aggregate per policy and
compares that against the leaderboard's "MolmoBot Combined" row -- a much tighter interval
over ~7xN episodes, and the load-bearing check per plans/BENCHMARK.md.

Two completeness gates, both of which additionally require every evaluated cell to PASS:

  --require-full-matrix   all 3 reproduction policies (pi05_droid, molmoact2_droid, tiptop)
                          x all 9 tasks = 27 cells, plus the 3 Group B aggregates.
  --require-group-b       the same 3 policies x the 7 Group B tasks = 21 cells, plus the 3
                          aggregates. This is the bar for a Group-B-first campaign, and it is
                          not merely a weaker version of the above: Group B is exactly the set
                          of tasks where all three reproduction policies HAVE a leaderboard
                          entry. TiPToP has none for Open-v1/Close-v1 (upstream did not report
                          them either), so those two cells can never produce a verdict for it
                          and --require-full-matrix can never be satisfied for TiPToP at all.

Without either flag (the default), reports on whatever cells are currently present -- useful
while only some policies have landed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (
    GROUP_B,
    GROUP_B_LEADERBOARD_TASK_NAME,
    REPRODUCTION_POLICIES,
    TASKS,
    wilson_interval,
)

METRIC_TO_COLUMNS = {
    "at-end": ("successes", "total"),
    "oracle": ("oracle_successes", "total"),
}


def latest_results_csv(runs_dir: Path, policy: str, task: str) -> Path | None:
    """Newest real results.csv for a cell, ignoring underscore-prefixed date directories.

    A leading underscore marks a directory as not-a-campaign-result. The convention already
    existed at the policy level (`_handshake/`, `_INVALID_*`, `_superseded_*`, which
    check_provenance.py skips by the same rule) but was not applied to the date level, and the
    omission actively corrupted output: `_` is 0x5F, which sorts AFTER digits, so a 60-episode
    A/B arm in `_ab_C_dt100/` beat the real `20260828_full/` cell to `candidates[-1]` and was
    reported as cosmos_edge's Pick-v1.5 verdict -- 26.7% (n=60) in place of 33.8% (n=1000).

    Same failure this project hit with the n=1 handshake cells, which "PASSED" everything
    because a Wilson interval at n=1 spans the range. Diagnostic runs must be inert to the
    comparison, not merely distinguishable by a human reading directory names.
    """
    cell_dir = runs_dir / policy / task
    if not cell_dir.exists():
        return None
    candidates = sorted(
        c for c in cell_dir.glob("*/results.csv") if not c.parent.name.startswith("_")
    )
    return candidates[-1] if candidates else None


def read_overall(results_csv: Path) -> pd.Series:
    df = pd.read_csv(results_csv, comment="#")
    overall = df[df["category"] == "OVERALL"]
    if overall.empty:
        raise ValueError(f"{results_csv} has no OVERALL row")
    return overall.iloc[0]


def successes_and_total(overall: pd.Series, metric: str) -> tuple[int, int]:
    if metric not in METRIC_TO_COLUMNS:
        raise ValueError(
            f"Unsupported leaderboard metric '{metric}'; expected one of {list(METRIC_TO_COLUMNS)}"
        )
    s_col, n_col = METRIC_TO_COLUMNS[metric]
    return int(overall[s_col]), int(overall[n_col])


# A partial cell does not merely lose precision -- it INVERTS the test. The Wilson interval
# on 1 sample spans nearly [0, 1], so it contains every possible leaderboard value and the
# cell "PASSES" whatever the truth is. On 2026-09-05 a GPU3 filament OOM left seven cells
# with partial trajectory sets, two of them n=1, and both were reported as PASS.
# The `_`-prefix skip already guards against A/B arms leaking in; this guards against a
# real cell that simply did not finish.
MIN_COVERAGE_FRACTION = float(os.environ.get("MLSPACES_MIN_COVERAGE", "0.8"))


def verdict_row(
    task: str, policy: str, successes: int, total: int, leaderboard_pct: float,
    leaderboard_n: int | None = None,
) -> dict:
    lo, hi = wilson_interval(successes, total)
    ours_pct = 100.0 * successes / total if total else 0.0
    passed = lo * 100.0 <= leaderboard_pct <= hi * 100.0
    verdict = "PASS" if passed else "FAIL"
    # Refuse to score a cell that did not cover the benchmark. Compared against the
    # leaderboard's own episode count, which is the only per-task size available here.
    if leaderboard_n and total < MIN_COVERAGE_FRACTION * leaderboard_n:
        verdict = "INCOMPLETE"
    return {
        "task": task,
        "policy": policy,
        "ours_pct": round(ours_pct, 1),
        "n": total,
        "leaderboard_pct": leaderboard_pct,
        "verdict": verdict,
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'task':<20}{'policy':<18}{'ours (n)':<16}{'leaderboard':<14}{'verdict'}"
    print(header)
    for r in rows:
        ours = f"{r['ours_pct']}% ({r['n']})"
        print(
            f"{r['task']:<20}{r['policy']:<18}{ours:<16}{r['leaderboard_pct']}%{'':<7}{r['verdict']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--leaderboard", type=Path, default=Path("reference/leaderboard_snapshot.csv")
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--embodiment", default="DROID", help="Filter the leaderboard snapshot to this embodiment."
    )
    gate = parser.add_mutually_exclusive_group()
    gate.add_argument(
        "--require-full-matrix",
        action="store_true",
        help="Fail unless all 27 required cells (3 policies x 9 tasks) + 3 Group B aggregates are present and pass.",
    )
    gate.add_argument(
        "--require-group-b",
        action="store_true",
        help="Fail unless all 21 Group B cells (3 policies x 7 tasks) + 3 Group B aggregates are present and pass.",
    )
    args = parser.parse_args()

    if not args.leaderboard.exists():
        print(
            f"{args.leaderboard} does not exist -- capture it first (see docs/eval_reproduction.md)."
        )
        sys.exit(1)

    snapshot = pd.read_csv(args.leaderboard)
    snapshot = snapshot[snapshot["embodiment"] == args.embodiment]

    per_task_rows = []
    missing = []
    pooled = {}  # policy -> (successes, total)
    pooled_task_count = {}  # policy -> how many of the 7 Group B tasks actually contributed

    for _, row in snapshot.iterrows():
        task, policy, metric = row["task"], row["policy"], row["metric"]
        if task == GROUP_B_LEADERBOARD_TASK_NAME:
            continue  # handled separately, after per-task cells below
        if task not in TASKS:
            continue  # leaderboard has tasks outside our 9-task table (e.g. Pick-v1, excluded)

        results_csv = latest_results_csv(args.runs_dir, policy, task)
        if results_csv is None:
            missing.append((policy, task))
            continue

        overall = read_overall(results_csv)
        successes, total = successes_and_total(overall, metric)
        per_task_rows.append(
            verdict_row(task, policy, successes, total, row["success_rate"],
                        leaderboard_n=int(row.get("n_episodes") or 0) or None)
        )

        if task in GROUP_B:
            s, t = pooled.get(policy, (0, 0))
            pooled[policy] = (s + successes, t + total)
            pooled_task_count[policy] = pooled_task_count.get(policy, 0) + 1

    print_table(per_task_rows)

    # Group B pooled aggregates, compared against the leaderboard's "MolmoBot Combined" row.
    aggregate_rows = []
    partial_aggregates = []
    combined_snapshot = snapshot[snapshot["task"] == GROUP_B_LEADERBOARD_TASK_NAME]
    for _, row in combined_snapshot.iterrows():
        policy = row["policy"]
        if policy not in pooled:
            missing.append((policy, GROUP_B_LEADERBOARD_TASK_NAME))
            continue
        # The pooled aggregate is only comparable to the leaderboard's own "MolmoBot
        # Combined" number when all 7 Group B tasks contributed. Pooling a subset compares
        # apples to oranges -- e.g. one easy task alone reads as a wild FAIL against the
        # 7-task average -- so report those separately rather than emitting a verdict.
        n_tasks = pooled_task_count.get(policy, 0)
        if n_tasks < len(GROUP_B):
            partial_aggregates.append((policy, n_tasks, *pooled[policy], row["success_rate"]))
            continue
        s, t = pooled[policy]
        aggregate_rows.append(
            verdict_row(GROUP_B_LEADERBOARD_TASK_NAME, policy, s, t, row["success_rate"])
        )

    if aggregate_rows:
        print()
        print_table(aggregate_rows)

    if partial_aggregates:
        print(
            f"\nGroup B pooled aggregate not yet comparable for "
            f"{len(partial_aggregates)} policy(ies) -- needs all {len(GROUP_B)} Group B tasks "
            f"(no verdict issued):"
        )
        for policy, n_tasks, s, t, lb in partial_aggregates:
            rate = 100.0 * s / t if t else 0.0
            print(
                f"  {policy}: {n_tasks}/{len(GROUP_B)} tasks so far, "
                f"{rate:.1f}% ({s}/{t}) vs leaderboard {lb}% over all {len(GROUP_B)}"
            )

    if missing:
        print(f"\n{len(missing)} cells missing (no results yet):")
        for policy, task in missing:
            print(f"  {policy}/{task}")

    all_rows = per_task_rows + aggregate_rows
    failures = [r for r in all_rows if r["verdict"] == "FAIL"]
    if failures:
        print(f"\n{len(failures)}/{len(all_rows)} evaluated cells FAILED.")

    if args.require_full_matrix or args.require_group_b:
        # The only difference between the two gates is which task set has to be complete; the
        # Group B pooled aggregate is required either way, since it is the load-bearing check.
        if args.require_full_matrix:
            flag, required_tasks = "--require-full-matrix", tuple(TASKS)
        else:
            flag, required_tasks = "--require-group-b", GROUP_B
        expected_cells = len(REPRODUCTION_POLICIES) * len(required_tasks)
        present_cells = [
            r
            for r in per_task_rows
            if r["policy"] in REPRODUCTION_POLICIES and r["task"] in required_tasks
        ]
        expected_aggregates = len(REPRODUCTION_POLICIES)
        present_aggregates = [r for r in aggregate_rows if r["policy"] in REPRODUCTION_POLICIES]
        if len(present_cells) < expected_cells or len(present_aggregates) < expected_aggregates:
            print(
                f"\n{flag}: expected {expected_cells} cells over {len(required_tasks)} tasks + "
                f"{expected_aggregates} aggregates for {REPRODUCTION_POLICIES}, "
                f"have {len(present_cells)} cells + {len(present_aggregates)} aggregates."
            )
            sys.exit(1)
        # Only the cells this gate actually requires should be able to fail it; a FAIL on an
        # out-of-scope cell (an informational policy, or Open-v1 during a Group B campaign)
        # is reported above but must not sink the gate.
        gating_failures = [
            r
            for r in failures
            if r["policy"] in REPRODUCTION_POLICIES
            and r["task"] in (*required_tasks, GROUP_B_LEADERBOARD_TASK_NAME)
        ]
        if gating_failures:
            sys.exit(1)
    elif failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
