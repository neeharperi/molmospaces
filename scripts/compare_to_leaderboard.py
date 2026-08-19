#!/usr/bin/env python
"""Compare our results against the frozen leaderboard snapshot.

    python scripts/compare_to_leaderboard.py \\
        --leaderboard reference/leaderboard_snapshot.csv --runs-dir runs

For each (policy, task) cell present in both the leaderboard snapshot and runs/, computes a
Wilson score 95% interval on our success rate and reports PASS if the leaderboard's point
estimate falls inside it. Also pools the 7 Group B tasks into one aggregate per policy and
compares that against the leaderboard's "MolmoBot Combined" row -- a much tighter interval
over ~7xN episodes, and the load-bearing check per plans/BENCHMARK.md.

With --require-full-matrix, exits non-zero unless all 3 reproduction policies (pi05_droid,
molmoact2_droid, tiptop) have results for all 9 tasks and all pass, plus all 3 Group B
aggregates pass. Without it (the default), reports on whatever cells are currently present --
useful while only some PRs (and thus some policies) have landed.
"""

from __future__ import annotations

import argparse
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
    candidates = sorted((runs_dir / policy / task).glob("*/results.csv")) if (runs_dir / policy / task).exists() else []
    return candidates[-1] if candidates else None


def read_overall(results_csv: Path) -> pd.Series:
    df = pd.read_csv(results_csv, comment="#")
    overall = df[df["category"] == "OVERALL"]
    if overall.empty:
        raise ValueError(f"{results_csv} has no OVERALL row")
    return overall.iloc[0]


def successes_and_total(overall: pd.Series, metric: str) -> tuple[int, int]:
    if metric not in METRIC_TO_COLUMNS:
        raise ValueError(f"Unsupported leaderboard metric '{metric}'; expected one of {list(METRIC_TO_COLUMNS)}")
    s_col, n_col = METRIC_TO_COLUMNS[metric]
    return int(overall[s_col]), int(overall[n_col])


def verdict_row(task: str, policy: str, successes: int, total: int, leaderboard_pct: float) -> dict:
    lo, hi = wilson_interval(successes, total)
    ours_pct = 100.0 * successes / total if total else 0.0
    passed = lo * 100.0 <= leaderboard_pct <= hi * 100.0
    return {
        "task": task,
        "policy": policy,
        "ours_pct": round(ours_pct, 1),
        "n": total,
        "leaderboard_pct": leaderboard_pct,
        "verdict": "PASS" if passed else "FAIL",
    }


def print_table(rows: list[dict]) -> None:
    header = f"{'task':<20}{'policy':<18}{'ours (n)':<16}{'leaderboard':<14}{'verdict'}"
    print(header)
    for r in rows:
        ours = f"{r['ours_pct']}% ({r['n']})"
        print(f"{r['task']:<20}{r['policy']:<18}{ours:<16}{r['leaderboard_pct']}%{'':<7}{r['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leaderboard", type=Path, default=Path("reference/leaderboard_snapshot.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--embodiment", default="DROID", help="Filter the leaderboard snapshot to this embodiment.")
    parser.add_argument("--require-full-matrix", action="store_true", help="Fail unless all 27 required cells + 3 Group B aggregates are present and pass.")
    args = parser.parse_args()

    if not args.leaderboard.exists():
        print(f"{args.leaderboard} does not exist -- capture it first (see docs/eval_reproduction.md).")
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
        per_task_rows.append(verdict_row(task, policy, successes, total, row["success_rate"]))

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
        aggregate_rows.append(verdict_row(GROUP_B_LEADERBOARD_TASK_NAME, policy, s, t, row["success_rate"]))

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

    if args.require_full_matrix:
        expected_cells = len(REPRODUCTION_POLICIES) * len(TASKS)
        present_cells = [r for r in per_task_rows if r["policy"] in REPRODUCTION_POLICIES]
        expected_aggregates = len(REPRODUCTION_POLICIES)
        present_aggregates = [r for r in aggregate_rows if r["policy"] in REPRODUCTION_POLICIES]
        if len(present_cells) < expected_cells or len(present_aggregates) < expected_aggregates:
            print(
                f"\n--require-full-matrix: expected {expected_cells} cells + "
                f"{expected_aggregates} aggregates for {REPRODUCTION_POLICIES}, "
                f"have {len(present_cells)} cells + {len(present_aggregates)} aggregates."
            )
            sys.exit(1)
        if failures:
            sys.exit(1)
    elif failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
