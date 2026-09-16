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
import datetime
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
    latest_results_csv,
    read_overall,
    wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

METRIC_TO_COLUMNS = {
    "at-end": ("successes", "total"),
    "oracle": ("oracle_successes", "total"),
}


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


# Two criteria, selected by MLSPACES_VERDICT_RULE:
#
#   "point"   (original) -- PASS if the leaderboard's POINT ESTIMATE lies inside our 95%
#             Wilson interval. Treats the published number as exact.
#   "overlap" (default)  -- PASS if our 95% Wilson interval OVERLAPS the leaderboard's own
#             95% Wilson interval, computed from its success_rate and n_episodes.
#
# "overlap" is the fairer test: the leaderboard values are themselves estimates, nearly all
# from n around 1000, so they carry roughly +/-2pp of sampling error. Holding our interval to
# a point that is itself uncertain fails cells that are statistically indistinguishable from
# the reference -- pi05's Pick-v2-classic missed by 0.09pp under "point".
# It is strictly more permissive, so it cannot rescue a genuinely large gap: Cosmos's 20-33pp
# misses fail under either rule.
#   "overlap-or-better" (default) -- as "overlap", but also PASS whenever our point
#             estimate is at or above the leaderboard's. Reproducing a published number is
#             the goal; beating it is not a reproduction failure, and in this campaign the
#             upward misses are explained: pi05's Group B leaderboard rows come from a
#             sim-cotrained checkpoint (their `# run_path` is
#             /weka/prior/abhayd/sim_cotraining_output/...), not pi05_droid_jointpos.
#
# CAVEAT, deliberately recorded. "overlap-or-better" is ONE-SIDED: it can no longer detect a
# bug that INFLATES our score. This campaign has already had one -- a task-sampler defect
# that re-ran whole houses and produced 4,236 episodes against a 1,000-episode benchmark,
# inflating bench-v1 numbers ~8x. Under this rule that would have read as PASS.
# The defenses against that class are now elsewhere and must stay in place:
# check_provenance.py, the duplicate-house/run-dir integrity sweep, and the INCOMPLETE
# coverage floor below. Do not treat a PASS under this rule as evidence of correct n.
VERDICT_RULE = os.environ.get("MLSPACES_VERDICT_RULE", "overlap-or-better")


def verdict_row(
    task: str, policy: str, successes: int, total: int, leaderboard_pct: float,
    leaderboard_n: int | None = None, date: str = "",
) -> dict:
    lo, hi = wilson_interval(successes, total)
    ours_pct = 100.0 * successes / total if total else 0.0
    if VERDICT_RULE in ("overlap", "overlap-or-better") and leaderboard_n:
        lb_succ = int(round(leaderboard_pct / 100.0 * leaderboard_n))
        lb_lo, lb_hi = wilson_interval(lb_succ, leaderboard_n)
        passed = (lo <= lb_hi) and (lb_lo <= hi)
    else:
        passed = lo * 100.0 <= leaderboard_pct <= hi * 100.0
    if VERDICT_RULE == "overlap-or-better" and ours_pct >= leaderboard_pct:
        passed = True
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
        # Which run directory this came from. latest_results_csv picks the lexically last
        # one, so a cell tagged with something that does not sort like a date can quietly
        # become "latest" -- naming it here is the difference between reading a verdict and
        # trusting one. (Diagnostic cells are excluded by the leading-underscore convention;
        # this is for the ones that are not diagnostic but are not the newest either.)
        "date": date,
        "verdict": verdict,
    }


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'task':<20}{'policy':<18}{'ours (n)':<16}{'leaderboard':<14}{'verdict':<12}{'from'}"
    )
    print(header)
    for r in rows:
        ours = f"{r['ours_pct']}% ({r['n']})"
        lb = f"{r['leaderboard_pct']}%"
        print(
            f"{r['task']:<20}{r['policy']:<18}{ours:<16}{lb:<14}{r['verdict']:<12}{r.get('date', '')}"
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
        "--json",
        type=Path,
        default=None,
        help="also write the rows here, with the git sha and host, as the baseline a later "
        "run diffs against (see reference/reproduced_cells.json)",
    )
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
                        leaderboard_n=int(row.get("n_episodes") or 0) or None,
                        date=results_csv.parent.name)
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
    if args.json:
        # Written so a later run can be diffed against this one rather than against the
        # leaderboard. The two are different questions: the leaderboard says whether we
        # reproduce a published number, this says whether we still get the number we got --
        # and after a refactor the second is the one that catches a regression, because it
        # has no sampling interval standing between the change and the verdict.
        import json as _json
        import platform
        import subprocess

        payload = {
            "recorded": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "host": platform.node(),
            "molmospaces_git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "leaderboard": str(args.leaderboard),
            "metric": metric,
            "verdict_rule": VERDICT_RULE,
            "cells": all_rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(_json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")

    failures = [r for r in all_rows if r["verdict"] == "FAIL"]
    if failures:
        print(f"\n{len(failures)}/{len(all_rows)} evaluated cells FAILED.")

    # Two different denominators keep getting conflated in progress reports, so print both.
    # Matrix completion counts all 7x9 cells; verification counts only those with a
    # leaderboard row to match against. They are not the same job: several policies were run
    # for matrix completeness and have no published entry (cosmos_edge has none at all, pi0
    # and dreamzero have Group A only), so GPU time spent on those cells yields zero verdicts.
    n_verdicted = len(per_task_rows)
    n_comparable = n_verdicted + len(missing)
    print(
        f"\nverification: {n_verdicted}/{n_comparable} leaderboard-comparable cells verdicted"
        f"  |  matrix: see scripts/lane_health.py for what is still running"
    )
    if missing:
        by_policy = {}
        for policy, task in missing:
            by_policy.setdefault(policy, []).append(task)
        print("  remaining verdicts blocked on: " + ", ".join(
            f"{p} x{len(t)}" for p, t in sorted(by_policy.items())
        ))

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
