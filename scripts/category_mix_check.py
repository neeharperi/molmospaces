#!/usr/bin/env python
"""Explain a leaderboard mismatch that is really a category-coverage artifact.

    python scripts/category_mix_check.py --policy pi05_droid --task Open-v1 \
        --leaderboard-csv /tmp/pi05_ms_open.csv

`--max_episodes N` selects whole *houses* from the first N raw episodes, so a truncated run
can cover only a fraction of a benchmark's object categories -- and if the categories it
misses are the easy ones, the overall rate lands far below the leaderboard's even when every
category we did sample agrees with it. That produces a FAIL in
`scripts/compare_to_leaderboard.py` that is a sampling artifact, not a policy or integration
defect. Open-v1 is the known case: `--max_episodes 50` yields 5 of 13 categories and misses
`Stand` and `Chestofdrawers` (204 leaderboard episodes each, and among the highest-scoring),
which alone accounts for the entire apparent gap for both pi05_droid and molmoact2_droid.

This script reweights the leaderboard's own per-category rates by OUR per-category episode
counts, giving the number the leaderboard would have reported on the exact category mix we
actually evaluated. If our result brackets *that*, the cells agree and the run is simply
under-covered -- re-run without `--max_episodes` for a real verdict rather than reading the
raw FAIL as a defect.

The per-(task, policy) leaderboard CSVs this needs are the same ones
`reference/leaderboard_snapshot.csv` was built from; see `reference/README.md` for the URL
pattern (`https://molmospaces.allen.ai/benchmark/data/<task_slug>/<policy_slug>.csv`).
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import wilson_interval

MIN_LARGE_CATEGORY = 50  # only call out unsampled categories big enough to move the overall


def load_per_category(path: Path) -> dict[str, tuple[int, int]]:
    """category -> (oracle_successes, total), skipping the OVERALL row and '#' preamble."""
    lines = [ln for ln in path.read_text().splitlines(keepends=True) if not ln.startswith("#")]
    out = {}
    for row in csv.DictReader(io.StringIO("".join(lines))):
        category = row["category"]
        if category == "OVERALL":
            continue
        out[category] = (int(row["oracle_successes"]), int(row["total"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--leaderboard-csv", type=Path, required=True, help="Per-(task,policy) leaderboard CSV.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    cells = sorted((args.runs_dir / args.policy / args.task).glob("*/results.csv"))
    if not cells:
        print(f"No results.csv under {args.runs_dir / args.policy / args.task}")
        sys.exit(1)
    ours = load_per_category(cells[-1])
    leaderboard = load_per_category(args.leaderboard_csv)

    our_successes = sum(s for s, _ in ours.values())
    our_total = sum(n for _, n in ours.values())
    lo, hi = (100.0 * v for v in wilson_interval(our_successes, our_total))

    lb_successes = sum(s for s, _ in leaderboard.values())
    lb_total = sum(n for _, n in leaderboard.values())
    lb_pct = 100.0 * lb_successes / lb_total if lb_total else 0.0

    # The leaderboard's own per-category rates, weighted by the episode counts WE evaluated.
    reweighted = sum(
        (leaderboard[c][0] / leaderboard[c][1]) * n
        for c, (_, n) in ours.items()
        if c in leaderboard and leaderboard[c][1] > 0
    )
    reweighted_pct = 100.0 * reweighted / our_total if our_total else 0.0

    def verdict(x: float) -> str:
        return "INSIDE our 95% CI" if lo <= x <= hi else "OUTSIDE our 95% CI"

    print(f"{args.policy} / {args.task}   (results: {cells[-1]})")
    print(f"  ours                                      {100.0 * our_successes / our_total:6.2f}%  "
          f"({our_successes}/{our_total})   Wilson95 [{lo:.2f}%, {hi:.2f}%]")
    print(f"  leaderboard, all {len(leaderboard):2d} categories       {lb_pct:6.2f}%  "
          f"({lb_successes}/{lb_total})   -> {verdict(lb_pct)}")
    print(f"  leaderboard reweighted to OUR mix         {reweighted_pct:6.2f}%"
          f"{'':<20}-> {verdict(reweighted_pct)}")

    print(f"\n  categories: we sampled {len(ours)} of {len(leaderboard)} present on the leaderboard")
    unsampled = sorted(
        (c for c in leaderboard if c not in ours and leaderboard[c][1] >= MIN_LARGE_CATEGORY),
        key=lambda c: -leaderboard[c][1],
    )
    if unsampled:
        print(f"  large categories never sampled (>= {MIN_LARGE_CATEGORY} leaderboard episodes):")
        for c in unsampled:
            s, n = leaderboard[c]
            print(f"    {c:<18} n={n:<5} leaderboard rate={100.0 * s / n:5.1f}%")
    if lo <= reweighted_pct <= hi and not (lo <= lb_pct <= hi):
        print(
            "\n  => Our result agrees with the leaderboard on the category mix we actually\n"
            "     evaluated; the raw gap is a coverage artifact. Re-run without\n"
            "     --max_episodes for a verdict that can be compared directly."
        )


if __name__ == "__main__":
    main()
