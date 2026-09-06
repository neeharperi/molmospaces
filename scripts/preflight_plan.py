#!/usr/bin/env python
"""Choose each policy's two pre-flight cells, and print them as runnable commands.

    python scripts/preflight_plan.py

Before committing days of GPU time to the full 63-cell matrix, every policy runs two cells at
FULL coverage and is checked against the leaderboard. The point is to catch a mis-wired policy
-- wrong camera mapping, wrong control rate, wrong action space -- while it is still cheap.

Selection rule:

1. The CHOSEN pair is Pick-v2-classic + PnP-NextTo-v2. It covers both renderers -- classic and
   filament are different MuJoCo builds selected by which conda env is active, so a pair drawn
   from one renderer leaves the other half unvalidated -- and PnP-NextTo-v2 (n=322) is by far
   the cheapest filament cell.
2. A pre-flight cell is only useful if the leaderboard has an entry for it: without a reference
   number it tells you the cell ran, not that the integration is right. So where a policy has
   no entry for a chosen task, fall back to the cheapest cells it DOES have an entry for,
   preferring one per renderer, ranking within a renderer by episode count. Filament tasks run
   single-worker and are roughly an order of magnitude slower per episode, so the two renderers
   are never ranked against each other on episode count alone.

In practice only pi0 and DreamZero fall back -- their sole leaderboard entries are Open-v1 and
Close-v1, both classic. That leaves those two lanes with no filament validation before the
full run, which is a real gap and is reported rather than papered over.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import POLICIES, RENDERER_TO_CONDA_ENV, TASKS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
LEADERBOARD = REPO_ROOT / "reference/leaderboard_snapshot.csv"

# Relative per-episode cost, used only to rank candidates within a renderer. Filament tasks
# run at --num_workers 1 (4 concurrent Vulkan contexts exhaust GPU handles), which is why they
# are never compared against classic tasks on episode count alone.
FILAMENT_WORKERS, CLASSIC_WORKERS = 1, 4

# The chosen pre-flight pair, one per renderer. Used wherever the policy has a leaderboard
# entry for it; policies that don't fall back to the cheapest comparable cells they have.
CHOSEN = ("Pick-v2-classic", "PnP-NextTo-v2")


def leaderboard_cells() -> dict[tuple[str, str], tuple[float, int]]:
    out = {}
    with LEADERBOARD.open() as f:
        for r in csv.DictReader(f):
            if r["task"] in TASKS:
                out[(r["policy"], r["task"])] = (float(r["success_rate"]), int(r["n_episodes"]))
    return out


def main() -> None:
    lb = leaderboard_cells()
    print(f"\npre-flight plan  ({len(POLICIES)} policies, full coverage, no --max_episodes)\n")
    gaps = []
    for policy in POLICIES:
        cands = [(t, TASKS[t].renderer, n, rate)
                 for (p, t), (rate, n) in lb.items() if p == policy]
        by_task = {c[0]: c for c in cands}

        picks = [by_task[t] for t in CHOSEN if t in by_task]
        if len(picks) < 2:
            # Fall back for whichever renderer the chosen pair could not supply.
            have = {c[1] for c in picks}
            for renderer in ("classic", "filament"):
                if renderer in have:
                    continue
                same = sorted([c for c in cands if c[1] == renderer and c not in picks],
                              key=lambda c: c[2])
                if same:
                    picks.append(same[0])
            # Still short: the policy only has entries for one renderer. Take the two cheapest
            # there rather than running a single cell -- one number is a coin flip, two in the
            # same direction is a signal.
            if len(picks) < 2:
                same = sorted([c for c in cands if c not in picks], key=lambda c: c[2])
                picks += same[: 2 - len(picks)]
        picks.sort(key=lambda c: c[1])
        if not picks:
            print(f"  {policy:<18} NO comparable cell on the leaderboard -- cannot pre-flight")
            gaps.append((policy, "no leaderboard entry at all"))
            continue
        covered = {p[1] for p in picks}
        if covered != {"classic", "filament"}:
            gaps.append((policy, f"only the {'/'.join(sorted(covered))} renderer is validated"))
        print(f"  {policy}")
        for task, renderer, n, rate in picks:
            env = RENDERER_TO_CONDA_ENV[renderer]
            workers = CLASSIC_WORKERS if renderer == "classic" else FILAMENT_WORKERS
            tag = "chosen  " if task in CHOSEN else "fallback"
            print(f"    {task:<18} {renderer:<9} n={n:<5} leaderboard={rate:>6.2f}%  "
                  f"[{tag}] env={env}")
            print(f"      conda activate {env} && python scripts/eval.py "
                  f"--policy {policy} --task {task} --num_workers {workers} --date $DATE_TAG")
    if gaps:
        print("\n  coverage gaps (stated, not silently accepted):")
        for policy, why in gaps:
            print(f"    {policy:<18} {why}")
    print()


if __name__ == "__main__":
    main()
