#!/usr/bin/env python3
"""What a running cell has covered, and what its number would be on the full category mix.

    python scripts/cell_progress.py --policy pi05_droid --task Open-v1
    python scripts/cell_progress.py --policy pi05_droid --task Open-v1 --date full20260916

A full-coverage bench-v1 cell takes tens of minutes to an hour here, and for all of that time
the only number available is the running raw rate -- which is misleading in a specific,
predictable way. `eval_main.py` dispatches whole *houses*, so a half-finished cell is not a
random half: it is whichever categories happen to live in the houses that finished. Open-v1's
13 object categories are wildly uneven (205 cabinets, 2 safes), so a prefix can sit several
points off the cell's eventual value purely from mix.

So this prints two things:

  * **coverage per category** -- how many of each the cell has done, out of how many exist.
    A category at 0% is contributing nothing to the running rate and will drag it toward its
    own value when it arrives.
  * **the rate reweighted to the benchmark's full mix**, over the categories with at least
    one completed episode. This is the cell's number *if* the covered categories are
    representative of themselves, which is a far weaker assumption than the raw prefix rate
    makes, and it is the one worth quoting while a cell is in flight.

**Neither is a verdict, and `scripts/compare_to_leaderboard.py` will still refuse the cell
until it is complete.** That refusal is right: this reweighting cannot speak for a category
with no episodes in it. Where it earns its keep is the other direction -- when the reweighted
number and the raw rate agree, the prefix is not mix-skewed and a gap against the leaderboard
is real rather than an artifact, hours before the cell can say so itself.

Reads only the cell's eval_stdout.log and the benchmark JSON, so it needs no molmo_spaces
import, no server, and no GPU -- and it works on a cell that is still running.

The complement is `scripts/category_mix_check.py`, which goes the other way: it reweights the
LEADERBOARD's per-category rates by our counts, and needs the per-(task, policy) leaderboard
CSV to do it.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_common import TASKS, resolve_benchmark_dir, wilson_interval  # noqa: E402

# pipeline.py's per-episode completion line. The object name is the benchmark's
# pickup_obj_name, which is what makes the join to the benchmark's categories possible
# without re-reading the run's own outputs.
DONE = re.compile(r"object (\S+) completed with success=(True|False)")


def category(raw: str) -> str:
    """The object's category: the leading alphabetic run of its asset name, lowercased.

    Deliberately the source benchmark_metadata.json's convention ('chestofdrawers', not
    'ChestOfDrawers') rather than eval_to_csv's _simplify, so the counts printed here can be
    read straight against that file. Nothing here needs the prettier labels.
    """
    cleaned = "".join(c if c.isalpha() else " " for c in raw).strip()
    return (cleaned.split()[0] if cleaned else "unknown").lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--date", default=None, help="cell date tag; the newest cell by default")
    ap.add_argument("--runs-dir", type=pathlib.Path, default=pathlib.Path("runs"))
    args = ap.parse_args()

    cells = args.runs_dir / args.policy / args.task
    if args.date:
        log = cells / args.date / "eval_stdout.log"
    else:
        # Newest by mtime, and `_`-prefixed cells are included: an in-flight probe run is
        # exactly the thing someone wants this for.
        found = sorted(cells.glob("*/eval_stdout.log"), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit(f"no eval_stdout.log under {cells}")
        log = found[-1]
    if not log.exists():
        raise SystemExit(f"{log} does not exist")

    benchmark = resolve_benchmark_dir(TASKS[args.task]) / "benchmark.json"
    episodes = json.loads(benchmark.read_text())
    episodes = episodes["episodes"] if isinstance(episodes, dict) else episodes
    full = collections.Counter(category((e.get("task") or {}).get("pickup_obj_name") or "unknown") for e in episodes)

    done: collections.Counter = collections.Counter()
    won: collections.Counter = collections.Counter()
    for match in DONE.finditer(log.read_text(errors="replace")):
        key = category(match.group(1))
        done[key] += 1
        won[key] += match.group(2) == "True"

    n, wins = sum(done.values()), sum(won.values())
    low, high = wilson_interval(wins, n) if n else (0.0, 0.0)
    print(f"{args.policy} / {args.task}   {log.parent.name}")
    print(f"  {n}/{len(episodes)} episodes done, {wins} success -- raw rate {100 * wins / max(n, 1):.1f}%"
          f"  95% CI [{100 * low:.1f}, {100 * high:.1f}]\n")
    print(f"  {'category':<18} {'done':>5} {'of':>5} {'cover':>6} {'rate':>7}")
    for key, total in full.most_common():
        did = done.get(key, 0)
        rate = f"{100 * won[key] / did:.1f}%" if did else "-"
        print(f"  {key:<18} {did:>5} {total:>5} {100 * did / total:>5.0f}% {rate:>7}")

    weight = sum(full[k] for k in done if done[k])
    if not weight:
        print("\n  nothing completed yet.")
        return 0
    reweighted = sum(full[k] * (won[k] / done[k]) for k in done if done[k]) / weight
    missing = sorted(k for k in full if not done.get(k))
    print(f"\n  reweighted to the full mix: {100 * reweighted:.1f}%"
          f"  (over the {weight} episodes in the {len(full) - len(missing)} covered categories)")
    if missing:
        print(f"  no episodes yet in: {', '.join(missing)}"
              f" -- {sum(full[k] for k in missing)} episodes the reweighting cannot speak for")
    if n < len(episodes):
        print("\n  Not a verdict: compare_to_leaderboard.py refuses an incomplete cell, correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
