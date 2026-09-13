#!/usr/bin/env python
"""Print the (policy x task) campaign grid: what's done, running, stalled or pending.

    python scripts/campaign_status.py --date 20260828_full

Reads runs/ directly rather than tracking state anywhere, so it is always correct after a
crash, a kill, or a resumed lane -- there is no second source of truth to fall out of sync.

Cell states come from what scripts/eval.py leaves behind, in the order it writes them:

  done     provenance.json exists. eval.py writes it LAST and only on success, so its
           presence is the completion signal -- the same one eval.py's own resume check uses.
  running  eval_stdout.log exists, no provenance.json, and the log was touched recently.
  stalled  same, but the log has not been touched in --stall-minutes. Worth a look: the
           reference campaign lost 8 hours to a lane whose server had died while its client
           retried forever, and TCP reachability alone would not have caught it -- the server
           had come back up. Log silence would have.
  pending  nothing on disk yet.

The stall threshold is per-task, not global: several filament cells legitimately run for
hours between log writes, so a single 45-minute rule would cry wolf on exactly the cells
that take longest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import GROUP_A, GROUP_B, POLICIES, read_overall  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tasks whose per-episode cost is high enough that a long quiet stretch is normal.
SLOW_TASKS = {"PnP-v2", "PnP-NextTo-v2", "PnP-Color-v2", "Pick-v2-filament", "Pick-v2-RandCam"}
SLOW_STALL_MINUTES = 180

_BATCHED = re.compile(r"Batched (\d+) episodes in")

MARK = {"done": "OK", "running": "..", "stalled": "!!", "pending": "--"}


def cell_state(d: Path, task: str, stall_minutes: float) -> tuple[str, str]:
    if not d.exists():
        return "pending", ""
    prov = d / "provenance.json"
    if prov.exists():
        # provenance.json records n_episodes but NOT the success count -- that lives in
        # results.csv's OVERALL row, which is also where compare_to_leaderboard.py reads it
        # from. Use the same source -- and the same by-name reader -- rather than a second one
        # that could disagree. This used to index the row positionally (row[8] for
        # oracle_rate_pct), which silently reports a different metric if eval_to_csv.py ever
        # grows a column. Oracle is what the leaderboard uses.
        try:
            overall = read_overall(d / "results.csv")
            return "done", f"oracle {float(overall['oracle_rate_pct']):.1f}%  n={overall['total']}"
        except Exception:
            pass
        try:
            return "done", f"n={json.loads(prov.read_text()).get('n_episodes')}"
        except Exception:
            return "done", ""
    log = d / "eval_stdout.log"
    if log.exists():
        # Progress for an in-flight cell comes from pipeline.py's own
        # "Batched N episodes in ..." line, emitted once per completed house with the exact
        # episode count. Earlier versions of this tool counted distinct "house H episode E"
        # markers instead, which undercounts badly -- it reported 34 episodes for a cell whose
        # own summary said 79, and produced a 2115 s/episode rate for a cell actually running
        # at ~218. Progress metrics that are wrong in the pessimistic direction are worse than
        # no metric, because they trigger restarts of healthy work.
        #
        # NOTE: this counter is still inert. The pattern itself was broken (`\\d` inside a
        # raw string matches a literal backslash, so it never fired) and is now fixed, but
        # pipeline.py:239 emits the line via worker_logger.debug() while eval_main.py
        # configures INFO -- so it reaches no log, and `grep -c Batched` is 0 across every
        # cell in runs/. Promote that call to info(), or lower the eval log level, to make
        # the "N ep done" readout live; until then it degrades to the quiet-time readout.
        done = 0
        try:
            for m in _BATCHED.finditer(log.read_text(errors="ignore")):
                done += int(m.group(1))
        except Exception:
            pass
        age_min = (time.time() - log.stat().st_mtime) / 60.0
        prog = f"{done} ep done, " if done else ""
        limit = SLOW_STALL_MINUTES if task in SLOW_TASKS else stall_minutes
        state = "running" if age_min < limit else "stalled"
        return state, f"{prog}{age_min:.0f}m quiet"
    return "pending", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="the --date / DATE_TAG the campaign uses")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--stall-minutes", type=float, default=45.0)
    a = ap.parse_args()

    runs = REPO_ROOT / a.runs_dir
    tasks = list(GROUP_A) + list(GROUP_B)
    counts: dict[str, int] = {}

    w = max(len(p) for p in POLICIES) + 1
    print(f"\ncampaign {a.date}   ({len(POLICIES)} policies x {len(tasks)} tasks = "
          f"{len(POLICIES) * len(tasks)} cells)\n")
    print(" " * w + "  ".join(f"{t[:14]:>14}" for t in tasks))
    # Once per cell: an in-flight cell's eval_stdout.log runs to 100+ MB, and this is the
    # command the campaign polls with.
    states = {(p, t): cell_state(runs / p / t / a.date, t, a.stall_minutes)
              for p in POLICIES for t in tasks}

    for policy in POLICIES:
        row = [f"{policy:<{w}}"]
        for task in tasks:
            st = states[(policy, task)][0]
            counts[st] = counts.get(st, 0) + 1
            row.append(f"{MARK[st]:>14}")
        print("  ".join(row))

    print("\n  " + "   ".join(f"{MARK[k]} {k}={counts.get(k, 0)}"
                              for k in ("done", "running", "stalled", "pending")))

    for state, header in (("done", "completed cells:"),
                          ("stalled", "STALLED -- check the lane's server log in runs/_servers/:")):
        rows = [(p, t, detail) for (p, t), (st, detail) in states.items() if st == state]
        if rows:
            print(f"\n  {header}")
            for p, t, detail in rows:
                print(f"    {p:<18} {t:<18} {detail}")
    print()


if __name__ == "__main__":
    main()
