#!/usr/bin/env python3
"""Per-cell health for the running campaign lanes.

Exists because three separate silent degradations cost this campaign days apiece before
anything in the logs made them visible:

  * tiptop/PnP-NextTo-v2  -- workers 0-2 wedged on an unbounded recv(); worker 3 ran the cell
                             alone for 29h at 1.06 houses/hr against 23-30 for the same policy
  * cosmos_edge, cosmos_nano -- every filament cell ran single-worker because a relaunch did
                             not export FILAMENT_WORKERS, at 2.4-4x below rate
  * cosmos_nano json A/B  -- both arms hit the campaign server, so the experiment was a null
                             by construction

All three were visible in the data the whole time. None raised an error. The two signals that
would have caught them are worker balance and rate-against-baseline, so print both.

Usage:  python3 scripts/lane_health.py [--date 20260828_full]
"""
import argparse
import os
import re
import subprocess
import time

WORKER_RE = re.compile(r"Worker (\d+).*pipeline\.py:1003")


def cells_running():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    cells = []
    for line in out.splitlines():
        if "eval.py --policy" not in line or "--max_episodes" in line:
            continue  # A/B arms carry --max_episodes and are not campaign cells
        m = re.search(r"--policy (\S+) --task (\S+).*--num_workers (\d+)", line)
        if m:
            cells.append((m.group(1), m.group(2), int(m.group(3))))
    return cells


def h5_stats(d):
    ts = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".h5"):
                try:
                    ts.append(os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
    return sorted(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260828_full")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    running = cells_running()
    if not running:
        print("no campaign cells running")
        return

    print(f"{'cell':<38} {'wrk':<4} {'houses':<7} {'h/hr':<7} {'stale':<8} balance")
    for policy, task, nworkers in sorted(running):
        d = os.path.join(args.runs_dir, policy, task, args.date)
        ts = h5_stats(d)
        n = len(ts)
        rate = "-"
        if n >= 2:
            hours = (ts[-1] - ts[0]) / 3600.0
            if hours > 0.01:
                rate = f"{n / hours:.1f}"
        stale = f"{(time.time() - ts[-1]) / 3600:.1f}h" if ts else "-"

        log = os.path.join(d, "eval_stdout.log")
        counts = {}
        try:
            with open(log, errors="ignore") as fh:
                for line in fh:
                    m = WORKER_RE.search(line)
                    if m:
                        counts[int(m.group(1))] = counts.get(int(m.group(1)), 0) + 1
        except OSError:
            pass

        bal = " ".join(f"w{w}:{c}" for w, c in sorted(counts.items())) or "-"
        flags = []
        # A wedged worker shows up as a share far below its fair split. Only meaningful once
        # the cell has enough episodes that the split should have evened out.
        if counts and sum(counts.values()) >= 4 * nworkers:
            if len(counts) < nworkers:
                flags.append(f"ONLY {len(counts)}/{nworkers} WORKERS")
            elif max(counts.values()) > 4 * max(1, min(counts.values())):
                flags.append("IMBALANCED")
        if ts and (time.time() - ts[-1]) > 3 * 3600:
            flags.append("NO H5 >3h")
        print(
            f"{policy + '/' + task:<38} {nworkers:<4} {n:<7} {rate:<7} {stale:<8} {bal}"
            + (("   ** " + ", ".join(flags)) if flags else "")
        )


if __name__ == "__main__":
    main()
