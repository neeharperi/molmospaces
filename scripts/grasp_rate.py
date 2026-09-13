#!/usr/bin/env python3
"""Grasp-command rate for a run directory -- the leading indicator for a base->DROID fine-tune.

Why this metric
---------------
`docs/eval_reproduction.md:3706` established that the entire 17.3pp gap between `pi05_base`
(6.0% on Pick-v1.5) and the reference `pi05_droid_jointpos` (23.3%) is very nearly explained by
one thing: base almost never closes the gripper. Over 30 episodes each,

    reference : 24/30 episodes (80.0%) ever command a grasp; 144/992 steps gripper-high
    pi05_base :  2/30 episodes ( 6.7%) ever command a grasp;  31/1088 steps gripper-high

A policy that never closes the gripper cannot complete a pick, so this is not a correlate of
the gap -- it is close to the whole of it.

It matters operationally because the training loss *cannot* gate this run. The loss is averaged
over a 32-dim action vector of which 24 dims are zero padding and 7 are joints; the gripper is a
single dimension, ~3% of the signal. Base posts an excellent loss (0.041) while being wrong on
the one channel that decides success. So the fine-tune ladder is gated on sim success, and this
number is the cheap leading indicator that should climb toward 80% well *before* the success
rate converges -- letting a doomed run be stopped at day 2 rather than day 13.

Usage
-----
    python scripts/grasp_rate.py runs/pi05_droid/Pick-v1.5/20260828_full
    python scripts/grasp_rate.py runs/pi05_droid/Pick-v1.5/gate_x_10000 --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

# The harness emits a binary gripper command (grasping_type="binary",
# grasping_threshold=0.5), so the decoded value is 0.0 or 255.0. Anything above this counts as
# a commanded grasp; the threshold is deliberately loose so a continuous-gripper policy is
# still measured sensibly rather than silently reading as all-zero.
GRIPPER_HIGH = 1.0


def decode_steps(h5file: h5py.File, traj: str) -> list[dict]:
    """Decode one trajectory's commanded actions.

    Same encoding as scripts/check_handshake.py:34 -- one UTF-8 JSON object per step, null-padded
    to a fixed row width. Empty objects (step 0 carries `{}`) are skipped.
    """
    raw = np.array(h5file[f"{traj}/actions/commanded_action"])
    out = []
    for row in raw:
        b = bytes(row).rstrip(b"\x00")
        if not b:
            continue
        d = json.loads(b.decode("utf-8"))
        if d:
            out.append(d)
    return out


def analyze(run_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(run_dir, "**", "*.h5"), recursive=True))
    episodes = 0
    episodes_with_grasp = 0
    steps = 0
    steps_high = 0
    for path in files:
        try:
            with h5py.File(path, "r") as h:
                trajs = [k for k in h.keys() if k.startswith("traj_")]
                for t in trajs:
                    try:
                        acts = decode_steps(h, t)
                    except (KeyError, ValueError, UnicodeDecodeError):
                        continue
                    if not acts:
                        continue
                    episodes += 1
                    grasped = False
                    for a in acts:
                        g = a.get("gripper")
                        if g is None:
                            continue
                        steps += 1
                        if float(np.asarray(g).ravel()[0]) > GRIPPER_HIGH:
                            steps_high += 1
                            grasped = True
                    episodes_with_grasp += int(grasped)
        except OSError:
            # A file still being written by a live eval; skip rather than abort, so this can be
            # run against an in-flight cell.
            continue
    return {
        "run_dir": run_dir,
        "h5_files": len(files),
        "episodes": episodes,
        "episodes_with_grasp": episodes_with_grasp,
        "grasp_episode_rate_pct": round(100.0 * episodes_with_grasp / episodes, 2) if episodes else None,
        "steps": steps,
        "steps_gripper_high": steps_high,
        "grasp_step_rate_pct": round(100.0 * steps_high / steps, 2) if steps else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", help="run directory/directories (globbed recursively for *.h5)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    a = ap.parse_args()

    rows = [analyze(d) for d in a.run_dirs]
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'run':52s} {'eps':>6s} {'grasp eps':>10s} {'rate':>8s} {'steps':>8s} {'hi':>7s} {'step%':>7s}")
    for r in rows:
        rate = "-" if r["grasp_episode_rate_pct"] is None else f"{r['grasp_episode_rate_pct']:.1f}%"
        srate = "-" if r["grasp_step_rate_pct"] is None else f"{r['grasp_step_rate_pct']:.1f}%"
        name = r["run_dir"] if len(r["run_dir"]) <= 52 else "..." + r["run_dir"][-49:]
        print(
            f"{name:52s} {r['episodes']:6d} {r['episodes_with_grasp']:10d} {rate:>8s} "
            f"{r['steps']:8d} {r['steps_gripper_high']:7d} {srate:>7s}"
        )
    print("\nreference points on Pick-v1.5 (docs/eval_reproduction.md:3706):")
    print("  pi05_droid_jointpos (reference): 24/30 eps = 80.0%   144/992 steps = 14.5%")
    print("  pi05_base                      :  2/30 eps =  6.7%    31/1088 steps =  2.8%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
