#!/usr/bin/env python
"""Validate a handshake cell's action stream.

    python scripts/check_handshake.py --policy molmoact2_droid --task Pick-v1.5 --date handshake

Asserts what BENCHMARK.md's Validation section asks for, per policy, before any full run:
`arm` is (N, 7), `gripper` is (N, 1), there are no NaNs, and the actions are not frozen.

Read the limits honestly. A frozen or all-zeros action stream is the standard silent failure
of a mis-wired policy server, and this catches it -- but that is ALL it catches. The reference
campaign's worst bugs all produced well-formed, non-constant, NaN-free actions: a control rate
3x too slow, a chunk truncated to 10 of 15 actions, and a missing grasp server that turned
6,486 of 6,486 planning calls into a well-formed `success=False`. Every one of those passes
this check. Only comparing a full-coverage cell against the leaderboard catches them.

Trajectories store each step as a NUL-padded JSON row in a uint8 array (the same encoding
scripts/benchmarks/eval_to_csv.py decodes). The first row is `{}` -- the pre-action step --
which is expected, not a defect.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys

import h5py
import numpy as np

EXPECTED_DIMS = {"arm": 7, "gripper": 1}


def decode(path: str, key: str = "traj_0/actions/commanded_action") -> list[dict]:
    with h5py.File(path, "r") as h:
        raw = np.array(h[key])
    out = []
    for row in raw:
        b = bytes(row).rstrip(b"\x00")
        if not b:
            continue
        d = json.loads(b.decode("utf-8"))
        if d:
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--date", default="handshake")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args()

    pattern = f"{a.runs_dir}/{a.policy}/{a.task}/{a.date}/eval_output/**/*.h5"
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        print(f"no trajectory h5 under {pattern}")
        return 2

    ok = True
    for f in files:
        steps = decode(f)
        print(f"\n  {f.split('/eval_output/')[-1]}  ({len(steps)} action steps)")
        if not steps:
            print("    FAIL: no non-empty action rows -- the policy never produced an action")
            ok = False
            continue
        for key, want in EXPECTED_DIMS.items():
            if key not in steps[0]:
                print(f"    FAIL: no '{key}' in the action dict (keys: {sorted(steps[0])})")
                ok = False
                continue
            arr = np.array([s[key] for s in steps], dtype=float)
            std = arr.std(axis=0)
            nan = int(np.isnan(arr).sum())
            frozen = float(np.max(std)) < 1e-9
            flag = ""
            if arr.shape[1] != want:
                flag, ok = f"  FAIL: expected {want} dims", False
            elif nan:
                flag, ok = "  FAIL: NaNs present", False
            elif frozen:
                # Not fatal for the gripper, which legitimately never moves in some episodes.
                flag = "  FAIL: constant across the episode" if key == "arm" else "  (constant -- ok for gripper)"
                if key == "arm":
                    ok = False
            print(f"    {key:8s} shape={arr.shape} NaN={nan} "
                  f"std[min={float(np.min(std)):.4f} max={float(np.max(std)):.4f}]{flag}")
    print(f"\n  HANDSHAKE {a.policy}/{a.task}: {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
