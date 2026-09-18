#!/usr/bin/env python
"""MolmoSpaces h5 -> a LeRobot v2.1 dataset in the DROID rig's own 8-dim schema.

    python scripts/data/format_conversion/mlspaces_to_lerobot_droid8.py \
        <run_dir> --root ~/Workspace/droid/data/lerobot/pick_one_env_v1

A sibling of `mlspaces_to_lerobot.py`, not a replacement: that one emits the 17-dim
GR00T-style schema (eef_9d + gripper + joints, 180x320) and has other consumers. This one
emits what `third_party/lerobot/examples/droid/serve_droid.py` and `droid/collect.py`
already speak, so a policy trained from it can be served to both harnesses with no mapping
layer -- which is the whole point, since a mapping layer is new untested surface in exactly
the place a cross-harness comparison is supposed to be clean.

| | |
|---|---|
| `action` | (8,) float32 -- 7 absolute joint targets in radians, then gripper 0 open .. 1 closed |
| `observation.state` | (8,) float32 -- the same layout, as measured |
| `observation.images.wrist` | `wrist_camera`, native 640x368 |
| `observation.images.exterior_1` | `exo_camera_1`, native 640x368 |

**No interpreter problem.** `droid/lerobot_writer.py` writes v2.1 with nothing but numpy,
pyarrow and an ffmpeg subprocess -- it exists precisely so the py3.8 workstation container,
which cannot install lerobot, can still produce the format. So this runs entirely in
`mlspaces-classic`, which already has h5py, decord and ffmpeg. Convert up with
`droid/scripts/convert.sh to-v30` afterwards if a v3.0 reader needs it.

**Native resolution, deliberately.** LeRobot keeps the dataset's own size in
`config.image_features` and `serve_droid.py` declares that size to clients with
`pad: false`, so a dataset at the renderer's 640x368 makes the rig's `resize_exact` a no-op
and leaves exactly one resize in the system -- the policy's own `--policy.resize_shape`,
identical in training and serving. Pre-resizing here would add a second one, and the two
harnesses letterbox differently.

**The gripper divisor is 0.824033**, measured: 2f85.xml's actuator comment says ctrl=255
commands 0.8 rad, and stepping the model to rest gives 0.823981. The 0.9 in the MJCF
`range` is a hard limit the actuator never approaches. See
docs/eval_reproduction.md, 2026-09-17.

**Index alignment**, per docs/data_format.md and kept identical to the 17-dim converter:
state `i` pairs with action `i+1`, the first action is a dummy and the last is a `done`
sentinel, so pair `obs[i]` with `act[i+1]` for `i in range(T-2)`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import h5py
import numpy as np
from tqdm import tqdm

DROID_ROOT = pathlib.Path(os.environ.get("DROID_ROOT", pathlib.Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(DROID_ROOT))
from droid import lerobot_writer  # noqa: E402

#: (feature suffix, MolmoSpaces sensor name). Matches FrankaDroidCameraSystem, which is the
#: camera pair both benchmarks render and the one sim_server.py maps to the rig's roles.
CAMERAS = [("wrist", "wrist_camera"), ("exterior_1", "exo_camera_1")]

GRIPPER_JOINT_MAX = 0.824033   # measured; see the module docstring
GRIPPER_CTRL_MAX = 255.0
JOINT_NAMES = tuple(f"joint_{i}" for i in range(7))


def decode_json(row) -> dict:
    """A null-padded JSON blob stored as a uint8 row."""
    return json.loads(bytes(np.asarray(row)).split(b"\x00")[0].decode())


def video_frames(path: pathlib.Path) -> np.ndarray:
    import decord

    decord.bridge.set_bridge("native")
    return decord.VideoReader(str(path))[:].asnumpy()  # (T, H, W, 3) RGB


def to_bgra(rgb: np.ndarray) -> np.ndarray:
    """lerobot_writer's VideoEncoder takes BGRA, matching what a ZED hands collect.py."""
    h, w, _ = rgb.shape
    out = np.empty((h, w, 4), np.uint8)
    out[..., 0] = rgb[..., 2]
    out[..., 1] = rgb[..., 1]
    out[..., 2] = rgb[..., 0]
    out[..., 3] = 255
    return out


def convert(run_dir: pathlib.Path, root: pathlib.Path, limit: int | None, only_success: bool) -> int:
    h5s = sorted(run_dir.rglob("*.h5"))
    if not h5s:
        raise SystemExit(f"no .h5 under {run_dir}")

    writer = None
    written = skipped = 0
    for path in tqdm(h5s, desc="files"):
        with h5py.File(path) as h:
            for name in sorted(k for k in h if k.startswith("traj")):
                if limit is not None and written >= limit:
                    break
                traj = h[name]
                scene = decode_json(traj["obs_scene"][()])
                if only_success and not bool(np.any(traj["success"][()])):
                    skipped += 1
                    continue

                sensors = traj["obs/sensor_data"]
                if not len(sensors):
                    raise SystemExit(
                        f"{path}:{name} has an empty obs/sensor_data. Run\n"
                        f"  python scripts/data/repair_video_paths.py {run_dir}\n"
                        "first -- datagen writes the group but not the mp4 names."
                    )

                # Resolve the mp4 beside the REAL h5, not beside the path we were handed:
                # a caller selecting a subset by symlinking h5s into a scratch directory
                # (which is how the train/held-out split is made) would otherwise look for
                # the videos in the scratch directory and find nothing.
                home = pathlib.Path(os.path.realpath(path)).parent
                frames = {}
                for suffix, sensor in CAMERAS:
                    rel = bytes(np.asarray(sensors[sensor])).split(b"\x00")[0].decode()
                    frames[suffix] = video_frames(home / rel)

                qpos = [decode_json(r) for r in traj["obs/agent/qpos"]]
                acts = [decode_json(r) for r in traj["actions/joint_pos"]]
                effective = len(qpos) - 2       # drop the last two states
                if effective < 1:
                    skipped += 1
                    continue

                if writer is None:
                    fps = int(round(1000.0 / float(scene["policy_dt_ms"])))
                    h_, w_ = frames["wrist"].shape[1:3]
                    features = {
                        "action": lerobot_writer.vector_feature("float32", (*JOINT_NAMES, "gripper")),
                        "observation.state": lerobot_writer.vector_feature("float32", (*JOINT_NAMES, "gripper")),
                    }
                    video_keys = {}
                    for suffix, _ in CAMERAS:
                        key = f"observation.images.{suffix}"
                        features[key] = lerobot_writer.video_feature(w_, h_, fps)
                        video_keys[key] = (w_, h_)
                    writer = lerobot_writer.DatasetWriter(
                        root=str(root), fps=fps, features=features, video_keys=video_keys,
                        depth_keys={}, robot_type="fr3",
                        extras={"source_run": str(run_dir),
                                "converter": "mlspaces_to_lerobot_droid8.py",
                                "gripper_joint_max": GRIPPER_JOINT_MAX,
                                "schema": "droid-8dim: 7 joint positions + gripper 0..1"},
                        chown_reference=str(DROID_ROOT))

                episode = writer.open_episode(str(scene["task_description"]))
                for i in range(effective):
                    grip_state = float(np.clip(
                        np.mean(np.asarray(qpos[i]["gripper"], np.float64)) / GRIPPER_JOINT_MAX, 0.0, 1.0))
                    grip_act = float(np.clip(
                        float(np.asarray(acts[i + 1]["gripper"]).reshape(-1)[0]) / GRIPPER_CTRL_MAX, 0.0, 1.0))
                    row = {
                        "observation.state": np.asarray(list(qpos[i]["arm"][:7]) + [grip_state], np.float32),
                        "action": np.asarray(list(acts[i + 1]["arm"][:7]) + [grip_act], np.float32),
                    }
                    images = {f"observation.images.{s}": to_bgra(frames[s][i]) for s, _ in CAMERAS}
                    episode.add_frame(row, images, {}, block=True)   # not a control loop; never drop
                writer.commit(episode)
                written += 1

    if writer is None:
        raise SystemExit("nothing converted")
    print(f"\n{written} episode(s) written to {root}; {skipped} skipped")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--root", type=pathlib.Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all-trajectories", action="store_true",
                    help="keep failures too; by default only episodes that reached success are written")
    args = ap.parse_args()
    return convert(args.run_dir, args.root, args.limit, not args.all_trajectories)


if __name__ == "__main__":
    raise SystemExit(main())
