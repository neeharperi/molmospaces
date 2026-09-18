#!/usr/bin/env python
"""Do the two harnesses hand the same model the same observation?

    conda activate mlspaces-classic
    python scripts/probe_cross_harness_payload.py --json /tmp/xharness.json

`scripts/probe_policy_payload.py` asks whether *this* repository's wrappers still send what
they sent yesterday. This asks the question one level out: for one observation, does
MolmoSpaces' own wrapper and DROID's `droid/policy/observation.py` build the same thing?

That is the half of a cross-harness comparison nothing measured. `check_sim_server_parity.py`
covers the environment (same house, same start pose, same translated state);
`compare_harnesses.py` covers the outcome. Between them sits the payload, and a difference
there is a harness bug that no success rate would ever name.

**Both arms run in this one process, and that is what makes the comparison exact.**
`droid/policy/observation.py` imports cleanly under `mlspaces-classic` -- the Python 3.8 pin
on droid's container is `pyzed`, which the payload path never touches -- so there is no
two-interpreter problem and no cross-process anything. Consequently there is also no
rendering: the observation is synthesised, identically for both sides, from one set of
arrays. A real episode would add nothing here and would cost the one thing that makes this
measurable, because a `molmospaces-bench-v1` `EpisodeSpec` carries `seed: null` and the same
episode renders differently in two processes.

**Differences are expected, and the point is to enumerate them rather than to pass.** Four
are already known and deliberate; they live in `xharness_allowlist.json` with the file:line
that causes each. Anything not on that list is a finding. Per field, not per payload: a
single hash over the whole dict tells you something moved and not what.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DROID_ROOT = HERE.parents[2]  # .../droid
sys.path.insert(0, str(HERE))  # probe_policy_payload, eval_common
sys.path.insert(0, str(DROID_ROOT))  # droid.policy.*

#: The benchmark's own camera size, width-first as MolmoSpaces records it
#: (`img_resolution: [624, 352]` in both smoke draws). Using it rather than
#: probe_policy_payload's 270x480 means the resize under test is the real one --
#: and 352 * 224/624 = 126.36, which rounds AND truncates to 126, so droid's
#: rounding and MolmoSpaces' truncation agree here by arithmetic accident. That
#: is exactly the kind of thing worth asserting instead of reasoning about.
BENCH_W, BENCH_H = 624, 352

#: One prompt, taken from probe_policy_payload so both arms are fed the identical string.
#: Mixed case on purpose: MolmoSpaces' wrappers call .lower() and the droid servers do not,
#: which is a real divergence and is invisible against an already-lowercase instruction like
#: Close-v1's "close the drawer.". Pick-v1.5's are mixed case, so this is not hypothetical.
PROMPT = None  # set from probe_policy_payload.PROMPT at run time

#: MolmoSpaces policy name -> the droid server fixture that serves the same model.
#: The fixtures are live captures under droid/scripts/tests/fixtures/policy_specs/.
PAIRS = {
    "pi05_droid": "openpi",
    "pi0_droid": "openpi",
    "molmoact2_droid": "molmoact2",
    "dreamzero": "dreamzero",
    "tiptop": "tiptop",
    # The droid fixture must be RE-CAPTURED after training a new checkpoint: serve_droid.py
    # derives its spec from the checkpoint, so the committed one describes whatever was
    # trained last. Comparing against a stale fixture reports the resolution change as a
    # cross-harness difference when it is really a fixture that needs refreshing:
    #   test_policy_servers.py --capture lerobot=ws://127.0.0.1:8200
    "lerobot_droid": "lerobot",
    "inspect_robots": "inspect-robots",
}


#: How to pull the canonical fields out of each side's model-call dict. The two harnesses
#: name their keys differently by construction -- MolmoSpaces' wrappers emit openpi's
#: `observation/...` names, droid emits roles and lets the model's own server rename -- so
#: comparing by key would compare spelling. These compare the thing.
def _native_fields(policy: str, payload: dict) -> dict:
    g = payload.get
    if policy in ("pi05_droid", "pi0_droid"):
        return {
            "external_image": g("observation/exterior_image_1_left"),
            "wrist_image": g("observation/wrist_image_left"),
            "joint_positions": g("observation/joint_position"),
            "gripper_position": g("observation/gripper_position"),
            "prompt": g("prompt"),
        }
    if policy == "molmoact2_droid":
        state = np.asarray(g("state")) if g("state") is not None else None
        return {
            "external_image": g("external_cam"),
            "wrist_image": g("wrist_cam"),
            "joint_positions": None if state is None else state[:7],
            "gripper_position": None if state is None else state[7:8],
            "prompt": g("instruction"),
        }
    if policy == "tiptop":
        # TiPToP is the only policy that consumes the rig's *geometry* rather than its
        # pixels, so it is the only one where depth, intrinsics and extrinsics are
        # comparable at all. Note `world_from_cam` vs the rig's base-framed extrinsic:
        # different frames by design, so it is reported and never asserted.
        return {
            "wrist_image": g("rgb"),
            "wrist_depth_m": g("depth"),
            "wrist_intrinsics": g("intrinsics"),
            # The wrist extrinsic is deliberately NOT compared here, and the reason
            # matters, because an earlier note here ("different frames by design; not
            # comparable") was wrong and let two frame bugs through a green report.
            #
            # They ARE the same quantity: TiPToP's `world_from_cam` is base-framed
            # (tiptop_policy.py composes inv(arm.root_frame_to_world) @ cam2world_gl),
            # and so is the rig's extrinsic. The obstacle is this probe, not the frames.
            # It SYNTHESISES the rig observation -- `build_droid_obs` writes a placeholder
            # cartesian_position and a placeholder wrist extrinsic -- so it never runs the
            # sim_server -> SimRig composition where the bugs actually were. Comparing a
            # value this file made up against MolmoSpaces' real one measures the
            # placeholder; deriving the placeholder from MolmoSpaces measures nothing.
            #
            # The composition needs a LIVE scene server, so the check lives in
            # check_sim_server_parity.py --check-extrinsics instead.
            "joint_positions": g("q_init"),
            "prompt": g("task"),
        }
    if policy in ("lerobot_droid", "inspect_robots"):
        # Role-keyed, like the rig's own payload, because this wrapper speaks droid's
        # protocol directly rather than openpi's key namespace.
        images = payload.get("images", {})
        state = payload.get("state", {})
        return {
            "external_image": images.get("external"),
            "wrist_image": images.get("wrist"),
            "joint_positions": state.get("joint_positions"),
            "gripper_position": state.get("gripper_position"),
            "joint_torques_external": state.get("joint_torques_external"),
            "wrist_depth_m": (payload.get("depth") or {}).get("wrist"),
            "prompt": payload.get("prompt"),
        }
    if policy == "dreamzero":
        # Zero-indexed, unlike pi's `exterior_image_1_left`. Getting this wrong compares
        # DreamZero's SECOND exterior against the rig's first and reports a 152-count
        # pixel difference that is entirely the probe's.
        return {
            "external_image": g("observation/exterior_image_0_left"),
            "external_2_image": g("observation/exterior_image_1_left"),
            "wrist_image": g("observation/wrist_image_left"),
            "joint_positions": g("observation/joint_position"),
            "gripper_position": g("observation/gripper_position"),
            "prompt": g("prompt"),
        }
    return {"_raw_keys": sorted(payload)}


def _rig_fields(payload: dict) -> dict:
    images = payload.get("images", {})
    state = payload.get("state", {})
    depth = payload.get("depth", {})
    intr = payload.get("intrinsics", {})
    out = {
        "external_image": images.get("external"),
        "external_2_image": images.get("external_2"),
        "wrist_image": images.get("wrist"),
        "joint_positions": state.get("joint_positions"),
        "gripper_position": state.get("gripper_position"),
        "prompt": payload.get("prompt"),
        "wrist_intrinsics": intr.get("wrist"),
        "joint_torques_external": state.get("joint_torques_external"),
    }

    # Already metres: observation.build calls depth_in_meters() when the spec declares
    # `units: meters`, which TiPToP's does (observation.py:298). The uint16-millimetre
    # form is what goes over the sim socket, not what reaches a policy -- so the
    # quantisation is still in this number, without a second conversion on top of it.
    if depth.get("wrist") is not None:
        out["wrist_depth_m"] = np.asarray(depth["wrist"], np.float64)
    return out


def build_droid_obs(ms_obs: dict, gripper_max: float) -> dict:
    """The same world, in the shape `RobotEnv.get_observation()` returns.

    This mirrors `scripts/sim_server.py`'s translation rather than importing it, for the
    reason `check_sim_server_parity.py` declares its own `GRIPPER_JOINT_MAX`: two
    independent expressions of one contract catch a change in either. The pieces:
    MolmoSpaces' RGB becomes BGRA the way `droid/sim/cameras.py` makes a rendered frame look
    like a ZED's, metres of depth become uint16 millimetres, and the Robotiq's driver-joint
    angle becomes a 0..1 closed fraction.
    """
    from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id

    def bgra(rgb):
        h, w, _ = rgb.shape
        out = np.empty((h, w, 4), np.uint8)
        out[..., 0] = rgb[..., 2]
        out[..., 1] = rgb[..., 1]
        out[..., 2] = rgb[..., 0]
        out[..., 3] = 255
        return out

    def mm(depth_m):
        d = np.asarray(depth_m, np.float32)
        d = np.where(np.isfinite(d), d, 0.0) * 1000.0
        return np.clip(d, 0, 65535).astype(np.uint16)

    arm = np.asarray(ms_obs["qpos"]["arm"], np.float64)[:7]
    grip_joint = float(np.mean(np.asarray(ms_obs["qpos"]["gripper"], np.float64)))
    closed = float(np.clip(grip_joint / gripper_max, 0.0, 1.0))

    intr = np.asarray(ms_obs["sensor_param_exo_camera_1"]["intrinsic_cv"], np.float32)
    return {
        "robot_state": {
            "joint_positions": list(arm),
            "joint_velocities": [0.0] * 7,
            "gripper_position": closed,
            "cartesian_position": [0.56, 0.0, 0.35, 3.14, 0.0, 0.0],
            "joint_torques_external": [0.0] * 7,
            "robot_mode": "RobotMode.kMove",
            "control_command_success_rate": 1.0,
        },
        "image": {
            hand_camera_id + "_left": bgra(ms_obs["wrist_camera"]),
            varied_camera_1_id + "_left": bgra(ms_obs["exo_camera_1"]),
            varied_camera_2_id + "_left": bgra(ms_obs.get("exo_camera_2", ms_obs["exo_camera_1"])),
        },
        "depth": {
            hand_camera_id: mm(ms_obs["wrist_camera_depth"]),
            varied_camera_1_id: mm(ms_obs["exo_camera_1_depth"]),
            varied_camera_2_id: mm(ms_obs.get("exo_camera_2_depth", ms_obs["exo_camera_1_depth"])),
        },
        "camera_intrinsics": {
            hand_camera_id + "_left": intr,
            varied_camera_1_id + "_left": intr,
            varied_camera_2_id + "_left": intr,
        },
        # sim_server.py duplicates the single exterior into external_2 when a policy asks
        # for it and the scene has only one, matching MolmoSpaces' DreamZero wrapper.
        # Mirror that here or the probe reports the deviation as a cross-harness difference.
        "camera_extrinsics": {
            hand_camera_id + "_left": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            varied_camera_1_id + "_left": [0.29, 0.52, 0.50, -1.99, 0.02, -2.11],
            varied_camera_2_id + "_left": [0.22, -0.43, 0.50, -2.02, 0.00, -1.06],
        },
        "timestamp": {},
    }


def compare_field(name: str, a, b) -> dict:
    """One field, both sides. Numeric fields get a magnitude, not just a verdict."""
    if a is None and b is None:
        return {"status": "absent both"}
    if a is None or b is None:
        return {"status": "MISSING", "native": a is not None, "rig": b is not None}
    if isinstance(a, str) or isinstance(b, str):
        return {"status": "same" if a == b else "DIFFERS", "native": str(a), "rig": str(b)}
    a = np.asarray(a)
    b = np.asarray(b)
    # A scalar and a one-element array holding the same number are the same observation.
    # The wrappers genuinely differ in wrapping -- openpi's client accepts either -- so
    # squeeze before comparing and record that we did.
    squeezed = False
    if a.shape != b.shape and a.size == b.size == 1:
        a, b, squeezed = a.reshape(()), b.reshape(()), True
    if a.shape != b.shape:
        return {"status": "SHAPE", "native": list(a.shape), "rig": list(b.shape)}
    if a.dtype != b.dtype:
        # Worth naming: float64 vs float32 moves a hash without moving a number.
        note = f"dtype {a.dtype} vs {b.dtype}"
    else:
        note = None
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    out = {
        "status": "same" if diff.max() == 0 else "DIFFERS",
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "shape": list(a.shape),
    }
    if note:
        out["note"] = note
    if squeezed:
        out["note"] = (out.get("note", "") + " scalar vs (1,), squeezed").strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", help="write the full per-field result here")
    ap.add_argument("--policy", action="append", help="restrict to these policies (repeatable)")
    ap.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="override the camera set (default: the two both benchmarks render)",
    )
    ap.add_argument(
        "--gripper-max",
        type=float,
        default=0.824033,
        help="the divisor sim_server.py applies to the Robotiq driver joint",
    )
    args = ap.parse_args()

    import probe_policy_payload as ppp

    # The real benchmark size, not the probe's default letterbox-forcing one.
    ppp.IMG_H, ppp.IMG_W = BENCH_H, BENCH_W

    global PROMPT
    PROMPT = ppp.PROMPT

    from molmo_spaces.evaluation.configs import evaluation_configs as E

    import droid.policy.observation as observation
    import droid.policy.protocol as protocol

    # TWO, because that is what both benchmarks actually render: every Close-v1 and
    # Pick-v1.5 episode carries exo_camera_1 and wrist_camera and nothing else (checked
    # against benchmark.json and against a live scene server's camera setup log). The
    # three-camera set is available with --cameras for probing the auto-detect branch,
    # but it does not describe any cell this campaign runs.
    cameras = args.cameras or ["exo_camera_1", "wrist_camera"]
    ms_obs = ppp.build_obs(cameras)
    droid_obs = build_droid_obs(ms_obs, args.gripper_max)

    fixtures = DROID_ROOT / "scripts" / "tests" / "fixtures" / "policy_specs"
    results: dict[str, dict] = {}

    for policy in args.policy or list(PAIRS):
        fixture_name = PAIRS[policy]
        entry: dict = {"fixture": fixture_name}
        try:
            cfg = getattr(E, ppp.POLICIES[policy])()
            ppp._ensure_camera_config(cfg, cameras)
            pol = cfg.policy_config.policy_cls(cfg)
            pol.task = ppp._FakeTask()
            pol.reset()
            native_payload = pol.obs_to_model_input([ms_obs])
            native = _native_fields(policy, native_payload)
        except Exception as exc:  # noqa: BLE001
            entry["native_error"] = f"{type(exc).__name__}: {exc}"
            results[policy] = entry
            continue

        try:
            meta = json.loads((fixtures / f"{fixture_name}.json").read_text())["metadata"]
            spec = protocol.parse_spec(meta)
            rig_payload = observation.build(spec, droid_obs, prompt=PROMPT, step=0)
            rig = _rig_fields(rig_payload)
        except Exception as exc:  # noqa: BLE001
            entry["rig_error"] = f"{type(exc).__name__}: {exc}"
            results[policy] = entry
            continue

        entry["fields"] = {
            k: compare_field(k, native.get(k), rig.get(k)) for k in sorted(set(native) | set(rig))
        }
        results[policy] = entry

    # ---- report -----------------------------------------------------------
    findings = 0
    for policy, entry in results.items():
        print(f"\n{policy}  (droid fixture: {entry['fixture']})")
        for key in ("native_error", "rig_error"):
            if key in entry:
                print(f"  {key}: {entry[key]}")
                findings += 1
        for field, verdict in (entry.get("fields") or {}).items():
            status = verdict.get("status")
            extra = ""
            if "max_abs_diff" in verdict:
                extra = f"  max|d|={verdict['max_abs_diff']:.6g} shape={verdict['shape']}"
            elif status in ("DIFFERS", "SHAPE", "MISSING"):
                extra = f"  native={verdict.get('native')!r} rig={verdict.get('rig')!r}"
            if verdict.get("note"):
                extra += f"  [{verdict['note']}]"
            print(f"  {status:9s} {field:18s}{extra}")
            if status != "same" and status != "absent both":
                findings += 1

    print(f"\n{findings} finding(s).")
    if args.json:
        pathlib.Path(args.json).write_text(
            json.dumps(
                {
                    "gripper_max": args.gripper_max,
                    "bench_wh": [BENCH_W, BENCH_H],
                    "prompt": PROMPT,
                    "results": results,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        print(f"wrote {args.json}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
