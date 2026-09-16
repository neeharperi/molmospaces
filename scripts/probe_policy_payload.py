#!/usr/bin/env python3
"""Hash the exact payload each policy wrapper would send, for a HEAD-vs-working-tree A/B.

Why this exists
---------------
The cheap way to ask "did our edits change what the harness sends to a policy server?" is a
deterministic replay of a real eval cell. That works for openpi (`DETERMINISTIC=1` in
`scripts/serve_openpi.sh` pins XLA autotuning) but it structurally cannot work for MolmoAct2:
`third_party/molmoact2/examples/droid/host_server_droid.py` calls `predict_action` without a
`generator=`, so sampling draws from the global torch RNG, which PyTorch seeds from OS entropy
per process. Two identical runs there are never bit-comparable.

This probe sidesteps the server entirely. It feeds each wrapper a synthetic observation and
hashes `obs_to_model_input()`'s output -- the complete payload, images included, after
`resize_with_pad` and state packing. That covers the whole class of change we actually make to
these files (camera-key selection, resize, normalisation, state layout, prompt handling) for
*every* policy, with no server, no GPU, and no sampling noise.

It also covers the 5-camera auto-detect branch, which `docs/eval_reproduction.md:3551` lists as
proven statically but never exercised end-to-end (PnP-NextTo-v2 costs 285 s/ep at one worker).

Usage
-----
Probe the working tree, probe a baseline worktree, then diff the two:

    git worktree add --detach /tmp/base <baseline-rev>
    python scripts/probe_policy_payload.py --json /tmp/now.json
    cd /tmp && PYTHONPATH=/tmp/base python <abs>/scripts/probe_policy_payload.py --json /tmp/base.json
    python scripts/probe_policy_payload.py --compare /tmp/base.json /tmp/now.json

Two details in that recipe are load-bearing:

  * **PYTHONPATH is what isolates the baseline arm.** Measured: setuptools *appends*
    `_EditableFinder`, so sys.meta_path is `[..., PathFinder, _EditableFinder]` and the path
    entries are consulted first. The editable install is therefore not the hazard. What can
    still beat PYTHONPATH is `sys.path[0]` -- the cwd for `python -c` and `python -m`, the
    script's own directory for `python some/script.py`. The `cd` above covers the first form;
    in the recipe as written sys.path[0] is `<main>/scripts`, which holds no `molmo_spaces`,
    so PYTHONPATH wins either way. Neither of these is a guarantee, which is why `--compare`
    re-checks the loaded module file and VOIDs instead of passing.
  * **Point PYTHONPATH at the baseline but run the working tree's probe.** The instrument has
    to be held fixed while the code under test changes. An old revision's probe against an old
    revision's package compares two measurements, not two payloads.

`--compare` checks the isolation for you: two arms reporting the same `molmo_spaces_file` are
VOID rather than PASS, which is the one way this comparison can lie.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

# (exterior key the wrapper should pick, wrist key, full obs camera set, camera_names override)
RIGS = {
    # bench-v1 (Open-v1, Close-v1): exactly one exterior, one wrist.
    "benchv1_2cam": {
        "cameras": ["exo_camera_1", "wrist_camera"],
        "camera_names": None,
    },
    # bench-v2 (Pick-*, PnP-*): the full rig, exercising the auto-detect branch. Both
    # DROID_EXTERIOR_CAMERA_KEYS[0] and DREAMZERO_EXTERIOR_CAMERA_KEYS[0] are present, which
    # is what makes pi/molmoact2 and dreamzero resolve to *different* exteriors here.
    "benchv2_5cam": {
        "cameras": [
            "droid_shoulder_light_randomization",
            "randomized_zed2_analogue_1",
            # DreamZero's second exterior slot is randomized_zed2_analogue_2 (not exo_camera_2);
            # without it this rig silently exercises the single-exterior duplication branch
            # instead of the real two-view one. benchv1_2cam still covers duplication.
            "randomized_zed2_analogue_2",
            "exo_camera_1",
            "exo_camera_2",
            "wrist_camera_zed_mini",
        ],
        "camera_names": None,
    },
    # Pick-v2-RandCam is the only task that passes --camera_names, and it passes it to every
    # policy (which is why camera_names had to move onto BasePolicyConfig).
    "randcam_override": {
        "cameras": [
            "droid_shoulder_light_randomization",
            "randomized_zed2_analogue_1",
            # DreamZero's second exterior slot is randomized_zed2_analogue_2 (not exo_camera_2);
            # without it this rig silently exercises the single-exterior duplication branch
            # instead of the real two-view one. benchv1_2cam still covers duplication.
            "randomized_zed2_analogue_2",
            "exo_camera_1",
            "exo_camera_2",
            "wrist_camera_zed_mini",
        ],
        "camera_names": ["randomized_zed2_analogue_1", "wrist_camera_zed_mini"],
    },
}

POLICIES = {
    "pi05_droid": "PiPolicyEvalConfig",
    "pi0_droid": "Pi0PolicyEvalConfig",
    "molmoact2_droid": "MolmoAct2PolicyEvalConfig",
    "dreamzero": "DreamZeroPolicyEvalConfig",
    "tiptop": "TiptopEvalConfig",
}

# Non-square on purpose: resize_with_pad must actually letterbox, so a padding regression moves
# the hash instead of hiding behind an identity resize.
IMG_H, IMG_W = 270, 480
PROMPT = "Pick up the Mug"

# Payload fields that are nondeterministic BY DESIGN and would otherwise mask the comparison.
# DreamZero mints a fresh uuid4() session id in reset() -- that is the mechanism its server keys
# AR frame history off, and the reason DreamZero must run at --num_workers 1. Normalising it is
# not hiding a difference; leaving it in would hide all the others behind it. Every normalised
# key is listed in the output so this stays visible.
VOLATILE_KEYS = {"session_id"}


# Output-side probe. obs_to_model_input covers everything going *to* the server; this covers
# everything coming *back*. It is where the gripper convention lives, and the gripper is the
# channel the whole reproduction turns on: docs/eval_reproduction.md:3706 shows pi05_base
# commanding a grasp in 2/30 episodes against the reference's 24/30, and that difference is
# very nearly the entire 17.3pp gap.
#
# The gripper values deliberately straddle grasping_threshold (0.5), including the exact
# boundary, so a `>` vs `>=` flip shows up -- the wrappers genuinely differ here today
# (pi/molmoact2 use `>`, dreamzero uses `>=`), and out-of-range values exercise DreamZero's clip.
_ARM7 = np.array([0.11, -0.62, 0.23, -1.84, 0.35, 1.46, -0.97], dtype=np.float64)
MODEL_OUTPUTS = {
    "grip_0.00": 0.0,
    "grip_0.49999": 0.49999,
    "grip_0.50000": 0.5,
    "grip_0.50001": 0.50001,
    "grip_1.00": 1.0,
    "grip_neg": -0.25,
    "grip_over": 1.75,
}


# TiPToP derives its planning frame from the live env:
#   self.task.env.current_robot.robot_view.get_move_group("arm").root_frame_to_world
# A fixed, non-identity, non-symmetric pose here means the base_from_cam matrix multiply in
# tiptop_policy.py is genuinely exercised -- an inverted or transposed frame moves the hash.
_ROOT_FRAME_TO_WORLD = np.array(
    [
        [0.0, -1.0, 0.0, 0.25],
        [1.0, 0.0, 0.0, 0.40],
        [0.0, 0.0, 1.0, 0.15],
        [0.0, 0.0, 0.0, 1.00],
    ],
    dtype=np.float64,
)


class _FakeMoveGroup:
    root_frame_to_world = _ROOT_FRAME_TO_WORLD


class _FakeRobotView:
    def get_move_group(self, name: str) -> "_FakeMoveGroup":
        assert name == "arm", f"unexpected move group {name!r}"
        return _FakeMoveGroup()


class _FakeRobot:
    robot_view = _FakeRobotView()


class _FakeEnv:
    current_robot = _FakeRobot()


class _FakeTask:
    """Stands in for BaseMujocoTask. The wrappers read get_task_description() and (TiPToP only)
    the robot base frame off .env."""

    env = _FakeEnv()

    def get_task_description(self) -> str:
        return PROMPT


class _FakeCamera:
    """Minimal stand-in for a camera spec: BasePolicy reads only .name and .record_depth."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.record_depth = True


class _FakeCameraConfig:
    def __init__(self, names: list[str]) -> None:
        self.cameras = [_FakeCamera(n) for n in names]


def _ensure_camera_config(cfg, cameras: list[str]) -> None:
    """TiPToP sets force_enable_depth=True, so BasePolicy.__init__ walks
    config.camera_config.cameras and requires record_depth on each. The eval configs leave
    camera_config as None (eval_main fills it in at runtime), so supply a stand-in rather than
    flipping force_enable_depth off -- that flag should stay honest, and none of this reaches
    obs_to_model_input, which is what we actually hash.
    """
    if getattr(cfg, "camera_config", None) is not None:
        return
    if not cfg.policy_config.force_enable_depth:
        return
    try:
        cfg.camera_config = _FakeCameraConfig(cameras)
    except Exception:
        # Pydantic refused the duck type; fall back to disabling the construction-time
        # assertion. Recorded in the output so it is never silently assumed.
        cfg.policy_config.force_enable_depth = False


def _image_for(camera_name: str) -> np.ndarray:
    """Deterministic, and *distinct per camera*, so selecting the wrong camera moves the hash."""
    seed = int.from_bytes(hashlib.sha256(camera_name.encode()).digest()[:4], "big")
    return np.random.default_rng(seed).integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)


def _depth_for(camera_name: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(("depth:" + camera_name).encode()).digest()[:4], "big")
    return np.random.default_rng(seed).uniform(0.2, 2.5, (IMG_H, IMG_W)).astype(np.float32)


def build_obs(cameras: list[str]) -> dict:
    """One observation carrying the superset of keys any of the five wrappers reads."""
    obs: dict = {
        # 7 arm joints + gripper. Fixed values, chosen to be asymmetric so a transposed or
        # truncated slice changes the hash.
        "qpos": {
            "arm": np.array([0.11, -0.62, 0.23, -1.84, 0.35, 1.46, -0.97, 0.031], dtype=np.float64),
            "gripper": np.array([0.412], dtype=np.float64),
        },
    }
    for cam in cameras:
        obs[cam] = _image_for(cam)
        # TiPToP additionally wants depth and camera params for whichever wrist camera it picks.
        obs[f"{cam}_depth"] = _depth_for(cam)
        obs[f"sensor_param_{cam}"] = {
            "intrinsic_cv": np.array(
                [[520.0, 0.0, 240.0], [0.0, 520.0, 135.0], [0.0, 0.0, 1.0]], dtype=np.float64
            ),
            "cam2world_gl": np.array(
                [
                    [0.0, -1.0, 0.0, 0.30],
                    [1.0, 0.0, 0.0, -0.12],
                    [0.0, 0.0, 1.0, 0.55],
                    [0.0, 0.0, 0.0, 1.00],
                ],
                dtype=np.float64,
            ),
        }
    return obs


def canonical(obj, out: list) -> None:
    """Append a stable byte representation of `obj` to `out`.

    Arrays contribute dtype, shape and raw bytes -- not a repr -- so a 1e-9 change in a single
    pixel or joint target is caught.
    """
    if isinstance(obj, np.ndarray):
        out.append(f"ndarray|{obj.dtype.str}|{obj.shape}|".encode())
        out.append(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        out.append(b"dict|")
        for k in sorted(obj, key=str):
            out.append(f"k={k!r}|".encode())
            if k in VOLATILE_KEYS:
                out.append(b"<normalized-volatile>|")
                continue
            canonical(obj[k], out)
    elif isinstance(obj, (list, tuple)):
        out.append(f"seq|{len(obj)}|".encode())
        for v in obj:
            canonical(v, out)
    elif isinstance(obj, (str, bytes, bool, int, float, type(None))):
        out.append(f"{type(obj).__name__}|{obj!r}|".encode())
    else:
        # Fail loudly: an unhashable-by-this-function type is a payload field we are silently
        # not covering, which is exactly the blind spot this probe exists to close.
        raise TypeError(f"no canonical form for {type(obj)!r}: {obj!r}")


def hash_payload(payload) -> str:
    parts: list[bytes] = []
    canonical(payload, parts)
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def summarize(payload) -> dict:
    """A human-readable shape/dtype summary, so a diff says *what* moved, not just that it did."""
    if isinstance(payload, dict):
        return {
            k: ("<normalized-volatile>" if k in VOLATILE_KEYS else summarize(v))
            for k, v in sorted(payload.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(payload, np.ndarray):
        return f"ndarray{payload.shape}:{payload.dtype}"
    if isinstance(payload, (list, tuple)):
        return [summarize(v) for v in payload]
    return repr(payload)


def compare_arms(path_a: str, path_b: str) -> int:
    """Diff two --json outputs and turn the pair into a verdict.

    Needs nothing but the two files -- no molmo_spaces, no GPU -- so it is the half of this
    probe that can run in CI while the two arms are produced wherever they have to be.

    Verdicts per (policy, rig) cell:

      same     identical hash, or identical error text
      MOVED    both arms hashed, and the hashes differ -- the payload changed
      fixed    A errored, B hashed
      BROKE    A hashed, B errored
      absent   only one arm probed the cell at all

    MOVED and BROKE fail. `fixed` does not, because turning a crash into a payload is the
    shape every deliberate improvement here has had. `absent` does not either -- it means the
    two arms probed different matrices (a `--policy` restriction, a rig added between
    revisions), so those cells were not compared and are reported as such rather than as a
    payload that moved.

    Exit status: 0 agree, 1 the payload changed, 2 VOID -- nothing was compared. The three
    are distinct because a caller that cannot tell VOID from a regression will treat an
    un-run check as a failure, or worse, the other way round. Same convention as
    scripts/compare_harnesses.py.
    """
    a = json.load(open(path_a))
    b = json.load(open(path_b))

    pa, pb = a["provenance"], b["provenance"]
    print(f"A  {path_a}\n     {pa['molmo_spaces_file']}")
    print(f"B  {path_b}\n     {pb['molmo_spaces_file']}")
    # The one way this comparison lies is both arms importing the same tree -- the editable
    # install's finder hard-codes the main checkout, so an unisolated arm looks like a PASS.
    if pa["molmo_spaces_file"] == pb["molmo_spaces_file"]:
        print("\nVOID: both arms loaded the same molmo_spaces; nothing was compared.")
        return 2

    ABSENT = "<absent>"
    moved, broke, fixed, same, absent = [], [], [], [], []
    for policy in sorted(set(a["hashes"]) | set(b["hashes"])):
        ha, hb = a["hashes"].get(policy, {}), b["hashes"].get(policy, {})
        for rig in sorted(set(ha) | set(hb)):
            va, vb = ha.get(rig, ABSENT), hb.get(rig, ABSENT)
            cell = f"{policy}/{rig}"
            if ABSENT in (va, vb):
                absent.append((cell, "B" if vb == ABSENT else "A"))
                continue
            ea, eb = va.startswith("ERROR"), vb.startswith("ERROR")
            if va == vb:
                same.append(cell)
            elif ea and not eb:
                fixed.append((cell, va))
            elif eb and not ea:
                broke.append((cell, vb))
            else:
                moved.append((cell, va, vb))

    aa, ab = a.get("actions", {}), b.get("actions", {})
    for policy in sorted(set(aa) | set(ab)):
        cell = f"{policy}/model_output_to_action"
        va = aa.get(policy, {}).get("hash", ABSENT)
        vb = ab.get(policy, {}).get("hash", ABSENT)
        if ABSENT in (va, vb):
            absent.append((cell, "B" if vb == ABSENT else "A"))
        elif va == vb:
            same.append(cell)
        else:
            moved.append((cell, va, vb))

    print(f"\n{len(same)} cell(s) identical")
    for cell, side in absent:
        print(f"absent {cell}\n         only arm {'A' if side == 'B' else 'B'} probed it; not compared")
    for cell, was in fixed:
        print(f"fixed  {cell}\n         A: {was}")
    for cell, now in broke:
        print(f"BROKE  {cell}\n         B: {now}")
    for cell, va, vb in moved:
        print(f"MOVED  {cell}\n         A: {va[:32]}\n         B: {vb[:32]}")

    if moved or broke:
        print("\nFAIL: the payload changed. Either the change was intended -- say so in "
              "docs/eval_reproduction.md and re-baseline -- or it is a regression.")
        return 1
    if not same:
        print("\nVOID: no cell was probed by both arms.")
        return 2
    print(f"\nPASS: all {len(same)} shared cell(s) send byte-identical payloads.")
    if absent:
        print(f"      {len(absent)} cell(s) only one arm probed, so they were not compared.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the full result (hashes + provenance) here")
    ap.add_argument("--policy", action="append", help="restrict to these policies (repeatable)")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="diff two --json outputs instead of probing; needs no molmo_spaces",
    )
    args = ap.parse_args()

    if args.compare:
        return compare_arms(*args.compare)

    import molmo_spaces
    from molmo_spaces.configs import policy_configs_baselines as pcb
    from molmo_spaces.evaluation.configs import evaluation_configs as E
    from molmo_spaces.policy.learned_policy import utils as lp_utils

    provenance = {
        "molmo_spaces_file": molmo_spaces.__file__,
        "has_resolve_camera_keys": hasattr(lp_utils, "resolve_camera_keys"),
        "PiPolicyConfig.camera_names": pcb.PiPolicyConfig().camera_names,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    wanted = args.policy or list(POLICIES)
    results: dict[str, dict[str, str]] = {}
    shapes: dict[str, dict] = {}
    normalized: dict[str, dict] = {}
    actions: dict[str, dict] = {}

    for policy in wanted:
        cfg_cls_name = POLICIES[policy]
        results[policy] = {}
        shapes[policy] = {}
        for rig_name, rig in RIGS.items():
            try:
                cfg = getattr(E, cfg_cls_name)()
                # camera_names=None in a rig means "leave the config's own default alone".
                # That is deliberate and load-bearing for the A/B: HEAD defaults to the literal
                # ['exo_camera_1','wrist_camera'] and the working tree defaults to None, and
                # BOTH mean "auto-detect" in their own tree. Forcing either value onto the other
                # tree measures the probe, not the code. Only Pick-v2-RandCam passes an explicit
                # pair, so only that rig assigns.
                #
                # At HEAD camera_names was declared per-policy, so this assignment raises for
                # the configs that lacked it (DreamZero, TiPToP). That is the very crash the
                # working tree fixes by moving the field onto BasePolicyConfig -- RandCam
                # attaches --camera_names for *every* policy. Report it per cell rather than
                # aborting, so the A/B still covers the remaining cells.
                if rig["camera_names"] is not None:
                    cfg.policy_config.camera_names = rig["camera_names"]
                _ensure_camera_config(cfg, rig["cameras"])
                pol = cfg.policy_config.policy_cls(cfg)
                pol.task = _FakeTask()
                pol.reset()  # DreamZero's AR frame history lives here; harmless for the rest.
                obs = build_obs(rig["cameras"])
                # Every wrapper accepts the list form; TiPToP *requires* it.
                payload = pol.obs_to_model_input([obs])
            except Exception as exc:  # noqa: BLE001 - report, don't abort the whole matrix
                results[policy][rig_name] = f"ERROR: {type(exc).__name__}: {exc}"
                shapes[policy][rig_name] = None
                continue
            results[policy][rig_name] = hash_payload(payload)
            shapes[policy][rig_name] = summarize(payload)
            if isinstance(payload, dict):
                hit = sorted(VOLATILE_KEYS & set(payload))
                if hit:
                    normalized.setdefault(policy, {})[rig_name] = hit

            # Output side. Rig-independent, so compute it once per policy.
            if policy not in actions:
                # Hash the REAL action dicts, not summarize()'d ones. summarize() records
                # shape and dtype but not value, so hashing it made every policy agree --
                # a false pass, and exactly the class of bug this probe exists to catch.
                # (TiPToP is continuous and DreamZero uses >= where pi/molmoact2 use >; if
                # all five hash alike, the probe is measuring nothing.)
                acts = {}
                detail = {}
                for label, grip in MODEL_OUTPUTS.items():
                    mo = np.concatenate([_ARM7, np.array([grip], dtype=np.float64)])
                    try:
                        act = pol.model_output_to_action(mo)
                        acts[label] = act
                        detail[label] = {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in act.items()
                        }
                    except Exception as exc:  # noqa: BLE001
                        acts[label] = f"ERROR: {type(exc).__name__}: {exc}"
                        detail[label] = acts[label]
                actions[policy] = {"hash": hash_payload(acts), "detail": detail}

    print("provenance:")
    for k, v in provenance.items():
        print(f"  {k}: {v}")
    print()
    rig_names = list(RIGS)
    print(f"{'policy':18s} " + " ".join(f"{r:>20s}" for r in rig_names))
    for policy in wanted:
        row = []
        for r in rig_names:
            v = results[policy][r]
            row.append(v[:20] if not v.startswith("ERROR") else "ERROR")
        print(f"{policy:18s} " + " ".join(f"{c:>20s}" for c in row))
    print()
    print("model_output_to_action (rig-independent; gripper values straddle the 0.5 threshold)")
    print(f"{'policy':18s} {'hash':>20s}   gripper at 0.49999 / 0.50000 / 0.50001")
    for policy in wanted:
        a = actions.get(policy)
        if not a:
            continue
        g = []
        for k in ("grip_0.49999", "grip_0.50000", "grip_0.50001"):
            d = a["detail"][k]
            g.append(str(d["gripper"][0]) if isinstance(d, dict) else "ERROR")
        print(f"{policy:18s} {a['hash'][:20]:>20s}   {' / '.join(g)}")

    if normalized:
        print("\nvolatile fields normalized before hashing (nondeterministic by design):")
        for pol, d in sorted(normalized.items()):
            for rig, keys in sorted(d.items()):
                print(f"  {pol}/{rig}: {', '.join(keys)}")
    errs = {(p, r): v for p, d in results.items() for r, v in d.items() if v.startswith("ERROR")}
    if errs:
        print("\nerrors:")
        for (p, r), v in errs.items():
            print(f"  {p}/{r}: {v}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "provenance": provenance,
                    "hashes": results,
                    "shapes": shapes,
                    "normalized_volatile": normalized,
                    "actions": actions,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nwrote {args.json}")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
