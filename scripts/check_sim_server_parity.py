#!/usr/bin/env python
"""Does droid's sim server hand the rig the same episode this harness would evaluate?

    $MOLMOSPACES_PYTHON scripts/sim_server.py --port 8600 --benchmark "$BENCH"   # over there
    conda activate mlspaces-classic
    python scripts/check_sim_server_parity.py --benchmark "$BENCH" --episodes 3

`scripts/compare_harnesses.py` compares *outcomes*, and cannot give a verdict for a policy
that does not reproduce itself -- which openpi does not in a 500-step closed loop, measured
three ways (see docs/eval_reproduction.md). This asks the question that does have an exact
answer: for a given episode index, is the **initial condition** the two sides construct the
same object?

That is the whole of the environment half of a cross-harness comparison. If the scene, the
robot's start pose and the camera views match, any remaining difference in outcome is the
policy or the rollout loop, not the world -- and if they do not match, no outcome comparison
was ever going to mean anything.

Both sides go through `JsonEvalTaskSampler`, so the scene *ought* to be identical by
construction. What is not shared, and is therefore what this actually tests, is
`scripts/sim_server.py`'s translation: world frame to robot base frame, MolmoSpaces sensor
names to DROID camera roles, the Robotiq's 0..255 control range to a 0..1 closed fraction,
and metres to uint16 millimetres. A mistake in any of those produces a rig that sees a
plausible scene which is not this one.

The reference episode is built in this process rather than read from a recorded run, so the
comparison needs no prior cell -- and it is built from the same benchmark directory the
server was given, which the server reports back so a mismatch is caught rather than compared.

**Images are reported, not asserted, and that is a fact about the benchmark rather than a
weakness here.** An `EpisodeSpec` carries `seed: null`, so whatever randomises a scene is not
seeded from the episode: building the same index twice *in one process* gives a bit-identical
arm pose and a different `exo_camera_1` image. Measured across processes the wrist view
sometimes agrees and sometimes does not, with no pattern tied to the episode -- consistent
with a process-level RNG whose position is not recoverable from the episode. So nothing
image-level is assertable across two processes, in any harness, including this one against
itself.

What is asserted is what the episode spec fully determines: the benchmark, the house, and
the arm's start state.

**Joint positions alone do not test the translation**, which is worth saying plainly because
this check used to claim they did. `sim_server.py`'s `state()` returns `arm.joint_pos[:7]`
untranslated, and `reference_state` below reads the same attribute -- so a 0.00e+00 agreement
there is one number compared with itself, and it proves the two sides built the same episode
and nothing more. It is still worth asserting for exactly that.

So the translated quantities are asserted too, each against a reference derived here rather
than borrowed from the server:

  * `cartesian_position` -- `inv(base_to_world) @ leaf_frame_to_world`, with the rotation
    checked by rebuilding it from the served roll/pitch/yaw through scipy's extrinsic "xyz",
    the convention `droid/misc/transformations.py` uses. A transposed base frame, the flange
    substituted for the tool centre point, or a different Euler convention all move this and
    none of them moves `joint_pos`.
  * `gripper_position` -- the Robotiq's joint range mapped to a 0..1 closed fraction.
    `GRIPPER_JOINT_MAX` is declared below independently of `sim_server.py`'s copy, on purpose:
    if one of the two changes, this is what says so.

Depth is not covered: the rig requests it per role and nothing here asks for it, so the
metres-to-uint16-millimetres conversion is still untested by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# MolmoSpaces sensor name -> DROID camera role. A copy of scripts/sim_server.py:60 in the
# droid checkout, which is a plain script rather than an importable module; the two must
# track each other. (link_serial() below can import ROLE_SERIALS because that one does live
# in a package.)
CAMERA_ROLES = {"wrist_camera": "wrist", "exo_camera_1": "external", "exo_camera_2": "external_2"}

# The Robotiq joint travel sim_server.py divides by to get a 0..1 closed fraction. Declared
# here rather than imported so that the two copies disagreeing is a test failure.
GRIPPER_JOINT_MAX = 0.9


def digest(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()[:16]


def reference_episode(benchmark_dir: pathlib.Path, index: int):
    """One episode, built the way eval_main.py builds it."""
    from molmo_spaces.configs.policy_configs import DummyPolicyConfig
    from molmo_spaces.configs.robot_configs import FrankaRobotConfig
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes
    from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler

    episodes = load_all_episodes(benchmark_dir)
    spec = episodes[index]
    config = JsonBenchmarkEvalConfig(
        robot_config=FrankaRobotConfig(), policy_config=DummyPolicyConfig()
    )
    sampler = JsonEvalTaskSampler(config, spec)
    sampler.reset()
    task = sampler.sample_task()
    observations, _ = task.reset()
    return spec, sampler, task, observations[0]


def reference_state(task, obs):
    """What MolmoSpaces itself holds, in its own units, for everything the server translates."""
    robot_view = task._env.robots[0].robot_view
    arm_group = robot_view.get_move_group("arm")
    # `joint_pos`, the attribute sim_server.py reads, not a getter -- FrankaFR3ArmGroup has
    # no get_joint_positions. See the docstring for why this one is a number compared with
    # itself, and what is asserted instead.
    arm = np.asarray(arm_group.joint_pos, dtype=np.float64)[:7]

    # The two frames the served cartesian_position is built from, kept as matrices so the
    # composition happens here and not on the server's terms.
    base_to_world = np.asarray(
        robot_view.get_move_group("base").leaf_frame_to_world, dtype=np.float64
    )
    # leaf_frame_to_world on the arm is the grasp site -- the tool centre point, which is
    # what libfranka's O_T_EE reports and therefore what the rig expects.
    tcp_to_world = np.asarray(arm_group.leaf_frame_to_world, dtype=np.float64)

    gripper_joints = np.asarray(robot_view.get_move_group("gripper").joint_pos, dtype=np.float64)

    images = {}
    for sensor, role in CAMERA_ROLES.items():
        entry = obs.get(sensor)
        if entry is None:
            continue
        rgb = entry["rgb"] if isinstance(entry, dict) else entry
        images[role] = np.asarray(rgb)
    return arm, base_to_world, tcp_to_world, gripper_joints, images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--sim", default="ws://127.0.0.1:8600")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6, help="rad, on the joint vector")
    parser.add_argument(
        "--pose-tolerance",
        type=float,
        default=1e-6,
        help="m / rad, on the translated pose and gripper fraction. Separate from --tolerance "
        "because these cross a float32 wire encoding and the joint vector does not",
    )
    args = parser.parse_args()

    # droid's client, imported from the checkout beside this one. The point is to ask the
    # server the way the rig asks it, not to reimplement the protocol.
    droid_root = pathlib.Path(os.environ.get("DROID_ROOT", pathlib.Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(droid_root))
    from droid.sim.client import SimClient

    link = SimClient(args.sim)
    failures = 0
    try:
        for index in range(args.episodes):
            print(f"\nepisode {index}")
            info = link.reset_episode(seed=index)
            served = info.get("episode") or {}
            spec, sampler, task, obs = reference_episode(args.benchmark, index)
            arm, base_to_world, tcp_to_world, gripper_joints, images = reference_state(task, obs)

            # First, because it is the input error every other line would misattribute: two
            # sides pointed at different benchmark directories disagree about the house and
            # about the start pose, and would read as a broken translation layer.
            # The server reports this twice and in two forms: the greeting carries the full
            # path (sim_server.py:465) and each episode reply carries just the directory name
            # (:419). Compared by basename, which is defined for both -- a benchmark
            # directory's name is its identity here, and two directories of that name holding
            # different episodes is not a confusion this can resolve anyway.
            served_bench = served.get("benchmark") or getattr(link, "benchmark", None)
            if served_bench:
                same_bench = os.path.basename(str(served_bench).rstrip("/")) == os.path.basename(
                    str(args.benchmark).rstrip("/")
                )
                print(f"  {'reads the same benchmark':<44} {'ok' if same_bench else 'FAILED'}"
                      f"  {os.path.basename(str(served_bench).rstrip('/'))}")
                if not same_bench:
                    print(f"       served    {served_bench}")
                    print(f"       reference {args.benchmark}")
                    print("       Nothing below is comparable; point both sides at one directory.")
                    failures += 1
                    break

            same_episode = str(served.get("house")) == str(getattr(spec, "house_index", None))
            print(f"  {'names the same house':<44} {'ok' if same_episode else 'FAILED'}"
                  f"  served {served.get('house')}, reference {getattr(spec, 'house_index', None)}")
            failures += not same_episode

            served_arm = np.asarray(link._state["joint_positions"], dtype=np.float64)
            delta = float(np.max(np.abs(served_arm[: len(arm)] - arm))) if len(arm) else float("nan")
            ok = delta <= args.tolerance
            print(f"  {'the same episode (joints, untranslated)':<44} {'ok' if ok else 'FAILED'}  max |d| {delta:.2e} rad")
            failures += not ok

            # ---- the translated quantities. These are what sim_server.py could get wrong.
            failures += check_cartesian(link, base_to_world, tcp_to_world, args)
            failures += check_gripper(link, gripper_joints, args)

            # Reported, not asserted. See the module docstring: with no seed in the episode
            # the scene randomisation is drawn from a process-level RNG, so two processes
            # agree on the first build and not afterwards -- an assertion here would fail for
            # a reason that has nothing to do with what this check exists to find. The frame
            # SIZE is still asserted, because that the episode does determine.
            for role, reference in sorted(images.items()):
                entry = link._frames.get(role_serial(role))
                if entry is None:
                    print(f"  {role + ': frame arrived':<44} FAILED  nothing under that role")
                    failures += 1
                    continue
                # As sent: the wire carries the renderer's RGB, and the BGRA conversion the
                # ZEDs would have produced happens later, in droid/sim/cameras.py. Comparing
                # here is therefore comparing the same layout on both sides.
                got = np.asarray(entry["rgb"])
                same_shape = got.shape == reference.shape
                identical = same_shape and bool(np.array_equal(got, reference))
                detail = f"{digest(got)} vs {digest(reference)}" if same_shape else f"{got.shape} vs {reference.shape}"
                verdict = "same" if identical else "differs (unseeded; see docstring)"
                print(f"  {role + ': reported, not asserted':<44} {verdict:<8} {detail}")
                # A shape mismatch is a real fault whichever camera it is: it means the two
                # sides disagree about what the episode asked for, not about a random draw.
                if not same_shape:
                    print(f"  {role + ': the same frame size':<44} FAILED  {got.shape} vs {reference.shape}")
                    failures += 1

            release(sampler, task)
    finally:
        link.close()

    print()
    if failures:
        raise SystemExit(f"FAILED: {failures} check(s). The two sides are not building the same episode.")
    print("OK -- the sim server hands the rig the episode this harness would evaluate.")
    return 0


def role_serial(role):
    """Which serial the client filed a role's frame under, without reimplementing the map."""
    from droid.sim.client import ROLE_SERIALS

    return ROLE_SERIALS.get(role)


def release(sampler, task) -> None:
    """Drop one episode's scene before building the next.

    Both halves matter and scripts/sim_server.py's _release() documents why: MolmoSpaces'
    BaseMujocoTask.__del__ raises on the way out, so an unguarded close() escapes the loop
    and takes the verdict with it; and a sampler still referenced keeps its scene's model and
    renderers alive, so a 20-episode run holds 20 of them beside the server's own.
    """
    for obj in (task, sampler):
        closer = getattr(obj, "close", None)
        if closer is None:
            continue
        try:
            closer()
        except Exception as exc:  # noqa: BLE001 - teardown must not decide the verdict
            print(f"  (ignoring {type(obj).__name__}.close(): {type(exc).__name__}: {exc})")
    gc.collect()


def check_cartesian(link, base_to_world, tcp_to_world, args) -> int:
    """The served base-frame TCP pose, against one composed here.

    Position and orientation are separated because they fail for different reasons: a wrong
    base frame or the flange in place of the tool centre point moves the position, while a
    different Euler convention moves only the rotation.
    """
    served = link._state.get("cartesian_position")
    if served is None:
        print(f"  {'cartesian_position arrived':<44} FAILED  the server sent none")
        return 1
    served = np.asarray(served, dtype=np.float64)

    expected = np.linalg.inv(base_to_world) @ tcp_to_world
    d_pos = float(np.max(np.abs(served[:3] - expected[:3, 3])))
    ok_pos = d_pos <= args.pose_tolerance
    print(f"  {'the same tool pose (base frame, position)':<44} {'ok' if ok_pos else 'FAILED'}  max |d| {d_pos:.2e} m")

    # Rebuilt through scipy rather than droid's matrix_to_pose, so the conversion itself is
    # under test rather than compared with its own output.
    from scipy.spatial.transform import Rotation

    got_rotation = Rotation.from_euler("xyz", served[3:6]).as_matrix()
    d_rot = float(np.max(np.abs(got_rotation - expected[:3, :3])))
    ok_rot = d_rot <= args.pose_tolerance
    print(f"  {'the same tool pose (base frame, rotation)':<44} {'ok' if ok_rot else 'FAILED'}  max |d| {d_rot:.2e}")
    return (not ok_pos) + (not ok_rot)


def check_gripper(link, gripper_joints, args) -> int:
    """The served 0..1 closed fraction, against the Robotiq joint travel it came from."""
    served = link._state.get("gripper_position")
    if served is None:
        print(f"  {'gripper_position arrived':<44} FAILED  the server sent none")
        return 1
    expected = float(np.clip(np.mean(gripper_joints) / GRIPPER_JOINT_MAX, 0.0, 1.0))
    delta = abs(float(served) - expected)
    ok = delta <= args.pose_tolerance
    print(f"  {'the same gripper fraction':<44} {'ok' if ok else 'FAILED'}  "
          f"served {float(served):.6f} vs {expected:.6f}")
    return not ok


if __name__ == "__main__":
    raise SystemExit(main())
