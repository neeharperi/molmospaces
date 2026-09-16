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

**The exterior view is reported, not asserted, and that is a fact about the benchmark rather
than a weakness here.** An `EpisodeSpec` carries `seed: null`, so whatever places the
exocentric camera is not seeded from the episode -- building the same index twice *in one
process* yields a different `exo_camera_1` image and an identical `wrist_camera` image. So no
two runs of an episode agree on the exterior view, in any harness, including this one against
itself. The wrist camera is robot-mounted and follows the arm, which is why it is exact and
is the thing worth asserting.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

CAMERA_ROLES = {"wrist_camera": "wrist", "exo_camera_1": "external", "exo_camera_2": "external_2"}


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
    return spec, task, observations[0]


def reference_state(task, obs):
    """Joints, gripper and the camera images, in the units MolmoSpaces itself uses."""
    robot_view = task._env.robots[0].robot_view
    # `joint_pos`, the attribute sim_server.py reads, not a getter -- FrankaFR3ArmGroup has
    # no get_joint_positions, and reading a different field would compare two numbers rather
    # than one number twice.
    arm = np.asarray(robot_view.get_move_group("arm").joint_pos, dtype=np.float64)[:7]
    images = {}
    for sensor, role in CAMERA_ROLES.items():
        entry = obs.get(sensor)
        if entry is None:
            continue
        rgb = entry["rgb"] if isinstance(entry, dict) else entry
        images[role] = np.asarray(rgb)
    return arm, images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=pathlib.Path, required=True)
    parser.add_argument("--sim", default="ws://127.0.0.1:8600")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-6, help="rad, on the joint vector")
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
            spec, task, obs = reference_episode(args.benchmark, index)
            arm, images = reference_state(task, obs)

            same_episode = str(served.get("house")) == str(getattr(spec, "house_index", None))
            print(f"  {'names the same house':<44} {'ok' if same_episode else 'FAILED'}"
                  f"  served {served.get('house')}, reference {getattr(spec, 'house_index', None)}")
            failures += not same_episode

            served_arm = np.asarray(link._state["joint_positions"], dtype=np.float64)
            delta = float(np.max(np.abs(served_arm[: len(arm)] - arm))) if len(arm) else float("nan")
            ok = delta <= args.tolerance
            print(f"  {'the arm starts in the same place':<44} {'ok' if ok else 'FAILED'}  max |d| {delta:.2e} rad")
            failures += not ok

            # Asserted for the wrist, reported for the rest. See the module docstring: the
            # exocentric camera is redrawn per build because the episode carries no seed, so
            # requiring it to match would make this check fail always and mean nothing.
            for role, reference in sorted(images.items()):
                exact = role == "wrist"
                entry = link._frames.get(link_serial(link, role))
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
                if exact:
                    print(f"  {role + ': the same pixels':<44} {'ok' if identical else 'FAILED'}  {detail}")
                    failures += not identical
                else:
                    verdict = "same" if identical else "differs (unseeded; see docstring)"
                    print(f"  {role + ': reported, not asserted':<44} {verdict:<8} {detail}")
                # A shape mismatch is a real fault whichever camera it is: it means the two
                # sides disagree about what the episode asked for, not about a random draw.
                if not same_shape:
                    print(f"  {role + ': the same frame size':<44} FAILED  {got.shape} vs {reference.shape}")
                    failures += 1

            task.close() if hasattr(task, "close") else None
    finally:
        link.close()

    print()
    if failures:
        raise SystemExit(f"FAILED: {failures} check(s). The two sides are not building the same episode.")
    print("OK -- the sim server hands the rig the episode this harness would evaluate.")
    return 0


def link_serial(link, role):
    """Which serial the client filed a role's frame under, without reimplementing the map."""
    from droid.sim.client import ROLE_SERIALS

    return ROLE_SERIALS.get(role)


if __name__ == "__main__":
    raise SystemExit(main())
