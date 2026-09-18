"""Datagen for a SINGLE environment, for training a policy on one scene.

A new file rather than an entry in object_manipulation_datagen_configs.py because
`auto_import_configs()` (data_generation/main.py:45-74) walks this package, so a new module
is picked up without touching a shared one.

Five things here are deliberate, and four of them are traps the defaults walk into.

**Action noise off.** `ActionNoiseConfig.enabled` defaults to **True** (configs/robot_configs.py:55)
and no `*DataGenConfig` turns it off, while every `*PolicyEvalConfig` does. Worse for
imitation than it first looks: `actions/joint_pos` records the *unnoised* command
(sensors.py's LastCommandedJointPosSensor) while the sim executed the noised one, so the
recorded (state, action) pair does not generate the recorded transition. Training on that
teaches the policy a dynamics model that is not the simulator's.

**episodes_per_batch is the unit of parallelism, not a batch size.** pipeline.py builds
`round(samples_per_house / episodes_per_batch)` work items per house, and a work item is
what a worker takes. With one house and the default 4, a 500-episode run would be 125 work
items -- fine -- but setting it to samples_per_house (the obvious "one batch" reading)
would serialise the entire run into a single worker.

**samples_per_house counts SUCCESSES.** pipeline.py budgets
`samples_per_house * max_total_attempts_multiplier` (6) attempts and stops at
`num_collected >= batch_samples`, with `filter_for_successful_trajectories` defaulting True.
So this asks for N good demos or 6N attempts, whichever comes first.

**seed stays None.** task_sampler.py:312 seeds once at sampler *construction*, and each
worker builds its own sampler -- so a fixed seed with num_workers > 1 gives every worker the
identical episode sequence. Reproducibility comes from the frozen benchmark JSON that
create_json_benchmark.py writes from the output, which is this repo's own convention.

**One house and one object type, but poses stay random.** That is the useful reading of "a
single environment": one scene, one task, one target class, with the object and base pose
still varying. Pinning the poses too would let a policy that replays one open-loop
trajectory score well, and the evaluation would measure nothing.
"""

import os
from pathlib import Path

from molmo_spaces.configs import BaseRobotConfig
from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import FrankaDroidCameraSystem
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.configs.robot_configs import ActionNoiseConfig, FrankaRobotConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler

#: Overridable so a 10-episode probe and a 500-episode run are the same code path.
_HOUSE = int(os.environ.get("ONE_ENV_HOUSE", "8"))
_SAMPLES = int(os.environ.get("ONE_ENV_SAMPLES", "500"))
_WORKERS = int(os.environ.get("ONE_ENV_WORKERS", "10"))
_TYPES = [t for t in os.environ.get("ONE_ENV_PICKUP_TYPES", "").split(",") if t] or None
_OUT = Path(os.environ.get("ONE_ENV_OUTPUT_DIR", "/home/nperi/Workspace/droid/data/datagen/pick_one_env_v1"))


@register_config("FrankaPickOneEnvDataGenConfig")
class FrankaPickOneEnvDataGenConfig(PickBaseConfig):
    """Franka pick, one ProcTHOR house, DROID-style fixed cameras.

    Inherits FrankaPickDroidDataGenConfig's camera choice deliberately:
    `FrankaDroidCameraSystem` is `wrist_camera` + `exo_camera_1` at 640x368, which is
    exactly the pair `droid/scripts/sim_server.py` maps onto the rig's wrist/external
    roles, and exactly what Close-v1 and Pick-v1.5 render. Anything else would train a
    policy on views neither harness can serve it.
    """

    robot_config: BaseRobotConfig = FrankaRobotConfig(
        action_noise_config=ActionNoiseConfig(enabled=False),
    )
    camera_config: FrankaDroidCameraSystem = FrankaDroidCameraSystem()
    num_workers: int = _WORKERS
    seed: int | None = None
    output_dir: Path = _OUT

    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        house_inds=[_HOUSE],
        samples_per_house=_SAMPLES,
        episodes_per_batch=max(1, _SAMPLES // _WORKERS),
        pickup_types=_TYPES,
    )

    @property
    def tag(self) -> str:
        return "franka_pick_one_env"
