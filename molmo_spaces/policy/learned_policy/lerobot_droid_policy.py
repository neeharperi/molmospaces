"""A LeRobot checkpoint served to MolmoSpaces' eval over droid's own policy protocol.

`lerobot_grpc_client.py` already exists and is the wrong thing here: it targets LeRobot's
async-inference gRPC server for a bimanual YAM. `third_party/lerobot/examples/droid/
serve_droid.py` instead puts a checkpoint behind the DROID rig's protocol -- it derives its
own spec from the checkpoint, so there is nothing to keep in sync by hand -- and that is the
server both harnesses can then share.

**Sharing the server is the point; sharing the client would defeat it.** This wrapper packs
its own observation out of MuJoCo sensors and decodes its own action, exactly as
`pi_policy.py` does. That independence is what makes a cross-harness comparison mean
anything: if this and droid's `observation.build` agree, two separately written adapters
agree. Reusing droid's client would guarantee agreement by construction and measure nothing.

Four things have to be right, and each is a way to be quietly wrong:

* **HTTP, not the websocket.** `serve_droid.py` serves both, but its websocket handler holds
  one `_last_step` behind a lock, so two eval workers would reset each other's observation
  history. The HTTP path is stateless per request, which is what lets `--num_workers > 1`
  work at all.
* **openpi's msgpack_numpy, not the PyPI package.** They disagree on the ndarray wire
  format (`__ndarray__` versus `nd`), and the PyPI one fails silently on send --
  `endpoint_ws_client.py` and `dreamzero_policy.py` both carry the same warning.
* **No resize.** The spec declares the dataset's own resolution with `pad: false`, and the
  dataset was built at the renderer's native size, so the frame goes out untouched.
  Borrowing `pi_policy`'s `resize_with_pad` here would letterbox an image the policy never
  saw in training.
* **`step` must advance and must reset.** Diffusion Policy conditions on `n_obs_steps`
  frames held server-side, and `serve_droid.py` resets that queue when `step <= last_step`.
  A frozen counter resets the history every tick; one that never resets carries the previous
  episode's frames into the next.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy
from molmo_spaces.policy.learned_policy.utils import resolve_camera_keys

log = logging.getLogger(__name__)

#: The Robotiq driver-joint travel actually reachable at ctrl=255, measured off 2f85.xml.
#: Must equal droid/scripts/sim_server.py's GRIPPER_JOINT_MAX or the two harnesses hand the
#: same policy different gripper readings -- see docs/eval_reproduction.md, 2026-09-17.
GRIPPER_JOINT_MAX = 0.824033
GRIPPER_CTRL_MAX = 255.0


@dataclass
class LeRobotDroidPolicyState:
    actions_buffer: np.ndarray | None = None
    current_buffer_index: int = 0
    step: int = 0
    starting_time: float | None = None


class LeRobotDroidHTTPClient:
    """POST /act in droid's msgpack, with the transport retry MolmoAct2's client uses."""

    def __init__(
        self,
        host: str,
        port: int,
        max_retries: int = 5,
        expect_action_space: str | None = "joint_position",
    ) -> None:
        import requests
        from openpi_client import msgpack_numpy

        self._requests = requests
        self._pack = msgpack_numpy.packb
        self._unpack = msgpack_numpy.unpackb
        self.url = f"http://{host}:{port}"
        self.max_retries = max_retries
        self.session = requests.Session()
        response = self.session.get(self.url + "/metadata", timeout=60)
        response.raise_for_status()
        self.spec = self._unpack(response.content)["droid_policy"]
        action = self.spec["action"]
        # A plan server declares kind=plan and carries no space/dim/horizon -- the rig's
        # parse_spec fills those in. Pass expect_action_space=None for one of those.
        if expect_action_space is not None:
            if action.get("space") != expect_action_space:
                raise ValueError(
                    f"expected {expect_action_space}, server declares {action.get('space')!r}"
                )
            if int(action.get("dim", 0)) != 8:
                raise ValueError(f"expected an 8-dim action, server declares {action.get('dim')!r}")
        self.horizon = int(action.get("horizon", 1))
        self.cameras = {c["role"]: c for c in self.spec["observation"]["cameras"]}
        log.info(
            "lerobot server: %s, horizon %d, cameras %s",
            self.spec.get("name"),
            self.horizon,
            sorted(self.cameras),
        )

    def infer(self, payload: dict) -> dict:
        last = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.url + "/act",
                    data=self._pack(payload),
                    headers={"Content-Type": "application/msgpack"},
                    timeout=120,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"{response.status_code}: {response.text[:400]}")
                return self._unpack(response.content)
            except Exception as exc:  # noqa: BLE001
                # Retry rather than drop the episode: a dropped episode is recorded as a
                # policy failure and biases the rate DOWNWARD, because long episodes issue
                # more requests and long episodes are disproportionately failures.
                last = exc
                log.warning(
                    "lerobot /act attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc
                )
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"lerobot server unreachable after {self.max_retries} attempts: {last}")


class LeRobotDroidPolicy(InferencePolicy, StatefulPolicy):
    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(exp_config)
        cfg = exp_config.policy_config
        self.remote_config = cfg.remote_config
        self.checkpoint_path = cfg.checkpoint_path
        self.grasping_type = cfg.grasping_type
        self.grasping_threshold = cfg.grasping_threshold
        self.chunk_size = cfg.chunk_size
        self.camera_names = cfg.camera_names
        self.model = None
        self.reset()

    def get_state(self):
        return LeRobotDroidPolicyState(
            self.actions_buffer, self.current_buffer_index, self.step, self.starting_time
        )

    def set_state(self, state: LeRobotDroidPolicyState):
        self.actions_buffer = state.actions_buffer
        self.current_buffer_index = state.current_buffer_index
        self.step = state.step
        self.starting_time = state.starting_time

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.step = 0  # zeroed per episode; the server keys its frame queue off this
        self.starting_time = None

    def prepare_model(self):
        self.model = LeRobotDroidHTTPClient(
            self.remote_config.get("host", "localhost"),
            int(self.remote_config["port"]),
            int(self.remote_config.get("max_retries", 5)),
        )

    def obs_to_model_input(self, obs):
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        exterior_key, wrist_key = resolve_camera_keys(obs, self.camera_names)
        qpos = obs["qpos"]
        gripper = float(
            np.clip(np.mean(np.asarray(qpos["gripper"], np.float64)) / GRIPPER_JOINT_MAX, 0.0, 1.0)
        )
        payload = {
            "protocol": 1,
            "step": int(self.step),
            "timestamp": 0.0,
            # Untouched: the spec declares the dataset's own resolution with pad: false, and
            # the dataset was built at this renderer's native size.
            "images": {
                "external": np.ascontiguousarray(obs[exterior_key]),
                "wrist": np.ascontiguousarray(obs[wrist_key]),
            },
            "state": {
                "joint_positions": np.asarray(qpos["arm"][:7], np.float32),
                "gripper_position": float(gripper),
            },
        }
        self.step += 1
        return payload

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            self.actions_buffer = np.asarray(self.model.infer(model_input)["actions"])
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        model_output = np.asarray(model_output).reshape(-1)
        raw = float(model_output[7])
        if self.grasping_type == "continuous":
            gripper = np.array([float(np.clip(raw, 0.0, 1.0)) * GRIPPER_CTRL_MAX])
        else:
            gripper = np.array([GRIPPER_CTRL_MAX if raw > self.grasping_threshold else 0.0])
        return {"arm": model_output[:7].reshape(7), "gripper": gripper}

    def get_action_chunk(self, observation: Any) -> list[dict[str, np.ndarray]]:
        first = self.get_action(observation)
        rest = self.actions_buffer[self.current_buffer_index : self.chunk_size]
        self.current_buffer_index = self.chunk_size
        return [first] + [self.model_output_to_action(o) for o in rest]

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = self.model.spec.get("name", "lerobot") if self.model else "lerobot"
        return info
