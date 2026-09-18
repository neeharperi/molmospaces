"""inspect-robots served to MolmoSpaces' eval over droid's own policy protocol.

Structurally this is `tiptop_policy.py` -- a remote planner whose reply is unrolled into
per-step actions -- with `lerobot_droid_policy.py`'s payload builder, because
inspect-robots speaks the DROID rig's protocol rather than a bespoke one. That makes the
observation half true-by-construction and leaves the plan half as the only real work.

Five details, each a way to be quietly wrong:

* **Send droid's gripper convention, not inspect-robots'.** Internally it is canonical
  0 = closed, 1 = open -- the inverse of the rig's. The conversion is the *server's*:
  `_observation.py:114` calls `canonical_gripper_from_rig()` on what arrives. So the wire
  carries 0 open .. 1 closed, exactly as `sim_server.py` reports it. Inverting here would
  be inverting twice.
* **`joint_torques_external` is required and is zeros.** The spec asks for it and it is
  the only policy that does. Zeros is not a fudge: `droid/sim/protocol.py` and
  `droid/sim/rig.py` both supply zeros in simulation, so this is exact parity.
* **Exact-resize to the declared 640x360, no letterbox.** The spec says `pad: false`.
* **Do not lowercase the prompt.** The rig passes it through; pi/molmoact2 lowercase and
  that difference is already an allowlisted cross-harness finding. Do not add a sixth.
* **There is no `q_init` in the plan envelope** (unlike tiptop), so the arm position the
  first gripper step should hold is seeded from the observation.

Run the server with `--mode joint`. `_safety.check_workspace` hard-codes the lab table
box (x 0.25..0.65, y -0.30..0.30, z 0.03..0.45), which is meaningless inside a ProcTHOR
house -- and it is only consulted on the cartesian path.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy, StatefulPolicy
from molmo_spaces.policy.learned_policy.utils import resolve_camera_keys

log = logging.getLogger(__name__)

GRIPPER_JOINT_MAX = 0.824033  # must match scripts/sim_server.py
GRIPPER_CTRL_MAX = 255.0
CAMERA_W, CAMERA_H = 640, 360  # the spec's declared size, pad: false


@dataclass
class InspectRobotsPolicyState:
    actions_buffer: np.ndarray | None = None
    current_buffer_index: int = 0
    step: int = 0
    starting_time: float | None = None


def _resize_exact(rgb: np.ndarray) -> np.ndarray:
    """PIL bilinear to exactly (CAMERA_W, CAMERA_H), matching droid/policy/images.py."""
    if rgb.shape[0] == CAMERA_H and rgb.shape[1] == CAMERA_W:
        return np.ascontiguousarray(rgb)
    return np.asarray(Image.fromarray(rgb).resize((CAMERA_W, CAMERA_H), Image.BILINEAR))


class InspectRobotsWebsocketClient:
    """droid's protocol over a websocket. NOT HTTP.

    The lerobot server offers a msgpack HTTP sidecar and this one does not: its
    ``/metadata`` answers **426 Upgrade Required** to a plain GET, because the whole
    protocol lives on the websocket. Only ``/healthz`` is HTTP.

    The shape, from droid/policy/client.py: the server pushes its metadata frame
    *unprompted* as the first frame after the handshake, then it is one request frame in,
    one response frame out. A text frame is a server-side traceback rather than a reply.

    Codec is openpi's msgpack_numpy (``__ndarray__``), which is what droid's protocol.py
    reproduces byte for byte -- the PyPI ``msgpack_numpy`` uses ``nd`` instead and fails
    silently on send.
    """

    def __init__(self, host: str, port: int, max_retries: int = 5) -> None:
        from openpi_client import msgpack_numpy

        self._pack = msgpack_numpy.packb
        self._unpack = msgpack_numpy.unpackb
        self.uri = f"ws://{host}:{port}"
        self.max_retries = max_retries
        self._conn = None
        self.spec = self._connect()

    def _connect(self):
        from websockets.sync.client import connect

        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
        # max_size None: a 640x360 RGB pair plus depth is well past the 1 MB default.
        self._conn = connect(self.uri, compression=None, max_size=None, open_timeout=60)
        greeting = self._conn.recv(timeout=120)
        if isinstance(greeting, str):
            raise RuntimeError(f"server sent a text greeting (a traceback?): {greeting[:400]}")
        meta = self._unpack(greeting)
        return meta.get("droid_policy", meta)

    def infer(self, payload: dict) -> dict:
        import time as _time

        last = None
        for attempt in range(self.max_retries):
            try:
                self._conn.send(self._pack(payload))
                reply = self._conn.recv(timeout=300)
                if isinstance(reply, str):
                    raise RuntimeError(f"server error frame: {reply[:400]}")
                return self._unpack(reply)
            except Exception as exc:  # noqa: BLE001
                # Reconnect and retry rather than drop the episode: a dropped episode is
                # recorded as success=False and biases the rate downward.
                last = exc
                log.warning(
                    "inspect-robots attempt %d/%d failed: %s", attempt + 1, self.max_retries, exc
                )
                _time.sleep(2.0 * (attempt + 1))
                try:
                    self._connect()
                except Exception as reconnect_exc:  # noqa: BLE001
                    last = reconnect_exc
        raise RuntimeError(f"inspect-robots unreachable after {self.max_retries} attempts: {last}")


class InspectRobotsPolicy(InferencePolicy, StatefulPolicy):
    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(exp_config)
        cfg = exp_config.policy_config
        self.remote_config = cfg.remote_config
        self.checkpoint_path = cfg.checkpoint_path
        self.camera_names = cfg.camera_names
        self.grasping_threshold = cfg.grasping_threshold
        self.model = None
        self._plan_exhausted = False
        self.reset()

    def get_state(self):
        return InspectRobotsPolicyState(
            self.actions_buffer, self.current_buffer_index, self.step, self.starting_time
        )

    def set_state(self, state: InspectRobotsPolicyState):
        self.actions_buffer = state.actions_buffer
        self.current_buffer_index = state.current_buffer_index
        self.step = state.step
        self.starting_time = state.starting_time

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.step = 0
        self.starting_time = None
        self._plan_exhausted = False
        # An explicit episode identity for the server to scope the agent's conversation
        # on. `step` alone is not enough: it is reset here, per episode, which is right,
        # but the steps that actually reach the server are the subset of ticks where the
        # action buffer happened to be empty, and in practice they arrive non-monotonic
        # and repeat 0 -- a server keyed on "step == 0" restarted the conversation on
        # roughly every other call (measured turn sequence 1,2,1,2,1,2,1,3 inside one
        # episode). A uuid minted here changes exactly when an episode does.
        self.episode_id = uuid.uuid4().hex

    def prepare_model(self):
        self.model = InspectRobotsWebsocketClient(
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
        depth = obs.get(f"{wrist_key}_depth")
        payload = {
            "protocol": 1,
            "step": int(self.step),
            "episode_id": self.episode_id,
            "timestamp": 0.0,
            "prompt": self.task.get_task_description(),  # not lowercased, deliberately
            "images": {
                "external": _resize_exact(np.asarray(obs[exterior_key], np.uint8)),
                "wrist": _resize_exact(np.asarray(obs[wrist_key], np.uint8)),
            },
            "state": {
                "joint_positions": np.asarray(qpos["arm"][:7], np.float32),
                "gripper_position": float(gripper),
                # Zeros is exact parity with what the rig reports in simulation.
                "joint_torques_external": np.zeros(7, np.float32),
            },
        }
        if depth is not None:
            # Metres, at native resolution: observation.build does not resize depth.
            payload["depth"] = {"wrist": np.asarray(depth, np.float32)}
        self.step += 1
        return payload

    def _unroll_plan(self, reply: dict, seed_arm: np.ndarray) -> np.ndarray:
        """A reply -> (N, 8) of [arm(7) | gripper(1)], gripper 0 open .. 1 closed.

        The steps are nested: the reply is ``{"success", "error", "plan"}`` and the plan is
        ``{"steps": [...]}``. Reading ``steps`` off the top level silently yields nothing,
        which this wrapper then treated as "planner found nothing" and ended the episode --
        every simulation episode ran 2 steps and the cell scored 0/50.
        """
        if not reply.get("success", True):
            log.warning("inspect-robots planning failed: %s", reply.get("error"))
            return np.empty((0, 8), np.float32)
        plan = reply.get("plan") or {}
        segments = []
        current_gripper = 0.0
        last_arm = np.asarray(seed_arm, np.float32)[:7]
        for step in plan.get("steps") or ():
            kind = step.get("type")
            if kind == "trajectory":
                arm = np.asarray(step["positions"], np.float32)[:, :7]
                col = np.full((len(arm), 1), current_gripper, np.float32)
                segments.append(np.hstack([arm, col]))
                last_arm = arm[-1]
            elif kind == "gripper":
                current_gripper = 1.0 if step.get("action") == "close" else 0.0
                segments.append(
                    np.concatenate([last_arm, [current_gripper]]).astype(np.float32)[None]
                )
        if not segments:
            return np.empty((0, 8), np.float32)
        return np.concatenate(segments, axis=0)

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= len(self.actions_buffer):
            seed = model_input["state"]["joint_positions"]
            reply = self.model.infer(model_input)
            self.actions_buffer = self._unroll_plan(reply, seed)
            self.current_buffer_index = 0
            if len(self.actions_buffer) == 0:
                # A planner that finds no plan is one episode's answer, not a crash --
                # simulate.py records it and moves on, and so should this.
                self._plan_exhausted = True
                log.warning("inspect-robots returned an empty plan; holding position")
                return np.concatenate([np.asarray(seed, np.float32), [0.0]])
        out = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return out

    def model_output_to_action(self, model_output):
        model_output = np.asarray(model_output).reshape(-1)
        gripper = GRIPPER_CTRL_MAX if model_output[7] > self.grasping_threshold else 0.0
        action = {"arm": model_output[:7].reshape(7), "gripper": np.array([gripper])}
        if self._plan_exhausted:
            action["done"] = True
        return action

    def get_action_chunk(self, observation: Any) -> list[dict[str, np.ndarray]]:
        return [self.get_action(observation)]

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "inspect-robots"
        return info
