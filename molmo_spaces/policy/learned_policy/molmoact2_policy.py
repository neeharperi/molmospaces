import logging
import time
from typing import Any

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MolmoAct2HTTPClient:
    """HTTP + json_numpy client for the MolmoAct2-DROID `/act` server.

    Wire protocol (see third_party/molmoact2/examples/droid/host_server_droid.py):
        POST /act  {external_cam, wrist_cam, instruction, state(8,), num_steps}
             -> {"actions": (N, 8) float32 absolute joint positions, "dt_ms": float}
    """

    def __init__(self, host: str = "localhost", port: int = 8000, timeout: float = 60.0) -> None:
        import json_numpy
        import requests

        json_numpy.patch()
        self._json_numpy = json_numpy
        self._session = requests.Session()
        # Captured here rather than imported at module scope: `requests` is deliberately a
        # lazy import in this class so the module stays importable before multiprocessing
        # forks. infer() needs the exception types, so bind them to the instance.
        self._transport_errors = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )
        self.url = f"http://{host}:{port}/act"
        self.timeout = timeout
        # Health check -- fails fast if the server isn't up yet.
        resp = self._session.get(self.url, timeout=self.timeout)
        resp.raise_for_status()

    # Transport-level failures are retried; HTTP errors are not. A dropped TCP connection
    # says nothing about the request, and POST /act is idempotent -- the server holds no
    # per-client state, so replaying the same observation is safe. An HTTP 500 is the server
    # telling us something real, and must still surface.
    _TRANSPORT_RETRIES = 3
    _RETRY_BACKOFF_SECS = 1.0

    def infer(self, payload: dict) -> dict:
        body = self._json_numpy.dumps(payload)
        last_exc: Exception | None = None
        for attempt in range(self._TRANSPORT_RETRIES):
            try:
                resp = self._session.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    data=body,
                    timeout=self.timeout,
                )
            except self._transport_errors as e:
                # Observed in this campaign as bursts of ConnectionReset/RemoteDisconnected
                # when MolmoAct2's server is contended by the other lanes sharing its GPU.
                # Without a retry each burst costs whole episodes: PnP-v2 lost 7 of 541.
                #
                # Those losses are not neutral. A longer episode issues more requests and is
                # therefore likelier to be hit, and long episodes are disproportionately
                # failures -- so silently dropping them biases the success rate UPWARD.
                last_exc = e
                if attempt + 1 < self._TRANSPORT_RETRIES:
                    time.sleep(self._RETRY_BACKOFF_SECS * (attempt + 1))
                    continue
                raise
            if resp.status_code != 200:
                raise RuntimeError(
                    f"MolmoAct2 server error {resp.status_code}: {resp.text[:500]}"
                )
            return resp.json()
        raise last_exc  # unreachable; the loop either returns or raises


class MolmoAct2Policy(InferencePolicy):
    def __init__(self, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.num_steps = exp_config.policy_config.num_steps
        self.action_horizon = exp_config.policy_config.action_horizon
        self.camera_names = exp_config.policy_config.camera_names
        self.model = None  # don't init model till inference to allow multiprocessing

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None

    def prepare_model(self):
        self.model_name = "molmoact2_droid"
        if self.remote_config is None:
            raise ValueError("MolmoAct2 policy only supports remote model inference")
        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8000)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = MolmoAct2HTTPClient(host=host, port=port)
                log.info(f"Successfully connected to MolmoAct2 model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def obs_to_model_input(self, obs):
        if isinstance(obs, list | tuple):
            obs = obs[0]
        prompt = self.task.get_task_description()

        # exactly two cameras needed (external_cam, wrist_cam) -- confirmed via
        # third_party/molmoact2/sim_eval/inference/common.py's MOLMOACT2_SCHEMAS["droid"].
        if self.camera_names != ["exo_camera_1", "wrist_camera"]:
            exo_camera_key, wrist_camera_key = self.camera_names[0], self.camera_names[1]
        else:
            exo_camera_key = (
                "droid_shoulder_light_randomization"
                if "droid_shoulder_light_randomization" in obs
                else "exo_camera_1"
            )
            wrist_camera_key = (
                "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
            )

        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        state = np.concatenate(
            [np.asarray(obs["qpos"]["arm"][:7], dtype=np.float32), np.array([grip], dtype=np.float32)]
        )
        model_input = {
            "external_cam": np.asarray(obs[exo_camera_key], dtype=np.uint8),
            "wrist_cam": np.asarray(obs[wrist_camera_key], dtype=np.uint8),
            "instruction": prompt.lower(),
            "state": state,
            "num_steps": self.num_steps,
        }
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self._chunk_len():
            result = self.model.infer(model_input)
            self.actions_buffer = np.asarray(result["actions"], dtype=np.float32)
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def _chunk_len(self) -> int:
        """How many actions of the current buffer to execute before re-querying.

        `action_horizon` (15, from the checkpoint's norm_stats) rather than `num_steps` (the
        server's flow-matching integration steps) -- these are unrelated quantities and
        conflating them capped the executed chunk at 10. Clamped to the buffer's real length
        so a server revision returning a shorter chunk can't index past the end.
        """
        if self.actions_buffer is None:
            return self.action_horizon
        return min(self.action_horizon, len(self.actions_buffer))

    def model_output_to_action(self, model_output):
        # Server returns absolute joint positions (not deltas), same handling as PI_Policy.
        if self.grasping_type == "continuous":
            gripper_pos = model_output[7] * np.array([255.0])
        else:  # binary
            gripper_pos = (
                np.array([255.0]) if model_output[7] > self.grasping_threshold else np.array([0.0])
            )

        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_action_chunk(self, observation: Any) -> list[dict[str, np.ndarray]]:
        """Return the whole predicted chunk -- mirrors PI_Policy.get_action_chunk."""
        first_action = self.get_action(observation)
        chunk_len = self._chunk_len()
        buffered_outputs = self.actions_buffer[self.current_buffer_index : chunk_len]
        self.current_buffer_index = chunk_len
        return [first_action] + [self.model_output_to_action(out) for out in buffered_outputs]

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "molmoact2_droid"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.action_horizon
        info["policy_num_integration_steps"] = self.num_steps
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
