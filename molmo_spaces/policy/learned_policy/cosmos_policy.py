import logging
import os
import time
from typing import Any

import numpy as np
from PIL import Image

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Cosmos3-{Edge,Nano}-Policy-DROID's server wraps openpi's own WebsocketPolicyServer, so the
# wire transport is identical to PI_Policy's -- but action_policy_server_robolab.py has its
# own request/response field names, confirmed directly against a sibling project's already-
# working client for this exact server (see docs/eval_reproduction.md):
#   - response key is "action" (singular), not "actions"
#   - two exterior slots, 1-indexed ("exterior_image_1_left"/"_2_left"), not 0-indexed
#   - no "observation/cartesian_position" field
#   - the server always bilinear-resizes (no antialiasing) both exterior images to half the
#     wrist frame's size before composing a canvas; pre-shrinking with LANCZOS here to that
#     exact half-size (rather than sending full-size and letting the server's own resize do
#     it) avoids stacking a second, non-antialiased downsample on top of this one. The
#     checkpoint's own training data is 640x360 (16:9); our benchmarks' camera aspect ratios
#     vary, so this is a plain resize (no letterboxing) to match the server's expected
#     composite geometry, not an aspect-preserving one -- a documented, minor deviation for
#     non-16:9 source cameras.
_WRIST_HEIGHT = 360
_WRIST_WIDTH = 640
_EXTERIOR_HEIGHT = _WRIST_HEIGHT // 2
_EXTERIOR_WIDTH = _WRIST_WIDTH // 2


def _lanczos_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(
        Image.fromarray(image).resize((width, height), Image.LANCZOS), dtype=np.uint8
    )


class Cosmos_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.chunk_size = exp_config.policy_config.chunk_size
        self.camera_names = exp_config.policy_config.camera_names
        self.model = None  # don't init model till inference to allow multiprocessing

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None

    def prepare_model(self):
        self.model_name = os.path.basename(self.checkpoint_path)
        if self.remote_config is not None:
            self._prepare_remote_model()
        else:
            raise NotImplementedError("Cosmos policy only supports remote model inference")

    def _prepare_remote_model(self):
        from openpi_client import websocket_client_policy

        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8003)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = websocket_client_policy.WebsocketClientPolicy(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to Cosmos model at {host}:{port}")
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
        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)

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

        # PolaRiS/this benchmark both expose one exterior camera; duplicated into both
        # exterior slots the server expects, same move dreamzero_client.py makes.
        exterior_image = _lanczos_resize(obs[exo_camera_key], _EXTERIOR_HEIGHT, _EXTERIOR_WIDTH)
        wrist_image = _lanczos_resize(obs[wrist_camera_key], _WRIST_HEIGHT, _WRIST_WIDTH)

        model_input = {
            "observation/exterior_image_1_left": exterior_image,
            "observation/exterior_image_2_left": exterior_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": np.array(
                obs["qpos"]["arm"][:7], dtype=np.float32
            ).reshape(
                7,
            ),
            "observation/gripper_position": np.array(grip, dtype=np.float32).reshape(
                1,
            ),
            "prompt": prompt.lower(),
        }
        return model_input

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            response = self.model.infer(model_input)
            # action_policy_server_robolab.py returns {"action": ndarray} (singular) --
            # unlike the openpi/DreamZero servers' plural "actions". Fail loudly rather than
            # silently guessing if a future server revision changes this.
            if "action" not in response:
                raise KeyError(
                    f"expected an 'action' key in the CosmosPolicy server response, got "
                    f"{list(response)}"
                )
            self.actions_buffer = np.asarray(response["action"], dtype=np.float32)
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        # No client-side gripper inversion needed: the server flips gripper_position once on
        # ingest and once again on the way out, so the round trip preserves this benchmark's
        # own "1 = closed" convention -- same as pi0.5/MolmoAct2, confirmed directly against
        # the training dataset's IS_GRIPPER_ACTION_FLIPPED entry for this checkpoint family
        # (see docs/eval_reproduction.md).
        gripper_pos = np.clip(model_output[7], 0.0, 1.0)
        if self.grasping_type == "binary":
            gripper_pos = (
                np.array([255.0]) if gripper_pos >= self.grasping_threshold else np.array([0.0])
            )
        elif self.grasping_type == "semi_binary":
            gripper_pos = (
                gripper_pos * np.array([255.0])
                if gripper_pos <= self.grasping_threshold
                else np.array([255.0])
            )
        elif self.grasping_type == "continuous":
            gripper_pos = gripper_pos * np.array([255.0])
        else:
            raise ValueError(f"Invalid grasping type: {self.grasping_type}")

        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_action_chunk(self, observation: Any) -> list[dict[str, np.ndarray]]:
        """Return the whole predicted chunk, so only its first action needs an observation.

        Mirrors PI_Policy.get_action_chunk -- same server family, same open-loop-chunk
        discipline (query once, drain chunk_size actions before re-querying).
        """
        first_action = self.get_action(observation)
        buffered_outputs = self.actions_buffer[self.current_buffer_index : self.chunk_size]
        self.current_buffer_index = self.chunk_size
        return [first_action] + [self.model_output_to_action(out) for out in buffered_outputs]

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "cosmos"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
