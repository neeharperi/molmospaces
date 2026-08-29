import collections
import logging
import os
import time
import uuid

import cv2
import numpy as np
import websockets.exceptions
import websockets.sync.client
# The standalone PyPI `msgpack_numpy` package encodes ndarrays with a different wire format
# than `openpi_client.msgpack_numpy` (a distinct implementation openpi ships and this server
# vendors as its own encode/decode convention -- confirmed against
# third_party/dreamzero/eval_utils/policy_server.py's own `from openpi_client import
# msgpack_numpy`). Using the standalone package here silently round-trips ndarrays as plain
# dicts server-side instead of raising -- it doesn't crash on send, only on the receiving
# end's first attempted array op.
from openpi_client import msgpack_numpy

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import shard_port, resize_with_pad

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600


class DreamZeroWebsocketClient:
    """Websocket client that adds endpoint field for DreamZero server."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        # store the URI that actually worked so reconnects reuse it
        self._connected_uri = self._uri

    def _connect_once(self, uri: str) -> tuple[websockets.sync.client.ClientConnection, dict]:
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn, metadata = self._connect_once(self._uri)
            return conn, metadata
        except Exception:
            logging.info("Connection with ws:// failed. Trying wss:// ...")

        wss_uri = "wss://" + self._uri.split("//")[1]
        conn, metadata = self._connect_once(wss_uri)
        self._uri = wss_uri
        return conn, metadata

    def _reconnect(self) -> None:
        retry_delay = 2
        while True:
            logging.warning(
                f"WebSocket connection closed. Reconnecting to {self._connected_uri}..."
            )
            try:
                self._ws, self._server_metadata = self._connect_once(self._connected_uri)
                logging.info("Reconnected to server.")
                return
            except Exception as e:
                logging.warning(f"Reconnect failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    def infer(self, obs: dict) -> dict:
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("ConnectionClosedError during infer. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: dict = None) -> None:
        if reset_info is None:
            reset_info = {}
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        try:
            self._ws.send(data)
            response = self._ws.recv()
        except websockets.exceptions.ConnectionClosedError:
            logging.warning("ConnectionClosedError during reset. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            response = self._ws.recv()
        return response


class DreamZero_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
    ) -> None:
        super().__init__(exp_config)
        self.remote_config = exp_config.policy_config.remote_config
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        self.chunk_size = exp_config.policy_config.chunk_size
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.model = None
        self.session_id = None

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None
        self.session_id = str(uuid.uuid4())
        # Rolling per-camera frame history the AR block-conditioning stack is drawn from
        # (oldest first). Cleared per episode -- see obs_to_model_input for why this exists.
        self.frame_history = {
            "exterior_0": collections.deque(maxlen=self.chunk_size),
            "exterior_1": collections.deque(maxlen=self.chunk_size),
            "wrist": collections.deque(maxlen=self.chunk_size),
        }

    def prepare_model(self):
        self.model_name = (
            os.path.basename(self.checkpoint_path) if self.checkpoint_path else "dreamzero"
        )
        if self.remote_config is not None:
            self._prepare_remote_model()
        else:
            raise NotImplementedError("DreamZero policy only supports remote model inference")

    def _prepare_remote_model(self):
        host = self.remote_config.get("host", "localhost")
        # Shard across server instances when several are running (see
        # molmo_spaces/policy/learned_policy/utils.py's shard_port). No-op at the
        # default of one instance.
        port = shard_port(self.remote_config.get("port", 6000))

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = DreamZeroWebsocketClient(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to DreamZero model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        # task.reset()/get_observations() always return list[dict] (one dict per batch
        # index, even for batch size 1) -- every other policy wrapper (pi_policy.py,
        # molmoact2_policy.py) unwraps this; this one never did, since it pre-dates this
        # project and was never actually run against the real observation pipeline before.
        if isinstance(obs, list | tuple):
            obs = obs[0]
        # self.render(obs)
        prompt = self.task.get_task_description()
        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        if grip < 0.1:
            grip = 0.00
        # DreamZero was trained on 3 distinct DROID views (2 exterior + 1 wrist). Confirmed
        # against real assets: FrankaOmniPurposeCameraSystem benchmarks (all of Group B
        # except Pick-v1.5) expose two genuinely distinct exterior views,
        # randomized_zed2_analogue_1/2. FrankaDroidCameraSystem benchmarks (Open-v1,
        # Close-v1, Pick-v1.5) only ever expose ONE exterior camera (exo_camera_1) -- there
        # is no second view to give DreamZero for these three tasks, so duplication there is
        # expected and unavoidable, not a bug. Still log it, since it's a real deviation from
        # DreamZero's training distribution and must be visible in reported numbers, not
        # silently absorbed (BENCHMARK.md's Risks section).
        if "randomized_zed2_analogue_1" in obs:
            exo_camera_key_0, exo_camera_key_1_fallback = (
                "randomized_zed2_analogue_1",
                "randomized_zed2_analogue_2",
            )
        else:
            exo_camera_key_0, exo_camera_key_1_fallback = "exo_camera_1", "exo_camera_2"
        exo_camera_key_1 = exo_camera_key_1_fallback if exo_camera_key_1_fallback in obs else exo_camera_key_0
        if exo_camera_key_1 == exo_camera_key_0:
            log.warning(
                f"Only one exterior camera ({exo_camera_key_0}) found in obs; duplicating it "
                f"into both DreamZero exterior slots. This is a known deviation -- see "
                f"docs/eval_reproduction.md."
            )
        wrist_camera_key = (
            "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        )

        # DreamZero's server is autoregressive over 24-step (chunk_size) video blocks: every
        # call after the first must condition on a 4-frame history stack spanning the block
        # (offsets -23,-16,-8,0), not just the current frame -- confirmed against upstream's
        # own third_party/dreamzero/test_client_AR.py reference client (RELATIVE_OFFSETS /
        # ACTION_HORIZON). Sending a single current frame on every call (the original bug
        # here) silently desyncs the server's causal KV-cache/RoPE positions from the actions
        # actually executed. The history is appended every step regardless of whether this
        # step will trigger a new server call, so the deque always reflects the true rollout.
        exterior_0_frame = resize_with_pad(obs[exo_camera_key_0], 180, 320)
        exterior_1_frame = resize_with_pad(obs[exo_camera_key_1], 180, 320)
        wrist_frame = resize_with_pad(obs[wrist_camera_key], 180, 320)
        self.frame_history["exterior_0"].append(exterior_0_frame)
        self.frame_history["exterior_1"].append(exterior_1_frame)
        self.frame_history["wrist"].append(wrist_frame)

        needs_new_chunk = self.actions_buffer is None or self.current_buffer_index >= self.chunk_size
        if needs_new_chunk and len(self.frame_history["exterior_0"]) > 1:
            exterior_0_image = self._history_stack(self.frame_history["exterior_0"])
            exterior_1_image = self._history_stack(self.frame_history["exterior_1"])
            wrist_image = self._history_stack(self.frame_history["wrist"])
        else:
            # First call of the episode (matches upstream's "send frame [0]" step), or a
            # mid-chunk step whose model_input will never actually be sent to the server
            # (inference_model returns a buffered action instead) -- either way a single
            # current frame is the correct/harmless payload.
            exterior_0_image, exterior_1_image, wrist_image = (
                exterior_0_frame,
                exterior_1_frame,
                wrist_frame,
            )

        model_input = {
            "observation/exterior_image_0_left": exterior_0_image,
            "observation/exterior_image_1_left": exterior_1_image,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": np.array(
                obs["qpos"]["arm"][:7], dtype=np.float64
            ).reshape(
                7,
            ),
            "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
            "observation/gripper_position": np.array(grip, dtype=np.float64).reshape(
                1,
            ),
            "prompt": prompt,
            "session_id": self.session_id,
        }
        return model_input

    @staticmethod
    def _history_stack(history: collections.deque) -> np.ndarray:
        """The 4 conditioning frames for one camera, as (4, H, W, 3), oldest first.

        Early in an episode the history is shorter than the block; the oldest available frame
        is repeated to fill, matching how the server itself pads a short buffer.
        """
        frames = list(history)
        newest = len(frames) - 1
        offsets = (-23, -16, -8, 0)
        return np.stack([frames[max(0, newest + offset)] for offset in offsets])

    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            # socket_test_optimized_AR.py's infer() returns a bare (N, 8) ndarray -- not a
            # dict -- confirmed directly against third_party/dreamzero's own server (Policy
            # Server "action" docstring) and test_client_AR.py's usage (`actions =
            # client.infer(obs)`, no further indexing).
            self.actions_buffer = self.model.infer(model_input)
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
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

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "dreamzero"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.task.get_task_description()
        info["session_id"] = self.session_id
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
