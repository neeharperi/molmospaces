"""Shared websocket client for inference servers that dispatch on an ``endpoint`` field.

DreamZero and TiPToP both run as separate remote inference servers speaking the same little
protocol: msgpack-packed dicts carrying an ``endpoint`` key (``"infer"`` or ``"reset"``), a
metadata frame on connect, and a plain-string body when the server wants to report an error.

The two clients were originally copied from one another, and the copies drifted in the one
place it mattered: TiPToP's reconnect loop was bounded after an unbounded one cost the
campaign about 8 hours on 2026-08-18 (the server was OOM-killed and the client logged
"Retrying in 2s..." all night while producing no episodes and no provenance), but that fix was
never carried back to DreamZero -- which runs at ``--num_workers 1``, so a wedge stalls the
whole lane rather than one worker. Sharing the client is what keeps the next hardening fix
from having to be written twice.
"""

import logging
import time

import websockets.exceptions
import websockets.sync.client

log = logging.getLogger(__name__)

PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 600

# Bound the reconnect loop. Generous but finite -- a server restart with model warmup
# legitimately takes minutes, so failing the cell after this budget means the matrix runner
# moves on and the loss is one task, not a night.
RECONNECT_MAX_ATTEMPTS = 15
RECONNECT_INITIAL_DELAY_SECS = 2
RECONNECT_MAX_DELAY_SECS = 30

# Bound every recv() as well. The reconnect path only covers a server that has *closed* the
# connection; a server that stays up but never answers is a different failure, and
# websockets.sync's recv() blocks forever on it. That is what happened to tiptop/PnP-NextTo-v2
# on 2026-09-04: workers 0, 1 and 2 wedged mid-episode with no traceback and zero CPU
# thereafter, leaving worker 3 to run the cell alone for 29 hours at a quarter of the
# requested throughput. The slowest legitimate calls observed are ~74s (TiPToP planning) and
# ~9s (DreamZero inference), so 600s is ample headroom and only fires on a genuine wedge.
RECV_TIMEOUT_SECS = 600


class EndpointWebsocketClient:
    """Websocket client that tags each request with an ``endpoint`` field.

    Subclasses supply the msgpack implementation, because the wire format is not shared: the
    standalone PyPI ``msgpack_numpy`` package encodes ndarrays differently from
    ``openpi_client.msgpack_numpy`` (a distinct implementation openpi ships and the DreamZero
    server vendors as its own convention). Using the wrong one silently round-trips ndarrays
    as plain dicts server-side instead of raising -- it doesn't fail on send, only on the
    receiving end's first attempted array op.
    """

    #: Module exposing ``Packer`` and ``unpackb``; set by each subclass.
    msgpack = None

    def __init__(self, host: str, port: int) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = self.msgpack.Packer()
        self._ws, self._server_metadata = self._wait_for_server()
        # _wait_for_server may have switched self._uri to wss://, so reconnects reuse whatever
        # actually worked.
        self._connected_uri = self._uri

    def _connect_once(self, uri: str) -> tuple[websockets.sync.client.ClientConnection, dict]:
        conn = websockets.sync.client.connect(
            uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = self.msgpack.unpackb(conn.recv(timeout=RECV_TIMEOUT_SECS))
        return conn, metadata

    def _wait_for_server(self) -> tuple[websockets.sync.client.ClientConnection, dict]:
        log.info(f"Waiting for server at {self._uri}...")
        try:
            return self._connect_once(self._uri)
        except Exception:
            log.info("Connection with ws:// failed. Trying wss:// ...")
        self._uri = "wss://" + self._uri.split("//")[1]
        return self._connect_once(self._uri)

    def _reconnect(self) -> None:
        """Reconnect with bounded exponential backoff, raising if the server stays down."""
        retry_delay = RECONNECT_INITIAL_DELAY_SECS
        last_error: Exception | None = None
        for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
            log.warning(
                f"WebSocket connection closed. Reconnecting to {self._connected_uri} "
                f"(attempt {attempt}/{RECONNECT_MAX_ATTEMPTS})..."
            )
            try:
                self._ws, self._server_metadata = self._connect_once(self._connected_uri)
                log.info("Reconnected to server.")
                return
            except Exception as e:
                last_error = e
                log.warning(f"Reconnect failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, RECONNECT_MAX_DELAY_SECS)
        raise RuntimeError(
            f"Server at {self._connected_uri} did not come back after "
            f"{RECONNECT_MAX_ATTEMPTS} attempts; last error: {last_error!r}. "
            f"Failing this cell rather than retrying indefinitely -- check the server log "
            f"(it is not restarted automatically) and re-run; scripts/eval.py is resumable."
        )

    def _request(self, endpoint: str, payload: dict):
        """Send one request, reconnecting and retrying once if the connection dropped."""
        payload["endpoint"] = endpoint
        data = self._packer.pack(payload)
        try:
            self._ws.send(data)
            return self._ws.recv(timeout=RECV_TIMEOUT_SECS)
        except (websockets.exceptions.ConnectionClosedError, TimeoutError) as e:
            log.warning(f"{type(e).__name__} during {endpoint}. Reconnecting and retrying...")
            self._reconnect()
            self._ws.send(data)
            return self._ws.recv(timeout=RECV_TIMEOUT_SECS)

    def _decode_str_response(self, response: str):
        """Handle a plain-string body. By default the server only sends one to report errors."""
        raise RuntimeError(f"Error in inference server:\n{response}")

    def infer(self, obs: dict) -> dict:
        response = self._request("infer", obs)
        if isinstance(response, str):
            return self._decode_str_response(response)
        return self.msgpack.unpackb(response)

    def reset(self, reset_info: dict | None = None):
        return self._request("reset", reset_info if reset_info is not None else {})

    def get_server_metadata(self) -> dict:
        return self._server_metadata
