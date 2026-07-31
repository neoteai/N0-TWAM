# Adapted from Physical Intelligence's openpi (https://github.com/Physical-Intelligence/openpi),
# Apache License 2.0.
import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from .msgpack_numpy import Packer, unpackb


class WebsocketClientPolicy:
    """Talks to a policy server over a websocket.

    See WebsocketPolicyServer for the corresponding server implementation.
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: Optional[int] = None,
                 api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(
            self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {
                    "Authorization": f"Api-Key {self._api_key}"
                } if self._api_key else None
                # disable ping so long-running inference calls don't trigger a timeout
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    close_timeout=10)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, OSError) as e:
                logging.info(f"Still waiting for server... (Error: {e})")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    @override
    def reset(self) -> None:
        self.infer(dict(reset=True))
