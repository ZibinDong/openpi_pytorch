import asyncio
import base64
import json
import pickle
from typing import Any, Dict, List, Union

import numpy as np
import websockets


class PolicyServer:
    def __init__(self, policy, host="0.0.0.0", port=12345):
        self.policy = policy
        self.host = host
        self.port = port

    def serialize_obs(self, obs: Dict[str, Union[np.ndarray, List[str]]]) -> str:
        serialized = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                serialized[key] = {
                    "type": "numpy",
                    "data": base64.b64encode(pickle.dumps(value)).decode("utf-8"),
                    "dtype": str(value.dtype),
                    "shape": value.shape,
                }
            elif isinstance(value, list):
                serialized[key] = {"type": "list", "data": value}
            else:
                raise ValueError(f"Unsupported data type for key {key}: {type(value)}")

        return json.dumps(serialized)

    def deserialize_obs(self, data: str) -> Dict[str, Union[np.ndarray, List[str]]]:
        serialized = json.loads(data)
        obs = {}

        for key, value in serialized.items():
            if value["type"] == "numpy":
                array_data = pickle.loads(base64.b64decode(value["data"]))
                obs[key] = array_data
            elif value["type"] == "list":
                obs[key] = value["data"]
            else:
                raise ValueError(f"Unknown data type: {value['type']}")

        return obs

    def serialize_action(self, action: Any) -> str:
        if isinstance(action, np.ndarray):
            return json.dumps(
                {
                    "type": "numpy",
                    "data": base64.b64encode(pickle.dumps(action)).decode("utf-8"),
                }
            )
        else:
            return json.dumps(
                {
                    "type": "other",
                    "data": base64.b64encode(pickle.dumps(action)).decode("utf-8"),
                }
            )

    async def handle_client(self, websocket):
        print(f"Client connected from {websocket.remote_address}")
        try:
            async for message in websocket:
                try:
                    obs = self.deserialize_obs(message)
                    print(f"Received obs with keys: {list(obs.keys())}")

                    action = self.policy(obs)
                    print(f"Computed action: {type(action)}")

                    action_data = self.serialize_action(action)
                    await websocket.send(action_data)

                except Exception as e:
                    error_msg = json.dumps({"error": str(e)})
                    await websocket.send(error_msg)
                    print(f"Error processing request: {e}")

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
        except Exception as e:
            print(f"Error in handle_client: {e}")

    async def start_server(self):
        print(f"Starting Policy server on {self.host}:{self.port}")

        server = await websockets.serve(
            self.handle_client, self.host, self.port, ping_interval=20, ping_timeout=10
        )

        print("Policy server is running and waiting for connections...")
        await server.wait_closed()

    def run(self):
        asyncio.run(self.start_server())
