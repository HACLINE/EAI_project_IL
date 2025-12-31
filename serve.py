from typing import Optional
from env_client import websocket_policy_server as _websocket_policy_server
from policy import LerobotPolicy, LerobotPolicyConfig
import dataclasses
import tyro


@dataclasses.dataclass
class PolicyServerConfig:
    policy: LerobotPolicyConfig
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None


def create_policy_server(config: PolicyServerConfig) -> _websocket_policy_server.WebsocketPolicyServer:
    policy = LerobotPolicy(config.policy)
    server = _websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )
    return server


if __name__ == "__main__":
    config = tyro.cli(PolicyServerConfig)
    server = create_policy_server(config)
    server.serve_forever()
