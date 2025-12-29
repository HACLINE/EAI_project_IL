import grasp_cube.real
import gymnasium as gym
import tyro
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
from grasp_cube.real.lerobot_env import LeRobotEnvConfig
from env_client import rtg_client_policy as _rtg_client_policy
from grasp_cube.real import MonitorWrapper, EvalRecordConfig, EvalRecordWrapper
from collections import deque

@dataclasses.dataclass
class Args:
    env: LeRobotEnvConfig 
    eval: EvalRecordConfig
    host: str = "0.0.0.0"
    port: int = 8000
    monitor_host: str = "0.0.0.0"
    monitor_port: int = 9000
    num_episodes: int = 10
    
    # RTG parameters
    max_velocity: float = 1.0
    blending_horizon: int = 10
    alpha: float = 0.5
    
def main(args: Args):
    env = gym.make(
        "LeRobotEnv-v0",
        config=args.env,
    )
    env = MonitorWrapper(env, port=args.monitor_port, host=args.monitor_host, include_images=True)
    env = EvalRecordWrapper(env, config=args.eval)
    
    client = _rtg_client_policy.RTGClientPolicy(
        host=args.host, 
        port=args.port,
        max_velocity=args.max_velocity,
        blending_horizon=args.blending_horizon,
        alpha=args.alpha
    )
    
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        client.reset()
        done = False
        
        while not done:
            # get_action will trigger inference if needed and return the next action
            action = client.get_action(obs)
            
            if action is None:
                print("Error: No action returned from policy")
                break
                
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
