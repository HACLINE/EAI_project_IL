import gymnasium as gym
from typing import Dict, Any, Literal
import numpy as np
import dataclasses
import tyro
import torch
from pathlib import Path
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.utils import get_safe_torch_device
from lerobot.envs.utils import preprocess_observation
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

# single arm observation space
# images {'front': Box(0, 255, (480, 640, 3), uint8), 'wrist': Box(0, 255, (480, 640, 3), uint8)}
# states {'left_arm': Box(-np.inf, np.inf, (6,), float32), 'right_arm': Box(-np.inf, np.inf, (6,), float32)} --- IGNORE ---
# actions Box(-np.inf, np.inf, (6,), float32)

# dual arm observation space
# images {'front': Box(0, 255, (480, 640, 3), uint8), 'left_wrist': Box(0, 255, (480, 640, 3), uint8), 'right_wrist': Box(0, 255, (480, 640, 3), uint8)}
# states {'arm': Box(-np.inf, np.inf, (6,), float32)}
# actions Box(-np.inf, np.inf, (12,), float32) [left, right]

class DummyEnv(gym.Env):
    def __init__(self, mode: Literal['single_arm', 'dual_arm'] = 'single_arm'):
        super(DummyEnv, self).__init__()
        if mode == 'single_arm':
            self.observation_space = gym.spaces.Dict({
                'images': gym.spaces.Dict({
                    'front': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                    'wrist': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                }),
                "states": gym.spaces.Dict({
                    'arm': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                }),
            })
            self.action_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        elif mode == 'dual_arm':
            self.observation_space = gym.spaces.Dict({
                'images': gym.spaces.Dict({
                    'front': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                    'left_wrist': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                    'right_wrist': gym.spaces.Box(0, 255, (480, 640, 3), dtype=np.uint8),
                }),
                "states": gym.spaces.Dict({
                    'left_arm': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                    'right_arm': gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32),
                }),
            })
            self.action_space = gym.spaces.Box(-np.inf, np.inf, (12,), dtype=np.float32)
        else:
            raise ValueError("mode must be 'single_arm' or 'dual_arm'")

    def get_observation(self) -> Dict[str, Any]:
        return self.observation_space.sample()
    
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        observation = self.get_observation()
        info = {}
        return observation, info
    
    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation = self.get_observation()
        reward = np.random.rand()
        terminated = np.random.rand() < 0.05  # 5% chance to terminate each step
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info
    
@dataclasses.dataclass
class LerobotPolicyConfig:
    path: str = ""
    policy_type: Literal['single_arm', 'dual_arm'] = 'single_arm'
    action_horizon: int = 16
    device: str = "cuda"
    
class LerobotPolicy:
    def __init__(self, config: LerobotPolicyConfig):
        cfg = PreTrainedConfig.from_pretrained(config.path)
        cfg.path = Path(config.path)
        self.robot_type = "so101" if config.policy_type == "single_arm" else "bi_so101"
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(pretrained_name_or_path=config.path, config=cfg)
        self.policy_type = config.policy_type
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            path=str(config.path),
            preprocessor_overrides={
                "device_processor": {"device": config.device},
            },
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
    def reset(self, **kwargs):
        self.policy.reset()
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        obs = {}
        if self.policy_type == "single_arm":
            obs["observation.state"] = observation["states"]["arm"]
            obs["observation.images.front"] = observation["images"]["front"]
            obs["observation.images.wrist"] = observation["images"]["wrist"]
        elif self.policy_type == "dual_arm":
            obs["observation.state"] = np.concatenate([
                observation["states"]["left_arm"],
                observation["states"]["right_arm"],
            ], axis=-1)
            obs["observation.images.front"] = observation["images"]["front"]
            obs["observation.images.left_wrist"] = observation["images"]["left_wrist"]
            obs["observation.images.right_wrist"] = observation["images"]["right_wrist"]
        obs_infer = prepare_observation_for_inference(obs, self.device, observation["task"], self.robot_type)
        obs_infer_processed = self.preprocessor(obs_infer)
        with torch.inference_mode():
            action = self.policy.predict_action_chunk(obs_infer_processed)
        action = self.postprocessor(action)
        
        return action[0].cpu().numpy()

def main(num_episodes: int, policy: LerobotPolicyConfig, prompt: str = 'hello EAI!', env_mode: Literal['single_arm', 'dual_arm'] = 'single_arm'):
    env = DummyEnv(mode=env_mode)
    policy = LerobotPolicy(config=policy)

    for episode in range(num_episodes):
        done = False
        step_count = 0
        obs, info = env.reset()
        policy.reset()
        while not done:
            obs['task'] = prompt
            action_seq = policy.get_action(obs)
            for action in action_seq:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                if done:
                    break
        print(f"Episode {episode + 1} finished in {step_count} steps.")
    
if __name__ == "__main__":
    tyro.cli(main)