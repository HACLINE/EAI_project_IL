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

@dataclasses.dataclass
class LerobotPolicyConfig:
    path: str = ""
    robot_type: Literal['so101', 'bi_so101'] = "so101"
    action_horizon: int = -1
    device: str = "cuda"
    
class LerobotPolicy:
    def __init__(self, config: LerobotPolicyConfig):
        cfg = PreTrainedConfig.from_pretrained(config.path)
        cfg.path = Path(config.path)
        self.action_horizon = config.action_horizon
        self.robot_type = config.robot_type
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(pretrained_name_or_path=config.path, config=cfg)
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(config.path),
            preprocessor_overrides={
                "device_processor": {"device": config.device},
            },
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        
    def reset(self, **kwargs):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
    
    def get_action(self, observation: Dict[str, Any]) -> np.ndarray:
        obs = {}
        if self.robot_type == "so101":
            obs["observation.state"] = observation["states"]["arm"]
            obs["observation.images.front"] = observation["images"]["front"]
            obs["observation.images.wrist"] = observation["images"]["wrist"]
        elif self.robot_type == "bi_so101":
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
        action = self.postprocessor(action)[0]
        if self.action_horizon > 0:
            action = action[:self.action_horizon]
        
        return action.cpu().numpy()