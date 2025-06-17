import pickle

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from torchvision.transforms.v2 import Normalize as ImgNorm
from transformers import AutoTokenizer

from pi0.modeling_pi0 import PI0Policy
import numpy as np

from cleandiffuser.env import libero
import gym
import robosuite.utils.transform_utils as T
import matplotlib.pyplot as plt
import json

# def dict_apply(func, d):
#     """
#     Recursively apply a function to all values in a dictionary.
#     """
#     if isinstance(d, dict):
#         return {k: dict_apply(func, v) for k, v in d.items()}
#     elif isinstance(d, list):
#         return [dict_apply(func, v) for v in d]
#     else:
#         return func(d)

# class Pi0FinetuneDatasetWrapper(torch.utils.data.Dataset):
#     def __init__(self, dataset: LeRobotDataset, stats_path: str):
#         self.dataset = dataset
#         with open(stats_path, "rb") as f:
#             self.stats = pickle.load(f)

#         self.action_max = self.stats["actions"]["max"][None]
#         self.action_min = self.stats["actions"]["min"][None]
#         self.state_mean = self.stats["state"]["mean"]
#         self.state_std = self.stats["state"]["std"]

#     def normalize_action(self, action):
#         return (action - self.action_min) / (self.action_max - self.action_min) * 2 - 1

#     def normalize_state(self, state):
#         return (state - self.state_mean) / self.state_std

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         lang_tokens = self.stats["lang_tokens"][item["task_index"]]
#         lang_masks = self.stats["lang_masks"][item["task_index"]]
#         item["image"] = {
#             "base_0_rgb": item["base_0_rgb"],
#             "right_wrist_0_rgb": item["right_wrist_0_rgb"],
#         }
#         del item["base_0_rgb"]
#         del item["right_wrist_0_rgb"]
#         del item["task"]
#         item["lang_tokens"] = lang_tokens
#         item["lang_masks"] = lang_masks
#         item["action"] = self.normalize_action(item["actions"]).to(torch.float32)
#         item["state"] = self.normalize_state(item["state"]).to(torch.float32)
#         return item

# config = PreTrainedConfig.from_pretrained(
#     pretrained_name_or_path="/home/dzb/pretrained/pi0"
# )
# config.device = "cuda:2"
# config.train_expert_only = True

# dataset = LeRobotDataset(
#     repo_id="ZibinDong/libero_goal",
#     delta_timestamps={"actions": [i / 20 for i in range(50)]},
#     image_transforms=ImgNorm(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
# )
# dataset = Pi0FinetuneDatasetWrapper(dataset, "stats/libero_goal_stats.pkl")
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
# batch = next(iter(dataloader))
# batch = dict_apply(lambda x: x.to(config.device), batch)

# policy = PI0Policy.from_pretrained("/home/dzb/pretrained/pi0", config=config)
policy = PI0Policy.from_pretrained(
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero_pytorch"
)

norm_stats_path = "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_libero/assets/physical-intelligence/libero/norm_stats.json"
with open(norm_stats_path) as f:
    norm_stats = json.load(f)

state_max = np.array(norm_stats['norm_stats']['state']['q99'][:8], dtype=np.float32)
state_min = np.array(norm_stats['norm_stats']['state']['q01'][:8], dtype=np.float32)
action_max = np.array(norm_stats['norm_stats']['actions']['q99'][:7], dtype=np.float32)
action_min = np.array(norm_stats['norm_stats']['actions']['q01'][:7], dtype=np.float32)
state_mean = np.array(norm_stats['norm_stats']['state']['mean'][:8], dtype=np.float32)
state_std = np.array(norm_stats['norm_stats']['state']['std'][:8], dtype=np.float32)
action_mean = np.array(norm_stats['norm_stats']['actions']['mean'][:7], dtype=np.float32)
action_std = np.array(norm_stats['norm_stats']['actions']['std'][:7], dtype=np.float32)

env = gym.make(
    "libero-10-v0",  # from ["libero-goal-v0", "libero-object-v0", "libero-spatial-v0", "libero-10-v0", "libero-90-v0"],
    task_id=1,  # task id from 0 to 9
    image_size=224,  # image size (height, width)
    camera_names=["agentview", "robot0_eye_in_hand"],  # camera names
    seed=0  # random seed
)

o = env.reset()
dummy_action = np.array([0,0,0,0,0,0,-1])
for _ in range(20):
    o, r, d, i = env.step(dummy_action)

unnorm_state = np.concatenate([
    o['robot0_eef_pos'], T.quat2axisangle(o["robot0_eef_quat"]), o['robot0_gripper_qpos']
], dtype=np.float32)
# state = (state - state_min) / (state_max - state_min) * 2 - 1
state = (unnorm_state - state_mean) / (state_std + 1e-6)
base_0_rgb = o['agentview_image'][:,:,::-1].copy()
left_wrist_0_rgb = o['robot0_eye_in_hand_image'][:,:,::-1].copy()

observation = {
    "image": {
        "base_0_rgb": torch.from_numpy(base_0_rgb).to("cuda:0")[None],
        "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to("cuda:0")[None],
    },
    "state": torch.from_numpy(state).to("cuda:0")[None],
    "prompt": [env.task_description]
}
action = policy.select_action(observation)[0, :, :7]
action = action.cpu().numpy()
action = action * (action_std + 1e-6) + action_mean
action[:, :6] += unnorm_state[None, :6]
for i in range(40):
    o, r, d, _ = env.step(action[i, :7])
plt.imshow(o['agentview_image'][:,:,::-1].transpose(1,2,0)) 