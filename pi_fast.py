import pickle

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.policies import PreTrainedConfig
from torchvision.transforms.v2 import Normalize as ImgNorm
from transformers import AutoTokenizer

from pi0.modeling_pi0fast import PI0FASTPolicy
import numpy as np

from cleandiffuser.env import libero
import gym
import robosuite.utils.transform_utils as T
import matplotlib.pyplot as plt
import json


policy = PI0FASTPolicy.from_pretrained(
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero_pytorch"
)

norm_stats_path = "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json"
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
    "libero-goal-v0",  # from ["libero-goal-v0", "libero-object-v0", "libero-spatial-v0", "libero-10-v0", "libero-90-v0"],
    task_id=0,  # task id from 0 to 9
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
# state = (unnorm_state - state_min) / (state_max - state_min + 1e-6) * 2 - 1
state = (unnorm_state - state_mean) / (state_std + 1e-6)
# state = np.clip(state, -1., 1.)
base_0_rgb = o['agentview_image'][:,:,::-1].copy()
left_wrist_0_rgb = o['robot0_eye_in_hand_image'][:,:,::-1].copy()

# with open("libero_example.pkl", "rb") as f:
#     example = pickle.load(f)
# unnorm_state = example['state']
# state = (unnorm_state - state_mean) / (state_std + 1e-6)
# base_0_rgb = example['image']['base_0_rgb']
# left_wrist_0_rgb = example['image']['left_wrist_0_rgb']

observation = {
    "image": {
        "base_0_rgb": torch.from_numpy(base_0_rgb).to("cuda:1")[None],
        "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to("cuda:1")[None],
        "right_wrist_0_rgb": torch.from_numpy(np.zeros_like(left_wrist_0_rgb)).to("cuda:1")[None]
    },
    "state": torch.from_numpy(state).to("cuda:1")[None],
    "prompt": [env.task_description]
}
# observation['prompt'] = example['prompt']

action = policy.select_action(observation)[0]
action = action.cpu().numpy()
# action = (action + 1) / 2 * (action_max - action_min + 1e-6) + action_min
action = action * (action_std + 1e-6) + action_mean
action[:, :6] += unnorm_state[None, :6]
for i in range(10):
    o, r, d, _ = env.step(action[i, :7])
plt.imshow(o['agentview_image'][:,:,::-1].transpose(1,2,0)) 