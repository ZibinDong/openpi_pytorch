import pickle

import torch

from pi0.new_modeling_pi0 import PI0Policy
import numpy as np

from cleandiffuser.env import libero
import gym
import robosuite.utils.transform_utils as T
import matplotlib.pyplot as plt
import json


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
    task_id=2,  # task id from 0 to 9
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