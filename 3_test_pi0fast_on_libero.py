import json
from pathlib import Path

import gym
import imageio
import numpy as np
import robosuite.utils.transform_utils as T
import torch
from cleandiffuser.env import libero  # noqa: F401
from termcolor import cprint

from pi0 import PI0FASTPolicy

PATH_TO_PI_MODEL = (
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero_pytorch"
)
PATH_TO_JAX_PI_MODEL = (
    "/home/dzb/.cache/openpi/openpi-assets/checkpoints/pi0_fast_libero"
)

# load model
cprint("Loading PI0 fast model...", "green")
policy = PI0FASTPolicy.from_pretrained(PATH_TO_PI_MODEL)
policy.model.action_dim = 7

# load normalization stats
device = policy.config.device
norm_stats_path = (
    Path(PATH_TO_JAX_PI_MODEL) / "assets/physical-intelligence/libero/norm_stats.json"
)
with open(norm_stats_path) as f:
    norm_stats = json.load(f)
state_mean = np.array(norm_stats["norm_stats"]["state"]["mean"][:8], dtype=np.float32)
state_std = np.array(norm_stats["norm_stats"]["state"]["std"][:8], dtype=np.float32)
action_mean = np.array(
    norm_stats["norm_stats"]["actions"]["mean"][:7], dtype=np.float32
)
action_std = np.array(norm_stats["norm_stats"]["actions"]["std"][:7], dtype=np.float32)

# create environment
# ** Change `env_name` and `task_id` to test different environments and tasks **
cprint("Creating Libero environment...", "green")
env = gym.make(
    "libero-goal-v0",  # from ["libero-goal-v0", "libero-object-v0", "libero-spatial-v0", "libero-10-v0", "libero-90-v0"],
    task_id=0,  # task id from 0 to 9
    image_size=224,  # image size (height, width)
    camera_names=["agentview", "robot0_eye_in_hand"],  # camera names
    seed=0,  # random seed
    max_episode_steps=300,
)

# reset environment
o = env.reset()
# important: do some `dummy` steps because the simulator drops object at the beginning
dummy_action = np.array([0, 0, 0, 0, 0, 0, -1])
for _ in range(10):
    o, r, d, i = env.step(dummy_action)

frames = []
cprint("Starting demo...", "green")
while not d:
    unnorm_state = np.concatenate(
        [
            o["robot0_eef_pos"],
            T.quat2axisangle(o["robot0_eef_quat"]),
            o["robot0_gripper_qpos"],
        ],
        dtype=np.float32,
    )
    state = (unnorm_state - state_mean) / (state_std + 1e-6)
    base_0_rgb = o["agentview_image"][:, :, ::-1].copy()
    left_wrist_0_rgb = o["robot0_eye_in_hand_image"][:, :, ::-1].copy()

    observation = {
        "image": {
            "base_0_rgb": torch.from_numpy(base_0_rgb).to(device)[None],
            "left_wrist_0_rgb": torch.from_numpy(left_wrist_0_rgb).to(device)[None],
            # "right_wrist_0_rgb": torch.from_numpy(np.zeros_like(left_wrist_0_rgb)).to(
            #     "cuda:1"
            # )[None],
        },
        "state": torch.from_numpy(state).to(device)[None],
        "prompt": [env.language],
    }
    # action = policy.select_action(observation)[0, :, :7]
    action = policy.select_action(observation)[0]
    action = action.cpu().numpy()
    action = action * (action_std + 1e-6) + action_mean
    action[:, :6] += unnorm_state[None, :6]
    for i in range(5):
        o, r, d, _ = env.step(action[i])
        frames.append(o["agentview_image"][:, :, ::-1].transpose(1, 2, 0).copy())
        if d:
            break

# save video
writer = imageio.get_writer("pi0fast_libero_demo.mp4", fps=30)
for frame in frames:
    writer.append_data(frame)
writer.close()
