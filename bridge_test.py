import uuid
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

path = Path("/mnt/20T/datasets/bridgev2/tf")


dataset = LeRobotDataset.create(
    repo_id="ZibinDong/bridgedatav2_train",
    root="/mnt/20T/datasets/bridgev2/lerobot/ZibinDong/bridgedatav2_train",
    robot_type="WidowX250",
    fps=5,
    features={
        "image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["action"],
        },
        "traj_idx": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["traj_idx"],
        },
    },
    image_writer_threads=10,
    image_writer_processes=5,
)

builder = tfds.builder_from_directory(path)
ds = builder.as_dataset(split="train")


for idx, episode in tqdm.tqdm(enumerate(tfds.as_numpy(ds))):
    images = [[], [], [], []]
    actions, states = [], []
    has_image = [False, False, False, False]

    for i, step in enumerate(episode["steps"]):
        if i == 0:
            language_instruction = step.get("language_instruction", b"").decode("utf-8")
            for img_idx in range(4):
                if step["observation"][f"image_{img_idx}"].mean() > 0:
                    has_image[img_idx] = True
            if not any(has_image):
                break

        for img_idx in range(4):
            if has_image[img_idx]:
                images[img_idx].append(step["observation"][f"image_{img_idx}"])

        actions.append(step["action"])
        states.append(step["observation"]["state"])

    if not any(has_image):
        continue

    for img_idx in range(4):
        if has_image[img_idx]:
            for image, action, state in zip(images[img_idx], actions, states):
                dataset.add_frame(
                    {
                        "image": image,
                        "action": action,
                        "state": state,
                        "traj_idx": np.array([idx], dtype=np.int64),
                    },
                    task=language_instruction,
                )
            dataset.save_episode()
