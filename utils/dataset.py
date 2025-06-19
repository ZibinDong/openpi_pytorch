import argparse
import os
from pathlib import Path

import h5py
import libero
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# --- Config ---

argparser = argparse.ArgumentParser()
argparser.add_argument("--image_size", type=int, default=224)
argparser.add_argument("--benchmark", type=str, default="libero_goal")
args = argparser.parse_args()

IMAGE_SIZE = args.image_size
LIBERO_PATH = Path(os.path.dirname(libero.libero.__file__)).parents[0]
DATASET_PATH = LIBERO_PATH / "datasets"
BENCHMARKS = [args.benchmark]

# benchmark for suite
benchmark_dict = benchmark.get_benchmark_dict()

# Total number of tasks
num_tasks = 0
for bm in BENCHMARKS:
    benchmark_path = DATASET_PATH / bm
    num_tasks += len(list(benchmark_path.glob("*.hdf5")))

tasks_stored = 0
for bm in BENCHMARKS:
    print(f"############################# {bm} #############################")
    benchmark_path = DATASET_PATH / bm

    # Init env benchmark suite
    task_suite = benchmark_dict[bm]()

    # Init lerobot dataset
    dataset = LeRobotDataset.create(
        repo_id=f"ZibinDong/{bm}",
        robot_type="panda",
        fps=20,
        features={
            "base_0_rgb": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "right_wrist_0_rgb": {
                "dtype": "video",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for task_file in benchmark_path.glob("*.hdf5"):
        print(f"Processing {tasks_stored + 1}/{num_tasks}: {task_file}")
        data = h5py.File(task_file, "r")["data"]

        # Init env
        task_name = str(task_file).split("/")[-1][:-10]
        # get task id from list of task names
        task_id = task_suite.get_task_names().index(task_name)
        # create environment
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": IMAGE_SIZE,
            "camera_widths": IMAGE_SIZE,
        }
        env = OffScreenRenderEnv(**env_args)

        obs = env.reset()

        states = []
        actions = []
        rewards = []
        episode_ends = []

        for demo in tqdm.tqdm(data.keys()):
            print(f"Processing demo {demo}")
            demo_data = data[demo]

            colors, colors_ego = [], []
            joint_states, eef_states, gripper_states = [], [], []

            for i in range(len(demo_data["states"])):
                obs = env.regenerate_obs_from_state(demo_data["states"][i])

                # get RGBD
                color = obs["agentview_image"][::-1]
                color_ego = obs["robot0_eye_in_hand_image"][::-1]
                eef_state = np.concatenate(
                    [obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                )
                gripper_state = obs["robot0_gripper_qpos"]

                dataset.add_frame(
                    {
                        "base_0_rgb": color,
                        "right_wrist_0_rgb": color_ego,
                        "state": np.concatenate([gripper_state, eef_state]).astype(np.float32),
                        "actions": demo_data["actions"][i].astype(np.float32),
                    },
                    task=env.language_instruction,
                )
            dataset.save_episode()

        print(f"{env.language_instruction}: Finish!")
        tasks_stored += 1

# dataset = LeRobotDataset("ZibinDong/libero_goal")
# dataset.push_to_hub(tags="libero", private=True)

"""
- data
    - demo_0
        - actions      float64(n, 7)
        - dones        uint8(n,)
        - rewards      uint8(n,)
        - robot_states float64(n, 9)  # (gripper_states(2), ee_pos(3), ee_quad(4))
        - states       float64(n, 79)
        - obs
            - agentview_rgb   uint8(n, 128, 128, 3)
            - eye_in_hand_rgb uint8(n, 128, 128, 3)
            - ee_ori          float64(n, 3)
            - ee_pos          float64(n, 3)
            - ee_states       float64(n, 6)
            - gripper_states  float64(n, 2)
            - joint_states    float64(n, 7)
    - demo_1
        - ...
"""


# REPO_NAME = "your_hf_username/libero"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_NAMES = [
#     "libero_10_no_noops",
#     "libero_goal_no_noops",
#     "libero_object_no_noops",
#     "libero_spatial_no_noops",
# ]  # For simplicity we will combine multiple Libero datasets into one training dataset


# def main(data_dir: str, *, push_to_hub: bool = False):
#     # Clean up any existing dataset in the output directory
#     output_path = LEROBOT_HOME / REPO_NAME
#     if output_path.exists():
#         shutil.rmtree(output_path)

#     # Create LeRobot dataset, define features to store
#     # OpenPi assumes that proprio is stored in `state` and actions in `action`
#     # LeRobot assumes that dtype of image data is `image`
#     dataset = LeRobotDataset.create(
#         repo_id=REPO_NAME,
#         robot_type="panda",
#         fps=10,
#         features={
#             "image": {
#                 "dtype": "image",
#                 "shape": (256, 256, 3),
#                 "names": ["height", "width", "channel"],
#             },
#             "wrist_image": {
#                 "dtype": "image",
#                 "shape": (256, 256, 3),
#                 "names": ["height", "width", "channel"],
#             },
#             "state": {
#                 "dtype": "float32",
#                 "shape": (8,),
#                 "names": ["state"],
#             },
#             "actions": {
#                 "dtype": "float32",
#                 "shape": (7,),
#                 "names": ["actions"],
#             },
#         },
#         image_writer_threads=10,
#         image_writer_processes=5,
#     )

#     # Loop over raw Libero datasets and write episodes to the LeRobot dataset
#     # You can modify this for your own data format
#     for raw_dataset_name in RAW_DATASET_NAMES:
#         raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
#         for episode in raw_dataset:
#             for step in episode["steps"].as_numpy_iterator():
#                 dataset.add_frame(
#                     {
#                         "image": step["observation"]["image"],
#                         "wrist_image": step["observation"]["wrist_image"],
#                         "state": step["observation"]["state"],
#                         "actions": step["action"],
#                     }
#                 )
#             dataset.save_episode(task=step["language_instruction"].decode())

#     # Consolidate the dataset, skip computing stats since we will do that later
#     dataset.consolidate(run_compute_stats=False)

#     # Optionally push to the Hugging Face Hub
#     if push_to_hub:
#         dataset.push_to_hub(
#             tags=["libero", "panda", "rlds"],
#             private=False,
#             push_videos=True,
#             license="apache-2.0",
#         )


# if __name__ == "__main__":
#     tyro.cli(main)
