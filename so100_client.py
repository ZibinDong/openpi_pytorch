import time

import cv2
import gym
import numpy as np
from lerobot.common.cameras.opencv import OpenCVCameraConfig
from lerobot.common.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.common.utils.robot_utils import busy_wait

from utils.server import PolicyClient


class SO100Env(gym.Env):
    def __init__(
        self,
        robot_id: str,
        port: str,
        cameras: dict[str, dict],
    ):
        super().__init__()

        cameras_config = {}
        for k, v in cameras.items():
            cameras_config[k] = OpenCVCameraConfig(**v)
            
        robot_config = SO100FollowerConfig(
            port=port,
            id=robot_id,
            cameras=cameras_config
        )
        self.robot = SO100Follower(robot_config)
        self.camera_names = list(cameras.keys())
        
        self.init_pos = None
    
    def get_observation(self):
        raw_observation = self.robot.get_observation()
        state = np.array([
            raw_observation['shoulder_pan.pos'],
            raw_observation['shoulder_lift.pos'],
            raw_observation['elbow_flex.pos'],
            raw_observation['wrist_flex.pos'],
            raw_observation['wrist_roll.pos'],
            raw_observation['gripper.pos'],
        ])
        observation = {'state': state}
        for camera_name in self.camera_names:
            _image = cv2.resize(raw_observation[camera_name], (224, 224))
            observation[camera_name] = _image
        return observation
    
    def reset(self):
        if self.init_pos is None:
            self.robot.connect()
            init_observation = self.get_observation()
            self.init_pos = init_observation['state']
        else:
            self.robot.send_action(self.init_pos)
            init_observation = self.get_observation()
        return init_observation
    
    def step(self, action):
        self.robot.send_action(action)
        return self.get_observation(), 0, False, {}

    def close(self):
        self.robot.disconnect()


client = PolicyClient(server_host='localhost', server_port=12346)

env = SO100Env(
    robot_id="so100_follower",
    port="/dev/ttyACM0",
    cameras={
        "base": {"fps": 25, "width": 640, "height": 480, "index_or_path": 0},
        "wrist": {"fps": 25, "width": 640, "height": 480, "index_or_path": 2},
    }
)

o = env.reset()
o['prompt'] = ["grab the screwdriver and put it to the right"]
busy_wait(1)

for _ in range(10):
    o = env.get_observation()
    o['prompt'] = ["grab the screwdriver and put it to the front"]
    action = client.get_action(o)[0]

    names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ]

    fps = 30
    start_episode_t = time.perf_counter()
    for i in range(0, 50):
        start_loop_t = time.perf_counter()

        act = {}
        for j, name in enumerate(names):
            act[name] = action[i, j]
            
        o, _, _, _ = env.step(act)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
    
env.close()

