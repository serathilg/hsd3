import logging
from typing import Optional

import fancy_gym
import gym
import numpy as np
from bisk import BiskSingleRobotEnv
from bisk.features.base import Featurizer
from gym.utils import seeding

from hucc.envs.features import (
    JointsReacherFeaturizer,
    JointsTaskReacherFeaturizer,
    JointValueReacherFeaturizer,
    JointValueVelReacherFeaturizer,
)

log = logging.getLogger(__name__)


def make_fancy_gym_featurizer(features: str, env: gym.Env, robot: str) -> Featurizer:
    if features == "joint_value" and robot == "reacher":
        return JointValueReacherFeaturizer(env)
    if features == "joint_value_vel" and robot == "reacher":
        return JointValueVelReacherFeaturizer(env)
    if features == "joints" and robot == "reacher":
        return JointsReacherFeaturizer(env)
    if features == "joints_task" and robot == "reacher":
        return JointsTaskReacherFeaturizer(env)
    else:
        raise NotImplementedError(f"make_fancy_gym_featurizer: {features=}, {robot=}")


class FancyGymAsBiskSingleRobotEnv(BiskSingleRobotEnv):
    """
    Imitate BiskSingleRobotEnv from existing fancy_gym env instance.
    """

    metadata = {"render.modes": ["rgb_array"]}  # TODO?

    def __init__(
        self,
        robot: str,
        features: str = "joints_task",  # proprioceptive + task-specific by default
        allow_fallover: bool = False,
        max_episode_steps: Optional[int] = None,
    ):
        if robot == "Reacher":
            self.env = fancy_gym.make(
                "Reacher5d-v0", seed=None, max_episode_steps=max_episode_steps
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(5,), dtype=np.float32
            )
        else:
            raise NotImplementedError(f"FancyGymAsBiskSingleRobotEnv: {robot=}")

        self.allow_fallover = allow_fallover
        self.robot = robot.lower()
        self.p = None

        self.featurizer = make_fancy_gym_featurizer(features, self.env, self.robot)
        self.observation_space = self.featurizer.observation_space
        self.seed()

    def seed(self, seed=None):
        # TODO: convert callers to reset(seed)
        # copied from BiskSingleRobotEnv
        self.np_random, sd = seeding.np_random(seed)
        if self.action_space is not None:
            self.action_space.seed(seed)
        if self.observation_space is not None:
            self.observation_space.seed(seed)
        # seed fancy_gym as well
        self.env.seed(seed)
        return sd

    def render(self, mode="rgb_array", **kwargs):
        width = kwargs.get("width", 480)
        height = kwargs.get("height", 480)
        camera = kwargs.get("camera", None)
        flags = kwargs.get("flags", {})
        assert flags == {}
        return self.env.render(
            mode=mode,
            width=width,
            height=height,
            camera_id=camera,
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.get_observation(), reward, done, info

    def reset(self):
        self.env.reset()
        return self.get_observation()

    def make_featurizer(self, features: str):
        return make_fancy_gym_featurizer(features, self.env, self.robot)

    def fell_over(self) -> bool:
        if self.robot.startswith("reacher"):
            # TODO: arbitrary decisions, maybe suboptimal
            pos = self.env.unwrapped.data.qpos[: self.env.unwrapped.n_links]
            vel = self.env.unwrapped.data.qvel[: self.env.unwrapped.n_links]
            # allow movement above 180 degrees but not full rotation
            angle_limit = np.any(np.abs(pos) >= 2 * np.pi)
            # hard to control if we spin too fast and need decelerate first when
            # switching goal, the choice of number 3 is arbitrary though
            vel_limit = np.any(np.abs(vel) >= 3 * np.pi)
            return bool(angle_limit or vel_limit)
        return False
