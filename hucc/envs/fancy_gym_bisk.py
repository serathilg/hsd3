import logging
from enum import Enum, auto

import fancy_gym
import gym
import numpy as np
from bisk import BiskSingleRobotEnv
from bisk.features.base import Featurizer
from gym.utils import seeding

from hucc.envs.features import (
    FingerPosdeltaVelYawPitchTableTennisFeaturizer,
    FingerPosFrankaFeaturizer,
    FingerPosPosdeltaVelEulerSwingTwistPosboxzFrankaFeaturizer,
    JointsFrankaFeaturizer,
    JointsReacherFeaturizer,
    JointsTableTennisFeaturizer,
    JointsTaskFrankaFeaturizer,
    JointsTaskReacherFeaturizer,
    JointsTaskReacherSparseFeaturizer,
    JointsTaskTableTennisFeaturizer,
    JointValueReacherFeaturizer,
    JointValueValuedeltaVelFingerPosPosdeltaVelReacherFeaturizer,
    JointValueVelReacherFeaturizer,
)

log = logging.getLogger(__name__)


class FancyGymTask(Enum):
    SANDBOX = auto()
    REACHER = auto()
    BOX_PUSH_DENSE = auto()
    BOX_PUSH_TEMPORAL_SPARSE = auto()
    REACHER_SPARSE = auto()
    TABLE_TENNIS = auto()
    BOX_PUSH_GOAL_SWITCH_DENSE = auto()


class _FGFeature(Enum):
    JOINT_VALUE = "joint_value"
    JOINT_VALUE_VEL = "joint_value_vel"
    JOINT_VALUE_VALUEDELTA_VEL_FINGERPOS_POSDELTA_VEL = (
        "jval_jvald_jvel_fpos_fposd_fvel"
    )
    JOINTS = "joints"
    JOINTS_TASK = "joints_task"
    FINGERPOS = "fingerpos"
    FINGERPOS_DELTA = "fingerpos_delta"
    FINGERPOS_POSDELTA_VEL_EULER_SWINGTWIST_POSBOXZ = (
        "fpos_fposd_fvel_feuler_fswitwi_fposboxz"
    )
    FINGERPOSDELTA_VEL_YAWPITCH = "fposd_fvel_fyawpitch"
    FINGERPOSDELTA_VELDELTA_YAWPITCHDELTA = "fposd_fveld_fyawpitchd"


class _FGRobot(Enum):
    REACHER = "reacher"
    FRANKA = "franka"
    WAM = "wam"


_FANCY_GYM_FEATURIZER = {
    _FGRobot.REACHER: {
        FancyGymTask.SANDBOX: {
            _FGFeature.JOINTS: JointsReacherFeaturizer,
            _FGFeature.JOINT_VALUE: JointValueReacherFeaturizer,
            _FGFeature.JOINT_VALUE_VEL: JointValueVelReacherFeaturizer,
            _FGFeature.JOINT_VALUE_VALUEDELTA_VEL_FINGERPOS_POSDELTA_VEL: JointValueValuedeltaVelFingerPosPosdeltaVelReacherFeaturizer,
        },
        FancyGymTask.REACHER: {
            _FGFeature.JOINTS: JointsReacherFeaturizer,
            _FGFeature.JOINTS_TASK: JointsTaskReacherFeaturizer,
            _FGFeature.JOINT_VALUE: JointValueReacherFeaturizer,
            _FGFeature.JOINT_VALUE_VEL: JointValueVelReacherFeaturizer,
            _FGFeature.JOINT_VALUE_VALUEDELTA_VEL_FINGERPOS_POSDELTA_VEL: JointValueValuedeltaVelFingerPosPosdeltaVelReacherFeaturizer,
        },
        FancyGymTask.REACHER_SPARSE: {
            _FGFeature.JOINTS: JointsReacherFeaturizer,
            _FGFeature.JOINTS_TASK: JointsTaskReacherSparseFeaturizer,
            _FGFeature.JOINT_VALUE: JointValueReacherFeaturizer,
            _FGFeature.JOINT_VALUE_VEL: JointValueVelReacherFeaturizer,
            _FGFeature.JOINT_VALUE_VALUEDELTA_VEL_FINGERPOS_POSDELTA_VEL: JointValueValuedeltaVelFingerPosPosdeltaVelReacherFeaturizer,
        },
    },
    _FGRobot.FRANKA: {
        FancyGymTask.SANDBOX: {
            _FGFeature.JOINTS: JointsFrankaFeaturizer,
            _FGFeature.FINGERPOS: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_DELTA: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_POSDELTA_VEL_EULER_SWINGTWIST_POSBOXZ: FingerPosPosdeltaVelEulerSwingTwistPosboxzFrankaFeaturizer,
        },
        FancyGymTask.BOX_PUSH_DENSE: {
            _FGFeature.JOINTS: JointsFrankaFeaturizer,
            _FGFeature.JOINTS_TASK: JointsTaskFrankaFeaturizer,
            _FGFeature.FINGERPOS: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_DELTA: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_POSDELTA_VEL_EULER_SWINGTWIST_POSBOXZ: FingerPosPosdeltaVelEulerSwingTwistPosboxzFrankaFeaturizer,
        },
        FancyGymTask.BOX_PUSH_TEMPORAL_SPARSE: {
            _FGFeature.JOINTS: JointsFrankaFeaturizer,
            _FGFeature.JOINTS_TASK: JointsTaskFrankaFeaturizer,
            _FGFeature.FINGERPOS: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_DELTA: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_POSDELTA_VEL_EULER_SWINGTWIST_POSBOXZ: FingerPosPosdeltaVelEulerSwingTwistPosboxzFrankaFeaturizer,
        },
        FancyGymTask.BOX_PUSH_GOAL_SWITCH_DENSE: {
            _FGFeature.JOINTS: JointsFrankaFeaturizer,
            _FGFeature.JOINTS_TASK: JointsTaskFrankaFeaturizer,
            _FGFeature.FINGERPOS: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_DELTA: FingerPosFrankaFeaturizer,
            _FGFeature.FINGERPOS_POSDELTA_VEL_EULER_SWINGTWIST_POSBOXZ: FingerPosPosdeltaVelEulerSwingTwistPosboxzFrankaFeaturizer,
        },
    },
    _FGRobot.WAM: {
        FancyGymTask.SANDBOX: {
            _FGFeature.JOINTS: JointsTableTennisFeaturizer,
            _FGFeature.FINGERPOSDELTA_VEL_YAWPITCH: FingerPosdeltaVelYawPitchTableTennisFeaturizer,
            _FGFeature.FINGERPOSDELTA_VELDELTA_YAWPITCHDELTA: FingerPosdeltaVelYawPitchTableTennisFeaturizer,
        },
        FancyGymTask.TABLE_TENNIS: {
            _FGFeature.JOINTS: JointsTableTennisFeaturizer,
            _FGFeature.JOINTS_TASK: JointsTaskTableTennisFeaturizer,
            _FGFeature.FINGERPOSDELTA_VEL_YAWPITCH: FingerPosdeltaVelYawPitchTableTennisFeaturizer,
            _FGFeature.FINGERPOSDELTA_VELDELTA_YAWPITCHDELTA: FingerPosdeltaVelYawPitchTableTennisFeaturizer,
        },
    },
}


def make_fancy_gym_featurizer(
    features: str, env: gym.Env, robot: str, task: FancyGymTask
) -> Featurizer:
    f = _FGFeature(features)
    r = _FGRobot(robot)
    try:
        featurizer = _FANCY_GYM_FEATURIZER[r][task][f]
    except KeyError:
        raise NotImplementedError(
            f"make_fancy_gym_featurizer: {features=}, {task=}, {robot=}"
        )
    return featurizer(env)


class FancyGymAsBiskSingleRobotEnv(BiskSingleRobotEnv):
    """
    Imitate BiskSingleRobotEnv from existing fancy_gym env instance.
    """

    metadata = {"render.modes": ["rgb_array"]}  # TODO?

    def __init__(
        self,
        robot: str,
        task: FancyGymTask,
        features: str = "joints_task",  # proprioceptive + task-specific by default
        allow_fallover: bool = False,
    ):
        self.allow_fallover = allow_fallover
        self.robot = robot.lower()
        self.p = None

        r = _FGRobot(self.robot)
        if r == _FGRobot.REACHER and task == FancyGymTask.SANDBOX:
            # hacky max_episode_steps instead of sandbox env
            self.env = fancy_gym.make("Reacher5d-v0", seed=None, max_episode_steps=1e9)
        elif r == _FGRobot.REACHER and task == FancyGymTask.REACHER:
            self.env = fancy_gym.make("Reacher5d-v0", seed=None)
        elif r == _FGRobot.REACHER and task == FancyGymTask.REACHER_SPARSE:
            self.env = fancy_gym.make("Reacher5dSparse-v0", seed=None)
        elif r == _FGRobot.FRANKA and task == FancyGymTask.SANDBOX:
            self.env = fancy_gym.make("FrankaRodSandbox-v0", seed=None)
        elif r == _FGRobot.FRANKA and task == FancyGymTask.BOX_PUSH_DENSE:
            self.env = fancy_gym.make("BoxPushingDense-v0", seed=None)
        elif r == _FGRobot.FRANKA and task == FancyGymTask.BOX_PUSH_TEMPORAL_SPARSE:
            self.env = fancy_gym.make("BoxPushingTemporalSparse-v0", seed=None)
        elif r == _FGRobot.FRANKA and task == FancyGymTask.BOX_PUSH_GOAL_SWITCH_DENSE:
            self.env = fancy_gym.make("BoxPushingGoalSwitchDense-v0", seed=None)
        elif r == _FGRobot.WAM and task == FancyGymTask.SANDBOX:
            self.env = fancy_gym.make("TableTennisSandbox-v0", seed=None)
        elif r == _FGRobot.WAM and task == FancyGymTask.TABLE_TENNIS:
            self.env = fancy_gym.make("TableTennis4D-v0", seed=None)
        else:
            raise NotImplementedError(
                f"FancyGymAsBiskSingleRobotEnv: {robot=}, {task=}"
            )
        self.action_space = self.env.action_space

        self._task = task
        self.featurizer = self.make_featurizer(features)
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
        return make_fancy_gym_featurizer(features, self.env, self.robot, self._task)

    def fell_over(self) -> bool:
        if _FGRobot(self.robot) == _FGRobot.REACHER:
            # TODO: arbitrary decisions, maybe suboptimal
            pos = self.env.unwrapped.data.qpos[: self.env.unwrapped.n_links]
            vel = self.env.unwrapped.data.qvel[: self.env.unwrapped.n_links]
            # allow movement above 180 degrees but not full rotation
            angle_limit = np.any(np.abs(pos) >= 2 * np.pi)
            # hard to control if we spin too fast and need decelerate first when
            # switching goal, the choice of number 3 is arbitrary though
            vel_limit = np.any(np.abs(vel) >= 3 * np.pi)
            return bool(angle_limit or vel_limit)
        if _FGRobot(self.robot) == _FGRobot.WAM:
            # TODO: arbitrary decisions, maybe suboptimal
            pos = self.env.unwrapped.data.qpos[:7]
            vel = self.env.unwrapped.data.qvel[:7]
            # allow movement above 180 degrees but not full rotation
            angle_limit = np.any(np.abs(pos) >= 2 * np.pi)
            # hard to control if we spin too fast and need decelerate first when
            # switching goal, the choice of number 3 is arbitrary though
            # (full rotation in 2/3s when full rotation over 1.2s already too much)
            vel_limit = np.any(np.abs(vel) >= 3 * np.pi)
            return bool(angle_limit or vel_limit)
        elif _FGRobot(self.robot) == _FGRobot.FRANKA:
            tip_x_pos, tip_y_pos, tip_z_pos = self.env.unwrapped.data.site(
                "rod_tip"
            ).xpos
            tip_quat = self.env.unwrapped.data.body("push_rod").xquat
            # stay above desk
            tip_between_desk_front_back = (
                0.2 - 0.49 < tip_x_pos and tip_x_pos < 0.2 + 0.49
            )
            tip_between_desk_left_right = -0.98 < tip_y_pos and tip_y_pos < 0.98
            below_desk_height = tip_z_pos < -0.02 + 0.001

            # this should not be possible because of contacts but mark it invalid anyway
            below_desk = (
                tip_between_desk_front_back
                and tip_between_desk_left_right
                and below_desk_height
            )
            # dont allow movement outside of frame
            outside_frame = not (
                tip_between_desk_front_back and tip_between_desk_left_right
            )

            # deviation from rod angle:
            angle_err = (
                fancy_gym.envs.mujoco.box_pushing.box_pushing_utils.rotation_distance(
                    tip_quat, self.env.unwrapped._desired_rod_quat
                )
            )
            # joints limits, soft eps limit only as violation penalty in task as well.
            q_min = self.env.unwrapped._q_min
            q_max = self.env.unwrapped._q_max
            q_pos_limit_eps = 0.1  # TODO make configurable
            joint_pos_limit = np.any(
                np.logical_or(
                    self.env.unwrapped.data.qpos[:7] > (1 + q_pos_limit_eps) * q_max,
                    self.env.unwrapped.data.qpos[:7] < (1 + q_pos_limit_eps) * q_min,
                )
            )
            q_vel_limit_eps = 0.5  # TODO make configurable
            q_dot_max = self.env.unwrapped._q_dot_max
            joint_vel_limit = np.any(
                np.abs(self.env.unwrapped.data.qvel[:7])
                > (1 + q_vel_limit_eps) * q_dot_max
            )

            # only allow ~45 degree tilt:
            tilt_limit_eps = 0.1  # TODO make configurable
            rod_tilted = angle_err > (1 + tilt_limit_eps) * np.pi / 4

            # TODO: limit around panda base

            return bool(
                below_desk
                or outside_frame
                or rod_tilted
                or joint_pos_limit
                or joint_vel_limit
            )
            # return bool(below_desk or outside_frame or rod_tilted)
        return False
