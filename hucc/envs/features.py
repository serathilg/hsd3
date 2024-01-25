# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List

import gym
import numpy as np
from bisk.features import Featurizer
from bisk.features.joints import JointsRelZFeaturizer
from dm_control import mujoco
from scipy.spatial.transform import Rotation


class BodyFeetWalkerFeaturizer(Featurizer):
    def __init__(self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        assert robot == "walker", f'Walker robot expected, got "{robot}"'
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        root = self.p.named.data.qpos[[f"{self.prefix}/root{p}" for p in "zxy"]]
        torso_frame = self.p.named.data.xmat[f"{self.prefix}/torso"].reshape(3, 3)
        torso_pos = self.p.named.data.xpos[f"{self.prefix}/torso"]
        positions = []
        for side in ("left", "right"):
            torso_to_limb = (
                self.p.named.data.xpos[f"{self.prefix}/{side}_foot"] - torso_pos
            )
            # We're in 2D effectively, y is constant
            positions.append(torso_to_limb.dot(torso_frame)[[0, 2]])
        extremities = np.hstack(positions)
        return np.concatenate([root, extremities])

    def feature_names(self) -> List[str]:
        names = ["rootz:p", "rootx:p", "rooty:p"]
        names += [f"left_foot:p{p}" for p in "xz"]
        names += [f"right_foot:p{p}" for p in "xz"]
        return names


class BodyFeetRelZWalkerFeaturizer(BodyFeetWalkerFeaturizer):
    def __init__(self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[0] = self.relz()
        return obs


class BodyFeetHumanoidFeaturizer(Featurizer):
    def __init__(
        self,
        p: mujoco.Physics,
        robot: str,
        prefix: str = "robot",
        exclude: str = None,
    ):
        super().__init__(p, robot, prefix, exclude)
        self.for_pos = None
        self.for_twist = None
        self.foot_anchor = "pelvis"
        self.reference = "torso"
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )

    @staticmethod
    def decompose_twist_swing_z(q):
        p = [0.0, 0.0, q[2]]
        twist = Rotation.from_quat(np.array([p[0], p[1], p[2], q[3]]))
        swing = Rotation.from_quat(q) * twist.inv()
        return twist, swing

    def __call__(self) -> np.ndarray:
        root = self.p.data.qpos[0:3]
        if self.for_pos is not None:
            root = root.copy()
            root[0:2] -= self.for_pos
            root[0:2] = self.for_twist.apply(root * np.array([1, 1, 0]))[0:2]
        q = self.p.data.qpos[3:7]
        t, s = self.decompose_twist_swing_z(q[[1, 2, 3, 0]])
        tz = t.as_rotvec()[2]
        e = s.as_euler("yzx")
        sy, sx = e[0], e[2]

        # Feet positions are relative to pelvis position and its heading
        # Also, exclude hands for now.
        pelvis_q = self.p.named.data.xquat[f"{self.prefix}/{self.foot_anchor}"]
        pelvis_t, pelvis_s = self.decompose_twist_swing_z(pelvis_q[[1, 2, 3, 0]])
        pelvis_pos = self.p.named.data.xpos[f"{self.prefix}/{self.foot_anchor}"]
        positions = []
        for ex in ("foot",):
            for side in ("left", "right"):
                pelvis_to_limb = (
                    self.p.named.data.xpos[f"{self.prefix}/{side}_{ex}"] - pelvis_pos
                )
                positions.append(pelvis_t.apply(pelvis_to_limb))
        extremities = np.hstack(positions)
        return np.concatenate([root, np.asarray([tz, sy, sx]), extremities])

    def feature_names(self) -> List[str]:
        names = [f"root:p{f}" for f in "xyz"]
        names += [f"root:t{f}" for f in "z"]
        names += [f"root:s{f}" for f in "yx"]
        names += [f"left_foot:p{f}" for f in "xyz"]
        names += [f"right_foot:p{f}" for f in "xyz"]
        return names


class BodyFeetRelZHumanoidFeaturizer(BodyFeetHumanoidFeaturizer):
    def __init__(self, p: mujoco.Physics, robot: str, prefix: str, exclude: str = None):
        super().__init__(p, robot, prefix, exclude)
        self.relzf = JointsRelZFeaturizer(p, robot, prefix, exclude)

    def relz(self):
        return self.relzf.relz()

    def __call__(self) -> np.ndarray:
        obs = super().__call__()
        obs[2] = self.relz()
        return obs


class JointValueReacherFeaturizer(Featurizer):
    """Angle/Position of each joint as features for Reacher5d-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped.data.qpos[: self.env.unwrapped.n_links].copy()

    def feature_names(self) -> List[str]:
        names = [f"joint{i}" for i in range(5)]
        return names


class JointValueVelReacherFeaturizer(Featurizer):
    """Angle/Position and rate of change of each joint as features for Reacher5d-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        pos = self.env.unwrapped.data.qpos[: self.env.unwrapped.n_links]
        vel = self.env.unwrapped.data.qvel[: self.env.unwrapped.n_links]
        return np.concatenate((pos, vel))

    def feature_names(self) -> List[str]:
        names = [f"joint{i}" for i in range(5)]
        names += [f"vel{i}" for i in range(5)]
        return names


class JointsReacherFeaturizer(Featurizer):
    """Regular proprioceptive observation for Reacher5d-v0.

    The name 'joints' is misleading but this is the named used by HSD-3 for
    the non goal-space and task-independent features.

    This only needs to exist for the adaptation of regular (fancy_gym) envs
    to work."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 * 5,), dtype=np.float32
        )
        # mask the task-specific dimensions
        self._obs_proprio = np.array(list(range(0, 10)) + list(range(12, 17)))

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped._get_obs()[self._obs_proprio].copy()

    def feature_names(self) -> List[str]:
        names = [f"cos{i}" for i in range(5)]
        names += [f"sin{i}" for i in range(5)]
        names += [f"vel{i}" for i in range(5)]
        return names


class JointsTaskReacherFeaturizer(Featurizer):
    """Regular task observation for Reacher5d-v0.

    This only needs to exist for the adaptation of regular (fancy_gym) envs
    to work."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 * 5 + 2 + 3,), dtype=np.float32
        )
        # NOTE: codebase implicitly expects task-specific obs to be last dims of obs,
        # so move the dims which are not in JointsReacherFeaturizer to the end.
        # HiToLoInterface uses obs_mask of (dummy) goal-space pretrain env to create
        # the low-level policy observation by masking the features; obs_mask is basically
        # range(n).
        self._obs_reorder = np.array(
            list(range(0, 10))
            + list(range(12, 17))
            + list(range(10, 12))
            + list(range(17, 20))
        )

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped._get_obs()[self._obs_reorder].copy()

    def feature_names(self) -> List[str]:
        names = [f"cos{i}" for i in range(5)]
        names += [f"sin{i}" for i in range(5)]
        names += [f"vel{i}" for i in range(5)]
        names += ["target_x", "target_y"]
        names += ["dist_x", "dist_y", "dist_z"]
        return names


class FingerPosFrankaFeaturizer(Featurizer):
    """Rod tip position as features for BoxPushingDense-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped.data.site("rod_tip").xpos.copy()

    def feature_names(self) -> List[str]:
        return ["tipx", "tipy", "tipz"]


class JointsFrankaFeaturizer(Featurizer):
    """Regular proprioceptive observation for BoxPushingDense-v0.

    The name 'joints' is misleading but this is the named used by HSD-3 for
    the non goal-space and task-independent features.

    This only needs to exist for the adaptation of regular (fancy_gym) envs
    to work."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * 7,), dtype=np.float32
        )
        # mask the task-specific dimensions
        self._obs_proprio = np.array(list(range(0, 14)))

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped._get_obs()[self._obs_proprio].copy()

    def feature_names(self) -> List[str]:
        names = [f"qpos{i}" for i in range(7)]
        names += [f"qvel{i}" for i in range(7)]
        return names


class JointsTaskFrankaFeaturizer(Featurizer):
    """Regular task observation for BoxPushingDense-v0.

    This only needs to exist for the adaptation of regular (fancy_gym) envs
    to work."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4 * 7,), dtype=np.float32
        )
        # NOTE: codebase implicitly expects task-specific obs to be last dims of obs,
        # for BPD, the dims which are not in JointsFrankaFeaturizer are already at the end.
        # HiToLoInterface uses obs_mask of (dummy) goal-space pretrain env to create
        # the low-level policy observation by masking the features; obs_mask is basically
        # range(n).

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped._get_obs().copy()

    def feature_names(self) -> List[str]:
        names = [f"qpos{i}" for i in range(7)]
        names += [f"qvel{i}" for i in range(7)]
        names += [f"box_0_pos{i}" for i in ("x", "y", "z")]
        names += [f"box_0_quat{i}" for i in ("w", "x", "y", "z")]
        names += [f"replan_target_pos{i}" for i in ("x", "y", "z")]
        names += [f"replan_target_quat{i}" for i in ("w", "x", "y", "z")]
        return names


def bodyfeet_featurizer(p: mujoco.Physics, robot: str, prefix: str, *args, **kwargs):
    if robot == "walker":
        return BodyFeetWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == "humanoid" or robot == "humanoidpc":
        return BodyFeetHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No bodyfeet featurizer for robot "{robot}"')


def bodyfeet_relz_featurizer(
    p: mujoco.Physics, robot: str, prefix: str, *args, **kwargs
):
    if robot == "walker":
        return BodyFeetRelZWalkerFeaturizer(p, robot, prefix, *args, **kwargs)
    elif robot == "humanoid" or robot == "humanoidpc":
        return BodyFeetRelZHumanoidFeaturizer(p, robot, prefix, *args, **kwargs)
    else:
        raise ValueError(f'No bodyfeet-relz featurizer for robot "{robot}"')
