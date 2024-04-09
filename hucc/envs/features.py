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


class JointValueValuedeltaVelFingerPosPosdeltaVelReacherFeaturizer(Featurizer):
    """Angle/Position and rate of change of each joint, and tip position and velocity
    as features for Reacher5d-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        # TODO
        pos = self.env.unwrapped.data.qpos[: self.env.unwrapped.n_links]
        vel = self.env.unwrapped.data.qvel[: self.env.unwrapped.n_links]
        fingerpos = self.env.unwrapped.data.body("fingertip").xpos[0:2]
        fingervel = self.env.unwrapped.data.body("fingertip").cvel[-3:-1]
        return np.concatenate((pos, pos, vel, fingerpos, fingerpos, fingervel))

    def feature_names(self) -> List[str]:
        xy = ("x", "y")
        names = [f"joint{i}" for i in range(5)]
        names += [f"joint{i}_delta" for i in range(5)]
        names += [f"vel{i}" for i in range(5)]
        names += [f"tip{i}" for i in xy]
        names += [f"tip{i}_delta" for i in xy]
        names += [f"tip{i}_vel" for i in xy]
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


class JointsTaskReacherSparseFeaturizer(Featurizer):
    """Regular task observation for Reacher5dSparse-v0.

    This only needs to exist for the adaptation of regular (fancy_gym) envs
    to work."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 * 5 + 2 + 3 + 1,), dtype=np.float32
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
        episode_progress = (
            self.env.unwrapped._steps / self.env._max_episode_steps
        ) * 2 - 1
        return np.concatenate(
            [self.env.unwrapped._get_obs()[self._obs_reorder], [episode_progress]]
        )

    def feature_names(self) -> List[str]:
        names = [f"cos{i}" for i in range(5)]
        names += [f"sin{i}" for i in range(5)]
        names += [f"vel{i}" for i in range(5)]
        names += ["target_x", "target_y"]
        names += ["dist_x", "dist_y", "dist_z"]
        names += ["episode_progress"]
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


class FingerPosPosdeltaVelEulerSwingTwistPosboxzFrankaFeaturizer(Featurizer):
    """Rod tip position, velocity, rod extrinsic xyz euler angles and swing twist around z
    as features for BoxPushingDense-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        # TODO: either data.body("push_rod") or add sensors to site
        pos = self.env.unwrapped.data.site("rod_tip").xpos
        vel = self.env.unwrapped.data.body("push_rod").cvel[-3:]
        quat = self.env.unwrapped.data.body("push_rod").xquat
        euler_xyz = quat_wxyz_to_extrinsic_euler_xyz(quat)
        swing_twist = quat_wxyz_to_swing_twist_z(quat)
        return np.concatenate(
            (pos, pos, vel, euler_xyz, swing_twist, pos[2, np.newaxis])
        )

    def feature_names(self) -> List[str]:
        xyz = ("x", "y", "z")
        names = [f"tip{i}" for i in xyz]
        names += [f"tip{i}_delta" for i in xyz]
        names += [f"tip{i}_vel" for i in xyz]
        names += [f"euler{i}" for i in xyz]
        names += ["swingxy", "twistz"]
        names += ["tipz"]
        return names


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
            low=-np.inf, high=np.inf, shape=(4 * 7 + 1,), dtype=np.float32
        )
        # NOTE: codebase implicitly expects task-specific obs to be last dims of obs,
        # for BPD, the dims which are not in JointsFrankaFeaturizer are already at the end.
        # HiToLoInterface uses obs_mask of (dummy) goal-space pretrain env to create
        # the low-level policy observation by masking the features; obs_mask is basically
        # range(n).

    def __call__(self) -> np.ndarray:
        episode_progress = (
            self.env.unwrapped._steps / self.env._max_episode_steps
        ) * 2 - 1
        return np.concatenate([self.env.unwrapped._get_obs(), [episode_progress]])

    def feature_names(self) -> List[str]:
        names = [f"qpos{i}" for i in range(7)]
        names += [f"qvel{i}" for i in range(7)]
        names += [f"box_0_pos{i}" for i in ("x", "y", "z")]
        names += [f"box_0_quat{i}" for i in ("w", "x", "y", "z")]
        names += [f"replan_target_pos{i}" for i in ("x", "y", "z")]
        names += [f"replan_target_quat{i}" for i in ("w", "x", "y", "z")]
        return names


class JointsTableTennisFeaturizer(Featurizer):
    """Regular proprioceptive observation for TableTennis4D-v0.

    The name 'joints' is misleading but this is the name used by HSD-3 for
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


class JointsTaskTableTennisFeaturizer(Featurizer):
    """Regular task observation for TableTennis4D-v0.

    This only needs to exist for the adaptation of regular (fancy_gym) envs
    to work."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 * 7 + 5,), dtype=np.float32
        )
        # NOTE: codebase implicitly expects task-specific obs to be last dims of obs,
        # for TT, the dims which are not in JointsTableTennisFeaturizer are already at the end.
        # HiToLoInterface uses obs_mask of (dummy) goal-space pretrain env to create
        # the low-level policy observation by masking the features; obs_mask is basically
        # range(n).

    def __call__(self) -> np.ndarray:
        return self.env.unwrapped._get_obs().copy()

    def feature_names(self) -> List[str]:
        names = [f"qpos{i}" for i in range(7)]
        names += [f"qvel{i}" for i in range(7)]
        names += [f"tar_{i}" for i in ("x", "y", "z")]
        names += [f"goal_{i}" for i in ("x", "y")]
        return names


class FingerPosdeltaVelYawPitchTableTennisFeaturizer(Featurizer):
    """Bat center position, velocity, and yaw, pitch (Z,Y of extrinsic ZYX euler) as
    features for TableTennis-v0."""

    def __init__(self, env: gym.Env):
        super().__init__(None, None)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

    def __call__(self) -> np.ndarray:
        pos = self.env.unwrapped.data.body("EE").xpos
        vel = self.env.unwrapped.data.body("EE").cvel[-3:]
        quat = self.env.unwrapped.data.body("EE").xquat
        # bat x is normal to (red) face, use place yaw putch roll decomp
        # and ignore roll which is irrelevant rotation of bat surface
        yaw_pitch = quat_wxyz_to_yaw_pitch(quat)
        return np.concatenate((pos, vel, yaw_pitch))

    def feature_names(self) -> List[str]:
        xyz = ("x", "y", "z")
        names = [f"tip{i}_delta" for i in xyz]
        names += [f"tip{i}_vel" for i in xyz]
        names += ["yaw", "pitch"]
        return names


def quat_wxyz_to_extrinsic_euler_xyz(quat: np.ndarray) -> np.ndarray:
    assert quat.ndim in (1, 2)
    assert quat.shape[-1] == 4
    # scipy quat in order x,y,z,w
    q_xyzw = quat[..., [1, 2, 3, 0]]
    return Rotation.from_quat(q_xyzw).as_euler("xyz")


def quat_wxyz_to_yaw_pitch(quat: np.ndarray) -> np.ndarray:
    assert quat.ndim in (1, 2)
    assert quat.shape[-1] == 4
    # scipy quat in order x,y,z,w
    q_xyzw = quat[..., [1, 2, 3, 0]]
    return Rotation.from_quat(q_xyzw).as_euler("ZYX")[..., :2]


def quat_wxyz_to_swing_twist_z(quat: np.ndarray) -> np.ndarray:
    assert quat.ndim == 1
    assert quat.shape[-1] == 4
    assert np.abs(np.linalg.norm(quat) - 1.0) <= 1e-6

    # swing and twist decomposition, code based on https://stackoverflow.com/a/22401169
    # and https://stackoverflow.com/a/63502201
    # twist_dir is normalized -> simpler vector projection
    twist_dir = np.array([0, 0, 1])

    q_xyzw = quat[[1, 2, 3, 0]]

    # project quat axis on twist_dir
    rot_axis = q_xyzw[:3]
    rot_dot_twist = np.dot(rot_axis, twist_dir)
    rot_proj_on_twist_dir = twist_dir * rot_dot_twist

    # determine twist part (non-unit quaternion) of quat rotation
    twist_xyzw = np.zeros_like(q_xyzw)
    twist_xyzw[0:3] = rot_proj_on_twist_dir[0:3]
    twist_xyzw[3] = q_xyzw[3]
    if np.abs(rot_dot_twist) < 1e-6:
        # rotation axis is orthogonal to twist axis, all twist angles equally valid
        # twist_xyz is 0,0,0 -> zero angle/identity rotation
        # NOTE: w component of quat might be zero (=cos(pi/2) for 180 degree rotation),
        # so twist would be zero-norm 0,0,0,0 quaternion -> set w to 1
        # TODO: pick twist_dir better. there would still be a twist value if q was not
        # as close to 180deg, so try to find that value
        twist_xyzw = np.array([0, 0, 0, 1])
    elif rot_dot_twist < 0:
        # if dot negative, then rot_proj_on_twist_dir is opposite direction -> negated angles
        # elementwise negation does not change rotation (as an operation) but flips axis and negates angles
        twist_xyzw = -twist_xyzw

    # normalize twist quaternion
    twist_xyzw = twist_xyzw / np.linalg.norm(twist_xyzw)

    # we only care about the angle of the twist
    twist_angle = 2 * np.arctan2(np.linalg.norm(twist_xyzw[0:3]), twist_xyzw[3])

    # swing = quat * twist.conjugated
    twist_conj = twist_xyzw * np.array([-1, -1, -1, 1])
    swing = Rotation.from_quat(q_xyzw) * Rotation.from_quat(twist_conj)
    swing_xyzw = swing.as_quat(canonical=True)

    # we only case about the angle of the swing, not the axis (in the xy plane orthogonal to twist)
    swing_angle = 2 * np.arctan2(np.linalg.norm(swing_xyzw[0:3]), swing_xyzw[3])
    # [0, 2pi]
    return np.array((swing_angle, twist_angle))


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
