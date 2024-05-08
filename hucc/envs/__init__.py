# Copyright (c) 2021-present, Facebook, Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import fancy_gym  # noqa: F401, needed for registrations of envs
from bisk.features import register_featurizer
from gym import spec
from gym.envs.registration import register

from hucc.envs.fancy_gym_bisk import FancyGymTask
from hucc.envs.features import (
    bodyfeet_featurizer,
    bodyfeet_relz_featurizer,
)

register(
    id="BiskReacher5d-v0",
    entry_point="hucc.envs.fancy_gym_bisk:FancyGymAsBiskSingleRobotEnv",
    kwargs={
        "allow_fallover": True,
        "task": FancyGymTask.REACHER,
    },
)

register(
    id="BiskReacher5dSparse-v0",
    entry_point="hucc.envs.fancy_gym_bisk:FancyGymAsBiskSingleRobotEnv",
    kwargs={
        "allow_fallover": True,
        "task": FancyGymTask.REACHER_SPARSE,
    },
)

register(
    id="BiskBoxPushingDense-v0",
    entry_point="hucc.envs.fancy_gym_bisk:FancyGymAsBiskSingleRobotEnv",
    kwargs={
        "allow_fallover": True,
        "task": FancyGymTask.BOX_PUSH_DENSE,
    },
)

register(
    id="BiskBoxPushingTemporalSparse-v0",
    entry_point="hucc.envs.fancy_gym_bisk:FancyGymAsBiskSingleRobotEnv",
    kwargs={
        "allow_fallover": True,
        "task": FancyGymTask.BOX_PUSH_TEMPORAL_SPARSE,
    },
)

register(
    id="BiskBoxPushingGoalSwitchDense-v0",
    entry_point="hucc.envs.fancy_gym_bisk:FancyGymAsBiskSingleRobotEnv",
    kwargs={
        "allow_fallover": True,
        "task": FancyGymTask.BOX_PUSH_GOAL_SWITCH_DENSE,
    },
)

register(
    id="FrankaRodSandbox-v0",
    entry_point="hucc.envs.franka_sandbox:FrankaRodSandbox",
)

register(
    id="TableTennisSandbox-v0",
    entry_point="hucc.envs.tabletennis_sandbox:TableTennisSandbox",
)

register(
    id="BiskTableTennis4D-v0",
    entry_point="hucc.envs.fancy_gym_bisk:FancyGymAsBiskSingleRobotEnv",
    kwargs={
        "allow_fallover": True,
        "task": FancyGymTask.TABLE_TENNIS,
    },
)

register(
    id="ContCtrlgsPreTraining-v1",
    # entry_point="hucc.envs.ctrlgs:CtrlgsPreTrainingEnv",
    entry_point="hucc.envs.fancy_gym_ctrlgs:FancyGymCtrlgsPreTrainingEnv",
    kwargs={
        "reward": "potential",
        "hard_reset_interval": 100,
        "resample_features": "soft",
    },
)

register_featurizer("bodyfeet", bodyfeet_featurizer)
register_featurizer("bodyfeet-relz", bodyfeet_relz_featurizer)
