import os

import mujoco
import numpy as np
from gym import spaces, utils
from gym.envs.mujoco import MujocoEnv


class TableTennisSandbox(MujocoEnv, utils.EzPickle):
    """
    7 DoF table tennis environment without ball

    action space:
        unnormalized joints torque * 7 , range [-1, 1]
    observation space:
        joint value * 7,
        joint velocity * 7
    """

    def __init__(self, frame_skip: int = 4, enable_artificial_wind: bool = False):
        utils.EzPickle.__init__(**locals())
        self._steps = 0

        self._terminated = False

        self._enable_artificial_wind = enable_artificial_wind

        self._artificial_force = 0.0

        MujocoEnv.__init__(
            self,
            model_path=os.path.join(
                os.path.dirname(__file__), "assets", "table_tennis_sandbox.xml"
            ),
            frame_skip=frame_skip,
            mujoco_bindings="mujoco",
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    def step(self, action):
        unstable_simulation = False

        for _ in range(self.frame_skip):
            if self._enable_artificial_wind:
                self.data.qfrc_applied[-2] = self._artificial_force
            try:
                self.do_simulation(action, 1)
            except Exception as e:
                print("Simulation get unstable return with MujocoException: ", e)
                unstable_simulation = True
                self._terminated = True
                break

        self._steps += 1

        reward = -25 if unstable_simulation else 0

        return (
            self._get_obs(),
            reward,
            self._terminated,
            {
                "num_steps": self._steps,
            },
        )

    def reset_model(self):
        self._steps = 0

        if self._enable_artificial_wind:
            self._artificial_force = self.np_random.uniform(low=-0.1, high=0.1)

        self.data.qpos[:7] = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5])
        self.data.qvel[:7] = np.zeros(7)

        mujoco.mj_forward(self.model, self.data)

        self._terminated = False
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.flat[:7].copy(),
                self.data.qvel.flat[:7].copy(),
            ]
        )
        return obs


class TableTennisWindSandbox(TableTennisSandbox):
    def __init__(self, frame_skip: int = 4):
        super().__init__(frame_skip=frame_skip, enable_artificial_wind=True)
