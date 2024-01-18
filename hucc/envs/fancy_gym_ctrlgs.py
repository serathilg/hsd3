from typing import Dict, List

import gym
import numpy as np
from bisk import BiskSingleRobotEnv

from hucc.envs.ctrlgs import CtrlgsPreTrainingEnv
from hucc.envs.fancy_gym_bisk import (
    FancyGymAsBiskSingleRobotEnv,
    make_fancy_gym_featurizer,
)
from hucc.envs.goal_spaces import g_delta_feats, g_goal_spaces


class FancyGymCtrlgsPreTrainingEnv(FancyGymAsBiskSingleRobotEnv, CtrlgsPreTrainingEnv):
    """Fancy gym adaption of CtrlgsPreTrainingEnv.

    See CtrlgsPreTrainingEnv for details, this class is mostly hacky multiple inheritance
    and correct redirections of function calls.
    """

    def __init__(
        self,
        robot: str,
        features: str,
        feature_dist: Dict[str, float],
        task_map: Dict[str, int],
        precision: float = 0.1,
        idle_steps: int = 0,
        max_steps: int = 20,
        backproject_goal: bool = True,
        reward: str = "potential",
        hard_reset_interval: int = 1,
        reset_p: float = 0.0,
        resample_features: str = "hard",
        full_episodes: bool = False,
        allow_fallover: bool = False,
        fallover_penalty: float = -1.0,
        implicit_soft_resets: bool = False,
        goal_sampling: str = "random",
        ctrl_cost: float = 0.0,
        normalize_gs_observation: bool = False,
        zero_twist_goals: bool = False,
        relative_frame_of_reference: bool = False,
    ):
        if robot in ("Franka", "Reacher"):
            self._is_fancy_gym = True
            FancyGymAsBiskSingleRobotEnv.__init__(
                self,
                robot=robot,
                features="joints",  # 'joints' means proprioceptive only
                allow_fallover=allow_fallover,
                max_episode_steps=1e9,  # effectively disable timelimit
            )

            # the features here define the goal-spaces
            self.goal_featurizer = make_fancy_gym_featurizer(
                features, self.env, self.robot
            )
            ########## identical to regular Ctrlgs from here ##########
            gsdim = self.goal_featurizer.observation_space.shape[0]
            self.goal_space = g_goal_spaces[features][robot]

            # Construct goal space
            self.psi, self.offset = self.abstraction_matrix(robot, features, gsdim)
            self.psi_1 = np.linalg.inv(self.psi)
            self.offset_1 = -np.matmul(self.offset, self.psi_1)

            assert len(self.observation_space.shape) == 1
            assert self.psi.shape == (gsdim, gsdim)
            assert self.offset.shape == (gsdim,)

            self.precision = precision
            self.idle_steps = idle_steps
            self.max_steps = max_steps
            self.backproject_goal = backproject_goal
            self.reward = reward
            self.hard_reset_interval = hard_reset_interval
            self.reset_p = reset_p
            self.resample_features = resample_features
            self.full_episodes = full_episodes
            self.fallover_penalty = fallover_penalty
            self.ctrl_cost = ctrl_cost
            self.implicit_soft_resets = implicit_soft_resets
            self.goal_sampling = goal_sampling
            self.normalize_gs_observation = normalize_gs_observation
            self.zero_twist_goals = zero_twist_goals
            self.relative_frame_of_reference = relative_frame_of_reference

            self.task_idx = [0] * len(task_map)
            for k, v in task_map.items():
                self.task_idx[v] = int(k)

            if len(self.goal_space["twist_feats"]) > 0:
                negpi = self.proj(
                    -np.pi * np.ones(gsdim), self.goal_space["twist_feats"]
                )
                pospi = self.proj(
                    np.pi * np.ones(gsdim), self.goal_space["twist_feats"]
                )
                if not np.allclose(-negpi, pospi):
                    # This could be supported by more elobarte delta computation
                    # logic in step()
                    raise ValueError("Twist feature ranges not symmetric")
                self.proj_pi = pospi

            if backproject_goal:
                all_feats = list(range(gsdim))
                gmin_back = self.backproj(-np.ones(gsdim), all_feats).astype(np.float32)
                gmax_back = self.backproj(np.ones(gsdim), all_feats).astype(np.float32)
                goal_space = gym.spaces.Box(gmin_back, gmax_back)
            else:
                max_features = max(
                    (len(f.replace("+", ",").split(",")) for f in feature_dist.keys())
                )
                goal_space = gym.spaces.Box(
                    low=-2, high=2, shape=(max_features,), dtype=np.float32
                )

            self.task_map = {int(k): v for k, v in task_map.items()}

            # Hide position-related invariant features from the observation, i.e.
            # X/Y or ant X for cheetah
            delta_feats = g_delta_feats[robot]
            self.obs_mask = list(range(self.observation_space.shape[0]))
            for d in delta_feats:
                self.obs_mask.remove(d)

            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(len(self.obs_mask),),
                        dtype=np.float32,
                    ),
                    "desired_goal": goal_space,
                    "task": gym.spaces.Box(
                        low=0, high=1, shape=(len(self.task_map),), dtype=np.float32
                    ),
                    "gs_observation": self.goal_featurizer.observation_space,
                }
            )

            self._do_hard_reset = True
            self._reset_counter = 0
            self.set_feature_dist(feature_dist)
            # Current features
            self._features: List[int] = []
            self._features_s = ""
            self._feature_mask = np.zeros(len(self.task_map))

            self.model = None
            self.gamma = 1.0

        else:
            self._is_fancy_gym = False
            CtrlgsPreTrainingEnv.__init__(
                self,
                robot=robot,
                features=features,
                feature_dist=feature_dist,
                task_map=task_map,
                precision=precision,
                idle_steps=idle_steps,
                max_steps=max_steps,
                backproject_goal=backproject_goal,
                reward=reward,
                hard_reset_interval=hard_reset_interval,
                reset_p=reset_p,
                resample_features=resample_features,
                full_episodes=full_episodes,
                allow_fallover=allow_fallover,
                fallover_penalty=fallover_penalty,
                implicit_soft_resets=implicit_soft_resets,
                goal_sampling=goal_sampling,
                ctrl_cost=ctrl_cost,
                normalize_gs_observation=normalize_gs_observation,
                zero_twist_goals=zero_twist_goals,
                relative_frame_of_reference=relative_frame_of_reference,
            )

    def hard_reset(self):
        if self._is_fancy_gym:
            self.env.reset()
        else:
            CtrlgsPreTrainingEnv.hard_reset(self)

    def step(self, action):
        # identical to Ctrlgs, except for redirection of step to FancyGymAsBiskSingleRobotEnv
        def distance_to_goal():
            gs = self.proj(self.goal_featurizer(), self._features)
            d = self.goal - gs
            for i, f in enumerate(self._features):
                if f in self.goal_space["twist_feats"]:
                    # Wrap around projected pi/-pi for distance
                    d[i] = (
                        np.remainder(
                            (self.goal[i] - gs[i]) + self.proj_pi,
                            2 * self.proj_pi,
                        )
                        - self.proj_pi
                    )
            return np.linalg.norm(d, ord=2)

        d_prev = distance_to_goal()
        if self._is_fancy_gym:
            ########## only this is different to regular Ctrlgs ##########
            next_obs, reward, done, info = FancyGymAsBiskSingleRobotEnv.step(
                self, action
            )
        else:
            next_obs, reward, done, info = BiskSingleRobotEnv.step(self, action)
        d_new = distance_to_goal()

        info["potential"] = d_prev - d_new
        info["distance"] = d_new
        info["reached_goal"] = bool(info["distance"] < self.precision)
        if self.reward == "potential":
            reward = info["potential"]
        elif self.reward == "potential2":
            reward = d_prev - self.gamma * d_new
        elif self.reward == "potential3":
            reward = 1.0 if info["reached_goal"] else 0.0
            reward += d_prev - self.gamma * d_new
        elif self.reward == "potential4":
            reward = (d_prev - d_new) / self._d_initial
        elif self.reward == "distance":
            reward = -info["distance"]
        elif self.reward == "sparse":
            reward = 1.0 if info["reached_goal"] else 0.0
        else:
            raise ValueError(f"Unknown reward: {self.reward}")
        reward -= self.ctrl_cost * np.square(action).sum()

        info["EpisodeContinues"] = True
        if info["reached_goal"] == True and not self.full_episodes:
            done = True
        info["time"] = self._step
        self._step += 1
        if self._step >= self.max_steps:
            done = True
        elif not info["reached_goal"] and self.np_random.random() < self.reset_p:
            info["RandomReset"] = True
            done = True

        if not self.allow_fallover and self.fell_over():
            reward = self.fallover_penalty
            done = True
            self._do_hard_reset = True
            info["reached_goal"] = False
            info["fell_over"] = True
        if done and (
            self._do_hard_reset or (self._reset_counter % self.hard_reset_interval == 0)
        ):
            del info["EpisodeContinues"]
        if done:
            info["LastStepOfTask"] = True

        if done and "EpisodeContinues" in info and self.implicit_soft_resets:
            need_hard_reset = self._do_hard_reset or (
                self.hard_reset_interval > 0
                and self._reset_counter % self.hard_reset_interval == 0
            )
            if not need_hard_reset:
                # Do implicit resets, let episode continue
                next_obs = self.reset()
                done = False
                del info["EpisodeContinues"]
                info["SoftReset"] = True

        info["features"] = self._features_s
        return next_obs, reward, done, info

    def reset(self):
        # dont use inherited reset of FancyGymAsBiskSingleRobotEnv,
        # CtrlgsPreTrainingEnv.reset might call self.hard_reset which then actually
        # resets the FancyGymAsBiskSingleRobotEnv
        return CtrlgsPreTrainingEnv.reset(self)

    def fell_over(self) -> bool:
        # fell_over is defined in external bisk package, which we cant edit, so redirect
        if self._is_fancy_gym:
            return FancyGymAsBiskSingleRobotEnv.fell_over(self)
        else:
            return CtrlgsPreTrainingEnv.fell_over(self)
