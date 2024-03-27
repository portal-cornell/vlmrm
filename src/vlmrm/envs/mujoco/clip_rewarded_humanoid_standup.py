import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoidstandup_v4 import HumanoidStandupEnv as GymHumanoidStandupEnv
from gymnasium.spaces import Box
from numpy.typing import NDArray


class CLIPRewardedHumanoidStandupEnv(GymHumanoidStandupEnv):
    def __init__(
        self,
        episode_length: int,
        render_mode: str = "rgb_array",
        camera_config: Optional[Dict[str, Any]] = None,
        textured: bool = True,
        **kwargs,
    ):
        terminate_when_unhealthy = False
        utils.EzPickle.__init__(
            self,
            render_mode=render_mode,
            **kwargs,
        )

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(376,), dtype=np.float64
        )
        env_file_name = None
        if textured:
            env_file_name = "humanoidstandup_textured.xml"
        else:
            env_file_name = "humanoidstandup.xml"
        model_path = str(pathlib.Path(__file__).parent / env_file_name)
        MujocoEnv.__init__(
            self,
            model_path,
            5,
            observation_space=observation_space,
            default_camera_config=camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        self.episode_length = episode_length
        self.num_steps = 0
        if camera_config:
            self.camera_id = -1

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length
        return (
            obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        return super().reset(seed=seed, options=options)
