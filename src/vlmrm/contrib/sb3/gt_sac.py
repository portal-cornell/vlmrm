import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import stable_baselines3.common.noise as sb3_noise
import torch
from einops import rearrange
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from stable_baselines3.common.type_aliases import MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import check_for_correct_spaces, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from vlmrm.trainer.config import Config

SelfGTSAC = TypeVar("SelfGTSAC", bound="GTSAC")


class GTSAC(SAC):
    def __init__(
        self,
        *,
        env: VecEnv,
        config: Config
    ):
        self.config = config
        stats_window_size = (
            (config.rl.learning_starts + config.rl.train_freq * env.num_envs)
            // config.rl.episode_length
            // env.num_envs
        ) * env.num_envs

        if config.rl.action_noise:
            mean = config.rl.action_noise.mean * np.ones(env.action_space.shape)
            sigma = config.rl.action_noise.sigma * np.ones(env.action_space.shape)
            if config.rl.action_noise.name == "NormalActionNoise":
                action_noise = sb3_noise.NormalActionNoise(mean=mean, sigma=sigma)
            elif config.rl.action_noise.name == "OrnsteinUhlenbeckActionNoise":
                action_noise = sb3_noise.OrnsteinUhlenbeckActionNoise(
                    mean=mean,
                    sigma=sigma,
                    theta=config.rl.action_noise.theta,
                    dt=config.rl.action_noise.dt,
                )
            else:
                raise ValueError(
                    f"Unknown action noise name: {config.rl.action_noise.name}"
                )
        else:
            action_noise = None

        super().__init__(
            env=env,
            policy=config.rl.policy_name,
            stats_window_size=stats_window_size,
            learning_starts=config.rl.learning_starts,
            train_freq=config.rl.train_freq,
            gradient_steps=config.rl.gradient_steps,
            batch_size=config.rl.batch_size,
            verbose=True,
            tensorboard_log=str(config.tb_dir),
            seed=config.seed,
            device="cuda:0",
            learning_rate=config.rl.learning_rate,
            tau=config.rl.tau,
            gamma=config.rl.gamma,
            action_noise=action_noise,
            buffer_size=config.rl.buffer_size,
            ent_coef=config.rl.ent_coef,
            use_sde=config.rl.use_sde,
            target_update_interval=config.rl.target_update_interval,
            policy_kwargs=config.rl.policy_kwargs,
            **(config.rl.rl_kwargs if config.rl.rl_kwargs else {}),
        )

    def _log(self) -> None:
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_gt_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self._log()
        super().train(gradient_steps, batch_size)

    @classmethod
    def load(
        cls: Type[SelfGTSAC],
        path: Union[str, pathlib.Path],
        *,
        env: Optional[VecEnv] = None,
        device: Union[torch.device, str] = "cuda:0",
        custom_objects: Optional[Dict[str, Any]] = None,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfGTSAC:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it
        in-place! For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to load the agent from
        :param env: the new environment to run the loaded model on (can be None if you
            only need prediction from a trained model) has priority over any saved
            environment.
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace upon loading. If a
            variable is present in this dictionary as a key, it will not be deserialized
            and the corresponding item will be used instead. Similar to custom_objects
            in ``keras.models.load_model``. Useful when you have an object in file that
            can not be deserialized.
        :param force_reset: Force call to ``reset()`` before training to avoid
            unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if (
                "net_arch" in data["policy_kwargs"]
                and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                    saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, "
                f"specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError(
                "The observation_space and action_space were not given, can't verify "
                "new environments."
            )

        # Gym -> Gymnasium space conversion
        for key in {"observation_space", "action_space"}:
            data[key] = _convert_space(
                data[key]
            )  # pytype: disable=unsupported-operands

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Check if given env is valid
            check_for_correct_spaces(
                env, data["observation_space"], data["action_space"]
            )
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated.
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for
            # predict)
            if "env" in data:
                env = data["env"]

        data["config"].rl.action_noise = None

        # pytype: disable=not-instantiable,wrong-keyword-args
        model = cls(
            env=env,
            config=data["config"]
        )
        # pytype: enable=not-instantiable,wrong-keyword-args

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for "
                    f"more info). Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                raise e
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward
                # compatibility). This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is
                # defined, otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()

        return model