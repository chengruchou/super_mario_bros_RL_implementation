"""Environment utilities and wrappers for PPO training on Super Mario Bros."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Tuple, Optional

import cv2
import gym
from gym.wrappers import StepAPICompatibility
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np

def _unpack_reset(reset_output: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Handle reset outputs from both old and new Gym APIs."""
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        obs, info = reset_output
        return obs, info
    return reset_output, {}


def _unpack_step(step_output: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    """Handle step outputs from both old and new Gym APIs."""
    if isinstance(step_output, tuple) and len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        return obs, float(reward), bool(terminated), bool(truncated), info
    obs, reward, done, info = step_output
    return obs, float(reward), bool(done), False, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame and max over the last two frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=2)

    def reset(self, **kwargs: Any) -> Any:
        reset_out = self.env.reset(**kwargs)
        obs, info = _unpack_reset(reset_out)
        self._obs_buffer.clear()
        self._obs_buffer.append(obs)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        for _ in range(self._skip):
            step_out = self.env.step(action)
            obs, reward, term, trunc, info = _unpack_step(step_out)
            total_reward += reward
            self._obs_buffer.append(obs)
            terminated = terminated or term
            truncated = truncated or trunc
            if term or trunc:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

class StickyJumpWrapper(gym.Wrapper):
    """
    Sticky Jump Wrapper:
    - 當 agent 選擇含有跳躍(A)的 action 時
    - 自動在接下來的 k 個 step 強制保留 A
    - 解決 Mario 高跳 / 連跳需要「持續按 A」的控制問題
    """

    def __init__(self, env: gym.Env, jump_hold_steps: int = 2):
        super().__init__(env)
        self.jump_hold_steps = jump_hold_steps
        self._jump_counter = 0
        self._last_action = None

    def reset(self, **kwargs):
        self._jump_counter = 0
        self._last_action = None
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        action: int (JoypadSpace 離散 action index)
        """
        # 如果還在 sticky jump 狀態
        if self._jump_counter > 0:
            action = self._last_action
            self._jump_counter -= 1
        else:
            # 判斷這個 action 是否包含跳躍 (A)
            action_buttons = self.env.unwrapped._actions[action]
            if "A" in action_buttons:
                self._jump_counter = self.jump_hold_steps
                self._last_action = action

        return self.env.step(action)

class EventRewardWrapper(gym.Wrapper):
    """
    給 Mario 的「語意事件」額外 reward
    - 吃金幣
    - 吃到 power-up（小 → 大）
    - 撞磚頭（score 變化）
    """

    def __init__(
        self,
        env: gym.Env,
        coin_reward: float = 1.0,
        powerup_reward: float = 5.0,
        score_scale: float = 0.01,
    ):
        super().__init__(env)
        self.coin_reward = coin_reward
        self.powerup_reward = powerup_reward
        self.score_scale = score_scale
        self.prev_info = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = reward

        if terminated and info.get("flag_get", False) is False:
            shaped_reward -= 50.0
            
        if self.prev_info is not None:
            # 1️⃣ 吃金幣
            if info.get("coins", 0) > self.prev_info.get("coins", 0):
                shaped_reward += self.coin_reward

            # 2️⃣ 吃到道具（小 → 大）
            if info.get("status") != self.prev_info.get("status"):
                shaped_reward += self.powerup_reward

            # 3️⃣ 撞磚 / 得分
            delta_score = info.get("score", 0) - self.prev_info.get("score", 0)
            if delta_score > 0:
                shaped_reward += self.score_scale * delta_score

        self.prev_info = info
        return obs, shaped_reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, 1}."""

    def reward(self, reward: float) -> float:  # type: ignore[override]
        return float(np.sign(reward))


class GrayScaleResizeObservation(gym.ObservationWrapper):
    """Convert RGB frames to 84x84 grayscale."""

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:  # type: ignore[override]
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)


class TransposeFrame(gym.ObservationWrapper):
    """Reorder observation to channel-first and normalize to [0, 1]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(f"Expected stacked observation with 3 dims, got {obs_shape}")
        # Heuristic: if first dimension is small (e.g., 4) and last is large (e.g., 84), assume channel-first.
        self.channel_first_input = obs_shape[0] <= 8 and obs_shape[-1] >= 32
        if self.channel_first_input:
            c, h, w = obs_shape
        else:
            h, w, c = obs_shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:  # type: ignore[override]
        obs = np.array(observation, copy=False)
        if obs.ndim == 2:
            obs = np.expand_dims(obs, axis=-1)
        if not self.channel_first_input:
            obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.float32) / 255.0


def make_env(
     world: int,
    stage: int,
    action_type: str = "simple",
    seed: int = 0,
    render: bool = False,
    clip_reward: bool = False,
    # ⭐ 新增
    use_event_reward: bool = False,
    coin_reward: float = 1.0,
    powerup_reward: float = 5.0,
    score_scale: float = 0.01,
) -> gym.Env:
    """
    Create a Super Mario Bros environment with preprocessing and wrappers.
    """
    level = f"SuperMarioBros-{world}-{stage}-v0"
    env = gym_super_mario_bros.make(level, disable_env_checker=True)

    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    env = StepAPICompatibility(env, output_truncation_bool=True)

    movement = SIMPLE_MOVEMENT if action_type == "simple" else RIGHT_ONLY
    env = JoypadSpace(env, movement)
    env = MaxAndSkipEnv(env, skip=4)
    env = EventRewardWrapper(env, coin_reward=1.0, powerup_reward=5.0, score_scale=0.01,)
    env = GrayScaleResizeObservation(env, width=84, height=84)
    env = gym.wrappers.FrameStack(env, num_stack=4)
    env = TransposeFrame(env)
    if clip_reward:
        env = ClipRewardEnv(env)

    def _safe_reset(e: gym.Env, s: int):
        try:
            return e.reset(seed=s)
        except TypeError:
            if hasattr(e, "seed"):
                e.seed(s)
            return e.reset()

    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    reset_out = _safe_reset(env, seed)
    _obs, _info = _unpack_reset(reset_out)

    if use_event_reward:
        env = EventRewardWrapper(
            env,
            coin_reward=coin_reward,
            powerup_reward=powerup_reward,
            score_scale=score_scale,
        )

    return env


__all__ = ["make_env"]
