"""Evaluation script for PPO agent on Super Mario Bros."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import gym
import numpy as np
import torch

from .envs import make_env
from .models import ActorCriticCNN
from .utils import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PPO agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action_type", type=str, default="simple", choices=["simple", "right"])
    parser.add_argument("--clip_reward", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy actions")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--record_dir", type=str, default=None, help="Directory to save videos")
    return parser.parse_args()


def _reset_env(env: gym.Env) -> Tuple[np.ndarray, Dict]:
    reset_out = env.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out
    return reset_out, {}


def _step_env(env: gym.Env, action: int):
    step_out = env.step(action)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        return step_out
    obs, reward, done, info = step_out
    return obs, reward, done, False, info


def load_checkpoint(path: str, model: torch.nn.Module, device: torch.device) -> Dict:
    """Load model and optimizer state."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def test() -> None:
    args = _parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(
        args.world,
        args.stage,
        action_type=args.action_type,
        seed=args.seed,
        render=args.render,
        clip_reward=args.clip_reward,
    )
    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=args.record_dir)

    obs, _ = _reset_env(env)
    obs_shape = obs.shape
    n_actions = env.action_space.n

    model = ActorCriticCNN(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    print(f"Loaded checkpoint from {args.checkpoint}, step={checkpoint.get('step', 'n/a')}")

    model.eval()
    for ep in range(1, args.episodes + 1):
        done = False
        ep_return = 0.0
        ep_len = 0
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model.forward(obs_tensor)
                if args.deterministic:
                    action = torch.argmax(logits, dim=-1)
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
            action_item = int(action.item())
            step_out = _step_env(env, action_item)
            obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
            ep_return += float(reward)
            ep_len += 1
            if args.render:
                env.render()

        print(f"Episode {ep}: return={ep_return:.1f}, length={ep_len}")
        obs, _ = _reset_env(env)

    env.close()


if __name__ == "__main__":
    test()
