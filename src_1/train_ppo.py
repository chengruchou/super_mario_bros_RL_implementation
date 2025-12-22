"""Training entrypoint for PPO on Super Mario Bros."""

from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import gym

from .config import PPOConfig
from .envs import make_env
from .models import ActorCriticCNN
from .ppo import PPOAgent
from .rollout import RolloutBuffer
from .utils import ensure_dir, set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # type: ignore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Super Mario Bros")
    parser.add_argument(
        "--event_reward",
        action="store_true",
        default=True,
        help="Enable event-based reward shaping (coin, power-up, score)",
    )
    parser.add_argument("--coin_reward", type=float, default=0.1)
    parser.add_argument("--powerup_reward", type=float, default=1.0)
    parser.add_argument("--score_scale", type=float, default=0.001)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=float, default=1e7)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="ckpt_ppo")
    parser.add_argument("--log_dir", type=str, default="log_ppo")
    parser.add_argument("--save_interval", type=int, default=200_000)
    parser.add_argument("--render", action="store_true", help="Enable env rendering")
    parser.add_argument("--action_type", type=str, default="simple", choices=["simple", "right"])
    parser.add_argument("--clip_reward", action="store_true", help="Clip reward to {-1,0,1}")
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


def train() -> None:
    args = _parse_args()

    config = PPOConfig(
        world=args.world,
        stage=args.stage,
        action_type=args.action_type,
        seed=args.seed,
        render=args.render,
        clip_reward=args.clip_reward,
        total_steps=int(args.total_steps),
        n_steps=args.n_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
    )

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(config.save_dir)

    writer = SummaryWriter(config.log_dir) if (config.log_dir and SummaryWriter) else None

    # -------- create env (舊 nes-py，不用 render_mode) --------
    env = make_env(
        config.world,
        config.stage,
        action_type=config.action_type,
        seed=config.seed,
        render=config.render,
        clip_reward=config.clip_reward,
        # ⭐ event reward
        use_event_reward=config.use_event_reward,
        coin_reward=config.coin_reward,
        powerup_reward=config.powerup_reward,
        score_scale=config.score_scale,
    )


    obs, _ = _reset_env(env)

    obs_shape = obs.shape
    n_actions = env.action_space.n

    model = ActorCriticCNN(
        in_channels=obs_shape[0],
        n_actions=n_actions
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)

    buffer = RolloutBuffer(config.n_steps, obs_shape, device)

    agent = PPOAgent(
        model=model,
        optimizer=optimizer,
        clip_eps=config.clip_eps,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
        device=device,
    )

    episode_returns: deque[float] = deque(maxlen=20)
    episode_lengths: deque[int] = deque(maxlen=20)

    ep_return = 0.0
    ep_len = 0
    global_step = 0
    update_idx = 0

    while global_step < config.total_steps:
        buffer.clear()

        for _ in range(config.n_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, logprob, value = model.act(obs_tensor)

            action_item = int(action.item())

            step_out = _step_env(env, action_item)
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated

            # 🔑 舊 nes-py：顯示畫面一定要手動 render
            if config.render:
                env.render()

            buffer.add(
                obs_tensor.squeeze(0),
                action,
                float(reward),
                bool(done),
                value,
                logprob,
            )

            ep_return += float(reward)
            ep_len += 1
            global_step += 1
            obs = next_obs

            if done:
                episode_returns.append(ep_return)
                episode_lengths.append(ep_len)
                ep_return = 0.0
                ep_len = 0
                obs, _ = _reset_env(env)

            if global_step >= config.total_steps:
                break

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, next_value = model.forward(next_obs_tensor)
            next_value = next_value.squeeze(-1)

        buffer.compute_returns_and_advantages(
            next_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        stats = agent.update(buffer)
        update_idx += 1

        avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        avg_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

        print(
            f"Update {update_idx} | Step {global_step}/{config.total_steps} | "
            f"Return {avg_return:.1f} | Len {avg_length:.1f} | "
            f"Policy {stats['policy_loss']:.4f} | "
            f"Value {stats['value_loss']:.4f} | "
            f"Entropy {stats['entropy']:.4f}"
        )

        if writer:
            writer.add_scalar("charts/avg_return", avg_return, global_step)
            writer.add_scalar("charts/avg_ep_length", avg_length, global_step)
            writer.add_scalar("losses/policy", stats["policy_loss"], global_step)
            writer.add_scalar("losses/value", stats["value_loss"], global_step)
            writer.add_scalar("losses/entropy", stats["entropy"], global_step)

        if global_step % config.save_interval < config.n_steps:
            ckpt = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": global_step,
                "config": config,
            }
            ckpt_path = os.path.join(config.save_dir, f"ppo_step_{global_step}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    env.close()
    if writer:
        writer.close()


if __name__ == "__main__":
    train()
