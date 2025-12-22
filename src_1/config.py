"""Default configuration for PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    """Hyperparameters for PPO."""

    world: int = 1
    stage: int = 1
    action_type: str = "simple"
    seed: int = 0
    render: bool = False
    clip_reward: bool = False

    # ===== Event-based reward shaping =====
    use_event_reward: bool = False
    coin_reward: float = 1.0
    powerup_reward: float = 5.0
    score_scale: float = 0.01

    total_steps: int = int(5e6)
    n_steps: int = 2048
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 256

    save_dir: str = "ckpt_ppo"
    log_dir: str | None = None
    save_interval: int = 200_000
    eval_interval: int = 0

    device: str | None = None
    extras: dict = field(default_factory=dict)


__all__ = ["PPOConfig"]
