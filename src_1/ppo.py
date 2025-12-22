"""Core PPO optimization loop."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .rollout import RolloutBuffer


class PPOAgent:
    """Proximal Policy Optimization trainer."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        gamma: float,
        gae_lambda: float,
        n_epochs: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update policy using collected rollouts."""
        assert buffer.advantages is not None and buffer.returns is not None
        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = advantages

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for batch in buffer.get_batches(self.batch_size, self.n_epochs):
            observations = batch.observations.to(self.device)
            actions = batch.actions.to(self.device)
            old_logprobs = batch.old_logprobs.to(self.device)
            returns = batch.returns.to(self.device)
            advs = batch.advantages.to(self.device)

            logprobs, entropy, values = self.model.evaluate(observations, actions)
            ratio = (logprobs - old_logprobs).exp()

            unclipped = ratio * advs
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": float(sum(policy_losses) / max(len(policy_losses), 1)),
            "value_loss": float(sum(value_losses) / max(len(value_losses), 1)),
            "entropy": float(sum(entropy_losses) / max(len(entropy_losses), 1)),
        }


__all__ = ["PPOAgent"]
