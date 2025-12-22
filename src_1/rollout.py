"""Rollout storage with GAE for PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, Tuple

import torch


@dataclass
class Batch:
    """Mini-batch of rollout data."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class RolloutBuffer:
    """Collect trajectories and compute advantages."""

    def __init__(self, buffer_size: int, obs_shape: Tuple[int, ...], device: torch.device) -> None:
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device
        self.observations: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        logprob: torch.Tensor,
    ) -> None:
        """Store a single transition."""
        self.observations.append(obs.detach())
        self.actions.append(action.detach())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
        self.values.append(value.detach())
        self.logprobs.append(logprob.detach())

    def __len__(self) -> int:
        return len(self.observations)

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        """Compute GAE-Lambda advantages and returns."""
        values = self.values + [last_value.detach()]
        advantages = []
        gae = 0.0
        for step in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * values[step + 1] * next_non_terminal - values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
        self.advantages = torch.stack(advantages)
        self.returns = self.advantages + torch.stack(self.values)

    def get_batches(self, batch_size: int, n_epochs: int) -> Generator[Batch, None, None]:
        """Yield mini-batches for multiple epochs."""
        assert self.advantages is not None and self.returns is not None, "Advantages not computed"
        dataset_size = len(self.observations)
        indices = torch.arange(dataset_size)

        obs = torch.stack(self.observations)
        actions = torch.stack(self.actions)
        old_logprobs = torch.stack(self.logprobs)
        returns = self.returns.squeeze(-1)
        advantages = self.advantages.squeeze(-1)
        values = torch.stack(self.values).squeeze(-1)

        for _ in range(n_epochs):
            permutation = torch.randperm(dataset_size)
            indices = indices[permutation]
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                yield Batch(
                    observations=obs[batch_idx],
                    actions=actions[batch_idx],
                    old_logprobs=old_logprobs[batch_idx],
                    returns=returns[batch_idx],
                    advantages=advantages[batch_idx],
                    values=values[batch_idx],
                )

    def clear(self) -> None:
        """Reset buffer storage."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.logprobs.clear()
        self.advantages = None
        self.returns = None
