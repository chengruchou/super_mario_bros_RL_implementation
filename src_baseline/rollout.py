import torch
from typing import List, Tuple, Generator


class RolloutBuffer:
    """
    On-policy rollout buffer with GAE(lambda) support.
    Stores transitions for PPO updates.
    """

    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        logprob: torch.Tensor,
    ) -> None:
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.dones.append(torch.tensor(done, dtype=torch.float32, device=self.device))
        self.values.append(value.detach())
        self.logprobs.append(logprob.detach())

    def compute_returns_and_advantages(
        self, last_value: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        values = self.values + [last_value.detach()]
        gae = 0.0
        advantages: List[torch.Tensor] = []
        for step in reversed(range(len(self.rewards))):
            non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages.insert(0, torch.tensor(gae, device=self.device))
        self.advantages = torch.stack(advantages)
        self.returns = self.advantages + torch.stack(self.values)

    def get_minibatches(self, batch_size: int, n_epochs: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        assert self.advantages is not None and self.returns is not None
        dataset_size = len(self.states)
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs)
        returns = self.returns
        advantages = self.advantages
        values = torch.stack(self.values)

        for _ in range(n_epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]
                yield (
                    states[mb_idx],
                    actions[mb_idx],
                    logprobs[mb_idx],
                    returns[mb_idx],
                    advantages[mb_idx],
                    values[mb_idx],
                )

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None
