import torch as T
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from collections import deque


# =====================================================
# Rollout Buffer (keeps name ReplayMemory for compatibility)
# =====================================================
class ReplayMemory:
    """
    On-policy rollout buffer compatible with existing run.py calls.
    Stores trajectories and yields the latest batch_size transitions in order,
    then clears for the next rollout.
    """

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            raise ValueError("Not enough samples")
        # Use the most recent batch_size steps to preserve temporal order
        batch = list(self.memory)[-batch_size:]
        self.memory.clear()
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


# =====================================================
# PPO Agent (keeps class name DQN for compatibility)
# =====================================================
class DQN:
    """
    PPO agent with actor-critic network, clipped objective, GAE advantages.
    Interface kept the same for run.py (take_action, train_per_step, epsilon attr, q_net attr).
    """

    def __init__(
        self,
        model,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
    ):
        self.device = device
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon  # kept for compatibility (not used in PPO policy)
        self.clip_eps = 0.1
        self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.n_epochs = 4
        self.max_grad_norm = 0.5
        self.minibatch_size = None  # default: use full batch

        self.q_net = self._build_net(model, state_dim, action_dim)
        self.optimizer = T.optim.Adam(self.q_net.parameters(), lr=learning_rate, eps=1e-5)

    def _build_net(self, model, state_dim, action_dim):
        return model(state_dim, action_dim).to(self.device)

    # =================================================
    # Action Selection (policy sampling)
    # =================================================
    def take_action(self, state):
        state_x = T.as_tensor(state, dtype=T.float32, device=self.device)
        if state_x.ndim == 3:
            state_x = state_x.unsqueeze(0)  # [1, C, H, W]

        with T.no_grad():
            logits, _ = self.q_net(state_x)
            dist = Categorical(logits=logits)
            action = dist.sample()

        return action.item()

    # =================================================
    # PPO Update
    # =================================================
    def train_per_step(self, state_dict):
        states, actions, rewards, next_states, dones = self._state_2_tensor(state_dict)
        batch_size = states.shape[0]
        mb_size = self.minibatch_size or batch_size

        # Compute values and logprobs for collected rollout
        with T.no_grad():
            logits, values = self.q_net(states)
            dist = Categorical(logits=logits)
            old_logprobs = dist.log_prob(actions)
            next_logits, next_values = self.q_net(next_states)
            next_values = next_values.detach()

        advantages = self._compute_gae(rewards, dones, values.detach(), next_values, self.gamma, self.gae_lambda)
        returns = advantages + values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.n_epochs):
            permutation = T.randperm(batch_size)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                idx = permutation[start:end]

                logits, values_pred = self.q_net(states[idx])
                dist = Categorical(logits=logits)
                logprobs = dist.log_prob(actions[idx])
                entropy = dist.entropy()

                ratio = (logprobs - old_logprobs[idx]).exp()
                surr1 = ratio * advantages[idx]
                surr2 = T.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]
                policy_loss = -T.min(surr1, surr2).mean()

                value_loss = 0.5 * (returns[idx] - values_pred).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_bonus.item())

        # Return average losses (not used by run.py but available)
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }

    # =================================================
    # Utilities
    # =================================================
    def _state_2_tensor(self, state_dict):
        states = T.tensor(state_dict["states"], dtype=T.float32, device=self.device)
        actions = T.tensor(state_dict["actions"], dtype=T.long, device=self.device)
        rewards = T.tensor(state_dict["rewards"], dtype=T.float32, device=self.device)
        next_states = T.tensor(state_dict["next_states"], dtype=T.float32, device=self.device)
        dones = T.tensor(state_dict["dones"], dtype=T.float32, device=self.device)
        return states, actions, rewards, next_states, dones

    def _compute_gae(self, rewards, dones, values, next_values, gamma, lam):
        advantages = T.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = next_values[t]
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
        return advantages
