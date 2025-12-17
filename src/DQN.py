import torch as T
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self,
                 model,
                 state_dim, action_dim, 
                 learning_rate, gamma,
                 epsilon, target_update, device):
        self.device = device
        self.action_dim = action_dim
        
        self.gamma = gamma  
        self.epsilon = epsilon
        self.target_update = target_update
        self.update_count = 0

        # Initialize [Q-net] and target [Q-net]
        self.model = model
        self.q_net = self._build_net(state_dim, action_dim)
        self.tgt_q_net = self._build_net(state_dim, action_dim)
        self.tgt_q_net.load_state_dict(self.q_net.state_dict())

        # Optimizer
        self.optimizer = T.optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def _build_net(self, state_dim, action_dim):
        return self.model(state_dim,action_dim).to(self.device)

    def take_action(self, state):
        # Exploration Unknown Policy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploitation Known Policy
        # Avoid wrapping a numpy array in a Python list (triggers slow-path warning)
        state_x = T.as_tensor(state, dtype=T.float32, device=self.device)
        if state_x.ndim == 3:
            state_x = state_x.unsqueeze(0)  # add batch dimension if missing
        with T.no_grad():
            action_probs = F.softmax(self.q_net(state_x), dim=1)
            action_dist  = T.distributions.Categorical(action_probs)
            return action_dist.sample().item()

    def get_loss(self, states, actions, rewards, next_states, dones):
        # Get current Q-values
        actions = actions.unsqueeze(1)
        q_val = self.q_net(states).gather(1, actions).squeeze(1)
        
        # Get maximum expected Q-values
        next_q_val = self.tgt_q_net(next_states).max(dim=1)[0]  # [B]
        
        # Compute target Q-values [custom-reward]
        q_target = rewards + self.gamma * next_q_val * (1 - dones.float())  # [B]
        
        return T.nn.functional.mse_loss(q_val, q_target.detach())

    def train_per_step(self, state_dict):
        # Convert one trajectory(s,a,r,n_s) to tensor
        states,actions,rewards,next_states,dones = self._state_2_tensor(state_dict)    

        # Compute loss 
        loss = self.get_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target-network periodically
        if self.update_count % self.target_update == 0:
            self.tgt_q_net.load_state_dict(self.q_net.state_dict())

        self.update_count += 1
    
    def _state_2_tensor(self,state_dict):
        # Convert one trajectory(s,a,r,n_s) to tensor
        states      = T.tensor(state_dict['states'], dtype=T.float32, device=self.device)
        actions     = T.tensor(state_dict['actions'], dtype=T.long, device=self.device)
        rewards     = T.tensor(state_dict['rewards'], dtype=T.float32, device=self.device)
        next_states = T.tensor(state_dict['next_states'], dtype=T.float32, device=self.device)
        dones       = T.tensor(state_dict['dones'], dtype=T.float32, device=self.device)

        return states,actions,rewards,next_states,dones
