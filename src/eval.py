import os
import numpy as np
import torch
from tqdm import tqdm

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from model import CustomCNN
from DQN import DQN

# ========== Config ===========
MODEL_PATH = os.path.join("ckpt","step_18_reward_536_custom_586.pth")

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# SIMPLE_MOVEMENT = [
#    # ["NOOP"],       # Do nothing.
#     ["right"],      # Move right.
#     ["right", "A"], # Move right and jump.
#     ["right", "B"], # Move right and run.
#     ["right", "A", "B"], # Move right, run, and jump.
#    # ["A"],          # Jump straight up.
#     ["left"],       # Move left.
#     ["left", "A"], # Move right and jump.
#     ["left", "B"], # Move right and run.
#     ["left", "A", "B"], # Move right, run, and jump.
# ]

env = JoypadSpace(env, SIMPLE_MOVEMENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_SHAPE = (1, 84, 84)
N_ACTIONS = len(SIMPLE_MOVEMENT)

VISUALIZE = True
TOTAL_EPISODES = 10

# ========== Initialize DQN ===========
dqn = DQN(
    model=CustomCNN, 
    state_dim=OBS_SHAPE,
    action_dim=N_ACTIONS,
    learning_rate=0.0001,  
    gamma=0.99,          
    epsilon=0.0,         
    target_update=1000, 
    device=device
)

if os.path.exists(MODEL_PATH):
    try:
        model_weights = torch.load(MODEL_PATH, map_location=device)
        dqn.q_net.load_state_dict(model_weights)
        dqn.q_net.eval() 
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        raise
else:
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# ========== Evaluation Loop ===========
for episode in range(1, TOTAL_EPISODES + 1):
    state = env.reset()
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)  # Add channel dimension
    state = np.expand_dims(state, axis=0)  # Add batch dimension

    done = False
    total_reward = 0

    while not done:
        # Take action using the trained policy
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            action_probs = torch.softmax(dqn.q_net(state_tensor), dim=1)
            action = torch.argmax(action_probs, dim=1).item()

        next_state, reward, done, info = env.step(action)

        # Preprocess next state
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)  # Add channel dimension
        next_state = np.expand_dims(next_state, axis=0)  # Add batch dimension

        # Accumulate rewards
        total_reward += reward
        state = next_state

        if VISUALIZE:
            env.render()

    print(f"Episode {episode}/{TOTAL_EPISODES} - Total Reward: {total_reward}")

env.close()
