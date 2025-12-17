import os
import warnings
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import gym
from gym.wrappers import StepAPICompatibility
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame
from reward import *
from model import CustomCNN
from DQN import DQN, ReplayMemory


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

# ========== config ===========

logging.getLogger("gym").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*old step API.*")
warnings.filterwarnings("ignore", message=".*No render modes was declared.*")

# 1) make（這裡可能會自動包 TimeLimit）
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", disable_env_checker=True)

# 2) 拆掉 TimeLimit（不拆一定炸 expected 5 got 4）
if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

# 3) 轉成新 step API（補 terminated/truncated）
env = StepAPICompatibility(env, output_truncation_bool=True)

# 4) 再包 JoypadSpace
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)

# basic train config
LR = 0.00001
BATCH_SIZE = 64
GAMMA = 0.99
MEMORY_SIZE = 10000
EPSILON_END = 0.3
TARGET_UPDATE = 50
TOTAL_TIMESTEPS = 1000000
VISUALIZE = True  # 建議先 False，避免 render 影響訓練/相容性
MAX_STAGNATION_STEPS = 80

# 🔑 自動判斷 CPU / GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== config ===========

# DQN Initialization
obs_shape = (1, 84, 84)
n_actions = len(SIMPLE_MOVEMENT)
model = CustomCNN  # CustomCNN should accept state_dim and action_dim as arguments
dqn = DQN(
    model=model,
    state_dim=obs_shape,
    action_dim=n_actions,
    learning_rate=LR,
    gamma=GAMMA,
    epsilon=EPSILON_END,
    target_update=TARGET_UPDATE,
    device=device
)

memory = ReplayMemory(MEMORY_SIZE)
os.makedirs("ckpt", exist_ok=True)
step = 0
best_reward = -float('inf')  # Track the best reward in each SAVE_INTERVAL
cumulative_reward = 0  # Track cumulative reward for the current timestep

def _reset_env(environment):
    """Handle both old (obs) and new (obs, info) reset signatures."""
    reset_out = environment.reset()
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out
    return reset_out, {}

for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training Progress"):
    state, _ = _reset_env(env)
    state = preprocess_frame(state)
    state = np.expand_dims(state, axis=0)  # Add channel dimension

    done = False
    prev_info = {
        "x_pos": 0,  # Starting horizontal position (int).
        "y_pos": 0,  # Starting vertical position (int).
        "score": 0,  # Initial score is 0 (int).
        "coins": 0,  # Initial number of collected coins is 0 (int).
        "time": 400,  # Initial time in most levels of Super Mario Bros is 400 (int).
        "flag_get": False,  # Player has not yet reached the end flag (bool).
        "life": 3  # Default initial number of lives is 3 (int).
    }

    cumulative_custom_reward = 0
    cumulative_reward = 0 
    stagnation_time = 0
    while not done:
        action = dqn.take_action(state)

        # ===== 正確的 step 寫法（新舊 Gym 相容）=====
        step_out = env.step(action)
        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_out
            terminated, truncated = done, False


        # preprocess image state
        next_state = preprocess_frame(next_state)
        next_state = np.expand_dims(next_state, axis=0)

        cumulative_reward += reward

        # ===========================
        custom_reward = get_coin_reward(info, reward, prev_info)

        # ===========================

        cumulative_custom_reward += custom_reward // 1

        # Check for x_pos stagnation
        if info["x_pos"] == prev_info["x_pos"]:
            stagnation_time += 1
            if stagnation_time >= MAX_STAGNATION_STEPS:
                print(f"Timestep {timestep} - Early stop triggered due to x_pos stagnation.")
                done = True
        else:
            stagnation_time = 0

        # Store transition in memory
        memory.push(state, action, custom_reward //1, next_state, done)
        state = next_state

        # Train DQN
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)

            state_dict = {
                'states': batch[0],
                'actions': batch[1],
                'rewards': batch[2],
                'next_states': batch[3],
                'dones': batch[4],
            }
            dqn.train_per_step(state_dict)

        # Update epsilon
        dqn.epsilon = EPSILON_END

        prev_info = info
        step += 1

        if VISUALIZE:
            env.render()
            viewer = getattr(env.unwrapped, "viewer", None)
            if viewer is not None and getattr(viewer, "isopen", True) is False:
                print(f"Timestep {timestep} - Viewer closed, stopping episode.")
                done = True

    # Print cumulative reward for the current timestep
    print(f"Timestep {timestep} - Total Reward: {cumulative_reward} - Total Custom Reward: {cumulative_custom_reward}")

    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        model_path = os.path.join("ckpt",f"step_{timestep}_reward_{int(best_reward)}_custom_{int(cumulative_custom_reward)}.pth")
        torch.save(dqn.q_net.state_dict(), model_path)
        print(f"Model saved: {model_path}")

env.close()
