import os
import warnings
import logging
import numpy as np
import random
import torch
from tqdm import tqdm
import gym
from gym.wrappers import StepAPICompatibility
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from utils import preprocess_frame, FrameStack
from reward import get_custom_reward
from model import CustomCNN
from DQN import DQN, ReplayMemory


# =====================================================
# Config & Environment Setup
# =====================================================
logging.getLogger("gym").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

env = gym_super_mario_bros.make("SuperMarioBros-v0", disable_env_checker=True)

if isinstance(env, gym.wrappers.TimeLimit):
    env = env.env

env = StepAPICompatibility(env, output_truncation_bool=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print("Final env:", env)

# =====================================================
# Hyperparameters
# =====================================================
NUM_FRAMES = 4
STICKY_ACTION_PROB = 0.25

LR = 1e-5
BATCH_SIZE = 64
GAMMA = 0.99
MEMORY_SIZE = 10000
TARGET_UPDATE = 50
TOTAL_TIMESTEPS = 1_000_000

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 300_000

VISUALIZE = True
MAX_STAGNATION_STEPS = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# DQN Initialization
# =====================================================
obs_shape = (NUM_FRAMES, 84, 84)
n_actions = len(SIMPLE_MOVEMENT)

dqn = DQN(
    model=CustomCNN,
    state_dim=obs_shape,
    action_dim=n_actions,
    learning_rate=LR,
    gamma=GAMMA,
    epsilon=EPSILON_START,
    target_update=TARGET_UPDATE,
    device=device,
)

memory = ReplayMemory(MEMORY_SIZE)
os.makedirs("ckpt", exist_ok=True)

best_reward = -float("inf")


def _reset_env(environment):
    reset_out = environment.reset()
    if isinstance(reset_out, tuple):
        return reset_out
    return reset_out, {}


# =====================================================
# Training Loop
# =====================================================
for timestep in tqdm(range(1, TOTAL_TIMESTEPS + 1), desc="Training"):

    obs, _ = _reset_env(env)
    frame = preprocess_frame(obs)

    frame_stack = FrameStack(NUM_FRAMES)
    state = frame_stack.reset(frame)  # (4, 84, 84)

    prev_action = 0
    done = False
    stagnation_time = 0

    prev_info = {
        "x_pos": 0,
        "y_pos": 0,
        "score": 0,
        "coins": 0,
        "time": 400,
        "flag_get": False,
        "life": 3,
    }

    cumulative_reward = 0.0
    cumulative_custom_reward = 0.0

    while not done:

        # -------------------------------
        # Sticky action
        # -------------------------------
        if random.random() < STICKY_ACTION_PROB:
            action = prev_action
        else:
            action = dqn.take_action(state)

        prev_action = action

        # -------------------------------
        # Step environment
        # -------------------------------
        step_out = env.step(action)
        if len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_out

        # -------------------------------
        # Frame stack update
        # -------------------------------
        next_frame = preprocess_frame(next_obs)
        next_state = frame_stack.append(next_frame)

        cumulative_reward += reward

        # -------------------------------
        # Custom reward (Day 1)
        # -------------------------------
        custom_reward = get_custom_reward(info, prev_info, done)
        cumulative_custom_reward += custom_reward

        # -------------------------------
        # Early stop if stagnation
        # -------------------------------
        if info["x_pos"] == prev_info["x_pos"]:
            stagnation_time += 1
            if stagnation_time >= MAX_STAGNATION_STEPS:
                done = True
        else:
            stagnation_time = 0

        # -------------------------------
        # Store transition
        # -------------------------------
        memory.push(state, action, custom_reward, next_state, done)
        state = next_state
        prev_info = info

        # -------------------------------
        # Train
        # -------------------------------
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            state_dict = {
                "states": batch[0],
                "actions": batch[1],
                "rewards": batch[2],
                "next_states": batch[3],
                "dones": batch[4],
            }
            dqn.train_per_step(state_dict)

        # -------------------------------
        # Epsilon decay
        # -------------------------------
        dqn.epsilon = max(
            EPSILON_END,
            EPSILON_START
            - (EPSILON_START - EPSILON_END) * timestep / EPSILON_DECAY_STEPS,
        )

        if VISUALIZE:
            env.render()

    # =================================================
    # Save best model
    # =================================================
    print(
        f"[Step {timestep}] Reward={cumulative_reward:.1f} | Custom={cumulative_custom_reward:.1f}"
    )

    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        path = f"ckpt/step_{timestep}_reward_{int(best_reward)}.pth"
        torch.save(dqn.q_net.state_dict(), path)
        print("Saved:", path)

env.close()
