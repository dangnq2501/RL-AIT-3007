import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random
import os
import cv2
from magent2.environments import battle_v4
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.kaggle_notebook import FunctionalPolicyAgent

class RLReplayDataset(Dataset):
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer

    def __len__(self):
        return len(self.replay_buffer)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.replay_buffer[idx]
        # state, next_state: (H,W,C)
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        states = state  # (H,W,C)
        next_states = next_state
        return states, action, reward, next_states, done

def collate_fn(batch):
    states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*batch)

    states = torch.stack(states_list, dim=0)        # (B,H,W,C)
    next_states = torch.stack(next_states_list,0)   # (B,H,W,C)
    actions = torch.stack(actions_list)
    rewards = torch.stack(rewards_list)
    dones = torch.stack(dones_list)

    return states, actions, rewards,  next_states, dones

env = battle_v4.env(map_size=45, max_cycles=300)
replay_buffer_blue = []
replay_buffer_red = []
max_buffer_size = 10000
batch_size = 64
n_episodes = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize agents
action_space_size = 21  
input_dim = 13 * 13 * 5  

blue_agent = FunctionalPolicyAgent(action_space_size)
red_agent = FunctionalPolicyAgent(action_space_size)

# Set up trainers
trainer_blue = pl.Trainer(max_epochs=n_episodes, devices=1 if torch.cuda.is_available() else 3, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
trainer_red = pl.Trainer(max_epochs=n_episodes, devices=1 if torch.cuda.is_available() else 3, accelerator='gpu' if torch.cuda.is_available() else 'cpu')

for episode in tqdm(range(n_episodes), desc="Training episodes"):
    env.reset()
    episode_buffer_blue = []
    episode_buffer_red = []
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        agent_team = agent_name.split('_')[0]

        if termination or truncation:
            action = None  # this agent has died
        else:
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            if agent_team == "blue":
                # Blue agents use their policy
                action = blue_agent.select_action(observation_tensor)
                next_observation, _, _, _, _ = env.last()
                next_observation_tensor = torch.tensor(next_observation, dtype=torch.float32)
                done = 1 if termination or truncation else 0
                episode_buffer_blue.append((observation_tensor, action, reward, next_observation_tensor, done))
            elif agent_team == "red":
                # Red agents use their policy
                action = red_agent.select_action(observation_tensor)
                next_observation, _, _, _, _ = env.last()
                next_observation_tensor = torch.tensor(next_observation, dtype=torch.float32)
                done = 1 if termination or truncation else 0
                episode_buffer_red.append((observation_tensor, action, reward, next_observation_tensor, done))

        env.step(action)

    replay_buffer_blue.extend(episode_buffer_blue)
    replay_buffer_red.extend(episode_buffer_red)

    if len(replay_buffer_blue) > max_buffer_size:
        replay_buffer_blue = replay_buffer_blue[-max_buffer_size:]
    if len(replay_buffer_red) > max_buffer_size:
        replay_buffer_red = replay_buffer_red[-max_buffer_size:]

    if len(replay_buffer_blue) >= batch_size:
        dataset_blue = RLReplayDataset(replay_buffer_blue)
        dataloader_blue = DataLoader(dataset_blue, batch_size=batch_size, shuffle=True)
        trainer_blue.fit(blue_agent, dataloader_blue)

    if len(replay_buffer_red) >= batch_size:
        dataset_red = RLReplayDataset(replay_buffer_red)
        dataloader_red = DataLoader(dataset_red, batch_size=batch_size, shuffle=True)
        trainer_red.fit(red_agent, dataloader_red)

torch.save(blue_agent.state_dict(), "blue_trained_agent.pth")
torch.save(red_agent.state_dict(), "red_trained_agent.pth")
print("Model parameters for both agents saved.")

env.close()
