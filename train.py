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
from models.functional_model import BasePolicyAgent

class RLReplayDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.buffer[idx]
        return state.float(), torch.tensor(action, dtype=torch.long), torch.tensor(reward, dtype=torch.float32), next_state.float(), torch.tensor(done, dtype=torch.float32)

if __name__ == "__main__":
    # Initialize environment
    env = battle_v4.env(map_size=45, max_cycles=300)
    replay_buffer = []
    max_buffer_size = 10000
    batch_size = 64
    n_episodes = 100

    # Initialize agent
    action_space_size = 21  # Reflecting possible actions in the Battle environment (move + attack)
    input_dim = 13 * 13 * 5  # Assuming the observation is a 13x13 grid with 5 features per cell
    agent = BasePolicyAgent(action_space_size, input_dim)

    # Training loop
    trainer = pl.Trainer(max_epochs=n_episodes, devices=1 if torch.cuda.is_available() else 3, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    for episode in tqdm(range(n_episodes), desc="Training episodes"):
        env.reset()
        episode_buffer = []
        for agent_name in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            agent_team = agent_name.split('_')[0]

            if termination or truncation:
                action = None  # this agent has died
            else:
                if agent_team == "blue":
                    # Blue agents use the learned policy
                    observation_tensor = torch.tensor(observation, dtype=torch.float32).flatten()
                    action = agent.select_action(observation_tensor)
                    next_observation, _, _, _, _ = env.last()
                    next_observation_tensor = torch.tensor(next_observation, dtype=torch.float32).flatten()
                    done = 1 if termination or truncation else 0
                    episode_buffer.append((observation_tensor, action, reward, next_observation_tensor, done))
                else:
                    # Red agents use a random policy
                    action = env.action_space(agent_name).sample()

            env.step(action)

        replay_buffer.extend(episode_buffer)
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]

        # Training step
        if len(replay_buffer) >= batch_size:
            dataset = RLReplayDataset(replay_buffer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            trainer.fit(agent, dataloader)

    # Evaluation loop
    env.reset()
    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        agent_team = agent_name.split('_')[0]

        if termination or truncation:
            action = None  # this agent has died
        else:
            if agent_team == "blue":
                # Blue agents use the learned policy in evaluation mode (greedy)
                observation_tensor = torch.tensor(observation, dtype=torch.float32).flatten()
                action = agent.select_action(observation_tensor, eval_mode=True)
            else:
                # Red agents use a random policy
                action = env.action_space(agent_name).sample()

        env.step(action)

    # Save model parameters
    torch.save(agent.q_network.state_dict(), "trained_agent.pth")
    print("Model parameters saved.")

    env.close()
