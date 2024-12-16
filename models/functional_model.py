import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random


class BasePolicyAgent(pl.LightningModule):
    def __init__(self, action_space_size, input_dim, epsilon=0.2):
        super(BasePolicyAgent, self).__init__()
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.q_network = self.create_q_network(input_dim, action_space_size)

    def create_q_network(self, input_dim, action_space_size):
        return nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_space_size)
        )

    def forward(self, observation):
        features = self.feature_extractor(observation)
        return self.q_network(features)

    def feature_extractor(self, observation):
        observation = torch.tensor(observation, dtype=torch.float)
        batchsize = observation.shape[0]
        observation = observation.reshape(batchsize, -1)
        return observation

    def select_action(self, observation, eval_mode=False):

        if not eval_mode and random.random() < self.epsilon:
            # Epsilon-greedy action selection during training
            return random.randint(0, self.action_space_size - 1)
        
        with torch.no_grad():
            q_values = self.forward(observation.unsqueeze(0))
        return torch.argmax(q_values).item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        q_values = self.forward(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.forward(next_states).max(1)[0]
        expected_q_values = rewards + (0.95 * next_q_values * (1 - dones))
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
class SubLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=None, relu=None):
        super(SubLayer, self).__init__()
        layers = [nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)]
        
        if dropout:
            layers.append(nn.Dropout(dropout))
        if relu:
            layers.append(nn.ReLU())
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random
import os
from magent2.environments import battle_v4
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


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

    # Trả về dạng phù hợp với training_step
    return {'blue': states}, actions, rewards, {'blue': next_states}, dones

class SpatialCNN(nn.Module):
    def __init__(self, in_channels=5, out_channels=32):
        super(SpatialCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv(x)  # (B, out_channels, H, W)

class FunctionalPolicyAgent(pl.LightningModule):
    def __init__(self, action_space_size, embed_dim=5, height=13, width=13, hidden_dim=256, dropout=0.3, epsilon=0.2):
        super(FunctionalPolicyAgent, self).__init__()
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.height = height
        self.width = width
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Spatial CNN
        self.spatial = SpatialCNN(in_channels=embed_dim, out_channels=32)

        # Q-network
        self.q_network = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Flatten(),
            nn.Linear(height*width*3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_space_size)
        )

    def forward(self, obs):
        # obs: (B,H,W,C)
        obs = obs.permute(0,3,1,2).contiguous()  # (B,C,H,W)
        spatial_features = self.spatial(obs)  # (B,32,H,W)
        q_values = self.q_network(spatial_features)
        return q_values

    def select_action(self, obs, eval_mode=False):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)  # (1,H,W,C)
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        with torch.no_grad():
            q_values = self.forward(obs)
        return torch.argmax(q_values, dim=-1).item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        # states['blue']: (B,H,W,C)
        blue_obs = states['blue']
        next_blue_obs = next_states['blue']
        actions = actions
        rewards = rewards
        dones = dones

        q_values = self.forward(blue_obs)
        with torch.no_grad():
            q_values_next = self.forward(next_blue_obs)
        max_next_q = q_values_next.max(dim=1)[0]
        target = rewards + 0.9 * max_next_q * (1 - dones)

        q_values_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_values_current, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)