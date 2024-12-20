import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random
import os

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