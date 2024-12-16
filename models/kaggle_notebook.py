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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class QNetwork(pl.LightningModule):
    def __init__(self, observation_shape=(13,13,5), action_shape=21, epsilon=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.action_shape = action_shape
        self.epsilon = epsilon
        C = observation_shape[-1]
        H, W = observation_shape[0], observation_shape[1]

        # ResNet-like structure
        self.stage1 = nn.Sequential(
            ResidualBlock(C, C, stride=1),
            ResidualBlock(C, C, stride=1)
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(C, C*2, stride=2),
            ResidualBlock(C*2, C*2, stride=1)
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(C*2, C*4, stride=2),
            ResidualBlock(C*4, C*4, stride=1)
        )

        self.upsample = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)

        with torch.no_grad():
            dummy_input = torch.randn(*observation_shape).permute(2,0,1).unsqueeze(0)
            x = self.stage1(dummy_input)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.upsample(x)
            flatten_dim = x.view(1, -1).shape[1]

        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, obs):
        # obs: (B,H,W,C)
        obs = obs.permute(0,3,1,2).contiguous() # (B,C,H,W)
        x = self.stage1(obs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.upsample(x)
        x = x.reshape(x.shape[0], -1)
        return self.network(x)

    def select_action(self, obs, eval_mode=False):
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_shape - 1)
        with torch.no_grad():
            q_values = self(obs)
        return torch.argmax(q_values, dim=-1).item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        blue_obs = states['blue']
        next_blue_obs = next_states['blue']
        q_values = self(blue_obs)
        with torch.no_grad():
            q_values_next = self(next_blue_obs)
        max_next_q = q_values_next.max(dim=1)[0]
        target = rewards + 0.9 * max_next_q * (1 - dones)

        q_values_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q_values_current, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)