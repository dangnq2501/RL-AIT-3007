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
    
class FunctionalPolicyAgent(pl.LightningModule):
    def __init__(self, action_space_size, input_dim, hidden_dim=256, dropout=0.3, epsilon=0.2):
        super(FunctionalPolicyAgent, self).__init__()
        self.action_space_size = action_space_size
        self.epsilon = epsilon
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.q_network = self.create_q_network(input_dim, action_space_size)

    def create_q_network(self, input_dim, action_space_size):
        return nn.Sequential(
            SubLayer(input_dim, self.hidden_dim, 0.3, True),
            SubLayer(self.hidden_dim, self.hidden_dim, 0.3, True),
            SubLayer(self.hidden_dim, action_space_size, None, True),
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
        expected_q_values = rewards + (0.9 * next_q_values * (1 - dones))
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)