import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn
import torch
class PPOActorCriticConv(nn.Module):
    def __init__(self, input_channels, input_size, action_space_size):
        super(PPOActorCriticConv, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # (input_channels x 13 x 13 -> 32 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),             # (32 x 13 x 13 -> 64 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                              # (64 x 13 x 13 -> 64 x 6 x 6)
        )
        # Flatten output from Conv2D
        conv_output_size = 64 * (input_size // 2) * (input_size // 2)  # For 13x13 input -> 64x6x6 = 2304

        # Fully Connected Shared Layer
        self.shared_layer = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU()
        )
        
        # Actor Head
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size),
            nn.Softmax(dim=-1)  # Output probabilities for actions
        )

        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output value of the state
        )

    def forward(self, x):
        # Input shape: (batch_size, input_channels, input_size, input_size)
        conv_output = self.conv_layers(x)  # Convolutional layers
        flat_output = torch.flatten(conv_output, start_dim=1)  # Flatten (batch_size, conv_output_size)
        shared_output = self.shared_layer(flat_output)  # Shared layer
        policy = self.actor(shared_output)  # Actor output
        value = self.critic(shared_output)  # Critic output
        return policy, value

class PPOAgentWithLightning(pl.LightningModule):
    def __init__(self, input_channels, input_size, action_space_size, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        super(PPOAgentWithLightning, self).__init__()
        self.model = PPOActorCriticConv(input_channels, input_size, action_space_size)
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def forward(self, x):
        policy, value = self.model(x)
        return policy, value

    def compute_loss(self, batch):
        states, actions, rewards, dones, old_policies = batch
        policy, value = self(states)
        value = value.squeeze(-1)

        # Compute Advantage
        returns, advantages = self.compute_advantages(rewards, value.detach(), dones)

        # Compute Policy Loss
        new_policies = policy.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        policy_ratio = new_policies / old_policies
        clipped_ratio = torch.clamp(policy_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(policy_ratio * advantages, clipped_ratio * advantages).mean()

        # Compute Value Loss
        value_loss = nn.MSELoss()(value, returns)

        # Combine Losses
        total_loss = policy_loss + 0.5 * value_loss
        return total_loss

    def compute_advantages(self, rewards, values, dones):
        returns = []
        advantages = []
        G = 0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            G = r + (1 - d) * self.gamma * G
            returns.insert(0, G)
            advantages.insert(0, G - v)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        return returns, advantages

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
from torch.utils.data import Dataset

class RLReplayDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, action, reward, next_state, done, old_policy = self.buffer[idx]
        return (
            torch.tensor(state, dtype=torch.float32).permute(2, 0, 1),  # (channels, height, width)
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
            torch.tensor(old_policy, dtype=torch.float32)
        )