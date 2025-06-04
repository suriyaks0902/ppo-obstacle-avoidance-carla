import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class CNNEncoder(nn.Module):
    def __init__(self, input_channels=3):
        super(CNNEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # 480x640 → 119x159
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # → 58x78
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # → 56x76
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 76, 512),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(Actor, self).__init__()
        self.mu_head = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable log_std

    def forward(self, x):
        mu = torch.tanh(self.mu_head(x))  # tanh to keep outputs in [-1, 1]
        std = torch.exp(self.log_std)
        return mu, std

    def sample(self, x):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob, dist.entropy().sum(axis=-1)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.value_head(x)


class PPOAgent(nn.Module):
    def __init__(self, input_channels, action_dim):
        super(PPOAgent, self).__init__()
        self.encoder = CNNEncoder(input_channels)
        self.actor = Actor(512, action_dim)
        self.critic = Critic(512)

    def act(self, state):
        with torch.no_grad():
            features = self.encoder(state)
            action, log_prob, _ = self.actor.sample(features)
        return action, log_prob

    def evaluate(self, state, action):
        features = self.encoder(state)
        mu, std = self.actor(features)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(features).squeeze()
        return log_prob, entropy, value

