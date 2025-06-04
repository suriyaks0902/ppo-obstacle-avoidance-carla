import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from collections import deque
from env.carla_env import CarlaEnv
from ppo.ppo_agent import PPOAgent
from ppo.utils import compute_gae, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_episodes = 2000
max_timesteps = 1000
update_every = 2048
mini_batch_size = 64
ppo_epochs = 10
gamma = 0.99
lam = 0.95
clip_eps = 0.2
entropy_coef = 0.01
value_coef = 0.5
lr = 3e-4

env = CarlaEnv()
input_channels = 6
action_dim = 2

agent = PPOAgent(input_channels, action_dim).to(device)
optimizer = Adam(agent.parameters(), lr=lr)

episode_rewards = []
trajectory = []

for episode in range(num_episodes):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, 6, 480, 640]
    ep_reward = 0

    for t in range(max_timesteps):
        action, log_prob = agent.act(state)
        clipped_action = action.squeeze().detach().cpu().numpy()
        next_state, reward, done, _ = env.step(clipped_action)

        trajectory.append({
            'state': state,
            'action': action,
            'log_prob': log_prob,
            'reward': reward,
            'done': done
        })

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        ep_reward += reward

        if len(trajectory) >= update_every:
            print(f"\n[UPDATE] Training on {len(trajectory)} steps...")
            # Stack trajectory tensors
            states = torch.cat([x['state'] for x in trajectory])
            actions = torch.cat([x['action'] for x in trajectory])
            log_probs_old = torch.cat([x['log_prob'].unsqueeze(0) for x in trajectory])
            rewards = [x['reward'] for x in trajectory]
            dones = [x['done'] for x in trajectory]

            returns, advantages = compute_gae(agent, states, rewards, dones, gamma, lam)

            for _ in range(ppo_epochs):
                for i in range(0, len(states), mini_batch_size):
                    idx = slice(i, i + mini_batch_size)
                    batch_states = states[idx]
                    batch_actions = actions[idx]
                    batch_old_log_probs = log_probs_old[idx].detach()
                    batch_returns = returns[idx].detach()
                    batch_advantages = advantages[idx].detach()

                    log_probs, entropy, values = agent.evaluate(batch_states, batch_actions)
                    ratios = torch.exp(log_probs - batch_old_log_probs)
                    surrogate1 = ratios * batch_advantages
                    surrogate2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * batch_advantages
                    actor_loss = -torch.min(surrogate1, surrogate2).mean()
                    critic_loss = F.mse_loss(values, batch_returns)
                    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            trajectory = []  # Clear after update

        if done:
            break

    episode_rewards.append(ep_reward)
    print(f"Episode {episode} → Reward: {ep_reward:.2f}")

    if episode % 50 == 0:
        save_model(agent, f"models/ppo_episode_{episode}.pt")

