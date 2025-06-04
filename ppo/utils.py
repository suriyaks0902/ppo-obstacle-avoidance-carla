import torch

def compute_gae(agent, states, rewards, dones, gamma=0.99, lam=0.95):
    with torch.no_grad():
        values = agent.critic(agent.encoder(states)).squeeze()
    returns = []
    advantages = []
    gae = 0
    value_next = 0

    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * value_next * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        value_next = values[step]
        returns.insert(0, gae + values[step])

    return torch.tensor(returns).float().to(values.device), torch.tensor(advantages).float().to(values.device)

def save_model(agent, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(agent.state_dict(), path)

