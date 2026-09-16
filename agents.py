import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RandomAgent:
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def act(self, state, training=True):
        return random.randrange(self.n_actions)

    def save(self, path):
        pass


def discretize_state(obs):
    """
    Convert continuous observations into a compact tuple for tabular RL.

    We deliberately do NOT discretize every floating-point coordinate.
    Instead we use track-relative/motion features that still come from
    the continuous (x, y) environment.
    """
    # x/y are useful but heavily binned.
    x = int(np.clip((obs[0] + 1.0) * 5, 0, 10))
    y = int(np.clip((obs[1] + 1.0) * 5, 0, 10))

    speed = int(np.clip((obs[4] + 0.0) * 5, 0, 5))
    heading_sin = int(np.clip((obs[7] + 1) * 2, 0, 4))
    heading_cos = int(np.clip((obs[8] + 1) * 2, 0, 4))
    distance = int(np.clip((obs[9] + 1) * 2, 0, 6))
    lateral = int(np.clip((obs[10] + 1) * 2, 0, 4))

    return (x, y, speed, heading_sin, heading_cos, distance, lateral)


class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.15,
        gamma=0.98,
        epsilon=1.0,
        epsilon_min=0.03,
        epsilon_decay=0.997,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = {}

    def _values(self, state):
        if state not in self.q:
            self.q[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q[state]

    def act(self, state, training=True):
        s = discretize_state(state)
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self._values(s)))

    def learn(self, state, action, reward, next_state, done):
        s = discretize_state(state)
        ns = discretize_state(next_state)
        q = self._values(s)
        target = reward if done else reward + self.gamma * np.max(self._values(ns))
        q[action] += self.alpha * (target - q[action])

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.save(path, self.q, allow_pickle=True)


class SarsaAgent(QLearningAgent):
    def learn(self, state, action, reward, next_state, next_action, done):
        s = discretize_state(state)
        ns = discretize_state(next_state)
        q = self._values(s)

        if done:
            target = reward
        else:
            target = reward + self.gamma * self._values(ns)[next_action]

        q[action] += self.alpha * (target - q[action])


class MonteCarloAgent(QLearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode = []

    def learn(self, state, action, reward, next_state, done):
        self.episode.append((discretize_state(state), action, reward))

    def end_episode(self):
        G = 0.0
        visited = set()

        # First-visit Monte Carlo update.
        for state, action, reward in reversed(self.episode):
            G = reward + self.gamma * G
            key = (state, action)
            if key not in visited:
                visited.add(key)
                q = self._values(state)
                q[action] += self.alpha * (G - q[action])

        self.episode.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, n_actions, lr=1e-3, gamma=0.99):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork(state_dim, n_actions).to(self.device)
        self.target = QNetwork(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.batch_size = 128
        self.learn_steps = 0

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(dim=1).item())

    def learn(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

        if len(self.buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, device=self.device)
        a = torch.tensor(a, device=self.device)
        r = torch.tensor(r, device=self.device)
        ns = torch.tensor(ns, device=self.device)
        d = torch.tensor(d, device=self.device)

        current = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target(ns).max(dim=1).values
            target = r + self.gamma * (1.0 - d) * next_q

        loss = nn.functional.smooth_l1_loss(current, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % 500 == 0:
            self.target.load_state_dict(self.online.state_dict())

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.online.state_dict(), path)

    def load(self, path):
        self.online.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.online.state_dict())


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class PPOAgent:
    """
    Compact educational PPO implementation.

    It collects a rollout and then performs several clipped policy updates.
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.rollout = []
        self.rollout_size = 2048
        self.epochs = 6
        self.batch_size = 256

    def act(self, state, training=True):
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)

        if training:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def store(self, state, action, reward, done, log_prob, value):
        self.rollout.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            float(done),
            float(log_prob),
            float(value),
        ))

    def update(self, last_value=0.0):
        if not self.rollout:
            return

        states = np.asarray([x[0] for x in self.rollout], dtype=np.float32)
        actions = np.asarray([x[1] for x in self.rollout], dtype=np.int64)
        rewards = np.asarray([x[2] for x in self.rollout], dtype=np.float32)
        dones = np.asarray([x[3] for x in self.rollout], dtype=np.float32)
        old_log_probs = np.asarray([x[4] for x in self.rollout], dtype=np.float32)
        values = np.asarray([x[5] for x in self.rollout], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_lp_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        n = len(states)
        indices = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                idx = torch.tensor(idx, dtype=torch.long, device=self.device)

                logits, value = self.net(states_t[idx])
                dist = torch.distributions.Categorical(logits=logits)

                new_lp = dist.log_prob(actions_t[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t[idx])

                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * adv_t[idx]

                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = nn.functional.mse_loss(value, ret_t[idx])

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

        self.rollout.clear()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
