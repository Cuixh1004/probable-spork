"""
Contextual reinforcement learning policy for coupon serving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Basic replay buffer for DQN training.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0

    def push(self, exp: Experience) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class CouponServingPolicy:
    """
    Deep Q-learning policy that chooses which coupon action to take.
    """

    state_dim: int
    action_dim: int
    gamma: float = 0.95
    learning_rate: float = 1e-3
    batch_size: int = 64
    device: str = "cpu"
    update_target_every: int = 100

    q_net: QNetwork | None = None
    target_q_net: QNetwork | None = None
    optimiser: Adam | None = None

    def __post_init__(self) -> None:
        self.q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimiser = Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer()
        self.step_count = 0

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b.action for b in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target_q_net(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def store_experience(self, exp: Experience) -> None:
        self.buffer.push(exp)
