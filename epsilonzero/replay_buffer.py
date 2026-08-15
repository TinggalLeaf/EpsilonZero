"""Replay buffer for training samples (states, MCTS policy, value target)."""
from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np

Sample = Tuple[np.ndarray, np.ndarray, float]  # (4*n*n planes, pi, z)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data: deque[Sample] = deque(maxlen=capacity)

    def add(self, samples: List[Sample]):
        self.data.extend(samples)

    def sample(self, batch_size: int):
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        states = np.stack([b[0] for b in batch])
        pis = np.stack([b[1] for b in batch])
        zs = np.array([b[2] for b in batch], dtype=np.float32)
        return states, pis, zs

    def __len__(self) -> int:
        return len(self.data)
