"""Environment adapter for ARC-AGI-3.

Provides a uniform interface whether using the official ARC-AGI Toolkit
or a local mock environment for testing.

The official toolkit integration will be updated once the arcagi package
API is confirmed. The adapter pattern keeps the agent decoupled.
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class ArcEnvironment(ABC):
    """Abstract environment interface for ARC-AGI-3."""

    @abstractmethod
    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset environment, return (initial_observation, info)."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Take action, return (observation, reward, done, info)."""
        ...

    @abstractmethod
    def close(self):
        ...


class MockArcEnvironment(ArcEnvironment):
    """Mock environment for testing. Simulates a simple pattern-matching game.

    The environment presents a grid with a pattern. The agent must reproduce
    a transformed version. Reward is based on cell-level accuracy.
    """

    def __init__(self, grid_size: int = 10, num_steps: int = 10, seed: int = 42):
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.rng = np.random.RandomState(seed)
        self._step = 0
        self._target: np.ndarray | None = None
        self._current: np.ndarray | None = None

    def reset(self) -> tuple[np.ndarray, dict]:
        self._step = 0
        # Generate a simple pattern
        self._current = self._generate_pattern()
        self._target = self._transform(self._current)
        return self._current.copy(), {"game": "mock", "step": 0}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self._step += 1
        action = np.asarray(action, dtype=int)

        # Reward = fraction of cells matching target
        if action.shape == self._target.shape:
            correct = np.sum(action == self._target)
            total = self._target.size
            reward = correct / total
        else:
            reward = 0.0

        done = self._step >= self.num_steps or reward > 0.99

        # Next observation: blend current with some new info
        if not done:
            noise = self.rng.randint(0, 10, size=self._current.shape)
            mask = self.rng.random(self._current.shape) > 0.9
            self._current = np.where(mask, noise, self._current)

        info = {"step": self._step, "accuracy": reward}
        return self._current.copy(), reward, done, info

    def close(self):
        pass

    def _generate_pattern(self) -> np.ndarray:
        """Simple colored blocks pattern."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Place a few colored rectangles
        for _ in range(3):
            color = self.rng.randint(1, 10)
            r1, c1 = self.rng.randint(0, self.grid_size - 2, size=2)
            r2 = min(r1 + self.rng.randint(1, 4), self.grid_size)
            c2 = min(c1 + self.rng.randint(1, 4), self.grid_size)
            grid[r1:r2, c1:c2] = color
        return grid

    def _transform(self, grid: np.ndarray) -> np.ndarray:
        """Simple transformation: horizontal flip."""
        return np.fliplr(grid)


class ToolkitArcEnvironment(ArcEnvironment):
    """Adapter for the official ARC-AGI Toolkit.

    Wraps the toolkit's environment API. Update the import and method calls
    once the toolkit package is available.
    """

    def __init__(self, **kwargs):
        self._env = None
        self._kwargs = kwargs

    def reset(self) -> tuple[np.ndarray, dict]:
        try:
            import arcagi  # type: ignore
            self._env = arcagi.make_env(**self._kwargs)
            obs = self._env.reset()
            return np.asarray(obs, dtype=int), {}
        except ImportError:
            raise ImportError(
                "ARC-AGI Toolkit not installed. Install with: pip install arcagi\n"
                "Or use MockArcEnvironment for local testing."
            )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        result = self._env.step(action.tolist())
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        return np.asarray(obs, dtype=int), float(reward), done, info

    def close(self):
        if self._env is not None:
            self._env.close()
