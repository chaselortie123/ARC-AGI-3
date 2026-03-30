"""Environment adapter for ARC-AGI-3.

Provides a uniform interface whether using the official ARC-AGI Toolkit
(pip install arc-agi) or a local mock environment for testing.

The adapter decouples the agent from the toolkit's specific API so the
engine code only sees (grid, actions, game_state) tuples.
"""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class GameStatus(Enum):
    PLAYING = "playing"
    WIN = "win"
    GAME_OVER = "game_over"


@dataclass
class Action:
    """Uniform action representation.

    action_id: int 0-7 mapping to GameAction.ACTION0..ACTION7
    data: optional dict with x,y for complex actions
    """
    action_id: int
    data: Optional[dict] = None
    reasoning: Optional[str] = None


@dataclass
class Observation:
    """Uniform observation from environment."""
    grid: np.ndarray               # (64, 64) int array, colors 0-15
    status: GameStatus
    levels_completed: int
    win_levels: int
    available_actions: list[int]   # list of valid action_ids
    raw: Any = None                # original FrameDataRaw if available


class ArcEnvironment(ABC):
    """Abstract environment interface for ARC-AGI-3."""

    @abstractmethod
    def reset(self) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action) -> Observation:
        ...

    @abstractmethod
    def close(self):
        ...


class ToolkitArcEnvironment(ArcEnvironment):
    """Adapter for the official ARC-AGI Toolkit (pip install arc-agi).

    Uses arc_agi.Arcade and arcengine.GameAction/GameState.
    """

    def __init__(
        self,
        game_id: str = "ls20",
        seed: int = 0,
        render_mode: str | None = "terminal",
        api_key: str | None = None,
        **kwargs,
    ):
        self._game_id = game_id
        self._seed = seed
        self._render_mode = render_mode
        self._api_key = api_key
        self._extra_kwargs = kwargs
        self._arcade = None
        self._env = None

    def _ensure_arcade(self):
        if self._arcade is not None:
            return
        try:
            import arc_agi
        except ImportError:
            raise ImportError(
                "ARC-AGI Toolkit not installed. Install with: pip install arc-agi\n"
                "Requires Python >= 3.12.\n"
                "Or use MockArcEnvironment for local testing."
            )
        kwargs = {}
        if self._api_key:
            kwargs["arc_api_key"] = self._api_key
        self._arcade = arc_agi.Arcade(**kwargs)

    def reset(self) -> Observation:
        self._ensure_arcade()
        self._env = self._arcade.make(
            self._game_id,
            seed=self._seed,
            render_mode=self._render_mode,
            **self._extra_kwargs,
        )
        frame = self._env.reset()
        return self._wrap_frame(frame)

    def step(self, action: Action) -> Observation:
        from arcengine import GameAction
        ga = GameAction(action.action_id)
        data = action.data or {}
        reasoning = {}
        if action.reasoning:
            reasoning["thought"] = action.reasoning
        frame = self._env.step(ga, data=data if data else None, reasoning=reasoning or None)
        return self._wrap_frame(frame)

    def close(self):
        pass

    def _wrap_frame(self, frame) -> Observation:
        from arcengine import GameState
        if frame is None:
            return Observation(
                grid=np.zeros((64, 64), dtype=int),
                status=GameStatus.PLAYING,
                levels_completed=0,
                win_levels=0,
                available_actions=list(range(8)),
            )

        # Extract grid from frame data
        grid = self._extract_grid(frame)

        # Map game state
        if frame.state == GameState.WIN:
            status = GameStatus.WIN
        elif frame.state == GameState.GAME_OVER:
            status = GameStatus.GAME_OVER
        else:
            status = GameStatus.PLAYING

        # Available actions
        avail = []
        if hasattr(frame, "available_actions") and frame.available_actions:
            avail = [a.value for a in frame.available_actions]
        elif self._env and hasattr(self._env, "action_space"):
            avail = [a.value for a in self._env.action_space]
        else:
            avail = list(range(8))

        return Observation(
            grid=grid,
            status=status,
            levels_completed=getattr(frame, "levels_completed", 0),
            win_levels=getattr(frame, "win_levels", 0),
            available_actions=avail,
            raw=frame,
        )

    def _extract_grid(self, frame) -> np.ndarray:
        """Extract 64x64 grid from FrameDataRaw."""
        # Try common attribute names
        for attr in ("grid", "frame", "data", "pixels", "image"):
            val = getattr(frame, attr, None)
            if val is not None:
                return np.asarray(val, dtype=int)
        # If frame itself is array-like
        try:
            arr = np.asarray(frame, dtype=int)
            if arr.ndim == 2:
                return arr
        except (TypeError, ValueError):
            pass
        # Fallback: check if it has frame_data or similar nested structure
        for attr in ("frame_data", "observation"):
            val = getattr(frame, attr, None)
            if val is not None:
                return np.asarray(val, dtype=int)
        return np.zeros((64, 64), dtype=int)


class MockArcEnvironment(ArcEnvironment):
    """Mock environment for testing. Simulates a multi-level ARC-AGI-3-like game.

    Presents a 64x64 grid with patterns. The agent must discover the
    correct action sequence. Mimics the toolkit's action/observation interface.
    """

    def __init__(self, grid_size: int = 10, num_steps: int = 10, seed: int = 42):
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.rng = np.random.RandomState(seed)
        self._step = 0
        self._level = 0
        self._target_action = 0
        self._current: np.ndarray | None = None

    def reset(self) -> Observation:
        self._step = 0
        self._level = 0
        self._target_action = self.rng.randint(0, 5)
        self._current = self._generate_pattern()
        return Observation(
            grid=self._current.copy(),
            status=GameStatus.PLAYING,
            levels_completed=0,
            win_levels=3,
            available_actions=list(range(6)),
        )

    def step(self, action: Action) -> Observation:
        self._step += 1

        # Check if action matches the hidden target
        correct = action.action_id == self._target_action
        if correct:
            self._level += 1
            self._target_action = self.rng.randint(0, 5)

        done = self._step >= self.num_steps
        won = self._level >= 3

        if won:
            status = GameStatus.WIN
        elif done:
            status = GameStatus.GAME_OVER
        else:
            status = GameStatus.PLAYING

        # Evolve the grid
        self._current = self._evolve(correct)

        return Observation(
            grid=self._current.copy(),
            status=status,
            levels_completed=self._level,
            win_levels=3,
            available_actions=list(range(6)) if status == GameStatus.PLAYING else [],
        )

    def close(self):
        pass

    def _generate_pattern(self) -> np.ndarray:
        """Generate a grid with colored blocks."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for _ in range(3):
            color = self.rng.randint(1, 10)
            r1, c1 = self.rng.randint(0, self.grid_size - 2, size=2)
            r2 = min(r1 + self.rng.randint(1, 4), self.grid_size)
            c2 = min(c1 + self.rng.randint(1, 4), self.grid_size)
            grid[r1:r2, c1:c2] = color
        return grid

    def _evolve(self, correct: bool) -> np.ndarray:
        """Evolve grid: subtle change if correct, noise if wrong."""
        grid = self._current.copy()
        if correct:
            # Shift pattern slightly
            grid = np.roll(grid, 1, axis=0)
        else:
            # Add noise
            noise = self.rng.randint(0, 10, size=grid.shape)
            mask = self.rng.random(grid.shape) > 0.9
            grid = np.where(mask, noise, grid)
        return grid
