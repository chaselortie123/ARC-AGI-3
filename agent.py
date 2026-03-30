"""ARC-AGI-3 Multigrid Agent: the main loop.

Architecture: Perceive → Understand → Act
Engine: Transport → Anneal → Compress → Repeat
"""

from __future__ import annotations
import logging
import numpy as np
from substrate.types import Frame, Substrate
from perceive.parser import parse_frame
from understand.multigrid import MultigridEngine
from act.strategy import ActionSelector, Candidate
from act.candidates import generate_candidates, score_candidates

logger = logging.getLogger("arc_agent")


class ArcMultigridAgent:
    """Multigrid agent for ARC-AGI-3.

    Maintains two-level multigrid state, runs partial V-cycles per tick,
    selects actions via mood-driven strategy.
    """

    def __init__(self, default_grid_shape: tuple[int, int] = (30, 30)):
        self.engine: MultigridEngine | None = None
        self.selector = ActionSelector()
        self.default_shape = default_grid_shape
        self._substrate: Substrate | None = None
        self._step = 0
        self._total_reward = 0.0
        self._log_history: list[dict] = []

    def reset(self):
        """Reset agent state for a new game."""
        self.engine = None
        self._substrate = None
        self._step = 0
        self._total_reward = 0.0
        self._log_history = []

    def observe(self, observation: np.ndarray | list, reward: float = 0.0, done: bool = False, info: dict | None = None) -> dict:
        """Process one observation from the environment.

        Returns diagnostics dict.
        """
        grid = np.asarray(observation, dtype=int)
        frame = Frame(grid=grid, step=self._step, reward=reward, done=done, info=info or {})

        # PERCEIVE
        self._substrate = parse_frame(frame)

        # Initialize engine on first observation
        if self.engine is None:
            self.engine = MultigridEngine(grid.shape)

        # UNDERSTAND: ingest + V-cycle
        self.engine.ingest(self._substrate)
        diag = self.engine.v_cycle(max_iters=3)

        self._step += 1
        self._total_reward += reward

        log_entry = {
            **diag,
            "agent_step": self._step,
            "reward": reward,
            "total_reward": self._total_reward,
            "done": done,
            "grid_shape": grid.shape,
        }
        self._log_history.append(log_entry)
        logger.info(
            "step=%d kappa=%.3f kappa_dot=%.4f T=%.3f committed=%s rollbacks=%d",
            self._step, diag["kappa"], diag["kappa_dot"],
            diag["temperature"], diag["committed"], diag["rollback_count"]
        )
        return log_entry

    def act(self) -> np.ndarray:
        """Select and return an action (grid) based on current state."""
        if self._substrate is None or self.engine is None:
            raise RuntimeError("Must call observe() before act()")

        grid = self._substrate.grid
        mood = self.engine.get_mood()

        # Generate candidates
        candidates = generate_candidates(self._substrate, grid)

        # Score candidates against current multigrid understanding
        candidates = score_candidates(
            candidates,
            self.engine.state.fine.data,
            self.engine.state.coarse.data,
        )

        # Select action via mood-driven strategy
        chosen = self.selector.select_action(candidates, mood)

        logger.info(
            "action selected: score=%.3f commitment=%.3f strategy_mood=(T=%.2f, kd=%.4f)",
            chosen.score, chosen.commitment, mood.temperature, mood.kappa_dot,
        )
        return np.asarray(chosen.action, dtype=int)

    @property
    def log_history(self) -> list[dict]:
        return self._log_history

    def summary(self) -> dict:
        """Summary stats for the game."""
        return {
            "total_steps": self._step,
            "total_reward": self._total_reward,
            "final_kappa": self._log_history[-1]["kappa"] if self._log_history else 0.0,
            "rollback_count": self.engine.state.rollback_count if self.engine else 0,
            "compression_count": self.engine._compression_count if self.engine else 0,
        }
