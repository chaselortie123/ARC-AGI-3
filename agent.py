"""ARC-AGI-3 Multigrid Agent: the main loop.

Architecture: Perceive → Understand → Act
Engine: Transport → Anneal → Compress → Repeat

ARC-AGI-3 actions are discrete (ACTION0-ACTION7), not grid submissions.
The agent observes 64x64 grids and selects from available actions each turn.
"""

from __future__ import annotations
import logging
import numpy as np
from substrate.types import Frame, Substrate
from perceive.parser import parse_frame
from understand.multigrid import MultigridEngine
from understand.operators import coherence_score, restrict
from act.strategy import ActionSelector, Candidate
from env_adapter import Observation, Action, GameStatus

logger = logging.getLogger("arc_agent")


class ArcMultigridAgent:
    """Multigrid agent for ARC-AGI-3.

    Maintains two-level multigrid state, runs partial V-cycles per tick,
    selects actions via mood-driven strategy from available discrete actions.
    """

    def __init__(self, default_grid_shape: tuple[int, int] = (64, 64)):
        self.engine: MultigridEngine | None = None
        self.selector = ActionSelector()
        self.default_shape = default_grid_shape
        self._substrate: Substrate | None = None
        self._last_obs: Observation | None = None
        self._step = 0
        self._levels_completed = 0
        self._log_history: list[dict] = []
        # Track action→outcome for learning within a game
        self._action_history: list[tuple[int, float]] = []  # (action_id, kappa_delta)
        self._prev_grid: np.ndarray | None = None

    def reset(self):
        """Reset agent state for a new game."""
        self.engine = None
        self._substrate = None
        self._last_obs = None
        self._step = 0
        self._levels_completed = 0
        self._log_history = []
        self._action_history = []
        self._prev_grid = None

    def observe(self, obs: Observation) -> dict:
        """Process one observation from the environment.

        Returns diagnostics dict.
        """
        grid = np.asarray(obs.grid, dtype=int)
        frame = Frame(
            grid=grid, step=self._step,
            reward=float(obs.levels_completed),
            done=obs.status != GameStatus.PLAYING,
            info={"status": obs.status.value, "levels": obs.levels_completed},
        )

        # PERCEIVE
        self._substrate = parse_frame(frame)
        self._last_obs = obs

        # Initialize engine on first observation
        if self.engine is None:
            self.engine = MultigridEngine(grid.shape)

        # Track grid change from last action
        kappa_before = self.engine._prev_kappa
        grid_changed = self._prev_grid is not None and not np.array_equal(grid, self._prev_grid)

        # UNDERSTAND: ingest + V-cycle
        self.engine.ingest(self._substrate)
        diag = self.engine.v_cycle(max_iters=3)

        # Record action outcome
        if self._action_history and grid_changed:
            last_action = self._action_history[-1][0]
            self._action_history[-1] = (last_action, diag["kappa_dot"])

        self._prev_grid = grid.copy()
        self._step += 1
        self._levels_completed = obs.levels_completed

        log_entry = {
            **diag,
            "agent_step": self._step,
            "levels_completed": obs.levels_completed,
            "win_levels": obs.win_levels,
            "status": obs.status.value,
            "grid_shape": grid.shape,
            "grid_changed": grid_changed,
        }
        self._log_history.append(log_entry)
        logger.info(
            "step=%d kappa=%.3f kappa_dot=%.4f T=%.3f committed=%s rollbacks=%d levels=%d/%d",
            self._step, diag["kappa"], diag["kappa_dot"],
            diag["temperature"], diag["committed"], diag["rollback_count"],
            obs.levels_completed, obs.win_levels,
        )
        return log_entry

    def act(self) -> Action:
        """Select and return an Action based on current state."""
        if self._last_obs is None or self.engine is None:
            raise RuntimeError("Must call observe() before act()")

        available = self._last_obs.available_actions
        if not available:
            return Action(action_id=0)

        mood = self.engine.get_mood()
        grid = self._substrate.grid.astype(float)

        # Generate candidates: one per available action
        candidates = []
        for aid in available:
            score = self._score_action(aid, grid)
            commitment = self._commitment_for_action(aid)
            candidates.append(Candidate(
                action=aid,
                score=score,
                commitment=commitment,
            ))

        # Select via mood-driven strategy
        chosen = self.selector.select_action(candidates, mood)

        # Build Action with optional data for complex actions
        action = Action(action_id=chosen.action)
        # For complex actions (typically ACTION6), include center coordinates as default
        if chosen.action == 6:
            H, W = self._substrate.grid.shape
            # Target the most interesting region: largest non-background object center
            cx, cy = W // 2, H // 2
            if self._substrate.objects:
                biggest = max(self._substrate.objects, key=lambda o: o.size)
                cy, cx = int(biggest.center[0]), int(biggest.center[1])
            action.data = {"x": cx, "y": cy}

        self._action_history.append((chosen.action, 0.0))

        logger.info(
            "action=%d score=%.3f commitment=%.3f mood=(T=%.2f, kd=%.4f)",
            chosen.action, chosen.score, chosen.commitment,
            mood.temperature, mood.kappa_dot,
        )
        return action

    def _score_action(self, action_id: int, grid: np.ndarray) -> float:
        """Score an action based on past outcomes and current state.

        Uses action history to build an empirical score, combined with
        the multigrid coherence to predict which actions are constructive.
        """
        # Base score from action history (actions that improved kappa get bonus)
        history_score = 0.0
        count = 0
        for aid, kd in self._action_history:
            if aid == action_id:
                history_score += kd
                count += 1
        if count > 0:
            history_score /= count

        # Novelty bonus: prefer less-tried actions
        times_used = sum(1 for aid, _ in self._action_history if aid == action_id)
        novelty = 1.0 / (1.0 + times_used)

        # Combine
        T = self.engine.get_mood().temperature
        return history_score + 0.3 * novelty * T

    def _commitment_for_action(self, action_id: int) -> float:
        """How much coherence does this action typically preserve?"""
        # Actions with positive historical kappa_dot are high-commitment
        positive = sum(1 for aid, kd in self._action_history if aid == action_id and kd > 0)
        total = max(1, sum(1 for aid, _ in self._action_history if aid == action_id))
        return positive / total

    @property
    def log_history(self) -> list[dict]:
        return self._log_history

    def summary(self) -> dict:
        """Summary stats for the game."""
        return {
            "total_steps": self._step,
            "levels_completed": self._levels_completed,
            "final_kappa": self._log_history[-1]["kappa"] if self._log_history else 0.0,
            "rollback_count": self.engine.state.rollback_count if self.engine else 0,
            "compression_count": self.engine._compression_count if self.engine else 0,
            "actions_tried": len(set(aid for aid, _ in self._action_history)),
        }
