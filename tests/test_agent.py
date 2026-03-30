"""End-to-end agent test on mock environment."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ArcMultigridAgent
from env_adapter import MockArcEnvironment, GameStatus, Action


class TestAgentEndToEnd:
    def test_runs_one_game(self):
        """Agent runs end-to-end on mock environment without crashing."""
        env = MockArcEnvironment(grid_size=6, num_steps=5, seed=42)
        agent = ArcMultigridAgent(default_grid_shape=(6, 6))

        obs = env.reset()
        agent.reset()

        for step in range(5):
            diag = agent.observe(obs)
            if obs.status != GameStatus.PLAYING:
                break
            action = agent.act()
            assert isinstance(action, Action)
            assert isinstance(action.action_id, int)
            obs = env.step(action)

        summary = agent.summary()
        assert summary["total_steps"] > 0
        assert "final_kappa" in summary
        assert "rollback_count" in summary
        assert "compression_count" in summary
        assert "levels_completed" in summary

    def test_log_history(self):
        env = MockArcEnvironment(grid_size=4, num_steps=3, seed=0)
        agent = ArcMultigridAgent()

        obs = env.reset()
        agent.reset()

        for _ in range(3):
            agent.observe(obs)
            if obs.status != GameStatus.PLAYING:
                break
            action = agent.act()
            obs = env.step(action)

        assert len(agent.log_history) >= 1
        entry = agent.log_history[0]
        assert "kappa" in entry
        assert "kappa_dot" in entry
        assert "agent_step" in entry
        assert "levels_completed" in entry

    def test_reset_clears_state(self):
        agent = ArcMultigridAgent()
        from env_adapter import Observation
        obs = Observation(
            grid=np.zeros((4, 4), dtype=int),
            status=GameStatus.PLAYING,
            levels_completed=0,
            win_levels=3,
            available_actions=[0, 1, 2],
        )
        agent.observe(obs)
        agent.act()
        assert agent._step > 0

        agent.reset()
        assert agent._step == 0
        assert agent.engine is None

    def test_handles_no_actions(self):
        """Agent handles empty action space gracefully."""
        agent = ArcMultigridAgent()
        from env_adapter import Observation
        obs = Observation(
            grid=np.zeros((4, 4), dtype=int),
            status=GameStatus.GAME_OVER,
            levels_completed=0,
            win_levels=3,
            available_actions=[],
        )
        agent.observe(obs)
        action = agent.act()
        assert action.action_id == 0

    def test_action_history_tracks(self):
        """Agent builds action history over multiple steps."""
        env = MockArcEnvironment(grid_size=6, num_steps=5, seed=42)
        agent = ArcMultigridAgent()

        obs = env.reset()
        agent.reset()

        for _ in range(3):
            agent.observe(obs)
            if obs.status != GameStatus.PLAYING:
                break
            action = agent.act()
            obs = env.step(action)

        assert len(agent._action_history) > 0
