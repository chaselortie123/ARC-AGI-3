"""End-to-end agent test on mock environment."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import ArcMultigridAgent
from env_adapter import MockArcEnvironment


class TestAgentEndToEnd:
    def test_runs_one_game(self):
        """Agent runs end-to-end on mock environment without crashing."""
        env = MockArcEnvironment(grid_size=6, num_steps=5, seed=42)
        agent = ArcMultigridAgent(default_grid_shape=(6, 6))

        obs, info = env.reset()
        agent.reset()

        for step in range(5):
            diag = agent.observe(obs)
            action = agent.act()
            assert isinstance(action, np.ndarray)
            obs, reward, done, info = env.step(action)
            if done:
                agent.observe(obs, reward=reward, done=True)
                break

        summary = agent.summary()
        assert summary["total_steps"] > 0
        assert "final_kappa" in summary
        assert "rollback_count" in summary
        assert "compression_count" in summary

    def test_log_history(self):
        env = MockArcEnvironment(grid_size=4, num_steps=3, seed=0)
        agent = ArcMultigridAgent()

        obs, _ = env.reset()
        agent.reset()

        for _ in range(3):
            agent.observe(obs)
            action = agent.act()
            obs, _, done, _ = env.step(action)
            if done:
                break

        assert len(agent.log_history) >= 1
        entry = agent.log_history[0]
        assert "kappa" in entry
        assert "kappa_dot" in entry
        assert "agent_step" in entry

    def test_reset_clears_state(self):
        agent = ArcMultigridAgent()
        obs = np.zeros((4, 4), dtype=int)
        agent.observe(obs)
        agent.act()
        assert agent._step > 0

        agent.reset()
        assert agent._step == 0
        assert agent.engine is None
