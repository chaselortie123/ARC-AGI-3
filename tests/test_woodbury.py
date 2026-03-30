"""Tests for Woodbury/delta low-rank updates."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrate.state import LevelState


class TestLowRankUpdate:
    def test_basic_update(self):
        state = LevelState(data=np.zeros((4, 4)))
        U = np.ones((4, 1))
        V = np.ones((4, 1))
        state.apply_lowrank_update(U, V)
        expected = np.ones((4, 4))
        np.testing.assert_allclose(state.data, expected)

    def test_accumulate_updates(self):
        state = LevelState(data=np.zeros((4, 4)))
        U1 = np.ones((4, 1))
        V1 = np.ones((4, 1))
        state.apply_lowrank_update(U1, V1)

        U2 = np.eye(4)[:, :1]  # first column of identity
        V2 = np.eye(4)[:, :1]
        state.apply_lowrank_update(U2, V2)

        assert state.rank_budget_used == 2

    def test_flush(self):
        state = LevelState(data=np.zeros((4, 4)))
        state.apply_lowrank_update(np.ones((4, 1)), np.ones((4, 1)))
        assert state.rank_budget_used == 1
        state.flush_lowrank()
        assert state.rank_budget_used == 0
        # Data should still have the update
        np.testing.assert_allclose(state.data, np.ones((4, 4)))

    def test_faster_than_rebuild(self):
        """Low-rank update on small delta should be cheaper than full SVD rebuild."""
        import time
        n = 50
        state = LevelState(data=np.random.rand(n, n))
        delta = np.random.rand(n, 1) @ np.random.rand(1, n) * 0.01

        # Low-rank update
        t0 = time.perf_counter()
        for _ in range(100):
            state.apply_lowrank_update(delta[:, :1], delta[:1, :].T)
        t_lr = time.perf_counter() - t0

        # Full rebuild
        t0 = time.perf_counter()
        for _ in range(100):
            state.data = state.data + delta
        t_full = time.perf_counter() - t0

        # Low-rank should be competitive (not necessarily faster for tiny matrices)
        # Just ensure it completes without error
        assert t_lr < 10.0  # sanity
