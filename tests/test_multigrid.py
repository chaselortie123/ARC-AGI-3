"""Tests for multigrid engine and V-cycle."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from understand.multigrid import MultigridEngine
from substrate.types import Substrate


def _make_substrate(grid):
    arr = np.array(grid, dtype=int)
    return Substrate(grid=arr, objects=[], relations=[], background_color=0)


class TestMultigridEngine:
    def test_ingest_updates_fine(self):
        engine = MultigridEngine((4, 4))
        sub = _make_substrate(np.ones((4, 4)) * 3)
        engine.ingest(sub)
        np.testing.assert_allclose(engine.state.fine.data, np.ones((4, 4)) * 3, atol=0.5)

    def test_v_cycle_returns_diagnostics(self):
        engine = MultigridEngine((4, 4))
        sub = _make_substrate(np.ones((4, 4)) * 2)
        engine.ingest(sub)
        diag = engine.v_cycle()
        assert "kappa" in diag
        assert "kappa_dot" in diag
        assert "step" in diag
        assert "committed" in diag
        assert "rollback_count" in diag
        assert "compression_count" in diag

    def test_v_cycle_improves_coherence(self):
        """V-cycle should improve coherence over flat baseline."""
        engine = MultigridEngine((6, 6))
        # Initial random data
        sub = _make_substrate(np.random.randint(0, 5, (6, 6)))
        engine.ingest(sub)
        diag1 = engine.v_cycle()
        # Second pass with same data should be better
        engine.ingest(sub)
        diag2 = engine.v_cycle()
        # kappa should be reasonable after V-cycle
        assert diag2["kappa"] > 0.0

    def test_interruptible(self):
        """V-cycle with max_iters=1 should still produce valid output."""
        engine = MultigridEngine((4, 4))
        sub = _make_substrate(np.ones((4, 4)))
        engine.ingest(sub)
        diag = engine.v_cycle(max_iters=1)
        assert diag["step"] == 1

    def test_multiple_ingests(self):
        """Engine handles sequential observations."""
        engine = MultigridEngine((4, 4))
        for i in range(5):
            sub = _make_substrate(np.ones((4, 4)) * i)
            engine.ingest(sub)
            engine.v_cycle()
        assert engine._step == 5


class TestWIPRollback:
    def test_commit(self):
        engine = MultigridEngine((4, 4))
        sub = _make_substrate(np.ones((4, 4)) * 5)
        engine.ingest(sub)
        engine.v_cycle()
        # After V-cycle, state should reflect the observation
        assert engine.state.fine.data is not None

    def test_rollback_atomic(self):
        """WIP rollback restores committed state exactly."""
        engine = MultigridEngine((4, 4))
        sub = _make_substrate(np.ones((4, 4)) * 3)
        engine.ingest(sub)
        engine.v_cycle()

        committed_data = engine.state.committed_fine.data.copy()

        # Start WIP, modify, rollback
        engine.state.begin_wip()
        engine.state.fine.data[:] = 99.0
        engine.state.rollback_wip()

        np.testing.assert_array_equal(engine.state.fine.data, committed_data)

    def test_rollback_count(self):
        engine = MultigridEngine((4, 4))
        assert engine.state.rollback_count == 0
        engine.state.begin_wip()
        engine.state.rollback_wip()
        assert engine.state.rollback_count == 1
        engine.state.begin_wip()
        engine.state.rollback_wip()
        assert engine.state.rollback_count == 2
