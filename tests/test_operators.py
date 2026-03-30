"""Tests for restriction/prolongation operators."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from understand.operators import restrict, prolongate, coherence_score


class TestRestrictionProlongation:
    def test_restrict_shape(self):
        fine = np.random.rand(6, 8)
        coarse = restrict(fine)
        assert coarse.shape == (3, 4)

    def test_restrict_odd_shape(self):
        fine = np.random.rand(5, 7)
        coarse = restrict(fine)
        assert coarse.shape == (3, 4)

    def test_prolongate_shape(self):
        coarse = np.random.rand(3, 4)
        fine = prolongate(coarse, (6, 8))
        assert fine.shape == (6, 8)

    def test_restrict_prolongate_approximate_inverse(self):
        """P^T(P(coarse)) ≈ coarse (interior cells, boundary may diverge)"""
        coarse = np.random.rand(4, 4)
        fine = prolongate(coarse, (8, 8))
        reconstructed = restrict(fine)
        # Interior cells should be close; boundary cells may diverge due to padding
        interior = slice(1, -1), slice(1, -1)
        np.testing.assert_allclose(reconstructed[interior], coarse[interior], atol=0.2)

    def test_boundary_preservation(self):
        """Restriction should approximately preserve boundary-relevant structure."""
        fine = np.zeros((6, 6))
        # Place a sharp boundary
        fine[:, 3:] = 1.0
        coarse = restrict(fine)
        # Coarse should show the transition
        assert coarse[0, 0] < 0.5  # left side
        assert coarse[0, 2] > 0.5  # right side

    def test_coherence_score_perfect(self):
        fine = np.ones((4, 4)) * 3.0
        coarse = restrict(fine)
        score = coherence_score(fine, coarse)
        assert score > 0.9

    def test_coherence_score_bad(self):
        fine = np.random.rand(4, 4)
        coarse = np.random.rand(2, 2) * 100
        score = coherence_score(fine, coarse)
        assert score < 0.5

    def test_coherence_score_range(self):
        fine = np.random.rand(6, 6)
        coarse = restrict(fine)
        score = coherence_score(fine, coarse)
        assert 0.0 <= score <= 1.0
