"""Test: one frame → factor graph → L → κ → spectrum.

The single concrete test that turns the Gossamer architecture into evidence.
"""

import numpy as np
import pytest
from substrate.types import Frame
from perceive.parser import parse_frame
from substrate.factor_graph import (
    FactorGraph, build_factor_graph, build_operator_L,
    commitment, spectrum,
)


# A small ARC-like frame: 5x5 grid with two colored objects on black background
SAMPLE_GRID = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 2, 2],
    [0, 0, 0, 2, 2],
], dtype=int)


@pytest.fixture
def substrate():
    frame = Frame(grid=SAMPLE_GRID, step=0)
    return parse_frame(frame)


@pytest.fixture
def factor_graph(substrate):
    return build_factor_graph(substrate)


class TestFactorGraphConstruction:
    def test_has_variables(self, factor_graph):
        fg = factor_graph
        # 25 cell variables + 2 object variables = 27
        assert fg.n_vars == 27

    def test_has_factors(self, factor_graph):
        fg = factor_graph
        # 25 observed + 8 color_match (4 cells per object * 2 objects) + relation factors
        assert fg.n_factors > 25  # at least the observed factors

    def test_variable_types(self, factor_graph):
        fg = factor_graph
        cell_vars = [v for v in fg.variables.values() if v.kind == "cell"]
        obj_vars = [v for v in fg.variables.values() if v.kind == "object"]
        assert len(cell_vars) == 25
        assert len(obj_vars) == 2

    def test_cell_states_are_one_hot(self, factor_graph):
        fg = factor_graph
        for v in fg.variables.values():
            if v.kind == "cell":
                assert abs(v.state.sum() - 1.0) < 1e-10
                assert v.state.shape == (10,)


class TestOperatorL:
    def test_L_shape(self, factor_graph):
        L = build_operator_L(factor_graph)
        dim = factor_graph.var_dim
        assert L.shape == (dim, dim)

    def test_L_symmetric(self, factor_graph):
        L = build_operator_L(factor_graph)
        assert np.allclose(L, L.T, atol=1e-10)

    def test_L_positive_semidefinite(self, factor_graph):
        L = build_operator_L(factor_graph)
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-10), f"Negative eigenvalue: {eigs.min()}"

    def test_L_not_zero(self, factor_graph):
        L = build_operator_L(factor_graph)
        assert np.linalg.norm(L) > 0


class TestCommitment:
    def test_commitment_in_range(self, factor_graph):
        kappa = commitment(factor_graph)
        assert 0.0 <= kappa <= 1.0

    def test_commitment_high_for_consistent_frame(self, factor_graph):
        """A fully observed, self-consistent frame should have high κ."""
        kappa = commitment(factor_graph)
        # Objects match their cells, so color_match residuals should be 0
        # same_color residuals are 0 (no same-color objects), adjacency is low
        assert kappa > 0.5, f"κ = {kappa}, expected > 0.5 for consistent frame"

    def test_commitment_drops_with_inconsistency(self, factor_graph):
        """Corrupting a variable's state should lower κ."""
        fg = factor_graph
        kappa_before = commitment(fg)

        # Corrupt an object variable's color to mismatch its cells
        for v in fg.variables.values():
            if v.kind == "object":
                v.state = np.zeros(10)
                v.state[9] = 1.0  # set to color 9, which no cell has
                break

        kappa_after = commitment(fg)
        assert kappa_after < kappa_before, \
            f"κ should drop after corruption: {kappa_before} -> {kappa_after}"


class TestSpectrum:
    def test_spectrum_sorted(self, factor_graph):
        L = build_operator_L(factor_graph)
        eigs = spectrum(L)
        assert np.all(np.diff(eigs) >= -1e-10)

    def test_spectrum_full_rank_with_observations(self, factor_graph):
        """With observed factors on every cell, L should be full rank."""
        L = build_operator_L(factor_graph)
        eigs = spectrum(L)
        # Every cell is pinned by an observed factor, so no null space
        assert np.all(eigs > 1e-8), "All modes should be constrained by observations"

    def test_spectrum_has_nonzero_modes(self, factor_graph):
        """L should also have positive eigenvalues (constrained directions)."""
        L = build_operator_L(factor_graph)
        eigs = spectrum(L)
        n_pos = np.sum(eigs > 1e-8)
        assert n_pos > 0, "Expected some positive eigenvalues"


class TestEndToEnd:
    def test_one_frame_one_graph_one_number(self):
        """The canonical test: frame → graph → L → κ → spectrum."""
        # 1. Parse frame
        frame = Frame(grid=SAMPLE_GRID, step=0)
        substrate = parse_frame(frame)

        # 2. Build factor graph
        fg = build_factor_graph(substrate)

        # 3. Build L = H^T Λ H
        L = build_operator_L(fg)

        # 4. Compute κ
        kappa = commitment(fg)

        # 5. Compute spectrum
        eigs = spectrum(L)

        # Print diagnostics
        print(f"\n{'='*60}")
        print(f"ONE FRAME, ONE GRAPH, ONE NUMBER")
        print(f"{'='*60}")
        print(f"Grid shape:       {substrate.grid.shape}")
        print(f"Objects:          {len(substrate.objects)}")
        print(f"Relations:        {len(substrate.relations)}")
        print(f"Variables:        {fg.n_vars}")
        print(f"Factors:          {fg.n_factors}")
        print(f"L shape:          {L.shape}")
        print(f"L rank:           {np.linalg.matrix_rank(L)}")
        print(f"κ (commitment):   {kappa:.6f}")
        print(f"Spectrum min:     {eigs[0]:.6f}")
        print(f"Spectrum max:     {eigs[-1]:.6f}")
        print(f"Zero modes:       {np.sum(np.abs(eigs) < 1e-8)}")
        print(f"Nonzero modes:    {np.sum(eigs > 1e-8)}")
        print(f"Spectral gap:     {eigs[eigs > 1e-8][0]:.6f}" if np.any(eigs > 1e-8) else "N/A")
        print(f"Top 10 eigenvals: {eigs[-10:]}")
        print(f"{'='*60}")

        # Assertions
        assert L.shape[0] == L.shape[1]
        assert np.allclose(L, L.T)
        assert 0.0 <= kappa <= 1.0
        assert len(eigs) == L.shape[0]
