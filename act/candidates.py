"""Candidate generation: separate from scoring.

Generates candidate actions from the current substrate and grid state.
"""

from __future__ import annotations
import numpy as np
from substrate.types import Substrate
from act.strategy import Candidate
from understand.operators import coherence_score, restrict


def generate_candidates(substrate: Substrate, grid: np.ndarray) -> list[Candidate]:
    """Generate candidate grid actions from current substrate.

    For ARC-AGI-3, actions are grid submissions. We generate candidates by:
    1. Identity (submit current grid as-is)
    2. Color fills (fill each object's bbox with its color)
    3. Single-cell modifications based on neighbor majority
    """
    candidates = []
    H, W = grid.shape

    # Candidate 0: identity (do nothing / submit as-is)
    candidates.append(Candidate(action=grid.copy(), score=0.0, commitment=0.5))

    # Candidate 1: majority-neighbor smoothing
    smoothed = _neighbor_majority(grid)
    if not np.array_equal(smoothed, grid):
        candidates.append(Candidate(action=smoothed, score=0.0, commitment=0.3))

    # Candidate 2-N: per-object fills
    for obj in substrate.objects[:5]:  # cap to avoid explosion
        filled = grid.copy()
        min_r, min_c, max_r, max_c = obj.bbox
        filled[min_r:max_r + 1, min_c:max_c + 1] = obj.color
        if not np.array_equal(filled, grid):
            candidates.append(Candidate(action=filled, score=0.0, commitment=0.2))

    return candidates


def score_candidates(
    candidates: list[Candidate],
    current_fine: np.ndarray,
    current_coarse: np.ndarray,
) -> list[Candidate]:
    """Score candidates by predicted coherence gain."""
    for cand in candidates:
        action_grid = cand.action.astype(float)
        restricted = restrict(action_grid)
        # Score = coherence of proposed action with current coarse understanding
        if restricted.shape == current_coarse.shape:
            cand.score = coherence_score(action_grid, current_coarse)
        else:
            cand.score = coherence_score(action_grid, restricted)
        # Commitment = how different from current state
        diff = np.linalg.norm(action_grid - current_fine) / (np.linalg.norm(current_fine) + 1e-12)
        cand.commitment = max(0.0, 1.0 - diff)
    return candidates


def _neighbor_majority(grid: np.ndarray) -> np.ndarray:
    """Replace each cell with majority color of its 4-neighbors + self."""
    H, W = grid.shape
    result = grid.copy()
    for r in range(H):
        for c in range(W):
            neighbors = [grid[r, c]]
            if r > 0: neighbors.append(grid[r - 1, c])
            if r < H - 1: neighbors.append(grid[r + 1, c])
            if c > 0: neighbors.append(grid[r, c - 1])
            if c < W - 1: neighbors.append(grid[r, c + 1])
            values, counts = np.unique(neighbors, return_counts=True)
            result[r, c] = values[np.argmax(counts)]
    return result
