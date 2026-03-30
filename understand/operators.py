"""Cross-scale adjoint pair: restriction (P^T) and prolongation (P).

Restriction  = Schur compression: coarsen fine-level data
Prolongation = harmonic completion: fill fine-level from coarse skeleton

Invariant: restrict(prolongate(coarse)) ≈ coarse  (approximate inverse pair)
"""

from __future__ import annotations
import numpy as np


def restrict(fine: np.ndarray) -> np.ndarray:
    """P^T: Schur compression. Coarsen fine grid by 2x block averaging.

    For non-square or odd-sized grids, pads to even dimensions first.
    """
    H, W = fine.shape
    # Pad to even
    pH = H + (H % 2)
    pW = W + (W % 2)
    padded = np.zeros((pH, pW), dtype=fine.dtype)
    padded[:H, :W] = fine

    # 2x2 block average
    coarse = (
        padded[0::2, 0::2] +
        padded[1::2, 0::2] +
        padded[0::2, 1::2] +
        padded[1::2, 1::2]
    ) / 4.0
    return coarse


def prolongate(coarse: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """P: Harmonic completion. Interpolate coarse grid to fine resolution.

    Uses bilinear interpolation as the approximate harmonic extension.
    """
    cH, cW = coarse.shape
    tH, tW = target_shape

    # Row indices in coarse space
    row_idx = np.linspace(0, cH - 1, tH)
    col_idx = np.linspace(0, cW - 1, tW)

    # Bilinear interpolation
    r0 = np.floor(row_idx).astype(int)
    r1 = np.minimum(r0 + 1, cH - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.minimum(c0 + 1, cW - 1)

    dr = row_idx - r0
    dc = col_idx - c0

    # Outer product interpolation
    fine = (
        np.outer(1 - dr, 1 - dc) * coarse[np.ix_(r0, c0)] +
        np.outer(1 - dr, dc) * coarse[np.ix_(r0, c1)] +
        np.outer(dr, 1 - dc) * coarse[np.ix_(r1, c0)] +
        np.outer(dr, dc) * coarse[np.ix_(r1, c1)]
    )
    return fine


def coherence_score(fine: np.ndarray, coarse: np.ndarray) -> float:
    """Proxy for kappa: how well does coarse represent fine?

    Returns value in [0, 1]. 1 = perfect representation.
    """
    reconstructed = prolongate(coarse, fine.shape)
    if np.max(np.abs(fine)) < 1e-12:
        return 1.0 if np.max(np.abs(reconstructed)) < 1e-12 else 0.0
    residual = np.linalg.norm(fine - reconstructed) / (np.linalg.norm(fine) + 1e-12)
    return float(max(0.0, 1.0 - residual))
