"""Multigrid engine: Transport → Anneal → Compress → Repeat.

Invariants:
- One substrate, one transport law
- Within-scale adaptation is low-rank (Woodbury)
- Search regime is continuous via mood = (T, kappa_dot)
- Obstruction triggers restructure, not shutdown
"""

from __future__ import annotations
import numpy as np
from substrate.state import MultigridState, Mood
from substrate.types import Substrate
from understand.operators import restrict, prolongate, coherence_score


class MultigridEngine:
    """Two-level multigrid engine with interruptible V-cycle."""

    def __init__(self, fine_shape: tuple[int, int], coarse_shape: tuple[int, int] | None = None):
        if coarse_shape is None:
            coarse_shape = ((fine_shape[0] + 1) // 2, (fine_shape[1] + 1) // 2)
        self.state = MultigridState(fine_shape, coarse_shape)
        self._prev_kappa = 0.0
        self._step = 0
        self._compression_count = 0

    def ingest(self, substrate: Substrate):
        """Update fine level from new observation."""
        fine_data = substrate.grid.astype(float)
        current_fine = self.state.fine.data

        # Reshape if grid dimensions changed
        if fine_data.shape != current_fine.shape:
            self._reinit(fine_data.shape)

        # Compute delta for low-rank update
        delta = fine_data - current_fine
        if np.linalg.norm(delta) > 1e-10:
            # Low-rank approximation of the delta via rank-1 SVD
            U_d, s_d, Vt_d = np.linalg.svd(delta, full_matrices=False)
            # Keep top-k singular values (k=min(2, rank))
            k = min(2, len(s_d))
            U_update = U_d[:, :k] * np.sqrt(s_d[:k])
            V_update = Vt_d[:k, :].T * np.sqrt(s_d[:k])
            self.state.fine.apply_lowrank_update(U_update, V_update)

            # Flush if rank budget exceeded
            if self.state.fine.rank_budget_used > 10:
                self.state.fine.flush_lowrank()

    def v_cycle(self, max_iters: int = 3) -> dict:
        """Run one interruptible V-cycle: Transport → Anneal → Compress.

        Returns diagnostics dict.
        """
        self.state.begin_wip()
        fine = self.state.fine
        coarse = self.state.coarse

        kappa_before = coherence_score(fine.data, coarse.data)

        for i in range(max_iters):
            # 1. RESTRICT (Transport down): compress fine → coarse
            new_coarse = restrict(fine.data)
            if new_coarse.shape != coarse.data.shape:
                # Reshape coarse to match
                coarse.data = new_coarse
            else:
                delta = new_coarse - coarse.data
                norm = np.linalg.norm(delta)
                if norm > 1e-10 and norm < 100:
                    # Low-rank update on coarse
                    U_d, s_d, Vt_d = np.linalg.svd(delta, full_matrices=False)
                    k = min(1, len(s_d))
                    coarse.apply_lowrank_update(
                        U_d[:, :k] * np.sqrt(s_d[:k]),
                        Vt_d[:k, :].T * np.sqrt(s_d[:k])
                    )
                else:
                    coarse.data = new_coarse
            self._compression_count += 1

            # 2. PROLONGATE (Transport up): harmonic completion coarse → fine correction
            reconstructed = prolongate(coarse.data, fine.data.shape)
            residual = fine.data - reconstructed

            # 3. ANNEAL: smooth the residual, blend back
            T = fine.mood.temperature
            smoothed = _smooth(residual, T)
            fine.data = reconstructed + smoothed

            # Check coherence for early exit
            kappa_now = coherence_score(fine.data, coarse.data)
            if kappa_now > 0.95:
                break

        kappa_after = coherence_score(fine.data, coarse.data)
        kappa_dot = kappa_after - kappa_before

        # Update moods
        fine.mood.kappa_dot = kappa_dot
        coarse.mood.kappa_dot = kappa_dot
        # Anneal temperature: cool if improving, heat if stuck
        if kappa_dot > 0.01:
            fine.mood.temperature *= 0.9
            coarse.mood.temperature *= 0.9
        elif kappa_dot < -0.01:
            fine.mood.temperature = min(fine.mood.temperature * 1.2, 10.0)
            coarse.mood.temperature = min(coarse.mood.temperature * 1.2, 10.0)

        # Decide commit vs rollback
        if kappa_after >= self._prev_kappa - 0.05:
            self.state.commit_wip()
            self._prev_kappa = kappa_after
            committed = True
        else:
            # Obstruction: rollback, restructure by heating
            self.state.rollback_wip()
            self.state.fine.mood.temperature = min(
                self.state.fine.mood.temperature * 1.5, 10.0
            )
            committed = False

        self._step += 1

        return {
            "step": self._step,
            "kappa": kappa_after,
            "kappa_dot": kappa_dot,
            "temperature": fine.mood.temperature,
            "committed": committed,
            "rollback_count": self.state.rollback_count,
            "compression_count": self._compression_count,
        }

    def get_mood(self) -> Mood:
        """Current mood from the fine level."""
        return self.state.fine.mood

    def _reinit(self, fine_shape: tuple[int, int]):
        """Reinitialize state for new grid dimensions."""
        coarse_shape = ((fine_shape[0] + 1) // 2, (fine_shape[1] + 1) // 2)
        self.state = MultigridState(fine_shape, coarse_shape)


def _smooth(residual: np.ndarray, temperature: float) -> np.ndarray:
    """Weighted Jacobi-like smoother. Temperature controls damping."""
    omega = 0.5 / (1.0 + temperature)  # damping factor, lower T → more aggressive
    H, W = residual.shape
    smoothed = residual.copy()

    # One pass of weighted averaging with neighbors
    padded = np.pad(residual, 1, mode='edge')
    neighbor_avg = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:]
    ) / 4.0
    smoothed = (1 - omega) * residual + omega * neighbor_avg
    return smoothed
