"""Multigrid state: two-level hierarchy with WIP/committed separation.

Invariants preserved:
- Committed state is only updated via explicit commit
- WIP can be rolled back atomically
- Each level carries its own mood (T, kappa_dot)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import copy


@dataclass
class Mood:
    """Continuous regime state: temperature and coherence rate."""
    temperature: float = 1.0
    kappa_dot: float = 0.0  # rate of coherence change


@dataclass
class LevelState:
    """State at one multigrid level."""
    # Core data: a matrix representation at this scale
    data: np.ndarray
    mood: Mood = field(default_factory=Mood)
    # Low-rank adaptation buffers (Woodbury)
    U: Optional[np.ndarray] = None  # low-rank left factor
    V: Optional[np.ndarray] = None  # low-rank right factor

    def apply_lowrank_update(self, delta_U: np.ndarray, delta_V: np.ndarray):
        """Woodbury-style low-rank update: data += delta_U @ delta_V.T"""
        if self.U is None:
            self.U = delta_U
            self.V = delta_V
        else:
            self.U = np.concatenate([self.U, delta_U], axis=1)
            self.V = np.concatenate([self.V, delta_V], axis=1)
        self.data = self.data + delta_U @ delta_V.T

    def flush_lowrank(self):
        """Absorb accumulated low-rank updates into data, reset buffers."""
        self.U = None
        self.V = None

    @property
    def rank_budget_used(self) -> int:
        return 0 if self.U is None else self.U.shape[1]


class MultigridState:
    """Two-level multigrid state with WIP/committed separation.

    fine: frame/object level substrate
    coarse: action-effect / dynamics summary

    Uses copy-on-write delta snapshots for WIP.
    """

    def __init__(self, fine_shape: tuple[int, int], coarse_shape: tuple[int, int]):
        self._committed_fine = LevelState(data=np.zeros(fine_shape))
        self._committed_coarse = LevelState(data=np.zeros(coarse_shape))
        self._wip_fine: Optional[LevelState] = None
        self._wip_coarse: Optional[LevelState] = None
        self._rollback_count = 0

    # -- WIP management --

    def begin_wip(self):
        """Start a WIP transaction by snapshotting committed state."""
        self._wip_fine = copy.deepcopy(self._committed_fine)
        self._wip_coarse = copy.deepcopy(self._committed_coarse)

    def commit_wip(self):
        """Promote WIP to committed."""
        if self._wip_fine is None:
            return
        self._committed_fine = self._wip_fine
        self._committed_coarse = self._wip_coarse
        self._wip_fine = None
        self._wip_coarse = None

    def rollback_wip(self):
        """Discard WIP, revert to committed."""
        self._wip_fine = None
        self._wip_coarse = None
        self._rollback_count += 1

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    @property
    def has_wip(self) -> bool:
        return self._wip_fine is not None

    # -- Accessors: prefer WIP if active, else committed --

    @property
    def fine(self) -> LevelState:
        return self._wip_fine if self._wip_fine is not None else self._committed_fine

    @property
    def coarse(self) -> LevelState:
        return self._wip_coarse if self._wip_coarse is not None else self._committed_coarse

    @property
    def committed_fine(self) -> LevelState:
        return self._committed_fine

    @property
    def committed_coarse(self) -> LevelState:
        return self._committed_coarse
