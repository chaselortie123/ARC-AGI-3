"""Typed bipartite factor graph and L = H^T Λ H operator.

This is the concrete implementation of the Gossamer spec's core substrate:
- Variables carry state (cell colors, object properties)
- Factors carry law (constraints between variables)
- L = H^T Λ H is the induced variable-space operator
- κ = commitment reading from the factor graph
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from substrate.types import Substrate, Object, Relation


# ---------------------------------------------------------------------------
# Variable and Factor nodes
# ---------------------------------------------------------------------------

@dataclass
class Variable:
    """Variable node: carries mutable state."""
    id: int
    kind: str          # "cell", "object", "relation"
    state: np.ndarray  # state vector (e.g. one-hot color for cells)


@dataclass
class Factor:
    """Factor node: carries law (constraint between variables)."""
    id: int
    kind: str              # "observed", "adjacency", "same_color", "color_match", etc.
    var_ids: list[int]     # which variables this factor connects
    precision: float       # λ_i in Λ — how strongly this factor matters
    hard: bool = False     # if True, this is an observation (infinite precision conceptually)

    def residual(self, var_states: dict[int, np.ndarray]) -> float:
        """Compute scalar residual: how violated is this factor?"""
        states = [var_states[vid] for vid in self.var_ids]

        if self.kind == "observed":
            # Observation factor: state should match observed value
            # var_ids[0] = cell variable, stored target in self._target
            return 0.0  # observations are always satisfied by construction

        elif self.kind == "same_color":
            # Two objects should have the same color distribution
            if len(states) == 2:
                return float(np.linalg.norm(states[0] - states[1]))
            return 0.0

        elif self.kind == "adjacency":
            # Adjacent objects: soft constraint that neighbors are related
            # Residual = 0 when acknowledged, but we just check consistency
            if len(states) == 2:
                # Lower residual when objects have distinct colors (typical ARC pattern)
                similarity = float(np.dot(states[0], states[1]))
                return similarity  # high similarity = high residual (adjacent objects usually differ)
            return 0.0

        elif self.kind == "color_match":
            # Cell should match its parent object's color
            if len(states) == 2:
                return float(np.linalg.norm(states[0] - states[1]))
            return 0.0

        else:
            return 0.0


# ---------------------------------------------------------------------------
# Factor Graph
# ---------------------------------------------------------------------------

@dataclass
class FactorGraph:
    """Typed bipartite factor graph. Variables <-> Factors."""
    variables: dict[int, Variable] = field(default_factory=dict)
    factors: dict[int, Factor] = field(default_factory=dict)
    _next_var_id: int = 0
    _next_fac_id: int = 0

    def add_variable(self, kind: str, state: np.ndarray) -> int:
        vid = self._next_var_id
        self.variables[vid] = Variable(id=vid, kind=kind, state=state)
        self._next_var_id += 1
        return vid

    def add_factor(self, kind: str, var_ids: list[int],
                   precision: float, hard: bool = False) -> int:
        fid = self._next_fac_id
        self.factors[fid] = Factor(
            id=fid, kind=kind, var_ids=var_ids,
            precision=precision, hard=hard
        )
        self._next_fac_id += 1
        return fid

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def var_dim(self) -> int:
        """Total dimension of the variable state space."""
        return sum(v.state.shape[0] for v in self.variables.values())


# ---------------------------------------------------------------------------
# Build factor graph from a parsed Substrate
# ---------------------------------------------------------------------------

def build_factor_graph(substrate: Substrate, n_colors: int = 10) -> FactorGraph:
    """Construct a typed bipartite factor graph from a parsed ARC frame.

    Variables:
      - One per cell (state = one-hot color vector)
      - One per object (state = one-hot color vector)

    Factors:
      - Observed: each cell is observed with its color (hard, high precision)
      - ColorMatch: each cell's color matches its parent object (medium precision)
      - SameColor: pairs of same-colored objects (medium precision)
      - Adjacency: pairs of adjacent objects (low precision)
    """
    fg = FactorGraph()

    # --- Cell variables ---
    H, W = substrate.grid.shape
    cell_var_ids = np.zeros((H, W), dtype=int)
    for r in range(H):
        for c in range(W):
            color = int(substrate.grid[r, c])
            state = np.zeros(n_colors)
            state[color] = 1.0
            vid = fg.add_variable("cell", state)
            cell_var_ids[r, c] = vid

    # --- Observed factors (one per cell) ---
    for r in range(H):
        for c in range(W):
            fg.add_factor("observed", [cell_var_ids[r, c]],
                          precision=100.0, hard=True)

    # --- Object variables ---
    obj_var_ids = {}
    for obj in substrate.objects:
        state = np.zeros(n_colors)
        state[obj.color] = 1.0
        vid = fg.add_variable("object", state)
        obj_var_ids[obj.id] = vid

    # --- ColorMatch factors (cell belongs to object) ---
    cell_owner = {}
    for obj in substrate.objects:
        for cell in obj.cells:
            cell_owner[(cell.row, cell.col)] = obj.id

    for r in range(H):
        for c in range(W):
            if (r, c) in cell_owner:
                obj_id = cell_owner[(r, c)]
                fg.add_factor("color_match",
                              [cell_var_ids[r, c], obj_var_ids[obj_id]],
                              precision=10.0)

    # --- Relation factors ---
    for rel in substrate.relations:
        if rel.src in obj_var_ids and rel.dst in obj_var_ids:
            if rel.kind == "same_color":
                fg.add_factor("same_color",
                              [obj_var_ids[rel.src], obj_var_ids[rel.dst]],
                              precision=5.0)
            elif rel.kind == "adjacent":
                fg.add_factor("adjacency",
                              [obj_var_ids[rel.src], obj_var_ids[rel.dst]],
                              precision=1.0)

    return fg


# ---------------------------------------------------------------------------
# L = H^T Λ H — the induced variable-space operator
# ---------------------------------------------------------------------------

def build_operator_L(fg: FactorGraph) -> np.ndarray:
    """Build L = H^T Λ H as a dense matrix.

    H is the factor-to-variable incidence Jacobian.
    Λ is the diagonal precision matrix over factors.

    Each variable contributes a block of size = len(variable.state).
    Each factor row in H has nonzero entries at the columns of its connected variables.

    For a factor connecting variables v_i and v_j:
      H has a row with identity blocks at the column positions of v_i and v_j.
      The precision λ weights this factor's contribution to L.

    Result: L is (var_dim x var_dim) symmetric positive semi-definite.
    """
    # Map variable ids to column offsets
    var_ids_sorted = sorted(fg.variables.keys())
    var_offset = {}
    offset = 0
    for vid in var_ids_sorted:
        var_offset[vid] = offset
        offset += fg.variables[vid].state.shape[0]
    total_dim = offset

    # Build H and Λ
    # Each factor creates one or more rows depending on variable dimensions
    factor_rows = []
    precisions = []

    for fac in fg.factors.values():
        if len(fac.var_ids) == 1:
            # Unary factor: single block
            vid = fac.var_ids[0]
            d = fg.variables[vid].state.shape[0]
            for k in range(d):
                row = np.zeros(total_dim)
                row[var_offset[vid] + k] = 1.0
                factor_rows.append(row)
                precisions.append(fac.precision)
        elif len(fac.var_ids) == 2:
            # Binary factor: difference constraint (v_i - v_j)
            vid_a, vid_b = fac.var_ids
            d_a = fg.variables[vid_a].state.shape[0]
            d_b = fg.variables[vid_b].state.shape[0]
            d = min(d_a, d_b)
            for k in range(d):
                row = np.zeros(total_dim)
                row[var_offset[vid_a] + k] = 1.0
                row[var_offset[vid_b] + k] = -1.0
                factor_rows.append(row)
                precisions.append(fac.precision)

    if not factor_rows:
        return np.zeros((total_dim, total_dim))

    H = np.array(factor_rows)           # (n_factor_rows, var_dim)
    Lambda = np.diag(precisions)         # (n_factor_rows, n_factor_rows)
    L = H.T @ Lambda @ H                # (var_dim, var_dim)
    return L


# ---------------------------------------------------------------------------
# κ — commitment reading
# ---------------------------------------------------------------------------

def commitment(fg: FactorGraph) -> float:
    """Inverse normalized total residual.

    κ → 1 when all soft factors are satisfied.
    κ → 0 when they're maximally violated.
    """
    var_states = {vid: v.state for vid, v in fg.variables.items()}
    total = sum(
        f.precision * f.residual(var_states) ** 2
        for f in fg.factors.values() if not f.hard
    )
    n = sum(1 for f in fg.factors.values() if not f.hard)
    return 1.0 / (1.0 + total / max(n, 1))


# ---------------------------------------------------------------------------
# Spectrum of L
# ---------------------------------------------------------------------------

def spectrum(L: np.ndarray) -> np.ndarray:
    """Eigenvalues of L, sorted ascending."""
    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues
