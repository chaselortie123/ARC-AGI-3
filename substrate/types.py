"""Core substrate types. Minimal, typed, no over-abstraction."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Cell:
    """Single grid cell."""
    row: int
    col: int
    color: int


@dataclass
class Object:
    """Connected component in the grid."""
    id: int
    cells: list[Cell]
    color: int
    bbox: tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def center(self) -> tuple[float, float]:
        r = sum(c.row for c in self.cells) / len(self.cells)
        c = sum(c.col for c in self.cells) / len(self.cells)
        return (r, c)


@dataclass
class Relation:
    """Pairwise relation between objects."""
    src: int  # object id
    dst: int  # object id
    kind: str  # "adjacent", "contains", "same_color", "aligned_h", "aligned_v"


@dataclass
class Substrate:
    """Parsed representation of a single grid frame."""
    grid: np.ndarray  # (H, W) int array, colors 0-9
    objects: list[Object] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    background_color: int = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    def object_by_id(self, oid: int) -> Optional[Object]:
        for o in self.objects:
            if o.id == oid:
                return o
        return None


@dataclass
class Frame:
    """One observation from the environment."""
    grid: np.ndarray  # raw grid
    step: int
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)
