"""Frame parser: raw grid → typed Substrate.

Invariant: every non-background cell belongs to exactly one object.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from substrate.types import Cell, Object, Relation, Substrate, Frame


def parse_frame(frame: Frame) -> Substrate:
    """Parse a Frame into a Substrate with objects and relations."""
    grid = np.asarray(frame.grid, dtype=int)
    bg = _detect_background(grid)
    objects = _extract_objects(grid, bg)
    relations = _extract_relations(objects, grid.shape)
    return Substrate(grid=grid, objects=objects, relations=relations, background_color=bg)


def _detect_background(grid: np.ndarray) -> int:
    """Background = most frequent color."""
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _extract_objects(grid: np.ndarray, bg: int) -> list[Object]:
    """Connected-component labeling (4-connected) for non-background cells."""
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    objects = []
    obj_id = 0

    for r in range(H):
        for c in range(W):
            if visited[r, c] or grid[r, c] == bg:
                continue
            # BFS flood fill
            color = int(grid[r, c])
            cells = []
            queue = deque([(r, c)])
            visited[r, c] = True
            min_r, min_c, max_r, max_c = r, c, r, c
            while queue:
                cr, cc = queue.popleft()
                cells.append(Cell(cr, cc, int(grid[cr, cc])))
                min_r, min_c = min(min_r, cr), min(min_c, cc)
                max_r, max_c = max(max_r, cr), max(max_c, cc)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            objects.append(Object(
                id=obj_id, cells=cells, color=color,
                bbox=(min_r, min_c, max_r, max_c)
            ))
            obj_id += 1

    return objects


def _extract_relations(objects: list[Object], grid_shape: tuple[int, int]) -> list[Relation]:
    """Extract pairwise relations between objects. Kept minimal for v1."""
    relations = []
    # Build a cell->object_id map for adjacency detection
    cell_owner = {}
    for obj in objects:
        for cell in obj.cells:
            cell_owner[(cell.row, cell.col)] = obj.id

    # Adjacency: objects whose cells are 4-adjacent
    adj_pairs = set()
    for obj in objects:
        for cell in obj.cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cell.row + dr, cell.col + dc
                key = (nr, nc)
                if key in cell_owner and cell_owner[key] != obj.id:
                    pair = (min(obj.id, cell_owner[key]), max(obj.id, cell_owner[key]))
                    if pair not in adj_pairs:
                        adj_pairs.add(pair)
                        relations.append(Relation(pair[0], pair[1], "adjacent"))

    # Same color
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            if a.color == b.color:
                relations.append(Relation(a.id, b.id, "same_color"))

    return relations
