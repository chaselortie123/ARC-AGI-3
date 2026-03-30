"""Tests for frame parser."""

import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from substrate.types import Frame
from perceive.parser import parse_frame


def _make_frame(grid):
    return Frame(grid=np.array(grid, dtype=int), step=0)


class TestParseFrame:
    def test_empty_grid(self):
        frame = _make_frame([[0, 0], [0, 0]])
        sub = parse_frame(frame)
        assert sub.background_color == 0
        assert len(sub.objects) == 0

    def test_single_object(self):
        grid = [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ]
        sub = parse_frame(_make_frame(grid))
        assert len(sub.objects) == 1
        assert sub.objects[0].color == 1
        assert sub.objects[0].size == 3

    def test_two_objects_different_colors(self):
        grid = [
            [1, 1, 0, 2, 2],
            [1, 0, 0, 0, 2],
            [0, 0, 0, 0, 0],
        ]
        sub = parse_frame(_make_frame(grid))
        assert len(sub.objects) == 2
        colors = {o.color for o in sub.objects}
        assert colors == {1, 2}

    def test_same_color_disconnected(self):
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
        sub = parse_frame(_make_frame(grid))
        # 4 separate single-cell objects
        assert len(sub.objects) == 4
        assert all(o.color == 1 for o in sub.objects)

    def test_adjacency_relations(self):
        grid = [
            [1, 2],
            [0, 0],
        ]
        sub = parse_frame(_make_frame(grid))
        assert len(sub.objects) == 2
        adj = [r for r in sub.relations if r.kind == "adjacent"]
        assert len(adj) == 1

    def test_bbox(self):
        grid = [
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 0],
        ]
        sub = parse_frame(_make_frame(grid))
        assert len(sub.objects) == 1
        assert sub.objects[0].bbox == (1, 1, 2, 2)

    def test_every_cell_assigned(self):
        """Invariant: every non-bg cell belongs to exactly one object."""
        grid = np.random.randint(0, 5, size=(8, 8))
        sub = parse_frame(_make_frame(grid))
        cell_set = set()
        for obj in sub.objects:
            for cell in obj.cells:
                key = (cell.row, cell.col)
                assert key not in cell_set, "Cell assigned to multiple objects"
                cell_set.add(key)
        # Every non-bg cell should be covered
        for r in range(8):
            for c in range(8):
                if grid[r, c] != sub.background_color:
                    assert (r, c) in cell_set
