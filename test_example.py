"""Example tests for ARC-AGI-3."""


def test_addition():
    assert 1 + 1 == 2


def test_string_concatenation():
    assert "hello" + " " + "world" == "hello world"


def test_list_operations():
    items = [1, 2, 3]
    items.append(4)
    assert items == [1, 2, 3, 4]
    assert len(items) == 4
