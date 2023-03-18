import pytest

from jigsaw import graph_utils


def test_topological_sort():
    dependency_graph = {
        0: {1, 2},
        1: {2},
        2: set(),
    }
    result = graph_utils.topological_sort(dependency_graph)
    expected = (2, 1, 0)
    assert result == expected


def test_topological_sort_short_cycle():
    dependency_graph = {
        0: {1},
        1: {0},
    }
    with pytest.raises(ValueError):
        graph_utils.topological_sort(dependency_graph)


def test_topological_sort_long_cycle():
    dependency_graph = {
        0: {6},
        1: {0},
        2: {1},
        3: {2},
        4: {3},
        5: {4},
        6: {5},
    }
    with pytest.raises(ValueError):
        graph_utils.topological_sort(dependency_graph)
