import pytest

from jigsaw import computation_graph



def test_build_dependency_graph():
    input_outputs = [
        (tuple(["input", "extra_input"]), tuple(["preprocessed"])),
        (tuple(["preprocessed"]), tuple(["output"])),
        (tuple(["preprocessed"]), tuple(["aux"])),
    ]
    dependency_graph = computation_graph.build_dependency_graph(input_outputs)
    expected = {
        0: set(),
        1: {0},
        2: {0},
    }
    assert dependency_graph == expected


def test_build_dependency_graph_with_conflicting_outputs():
    input_outputs = [
        (tuple(["input"]), tuple(["output"])),
        (tuple(["input"]), tuple(["output"])),
    ]
    with pytest.raises(computation_graph.OutputRedefinitionError):
        computation_graph.build_dependency_graph(input_outputs)


def test_topological_sort():
    dependency_graph = {
        0: {1, 2},
        1: {2},
        2: set(),
    }
    result = computation_graph.topological_sort(dependency_graph)
    expected = (2, 1, 0)
    assert result == expected


def test_topological_sort_short_cycle():
    dependency_graph = {
        0: {1},
        1: {0},
    }
    with pytest.raises(computation_graph.CyclicDependencyError):
        computation_graph.topological_sort(dependency_graph)


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
    with pytest.raises(computation_graph.CyclicDependencyError):
        computation_graph.topological_sort(dependency_graph)
