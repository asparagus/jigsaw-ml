"""Module containing utility functions related to graphs."""
from typing import Dict, Set, Tuple


def topological_sort(dependency_graph: Dict[int, Set[int]]) -> Tuple[int, ...]:
    """Compute the topological sorting of the given dependency graph.

    Args:
        dependency_graph:
            A dict pointing from indices to the indices this node depends on.
            If a node contains no dependencies, its corresponding index should
            contain an empty set.

    Returns:
        A tuple containing the indices that can be used to sort the graph.
    """
    sorting = list()
    resolved = [False] * len(dependency_graph)
    while len(sorting) < len(dependency_graph):
        initial_length = len(sorting)
        for i, deps in dependency_graph.items():
            if not resolved[i] and len(deps.difference(sorting)) == 0:
                sorting.append(i)
                resolved[i] = True
        any_added = len(sorting) > initial_length
        if not any_added:
            break
    if len(sorting) < len(dependency_graph):
        missing_indices = set(dependency_graph.keys()).difference(sorting)
        raise ValueError(
            f"Cyclic dependency between indices {missing_indices}"
        )

    return tuple(sorting)
