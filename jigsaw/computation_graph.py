"""Module containing utility functions related to graphs."""
from typing import Dict, List, Sequence, Set, Tuple
import collections


InputSpec = Tuple[str, ...]
OutputSpec = Tuple[str, ...]
IOSpec = Tuple[InputSpec, OutputSpec]


class GraphDefinitionError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class OutputRedefinitionError(GraphDefinitionError):
    def __init__(self, indices: Set[int], redefined_output: str):
        super().__init__(f"Multiple components redefine the same output ({redefined_output}): {indices}")
        self.indices = indices
        self.redefined_output = redefined_output


class CyclicDependencyError(GraphDefinitionError):
    def __init__(self, indices: Set[int]):
        super().__init__(f"Cyclic dependency between components ({indices})")
        self.indices = indices


def build_dependency_graph(inputs_and_outputs: Sequence[IOSpec]) -> Dict[int, Set[int]]:
    """Builds a dependency graph between the given inputs and outputs.

    Assumptions:
    - No output is repeated

    Args:
        inputs_and_outputs: Sequence of I/O specs from multiple components.

    Returns:
        A dictionary containing the dependencies for each component.
    """
    producers: Dict[str, int] = {}
    consumers = collections.defaultdict(list)
    for component_idx, (inputs, outputs) in enumerate(inputs_and_outputs):
        for output_name in outputs:
            existing_idx = producers.setdefault(output_name, component_idx)
            if existing_idx != component_idx:
                raise OutputRedefinitionError({existing_idx, component_idx}, output_name)
        for input_name in inputs:
            consumers[input_name].append(component_idx)

    dependency_graph: Dict[int, Set[int]] = {
        idx: set()
        for idx, _ in enumerate(inputs_and_outputs)
    }
    for input_name, input_consumers in consumers.items():
        producer = producers.get(input_name)
        if producer is not None:
            for consumer in input_consumers:
                dependency_graph[consumer].add(producer)
    return dependency_graph


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
    sorting: List[int] = list()
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
        raise CyclicDependencyError(missing_indices)

    return tuple(sorting)
