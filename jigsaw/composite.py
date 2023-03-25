"""Module containing the Composite class which stitches together Pieces."""
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar
import collections

from jigsaw import graph_utils
from jigsaw import piece


T = TypeVar("T", piece.Piece)


class Composite(piece.Piece):
    """This class allows abstracting multiple pieces into a single instance.

    The component pieces are merged such that:
    - The inputs to this instance will be the union of all inputs
    - The outputs of this instance will be the union of all outputs

    It's important to note that no two outputs can be the same within a Composite.
    If two different Pieces share the same output, an error will be thrown.

    The compute function will run all component pieces in topological order
    and make the outputs of pieces run earlier available to pieces run later on.
    """

    def __init__(self, components: Sequence[piece.Piece], *, name: Optional[str] = None):
        """Initializes the Composite.

        Args:
            components: The pieces that are being merged into this Composite
            name: Optional name to use for this instance.
                If missing will use the class name.
        """
        dependency_graph = Composite.build_dependency_graph(components)
        sorted_indices = graph_utils.topological_sort(dependency_graph)
        self._components = tuple(components[i] for i in sorted_indices)
        self._inputs = {i for piece in self._components for i in piece.inputs()}
        self._outputs = {o for piece in self._components for o in piece.outputs()}
        self.name = name or self.__class__.__qualname__

    def inputs(self) -> Tuple[str, ...]:
        """Gets the names of the inputs required by this composite."""
        return self._inputs

    def outputs(self) -> Tuple[str, ...]:
        """Gets the names of the outputs produced by this composite."""
        return self._outputs

    def compute(self, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Performs the computation.

        Components are pre-sorted in topological order such that the outputs of
        earlier pieces can be used by the later ones.

        Args:
            inputs: Dict containing the tensors matching the required inputs.

        Returns:
            A Dict with the tensors matching the declared outputs.
        """
        cumulative_inputs = dict(inputs)
        outputs = dict()
        for piece in self._components:
            piece_outputs = piece.compute(cumulative_inputs)
            cumulative_inputs.update(piece_outputs)
            outputs.update(piece_outputs)
        return outputs

    def extract(self, component_type: Type[T]) -> List[T]:
        """Extract components that match a given type.

        Args:
            component_type: Type of component to extract deriving from piece.Piece

        Returns:
            A list of the components of this type.
        """
        matching_components = []
        for component in self._components:
            if isinstance(component, component_type):
                matching_components.append(component)
            elif isinstance(component, Composite):
                matching_components.extend(
                    component.extract(component_type))
        return matching_components

    @classmethod
    def assert_non_conflicting_outputs(cls, components: Sequence[piece.Piece]):
        """Verifies that no two components have the same outputs.

        Args:
            components: Collection of pieces to verify.

        Raises:
            AssertionError if multiple components have outputs sharing the same name.
        """
        all_outputs = [out for piece in components for out in piece.outputs()]
        counts = collections.Counter(all_outputs)
        repeated_outputs = (out for out, count in counts.items() if count > 1)
        assert not any(repeated_outputs), (
            "There are conflicting outputs from multiple components: {}".format(
                repeated_outputs
            )
        )

    @classmethod
    def build_dependency_graph(
            cls,
            components: Sequence[piece.Piece]
        ) -> Dict[int, Set[int]]:
        """Builds a dependency graph between the given components.

        Dependency graph is constructed based on the inputs / outputs that
        the components themselves declare.

        Args:
            components: Collection of pieces to build the graph for.

        Returns:
            A dictionary containing the dependencies for each component.
        """
        cls.assert_non_conflicting_outputs(components)
        inputs_by_piece = {
            i: set(piece.inputs())
            for i, piece in enumerate(components)
        }
        output_mapping = {
            o: i
            for i, piece in enumerate(components)
            for o in piece.outputs()
        }
        dependency_graph = dict()
        for i, inputs in inputs_by_piece.items():
            dependency_graph[i] = set(
                output_mapping[i]
                for i in inputs
                if i in output_mapping
            )
        return dependency_graph
