"""Module containing the Composite class which stitches together Pieces."""
from typing import Dict, List, Optional, Sequence, Set, Tuple, Type, TypeVar

from jigsaw import computation_graph
from jigsaw import Piece

import torch
from torch import nn


T = TypeVar("T", bound=Piece)


class GraphDefinitionError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class OutputRedefinitionError(GraphDefinitionError):
    def __init__(self, composite_name: str, names: List[str], indices: List[int], redefined_output: str):
        super().__init__(
            f"Multiple components within {composite_name} "
            f"redefine the same output ({redefined_output}): "
            f"{names} at indices {indices}"
        )
        self.composite_name = composite_name
        self.names = names
        self.indices = indices
        self.redefined_output = redefined_output


class CyclicDependencyError(GraphDefinitionError):
    def __init__(self, composite_name: str, names: List[str], indices: List[int]):
        super().__init__(
            f"Cyclic dependency between components within {composite_name} :"
            f"{names} at indices {indices}"
        )
        self.composite_name = composite_name
        self.names = names
        self.indices = indices


class Composite(Piece):
    """This class allows abstracting multiple pieces into a single instance.

    The component pieces are merged such that:
    - The inputs to this instance will be the union of all inputs
    - The outputs of this instance will be the union of all outputs

    It's important to note that no two outputs can be the same within a Composite.
    If two different Pieces share the same output, an error will be thrown.

    The compute function will run all component pieces in topological order
    and make the outputs of pieces run earlier available to pieces run later on.
    """

    def __init__(self, components: Sequence[Piece], *, name: Optional[str] = None):
        """Initializes the Composite.

        Args:
            components: The pieces that are being merged into this Composite
            name: Optional name to use for this instance.
                If missing will use the class name.
        """
        super().__init__()
        self.name = name or self.__class__.__qualname__
        dependency_graph = self.try_build_dependency_graph(
            composite_name=self.name,
            components=components,
        )
        sorted_indices = self.try_topological_sort(
            composite_name=self.name,
            components=components,
            dependency_graph=dependency_graph,
        )
        self._components = nn.ModuleList(modules=[components[i] for i in sorted_indices])
        self._inputs = tuple([i for piece in self._components for i in piece.inputs()])
        self._outputs = tuple([o for piece in self._components for o in piece.outputs()])

    def inputs(self) -> Tuple[str, ...]:
        """Gets the names of the inputs required by this composite."""
        return self._inputs

    def outputs(self) -> Tuple[str, ...]:
        """Gets the names of the outputs produced by this composite."""
        return self._outputs

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
            piece_outputs = piece(cumulative_inputs)
            cumulative_inputs.update(piece_outputs)
            outputs.update(piece_outputs)
        return outputs

    def extract(self, component_type: Type[T]) -> List[T]:
        """Extract components that match a given type.

        Args:
            component_type: Type of component to extract deriving from Piece

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
    def try_build_dependency_graph(
            cls,
            composite_name: str,
            components: Sequence[Piece],
        ):
        component_ios = [(c.inputs(), c.outputs()) for c in components]
        try:
            return computation_graph.build_dependency_graph(component_ios)
        except computation_graph.OutputRedefinitionError as error:
            raise OutputRedefinitionError(
                composite_name=composite_name,
                names=[str(components[idx]) for idx in error.indices],
                indices=list(error.indices),
                redefined_output=error.redefined_output,
            )

    @classmethod
    def try_topological_sort(
            cls,
            composite_name: str,
            components: Sequence[Piece],
            dependency_graph: Dict[int, Set[int]],
        ) -> Tuple[int, ...]:
        try:
            return computation_graph.topological_sort(dependency_graph)
        except computation_graph.CyclicDependencyError as error:
            raise CyclicDependencyError(
                composite_name=composite_name,
                names=[str(components[idx]) for idx in error.indices],
                indices=list(error.indices),
            )
