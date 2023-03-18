"""Module containing the base class for Piece and its immediate children."""
from typing import Dict, Optional, Tuple

import abc


class Piece(abc.ABC):
    """Piece is the core class of Jigsaw.

    A piece declares its inputs, outputs and the corresponding compute function.

    A piece should not be too complex by itself, but grows in complexity as
    it's put together with other pieces.
    """

    @abc.abstractmethod
    def inputs(self) -> Tuple[str]:
        """Gets the names of the inputs required by this piece."""
        raise NotImplementedError()

    @abc.abstractmethod
    def outputs(self) -> Tuple[str]:
        """Gets the names of the outputs produced by this piece."""
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Performs the computation.

        Args:
            inputs: Dict containing the tensors matching the required inputs.

        Returns:
            A Dict with the tensors matching the declared outputs.
        """
        raise NotImplementedError()

    # def validate_inputs(self, inputs: Dict[str, "torch.Tensor"]):
    #     """Validates that the required inputs are available."""
    #     missing_inputs = set(self.inputs()).difference(inputs.keys())
    #     assert not missing_inputs, f"Missing required inputs {missing_inputs}"


class LossFunction(Piece):
    """Loss function base class implemented as a piece."""

    def __init__(self, *, name: Optional[str] = None):
        """Initializes the loss function.

        Args:
            name:
                Optional name given to this loss.
                If missing will use the class name.
        """
        self.name = name or self.__class__.__qualname__


class Module(Piece):
    """Module base class implemented as a piece."""

    def __init__(self, *, name: Optional[str] = None):
        """Initializes the module.

        Args:
            name:
                Optional name given to this module.
                If missing will use the class name.
        """
        self.name = name or self.__class__.__qualname__
