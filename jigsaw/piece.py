"""Module containing the base class for Piece and its immediate children."""
from typing import Callable, Dict, Optional, Tuple

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


class WrappedLoss(LossFunction):
    def __init__(
            self,
            loss_fn: Callable,
            *args,
            input_name: str,
            target_name: str,
            name: Optional[str] = None,
            **kwargs,
        ):
        super().__init__(name=name or loss_fn.__name__)
        self.wrapped_loss_fn = loss_fn
        self.input_name = input_name
        self.target_name = target_name
        self.args = args
        self.kwargs = kwargs

    def inputs(self) -> Tuple[str]:
        """Gets the name of the input which is just the output."""
        return tuple([self.input_name])

    def outputs(self) -> Tuple[str]:
        """Gets the names of the outputs produced by this piece."""
        return tuple([self.name])

    def compute(self, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        """Computes the loss."""
        inpt = inputs[self.input_name]
        trgt = inputs[self.target_name]
        loss_value = self.wrapped_loss_fn(inpt, trgt, *self.args, **self.kwargs)
        output = {
            self.name: loss_value
        }
        return output


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
