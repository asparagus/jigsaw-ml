"""Module containing the base class for Piece and its immediate children."""
from typing import Callable, Dict, Optional, Tuple

import abc

import torch
from torch import nn


class Piece(nn.Module, abc.ABC):
    """Piece is the core class of Jigsaw.

    A piece declares its inputs, outputs and the corresponding compute function.

    A piece should not be too complex by itself, but grows in complexity as
    it's put together with other pieces.
    """
    def __init__(self, *, name: Optional[str] = None):
        """Initializes the underlying nn.Module.

        Args:
            name:
                Optional name given to this piece.
                If missing will use the class name.
        """
        super().__init__()
        self.name = name or self.__class__.__qualname__

    @abc.abstractmethod
    def inputs(self) -> Tuple[str, ...]:
        """Gets the names of the inputs required by this piece."""
        raise NotImplementedError()

    @abc.abstractmethod
    def outputs(self) -> Tuple[str, ...]:
        """Gets the names of the outputs produced by this piece."""
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Performs the computation.

        Args:
            inputs: Dict containing the tensors matching the required inputs.

        Returns:
            A Dict with the tensors matching the declared outputs.
        """
        raise NotImplementedError()

    # def validate_inputs(self, inputs: Dict[str, torch.Tensor]):
    #     """Validates that the required inputs are available."""
    #     missing_inputs = set(self.inputs()).difference(inputs.keys())
    #     assert not missing_inputs, f"Missing required inputs {missing_inputs}"


class Loss(Piece):

    @classmethod
    def wrap(cls, loss_factory: Callable):
        class WrappedLoss(Loss):
            def __init__(self, *args, input_name: str, target_name: str, name: Optional[str] = None, **kwargs):
                loss = loss_factory(*args, **kwargs)
                super().__init__(name=name or loss.__class__.__qualname__)
                self.loss = loss
                self.input_name = input_name
                self.target_name = target_name

            def inputs(self) -> Tuple[str, ...]:
                return tuple([self.input_name, self.target_name])

            def outputs(self) -> Tuple[str, ...]:
                return tuple([self.name])

            def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                inpt = inputs[self.input_name]
                trgt = inputs[self.target_name]
                output = {
                    self.name: self.loss(inpt, trgt)
                }
                return output

        return WrappedLoss


class Module(Piece):

    @classmethod
    def wrap(cls, module_factory: Callable):
        class WrappedModule(Module):
            def __init__(self, *args, input_name: str, output_name: str, name: Optional[str] = None, **kwargs):
                module = module_factory(*args, **kwargs)
                super().__init__(name=name or module.__class__.__qualname__)
                self.inner_module = module
                self.input_name = input_name
                self.output_name = output_name

            def inputs(self) -> Tuple[str, ...]:
                return tuple([self.input_name])

            def outputs(self) -> Tuple[str, ...]:
                return tuple([self.output_name])

            def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                inpt = inputs[self.input_name]
                output = {
                    self.output_name: self.inner_module(inpt)
                }
                return output

        return WrappedModule
