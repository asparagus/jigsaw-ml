from typing import Dict, Tuple

import pytest

from jigsaw import piece
from jigsaw import composite


class DummyModule(piece.Module):
    def __init__(self, inputs: Tuple[str], outputs: Tuple[str]):
        super().__init__()
        self._inputs = inputs
        self._outputs = outputs

    def inputs(self) -> Tuple[str]:
        return self._inputs

    def outputs(self) -> Tuple[str]:
        return self._outputs

    def forward(self, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        return {}


def test_composite_init():
    preprocessor_module = DummyModule(
        inputs=tuple(["input", "extra_input"]),
        outputs=tuple(["preprocessed"]),
    )
    main_module = DummyModule(
        inputs=tuple(["preprocessed"]),
        outputs=tuple(["output"]),
    )
    auxiliary_module = DummyModule(
        inputs=tuple(["preprocessed"]),
        outputs=tuple(["aux"]),
    )
    pieces = [preprocessor_module, main_module, auxiliary_module]
    composite.Composite(components=pieces)


def test_composite_fails_with_repeated_outputs():
    main_module = DummyModule(
        inputs=tuple(["input"]),
        outputs=tuple(["output"]),
    )
    alternative_module = DummyModule(
        inputs=tuple(["input"]),
        outputs=tuple(["output"]),
    )
    pieces = [main_module, alternative_module]
    with pytest.raises(composite.OutputRedefinitionError):
        composite.Composite(components=pieces)
