from typing import Dict, Tuple

import pytest

from jigsaw import piece
from jigsaw import composite


class DummyModule(piece.Module):
    def __init__(self, inputs: Tuple[str], outputs: Tuple[str]):
        self._inputs = inputs
        self._outputs = outputs

    def inputs(self) -> Tuple[str]:
        return self._inputs

    def outputs(self) -> Tuple[str]:
        return self._outputs

    def forward(self, inputs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        return {}


def test_build_dependency_graph():
    KEY_INPUT = "input"
    KEY_EXTRA_INPUT = "extra_input"
    KEY_PREPROCESSED = "preprocessed"
    KEY_OUTPUT = "output"
    KEY_AUXILIARY_OUTPUT = "aux"
    preprocessor_module = DummyModule(
        inputs=tuple([KEY_INPUT, KEY_EXTRA_INPUT]),
        outputs=tuple([KEY_PREPROCESSED]),
    )
    main_module = DummyModule(
        inputs=tuple([KEY_PREPROCESSED]),
        outputs=tuple([KEY_OUTPUT]),
    )
    auxiliary_module = DummyModule(
        inputs=tuple([KEY_PREPROCESSED]),
        outputs=tuple([KEY_AUXILIARY_OUTPUT]),
    )
    pieces = [preprocessor_module, main_module, auxiliary_module]
    dependency_graph = composite.Composite.build_dependency_graph(pieces)
    expected = {
        0: set(),
        1: {0},
        2: {0},
    }
    assert dependency_graph == expected


def test_build_dependency_graph_with_conflicting_outputs():
    KEY_INPUT = "input"
    KEY_OUTPUT = "output"
    main_module = DummyModule(
        inputs=tuple([KEY_INPUT]),
        outputs=tuple([KEY_OUTPUT]),
    )
    alternative_module = DummyModule(
        inputs=tuple([KEY_INPUT]),
        outputs=tuple([KEY_OUTPUT]),
    )
    pieces = [main_module, alternative_module]
    with pytest.raises(AssertionError):
        composite.Composite.build_dependency_graph(pieces)
