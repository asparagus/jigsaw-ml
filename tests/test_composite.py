import pytest

import torch
from torch import nn

from jigsaw import (
    Composite,
    Module,
    CyclicDependencyError,
    OutputRedefinitionError,
)


def linear_factory(in_features, out_features, **kwargs) -> nn.Module:
    return nn.Linear(in_features=in_features, out_features=out_features, **kwargs)


LinearModule = Module.wrap(linear_factory)


def test_composite_init():
    preprocessor = LinearModule(16, 16, input_name="input", output_name="preprocessed")
    main = LinearModule(16, 16, input_name="preprocessed", output_name="output")
    aux = LinearModule(16, 16, input_name="preprocessed", output_name="aux")
    _ = Composite(components=[preprocessor, main, aux])


def test_composite_fails_with_repeated_outputs():
    main = LinearModule(16, 16, input_name="input", output_name="output")
    alt = LinearModule(16, 16, input_name="input", output_name="output")
    with pytest.raises(OutputRedefinitionError):
        _ = Composite(components=[main, alt])


def test_composite_fails_with_cyclic_dependencies():
    ab = LinearModule(16, 16, input_name="a", output_name="b")
    ba = LinearModule(16, 16, input_name="b", output_name="a")
    with pytest.raises(CyclicDependencyError):
        _ = Composite(components=[ab, ba])


def test_composite_run_sorted():
    ab = LinearModule(in_features=16, out_features=8, input_name="a", output_name="b")
    bc = LinearModule(in_features=8, out_features=4, input_name="b", output_name="c")
    cd = LinearModule(in_features=4, out_features=2, input_name="c", output_name="d")
    composite = Composite(components=[cd, bc, ab])
    inputs = {"a": torch.rand(32, 16)}
    outputs = composite.forward(inputs)
    outputs.keys() == {"a", "b", "c", "d"}
