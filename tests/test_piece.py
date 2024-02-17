import pytest

import torch
from torch import nn
from jigsaw import (
    Module,
    Loss,
    Piece,
)


def test_abstract_classes():
    with pytest.raises(TypeError):
        Piece()
    with pytest.raises(TypeError):
        Loss()
    with pytest.raises(TypeError):
        Module()


def test_module_wrap():
    def linear_factory(in_features, out_features, **kwargs) -> nn.Module:
        return nn.Linear(in_features=in_features, out_features=out_features, **kwargs)
    LinearModule = Module.wrap(linear_factory)
    linear_instance = LinearModule(
        in_features=32,
        out_features=1,
        bias=False,
        input_name="a",
        output_name="b",
    )
    assert linear_instance.inputs() == tuple(["a"],)
    assert linear_instance.outputs() == tuple(["b"],)
    assert linear_instance.name == "Linear"
    inputs = {"a": torch.randn(16, 32)}
    outputs = linear_instance.forward(inputs)
    assert outputs.keys() == {"b"}
    assert outputs["b"].shape == torch.Size([16, 1])


def test_loss_wrap():
    loss_factory = nn.BCELoss
    BCELoss = Loss.wrap(loss_factory)
    bce_instance = BCELoss(input_name="o", target_name="y", reduction='none')
    assert bce_instance.inputs() == tuple(["o", "y"])
    assert bce_instance.outputs() == tuple(["BCELoss"])
    assert bce_instance.name == "BCELoss"
    inputs = {
        "o": torch.rand(16, 1),
        "y": (torch.rand(16, 1) > 0.5).float(),
    }
    outputs = bce_instance.forward(inputs)
    assert outputs.keys() == {"BCELoss"}
    assert outputs["BCELoss"].shape == torch.Size([16, 1])
