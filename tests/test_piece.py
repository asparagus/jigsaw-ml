import pytest

from jigsaw import piece


def test_abstract_classes():
    with pytest.raises(TypeError):
        piece.Piece()
    with pytest.raises(TypeError):
        piece.LossFunction()
    with pytest.raises(TypeError):
        piece.Module()
