import numpy as np
import pytest

from AlgebraCore.basis import Basis
from AlgebraCore.basis import TensorBasis


def test_basis_multi_axis_flatten_and_indexing():
    b = Basis([["i", "j"], ["k", "l"]])

    assert b.shape == (2, 2)
    assert b.flatten() == ["i.k", "i.l", "j.k", "j.l"]

    # Access by multi-index and flattened names
    assert b.at(0, 1) == "i.l"
    assert b.index_tuple("i.k") == (0, 0)
    assert b.index_tuple("j", "l") == (1, 1)


def test_basis_single_axis_backward_compat():
    b = Basis(["x", "y", "z"])
    assert b.shape == (3,)
    assert b.names == ["x", "y", "z"]
    assert b.index_tuple("y") == (1,)


def test_basis_validation_errors():
    # Non-string labels rejected
    with pytest.raises(TypeError):
        Basis([1, 2, 3])

    # Empty label rejected
    with pytest.raises(ValueError):
        Basis(["", "a"])

    # Empty axis rejected
    with pytest.raises(ValueError):
        Basis([[]])

    # Duplicate labels within an axis rejected
    with pytest.raises(ValueError):
        Basis([["i", "i"], ["k"]])


def test_tensor_basis_builder_combines_axes():
    b1 = Basis(["a", "b"])
    b2 = Basis(["x", "y", "z"])

    tb = TensorBasis([b1, b2])

    assert tb.shape == (2, 3)
    assert tb.axes == [["a", "b"], ["x", "y", "z"]]
    assert tb.flatten() == ["a.x", "a.y", "a.z", "b.x", "b.y", "b.z"]
