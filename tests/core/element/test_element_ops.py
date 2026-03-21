import numpy as np
import pytest

from AlgebraCore.basis import Basis
from AlgebraCore.element import Element


def test_outer_element_builds_tensor_basis_and_outer_coeffs():
    basis_ab = Basis([["a", "b"]])
    basis_ghl = Basis([["g", "h", "l"]])

    left = Element(basis_ab, np.array([1.0, 2.0]))
    right = Element(basis_ghl, np.array([3.0, 4.0, 5.0]))

    out = left & right

    assert out.basis.shape == (2, 3)
    expected = np.tensordot(left.coeffs, right.coeffs, axes=0)
    np.testing.assert_allclose(out.coeffs, expected)


def test_outer_element_basis_names_and_order():
    basis_left = Basis([["a0", "a1"]])
    basis_right = Basis([["b0", "b1", "b2"]])
    left = Element(basis_left, np.array([1.0, 2.0]))
    right = Element(basis_right, np.array([3.0, 4.0, 5.0]))

    out_lr = left & right
    out_rl = right & left

    assert out_lr.basis.shape == (2, 3)
    assert out_lr.basis.names == ["a0.b0", "a0.b1", "a0.b2", "a1.b0", "a1.b1", "a1.b2"]

    # reversed order swaps axes; flattened names differ
    assert out_rl.basis.shape == (3, 2)
    assert out_rl.basis.names == ["b0.a0", "b0.a1", "b1.a0", "b1.a1", "b2.a0", "b2.a1"]


def test_outer_element_mixed_rank():
    basis_left = Basis([["a0", "a1"], ["b0", "b1"]])
    basis_right = Basis([["c0", "c1"]])

    left = Element(basis_left, np.arange(4).reshape(2, 2))
    right = Element(basis_right, np.array([10.0, 20.0]))

    out = left & right
    assert out.basis.shape == (2, 2, 2)
    expected = np.tensordot(left.coeffs, right.coeffs, axes=0)
    np.testing.assert_allclose(out.coeffs, expected)


def test_outer_element_both_rank_two():
    basis_left = Basis([["a0", "a1"], ["b0", "b1"]])
    basis_right = Basis([["c0", "c1", "c2"], ["d0", "d1"]])

    left = Element(basis_left, np.arange(4).reshape(2, 2))
    right = Element(basis_right, np.arange(6).reshape(3, 2))

    out = left & right
    assert out.basis.shape == (2, 2, 3, 2)
    expected = np.tensordot(left.coeffs, right.coeffs, axes=0)
    np.testing.assert_allclose(out.coeffs, expected)


def test_element_matmul_uses_outer_element():
    basis_left = Basis([["x0", "x1"]])
    basis_right = Basis([["y0", "y1", "y2"]])

    left = Element(basis_left, np.array([1.0, -1.0]))
    right = Element(basis_right, np.array([0.5, 2.0, -3.0]))

    with pytest.raises(TypeError):
        _ = left @ right


def test_element_and_operator_outer_product():
    basis_left = Basis([["a0", "a1"]])
    basis_right = Basis([["b0", "b1"]])
    left = Element(basis_left, np.array([2.0, 3.0]))
    right = Element(basis_right, np.array([5.0, 7.0]))

    out = left & right
    expected = np.tensordot(left.coeffs, right.coeffs, axes=0)
    np.testing.assert_allclose(out.coeffs, expected)
