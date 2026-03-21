import numpy as np
import pytest

from AlgebraCore.basis import Basis
from AlgebraCore.element import Element
from AlgebraCore.product import AlgebraProduct
from AlgebraCore.transformation import Transformation


def test_transformation_shape_and_apply_multi_axis():
    b_old = Basis([["i", "j"], ["k", "l"]])
    b_new = Basis([["i", "j"], ["k", "l"]])

    # Identity over flattened dimension (4x4) reshaped to (2,2,2,2)
    T = np.eye(4).reshape(2, 2, 2, 2)
    tf = Transformation(old_basis=b_old, new_basis=b_new, T=T)

    e = Element.basis_vector(b_old, "i.k")
    res = tf @ e
    assert np.isclose(res["i", "k"], 1.0)
    assert np.isclose(res.coeffs.sum(), 1.0)


def test_transformation_getitem_by_labels():
    b = Basis(["a", "b"])
    T = np.array([[1.0, 2.0], [3.0, 4.0]])
    tf = Transformation(old_basis=b, new_basis=b, T=T)

    assert np.isclose(tf["a", "a"], 1.0)
    assert np.isclose(tf["a", "b"], 2.0)
    assert np.isclose(tf["b", "a"], 3.0)
    assert np.isclose(tf["b", "b"], 4.0)


def test_transformation_from_columns_and_inverse_roundtrip():
    basis = Basis(["e0", "e1"])
    columns = np.array([[1.0, 1.0], [0.0, 1.0]])
    tf = Transformation.from_columns(basis, ["f0", "f1"], columns)

    elem = Element(basis, np.array([1.0, 2.0]))
    roundtrip = tf.invert() @ (tf @ elem)

    assert tf.new_basis.names == ["f0", "f1"]
    np.testing.assert_allclose(tf._T2d, columns)
    np.testing.assert_allclose(roundtrip.coeffs, elem.coeffs)


def test_transformation_powers_and_zero_power_identity():
    basis = Basis(["a", "b"])
    T = np.array([[2.0, 0.0], [0.0, 3.0]])
    tf = Transformation(basis, basis, T, allow_singular=True)

    np.testing.assert_allclose((tf**0)._T2d, np.eye(2))
    np.testing.assert_allclose((tf**2)._T2d, T @ T)


def test_transformation_power_validation_errors():
    basis_a = Basis(["a", "b"])
    basis_b = Basis(["x", "y"])
    tf = Transformation(basis_a, basis_b, np.eye(2), allow_singular=True)

    with pytest.raises(ValueError, match="Negative powers"):
        _ = tf ** -1

    with pytest.raises(ValueError, match="old_basis == new_basis"):
        _ = tf ** 2


def test_transformation_apply_to_product_matches_product_transform():
    basis = Basis(["e0", "e1"])
    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0
    product = AlgebraProduct(basis, C)

    tf = Transformation(basis, basis, np.eye(2))
    via_method = tf.apply_to_product(product)
    via_product = product.transform(tf)

    np.testing.assert_allclose(via_method.C, via_product.C)


def test_transformation_mul_operator_is_rejected():
    basis = Basis(["a", "b"])
    tf = Transformation(basis, basis, np.eye(2))

    with pytest.raises(TypeError, match="Use the @ operator"):
        _ = tf * tf
