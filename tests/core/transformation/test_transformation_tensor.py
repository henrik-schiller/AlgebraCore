import numpy as np

from AlgebraCore.basis import Basis, TensorBasis
from AlgebraCore.element import Element
from AlgebraCore.transformation import Transformation


def test_transformation_tensor_with_identity_on_second_factor():
    basis_a = Basis([["a0", "a1"]])
    basis_b = Basis([["b0", "b1"]])

    T_a = np.eye(2) * 2.0  # scales first factor by 2
    T_b = np.eye(2)

    tf_a = Transformation(basis_a, basis_a, T_a, allow_singular=True)
    tf_b = Transformation(basis_b, basis_b, T_b, allow_singular=True)

    tf = tf_a & tf_b

    basis_ab = tf.old_basis
    coeffs = np.array([[1.0, 2.0], [3.0, 4.0]])
    elem = Element(basis_ab, coeffs)

    res = tf @ elem
    expected = Element(basis_ab, coeffs * 2.0)

    np.testing.assert_allclose(res.coeffs, expected.coeffs)
