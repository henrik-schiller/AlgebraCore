import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.transformation import Transformation
from AlgebraCore.element import Element


def test_transformation_matmul_tensor_contraction():
    """
    Transformation @ Element should perform a tensor contraction over old axes
    (Einstein sum generalization of mat-vec).
    """
    old_basis = Basis([["i0", "i1"], ["j0", "j1", "j2"]])
    new_basis = Basis([["a0", "a1"], ["b0", "b1", "b2"]])

    # T shape: old_shape + new_shape
    T = np.arange(2 * 3 * 2 * 3, dtype=float).reshape(2, 3, 2, 3)
    tf = Transformation(old_basis, new_basis, T, allow_singular=True)

    v = np.arange(6, dtype=float).reshape(2, 3)  # Element coefficients on old basis
    elem = Element(old_basis, v)

    res = tf @ elem

    expected = np.tensordot(T, v, axes=((0, 1), (0, 1)))
    np.testing.assert_allclose(res.coeffs, expected)
