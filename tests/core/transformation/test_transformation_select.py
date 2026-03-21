import numpy as np
import pytest

from AlgebraCore.basis import Basis
from AlgebraCore.element import Element
from AlgebraCore.transformation import Transformation


def test_transformation_select_subbasis():
    basis = Basis(["a", "b", "c"])
    T = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    tf = Transformation(basis, basis, T, allow_singular=True)

    sub = tf.select(["a", "c"], ["c", "a"])

    assert sub.old_basis.names == ["a", "c"]
    assert sub.new_basis.names == ["c", "a"]
    expected = np.array([[0.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(sub._T2d, expected)

    # Applying the sub-transformation should follow the same restricted action
    v = Element(sub.old_basis, np.array([5.0, 7.0]))
    out = sub @ v
    np.testing.assert_allclose(out.coeffs, np.array([0.0, 5.0]))


def test_transformation_select_requires_square():
    basis = Basis(["a", "b", "c"])
    tf = Transformation(basis, basis, np.eye(3), allow_singular=True)
    with pytest.raises(ValueError):
        tf.select(["a", "b"], ["a"])
