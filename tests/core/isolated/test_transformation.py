import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.element import Element
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
