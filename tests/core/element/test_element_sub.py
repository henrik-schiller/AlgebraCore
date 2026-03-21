import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.element import Element


def test_element_select_subbasis():
    basis = Basis([["id", "e1", "e2"]])
    coeffs = np.array([1.0, 2.0, 3.0])
    elem = Element(basis, coeffs)

    sub = elem.select(["e2", "id"])
    assert sub.basis.names == ["e2", "id"]
    np.testing.assert_allclose(sub.coeffs, np.array([3.0, 1.0]))
