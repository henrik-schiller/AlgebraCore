import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.element import Element, UnitElements


def test_element_shape_and_indexing():
    b = Basis([["i", "j"], ["k", "l"]])
    e = Element.zero(b)
    assert e.coeffs.shape == (2, 2)

    e["i", "k"] = 1.5
    assert np.isclose(e["i", "k"], 1.5)
    assert np.isclose(e.coeffs[b.index_tuple("i", "k")], 1.5)

    e["j.l"] = 2.0
    assert np.isclose(e["j", "l"], 2.0)

    v = Element.basis_vector(b, "i.k")
    assert np.isclose(v["i", "k"], 1.0)
    assert np.isclose(v.coeffs.sum(), 1.0)


def test_unit_elements_chained_access():
    b = Basis([["a", "b"], ["x", "y"], ["p"]])
    u = UnitElements(b)

    elem = u.a.x.p
    assert isinstance(elem, Element)
    idx = b.index_tuple("a", "x", "p")
    assert np.isclose(elem.coeffs[idx], 1.0)
    assert np.isclose(elem.coeffs.sum(), 1.0)

    # Flattened/chained access
    elem2 = u.a.x.p
    assert np.isclose(elem2.coeffs[idx], 1.0)
    assert np.isclose(elem2.coeffs.sum(), 1.0)
