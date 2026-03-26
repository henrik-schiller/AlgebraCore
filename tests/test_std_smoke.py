import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.element import UnitElements
from AlgebraCore.std import complex_basis, complex_product, polynomial_basis


def test_complex_std_product_smoke():
    basis = complex_basis
    product = complex_product
    u = UnitElements(basis)

    result = (2 * u.id + 3 * u.i) @ product @ (-1 * u.id + 4 * u.i)
    np.testing.assert_allclose(result.coeffs, (-14 * u.id + 5 * u.i).coeffs)


def test_polynomial_std_basis_names():
    basis = polynomial_basis(max_degree=3, var="x")
    assert basis.names == ["x0", "x1", "x2", "x3"]


def test_canonical_std_objects_remain_callable_for_compatibility():
    copied_basis = complex_basis()
    assert copied_basis is not complex_basis
    assert copied_basis.names == complex_basis.names

    rebound_product = complex_product(copied_basis)
    assert rebound_product is not complex_product
    assert rebound_product.basis is copied_basis
    np.testing.assert_allclose(rebound_product.C, complex_product.C)


def test_canonical_product_rejects_incompatible_basis():
    other_basis = Basis(["id", "j"])
    try:
        complex_product(other_basis)
    except ValueError as exc:
        assert "canonical basis labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible basis")
