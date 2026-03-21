import numpy as np

from AlgebraCore.element import UnitElements
from AlgebraCore.std import complex_basis, complex_product, polynomial_basis


def test_complex_std_product_smoke():
    basis = complex_basis()
    product = complex_product(basis)
    u = UnitElements(basis)

    result = (2 * u.id + 3 * u.i) @ product @ (-1 * u.id + 4 * u.i)
    np.testing.assert_allclose(result.coeffs, (-14 * u.id + 5 * u.i).coeffs)


def test_polynomial_std_basis_names():
    basis = polynomial_basis(max_degree=3, var="x")
    assert basis.names == ["x0", "x1", "x2", "x3"]
