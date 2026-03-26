import numpy as np

from AlgebraCore.element import UnitElements
from AlgebraCore.std import (
    heisenberg_lie_basis,
    heisenberg_lie_product,
    octonion_basis,
    octonion_product,
    so3_lie_basis,
    so3_lie_product,
    split_complex_basis,
    split_complex_product,
)


def test_split_complex_square_is_identity():
    basis = split_complex_basis()
    product = split_complex_product(basis)
    u = UnitElements(basis)

    result = u.j @ product @ u.j
    np.testing.assert_allclose(result.coeffs, u.id.coeffs)


def test_octonion_product_is_nonassociative_showcase_example():
    basis = octonion_basis()
    product = octonion_product(basis)
    u = UnitElements(basis)

    left = (u.e1 @ product @ u.e2) @ product @ u.e4
    right = u.e1 @ product @ (u.e2 @ product @ u.e4)

    np.testing.assert_allclose(left.coeffs, u.e7.coeffs)
    np.testing.assert_allclose(right.coeffs, (-1.0 * u.e7).coeffs)


def test_so3_lie_bracket_matches_cross_product_pattern():
    basis = so3_lie_basis()
    product = so3_lie_product(basis)
    u = UnitElements(basis)

    np.testing.assert_allclose((u.e1 @ product @ u.e2).coeffs, u.e3.coeffs)
    np.testing.assert_allclose((u.e2 @ product @ u.e1).coeffs, (-1.0 * u.e3).coeffs)


def test_heisenberg_lie_bracket_has_single_nontrivial_pair():
    basis = heisenberg_lie_basis()
    product = heisenberg_lie_product(basis)
    u = UnitElements(basis)

    np.testing.assert_allclose((u.x @ product @ u.y).coeffs, u.z.coeffs)
    np.testing.assert_allclose((u.y @ product @ u.x).coeffs, (-1.0 * u.z).coeffs)
    np.testing.assert_allclose((u.x @ product @ u.z).coeffs, np.zeros_like(u.z.coeffs))
