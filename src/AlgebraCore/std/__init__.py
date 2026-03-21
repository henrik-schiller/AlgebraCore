"""Small stable standard catalog for AlgebraCore."""

from .bases import (
    basis_name,
    basis_name_from_axes,
    complex_basis,
    dual_basis,
    exterior_matrix_basis,
    generate_exponents,
    matrix_basis,
    matrix_basis_from_axes,
    monomial_name,
    polynomial_basis,
    quaternion_basis,
)
from .products import (
    complex_product,
    dual_product,
    matrix_product,
    polynomial_product,
    quaternion_product,
)

__all__ = [
    "complex_basis",
    "dual_basis",
    "basis_name",
    "basis_name_from_axes",
    "matrix_basis",
    "matrix_basis_from_axes",
    "exterior_matrix_basis",
    "generate_exponents",
    "monomial_name",
    "polynomial_basis",
    "quaternion_basis",
    "complex_product",
    "dual_product",
    "matrix_product",
    "polynomial_product",
    "quaternion_product",
]
