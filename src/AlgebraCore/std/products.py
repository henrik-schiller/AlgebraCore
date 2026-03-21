from __future__ import annotations

from typing import Dict, List

import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.product import AlgebraProduct
from .bases import (
    basis_name,
    complex_basis,
    dual_basis,
    generate_exponents,
    matrix_basis,
    monomial_name,
    polynomial_basis,
    quaternion_basis,
)


def complex_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for complex numbers."""
    if basis is None:
        basis = complex_basis()

    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0
    C[1, 1, 0] = -1.0
    return AlgebraProduct(basis, C)


def dual_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for dual numbers."""
    if basis is None:
        basis = dual_basis()

    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0
    return AlgebraProduct(basis, C)


def matrix_product(n: int, basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for the matrix algebra M_n(R)."""
    if n <= 0:
        raise ValueError("n must be positive")

    if basis is None:
        basis = matrix_basis(n)

    name_to_idx = basis.name_to_idx
    dim = len(basis)
    C = np.zeros((dim, dim, dim), dtype=float)

    for i in range(n):
        for j in range(n):
            left_name = basis_name(i, j)
            left_idx = name_to_idx[left_name]
            for k in range(n):
                for l in range(n):
                    right_name = basis_name(k, l)
                    right_idx = name_to_idx[right_name]
                    if j == k:
                        out_name = basis_name(i, l)
                        out_idx = name_to_idx[out_name]
                        C[left_idx, right_idx, out_idx] = 1.0

    return AlgebraProduct(basis, C)


def polynomial_product(
    max_degree: int = 10,
    var: str = "y",
    basis: Basis | None = None,
) -> AlgebraProduct:
    """Product table for truncated univariate polynomials."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")

    exps: List[int] = generate_exponents(max_degree)
    names = [monomial_name(e, var) for e in exps]

    if basis is None:
        basis = polynomial_basis(max_degree=max_degree, var=var)
    elif basis.names != names:
        raise ValueError("Provided basis does not match canonical polynomial basis.")

    name_to_exp: Dict[str, int] = {name: exp for name, exp in zip(names, exps)}
    C = np.zeros((len(basis), len(basis), len(basis)), dtype=float)
    idx = basis.name_to_idx

    for left_name, left_exp in name_to_exp.items():
        left_idx = idx[left_name]
        for right_name, right_exp in name_to_exp.items():
            right_idx = idx[right_name]
            out_exp = left_exp + right_exp
            if out_exp > max_degree:
                continue
            out_name = monomial_name(out_exp, var)
            out_idx = idx[out_name]
            C[left_idx, right_idx, out_idx] += 1.0

    return AlgebraProduct(basis, C)


def quaternion_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for the quaternions H."""
    if basis is None:
        basis = quaternion_basis()

    name_to_idx = basis.name_to_idx
    C = np.zeros((len(basis), len(basis), len(basis)), dtype=float)

    def set_prod(left: str, right: str, out: str | dict[str, float]) -> None:
        left_idx = name_to_idx[left]
        right_idx = name_to_idx[right]
        if isinstance(out, str):
            out_idx = name_to_idx[out]
            C[left_idx, right_idx, out_idx] += 1.0
            return
        for name, coeff in out.items():
            out_idx = name_to_idx[name]
            C[left_idx, right_idx, out_idx] += float(coeff)

    for name in basis.names:
        set_prod("1", name, name)
        set_prod(name, "1", name)

    set_prod("i", "i", {"1": -1.0})
    set_prod("j", "j", {"1": -1.0})
    set_prod("k", "k", {"1": -1.0})

    set_prod("i", "j", "k")
    set_prod("j", "i", {"k": -1.0})
    set_prod("j", "k", "i")
    set_prod("k", "j", {"i": -1.0})
    set_prod("k", "i", "j")
    set_prod("i", "k", {"j": -1.0})

    return AlgebraProduct(basis, C)
