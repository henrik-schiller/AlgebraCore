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
    heisenberg_lie_basis,
    matrix_basis,
    monomial_name,
    octonion_basis,
    polynomial_basis,
    quaternion_basis,
    so3_lie_basis,
    split_complex_basis,
)


class CanonicalProduct(AlgebraProduct):
    """A canonical product constant that can rebuild itself on a matching basis."""

    def __call__(self, basis: Basis | None = None) -> AlgebraProduct:
        if basis is None or basis is self.basis:
            return self
        if basis.names != self.basis.names:
            raise ValueError("Provided basis does not match the canonical basis labels.")
        return AlgebraProduct(basis, self.C.copy())


def _build_complex_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for complex numbers."""
    if basis is None:
        basis = complex_basis

    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0
    C[1, 1, 0] = -1.0
    return AlgebraProduct(basis, C)


def _build_dual_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for dual numbers."""
    if basis is None:
        basis = dual_basis

    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0
    return AlgebraProduct(basis, C)


def _build_split_complex_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for split-complex / perplex numbers."""
    if basis is None:
        basis = split_complex_basis

    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0
    C[1, 1, 0] = 1.0
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


def _build_quaternion_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for the quaternions H."""
    if basis is None:
        basis = quaternion_basis

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


def _build_octonion_product(basis: Basis | None = None) -> AlgebraProduct:
    """AlgebraProduct for the octonions via oriented Fano-plane triples."""
    if basis is None:
        basis = octonion_basis

    name_to_idx = basis.name_to_idx
    C = np.zeros((len(basis), len(basis), len(basis)), dtype=float)

    def set_prod(left: str, right: str, out: str, coeff: float = 1.0) -> None:
        C[name_to_idx[left], name_to_idx[right], name_to_idx[out]] += float(coeff)

    for name in basis.names:
        set_prod("1", name, name)
        set_prod(name, "1", name)

    for name in basis.names[1:]:
        set_prod(name, name, "1", -1.0)

    triples = [
        ("e1", "e2", "e3"),
        ("e1", "e4", "e5"),
        ("e1", "e7", "e6"),
        ("e2", "e4", "e6"),
        ("e2", "e5", "e7"),
        ("e3", "e4", "e7"),
        ("e3", "e5", "e6"),
    ]

    def add_triple(a: str, b: str, c: str) -> None:
        set_prod(a, b, c, 1.0)
        set_prod(b, c, a, 1.0)
        set_prod(c, a, b, 1.0)
        set_prod(b, a, c, -1.0)
        set_prod(c, b, a, -1.0)
        set_prod(a, c, b, -1.0)

    for triple in triples:
        add_triple(*triple)

    return AlgebraProduct(basis, C)


def _build_so3_lie_bracket(basis: Basis | None = None) -> AlgebraProduct:
    """Lie bracket for so(3) in the basis {e1, e2, e3}."""
    if basis is None:
        basis = so3_lie_basis

    name_to_idx = basis.name_to_idx
    C = np.zeros((len(basis), len(basis), len(basis)), dtype=float)

    def set_bracket(left: str, right: str, out: str, coeff: float = 1.0) -> None:
        C[name_to_idx[left], name_to_idx[right], name_to_idx[out]] += float(coeff)

    set_bracket("e1", "e2", "e3")
    set_bracket("e2", "e3", "e1")
    set_bracket("e3", "e1", "e2")
    set_bracket("e2", "e1", "e3", -1.0)
    set_bracket("e3", "e2", "e1", -1.0)
    set_bracket("e1", "e3", "e2", -1.0)

    return AlgebraProduct(basis, C)


def _build_heisenberg_lie_bracket(basis: Basis | None = None) -> AlgebraProduct:
    """Lie bracket for the 3D Heisenberg algebra with [x, y] = z."""
    if basis is None:
        basis = heisenberg_lie_basis

    name_to_idx = basis.name_to_idx
    C = np.zeros((len(basis), len(basis), len(basis)), dtype=float)
    C[name_to_idx["x"], name_to_idx["y"], name_to_idx["z"]] = 1.0
    C[name_to_idx["y"], name_to_idx["x"], name_to_idx["z"]] = -1.0
    return AlgebraProduct(basis, C)


complex_product = CanonicalProduct(complex_basis, _build_complex_product(complex_basis).C)
dual_product = CanonicalProduct(dual_basis, _build_dual_product(dual_basis).C)
split_complex_product = CanonicalProduct(
    split_complex_basis,
    _build_split_complex_product(split_complex_basis).C,
)
perplex_product = split_complex_product
quaternion_product = CanonicalProduct(
    quaternion_basis,
    _build_quaternion_product(quaternion_basis).C,
)
octonion_product = CanonicalProduct(
    octonion_basis,
    _build_octonion_product(octonion_basis).C,
)
so3_lie_bracket = CanonicalProduct(
    so3_lie_basis,
    _build_so3_lie_bracket(so3_lie_basis).C,
)
heisenberg_lie_bracket = CanonicalProduct(
    heisenberg_lie_basis,
    _build_heisenberg_lie_bracket(heisenberg_lie_basis).C,
)

# Backward-compatible aliases for older naming.
so3_lie_product = so3_lie_bracket
heisenberg_lie_product = heisenberg_lie_bracket
