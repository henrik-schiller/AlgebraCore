from __future__ import annotations

from itertools import combinations
from typing import List

from AlgebraCore.basis import Basis


def complex_basis() -> Basis:
    """Standard basis for complex numbers: {id, i}."""
    return Basis(["id", "i"])


def dual_basis() -> Basis:
    """Basis for dual numbers R[eps] / (eps^2): {id, eps}."""
    return Basis(["id", "eps"])


def split_complex_basis() -> Basis:
    """Basis for split-complex / perplex numbers: {id, j} with j^2 = id."""
    return Basis(["id", "j"])


def perplex_basis() -> Basis:
    """Alias for split-complex numbers."""
    return split_complex_basis()


def basis_name(i: int, j: int) -> str:
    """Name basis element E_ij as 'Eij'."""
    return f"E{i}{j}"


def basis_name_from_axes(row: str, col: str) -> str:
    """Name basis element with axis labels, e.g. E[x,y]."""
    return f"E[{row},{col}]"


def matrix_basis(n: int) -> Basis:
    """Basis for M_n(R) with basis {E_ij}."""
    if n <= 0:
        raise ValueError("n must be positive")
    names = [basis_name(i, j) for i in range(n) for j in range(n)]
    return Basis(names)


def matrix_basis_from_axes(rows: list[str], cols: list[str]) -> Basis:
    """Basis for labeled matrices M_{len(rows) x len(cols)}."""
    if not rows or not cols:
        raise ValueError("rows and cols must be non-empty")
    names = [basis_name_from_axes(r, c) for r in rows for c in cols]
    return Basis(names)


def exterior_matrix_basis(old_axes: list[str], new_axes: list[str]) -> Basis:
    """Basis for exterior-lift matrix blocks with labeled axes."""
    if len(old_axes) != len(new_axes):
        raise ValueError("old_axes and new_axes must have the same length")
    n = len(old_axes)
    names = ["m"]
    for k in range(1, n + 1):
        old_combos = ["".join(c) for c in combinations(old_axes, k)]
        new_combos = ["".join(c) for c in combinations(new_axes, k)]
        for old_combo in old_combos:
            for new_combo in new_combos:
                names.append(f"m{old_combo}{new_combo}")
    return Basis(names)


def generate_exponents(max_degree: int) -> List[int]:
    """Generate all exponents e with 0 <= e <= max_degree."""
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    return list(range(0, max_degree + 1))


def monomial_name(exp: int, var: str = "y") -> str:
    """Map an exponent to a canonical monomial basis label."""
    return f"{var}{exp}"


def polynomial_basis(max_degree: int = 10, var: str = "y") -> Basis:
    """Basis for truncated univariate polynomial algebra."""
    exps = generate_exponents(max_degree)
    names = [monomial_name(e, var) for e in exps]
    return Basis(names)


def quaternion_basis() -> Basis:
    """Basis for the quaternions H: {1, i, j, k}."""
    return Basis(["1", "i", "j", "k"])


def octonion_basis() -> Basis:
    """Basis for the octonions O: {1, e1, ..., e7}."""
    return Basis(["1", "e1", "e2", "e3", "e4", "e5", "e6", "e7"])


def so3_lie_basis() -> Basis:
    """Basis for the Lie algebra so(3): {e1, e2, e3}."""
    return Basis(["e1", "e2", "e3"])


def heisenberg_lie_basis() -> Basis:
    """Basis for the 3D Heisenberg Lie algebra: {x, y, z}."""
    return Basis(["x", "y", "z"])
