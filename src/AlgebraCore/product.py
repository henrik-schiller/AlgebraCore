from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Union, TYPE_CHECKING, Iterable

import numpy as np

from AlgebraCore.contraction import contract_product_element

from AlgebraCore.basis import Basis
from AlgebraCore.basis import TensorBasis
from AlgebraCore.element import Element

# Use TYPE_CHECKING to avoid circular import at runtime
if TYPE_CHECKING:
    from AlgebraCore.transformation import Transformation

Number = float | int
MulEntry = Union[str, Dict[str, Number]]
MulTable = Dict[Tuple[str, str], MulEntry]



# ----------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------
# N=512 results in a (512, 512, 512) float array which is ~1 GB of RAM.
# This allows up to Clifford(9) or Matrices M_22(R).
MAX_BASIS_DIMENSION = 512 


@dataclass
class AlgebraProduct:
    """
    AlgebraProduct = (basis, structure constants C[i,j,k]).

    - basis: Basis
    - C: numpy array, shape (n, n, n), where n = len(basis)
    """
    basis: Basis
    C: np.ndarray

    def __init__(self, basis: Basis, C: np.ndarray, allow_large: bool = False):
        self.basis = basis

        C = np.asarray(C, dtype=float)
        n = len(basis)
        
        # --- SAFETY CHECK ---
        if n > MAX_BASIS_DIMENSION and not allow_large:
            # Calculate approx RAM usage (8 bytes per float)
            mem_size_gb = (n**3 * 8) / (1024**3)
            raise ValueError(
                f"Algebra dimension {n} is too large for a dense AlgebraProduct.\n"
                f"This would require a ({n}x{n}x{n}) tensor (~{mem_size_gb:.2f} GB RAM).\n"
                f"Limit is {MAX_BASIS_DIMENSION}.\n"
                "To override, pass 'allow_large=True' to AlgebraProduct constructor."
            )
        # --------------------

        if C.shape != (n, n, n):
            raise ValueError(
                f"Structure constants must have shape ({n}, {n}, {n}), got {C.shape}"
            )

        self.C = np.ascontiguousarray(C)

        # helpers
        self.basis_names = list(self.basis)
        self.name_to_idx = {name: i for i, name in enumerate(self.basis_names)}
        self.n = n

    def __repr__(self) -> str:
        return f"AlgebraProduct(basis_dim={self.n}, shape={self.C.shape})"

    # ------------------------------------------------------------------
    # Convert structure constants → symbolic multiplication table (dict)
    # ------------------------------------------------------------------
    def to_table(self) -> MulTable:
        """
        Convert structure constants C[i,j,k] into a symbolic table:

            {
                (a, b): "c"
            }

        or:

            {
                (a, b): {"c": 1.0, "d": -2.0}
            }
        """
        table: MulTable = {}
        names = self.basis_names
        C = self.C
        n = self.n

        for i in range(n):
            for j in range(n):
                # collect all outputs for basis[i] * basis[j]
                outputs: Dict[str, float] = {}
                for k in range(n):
                    coef = C[i, j, k]
                    if coef != 0.0:
                        outputs[names[k]] = float(coef)

                if not outputs:
                    # product = 0 → skip
                    continue

                # simplify: if exactly one nonzero and coef == 1.0 → use string
                if len(outputs) == 1:
                    (name, val), = outputs.items()
                    if val == 1.0:
                        table[(names[i], names[j])] = name
                        continue

                # otherwise store full dict
                table[(names[i], names[j])] = outputs

        return table

    # ------------------------------------------------------------------
    # Pretty-printed multiplication table as a matrix
    # ------------------------------------------------------------------
    def to_table_string(self) -> str:
        """
        Return an ASCII matrix-style table of the product.
        """
        # --- PRINT PROTECTION ---
        if self.n > 20:
             return f"<AlgebraProduct: {self.n}x{self.n} matrix is too large to print nicely>"
        # ------------------------

        names = self.basis_names
        n = self.n
        C = self.C

        # build cell strings
        def cell_str(i: int, j: int) -> str:
            parts = []
            for k in range(n):
                coef = C[i, j, k]
                if coef == 0:
                    continue
                name = names[k]
                if coef == 1:
                    parts.append(name)
                elif coef == -1:
                    parts.append(f"-{name}")
                else:
                    parts.append(f"{coef:g}*{name}")
            return "0" if not parts else " + ".join(parts)

        cells: list[list[str]] = [
            [cell_str(i, j) for j in range(n)]
            for i in range(n)
        ]

        # column widths (max over header + column cells)
        col_widths = []
        for j in range(n):
            col_vals = [cells[i][j] for i in range(n)]
            col_widths.append(
                max(len(names[j]), *(len(v) for v in col_vals))
            )

        row_label_width = max(len("·"), *(len(name) for name in names))

        lines: list[str] = []

        # header row
        header = " " * (row_label_width + 3) + " ".join(
            f"{names[j]:>{col_widths[j]}}"
            for j in range(n)
        )
        lines.append(header)

        # separator
        lines.append("-" * len(header))

        # body rows: left factor · right factor
        for i in range(n):
            row = f"{names[i]:>{row_label_width}} · " + " ".join(
                f"{cells[i][j]:>{col_widths[j]}}"
                for j in range(n)
            )
            lines.append(row)

        return "\n".join(lines)

    def __str__(self) -> str:
        # Nice table by default when you do print(P)
        return self.to_table_string()

    # ------------------------------------------------------------------
    # Basis transformation
    # ------------------------------------------------------------------
    def transform(self, tf: "Transformation") -> "AlgebraProduct":
        """
        Change basis of the product using a Transformation object.

        If C are the old structure constants for basis {e_i}, and the transformation
        defines new basis {f_a} via f_a = sum_i e_i * T[i, a], then the new
        structure constants C' for {f_a} are:

            C'[a,b,c] = sum_{i,j,k} T[i,a] T[j,b] C[i,j,k] (T^{-1})[c,k]
        
        Args:
            tf: Transformation where tf.old_basis matches self.basis.
            
        Returns:
            New AlgebraProduct with tf.new_basis and transformed constants.
        """
        # Local import to prevent circular dependency
        # AlgebraProduct -> Transformation -> AlgebraProduct
        from AlgebraCore.transformation import Transformation

        if not isinstance(tf, Transformation):
             raise TypeError(f"Expected a Transformation, got {type(tf)}")

        if tf.old_basis is not self.basis:
             raise ValueError("Transformation source basis must match the product's current basis")

        T = tf._T2d
        new_basis = tf.new_basis

        # We assume Transformation has verified T is invertible and square.
        T_inv = np.linalg.inv(T)
        C_old = self.C
        
        # Tensor contraction for basis change
        C_new = np.einsum("ia,jb,ijk,ck->abc", T, T, C_old, T_inv, optimize=True)

        return AlgebraProduct(new_basis, C_new)

    # ------------------------------------------------------------------
    # Elementwise arithmetic on structure constants
    # ------------------------------------------------------------------
    def _check_compatible(self, other: "AlgebraProduct"):
        if not isinstance(other, AlgebraProduct):
            raise TypeError("Expected a AlgebraProduct")
        if other.basis is not self.basis:
            raise ValueError("AlgebraProducts must share the same Basis object")

    def __add__(self, other: "AlgebraProduct") -> "AlgebraProduct":
        self._check_compatible(other)
        return AlgebraProduct(self.basis, self.C + other.C)

    def __sub__(self, other: "AlgebraProduct") -> "AlgebraProduct":
        self._check_compatible(other)
        return AlgebraProduct(self.basis, self.C - other.C)

    def __neg__(self) -> "AlgebraProduct":
        return AlgebraProduct(self.basis, -self.C)

    def __mul__(self, other: Union["AlgebraProduct", Number]) -> "AlgebraProduct":
        if isinstance(other, AlgebraProduct):
            raise TypeError("Element-wise multiplication of AlgebraProducts is disabled")

        if isinstance(other, (int, float)):
            return AlgebraProduct(self.basis, self.C * float(other))

        return NotImplemented

    def __rmul__(self, other: Number) -> "AlgebraProduct":
        if isinstance(other, (int, float)):
            return AlgebraProduct(self.basis, float(other) * self.C)
        return NotImplemented

    def __matmul__(self, other: Any):
        """
        Interpret this product as right-multiplication when paired with an Element.
        """
        from AlgebraCore.element import Element
        from AlgebraCore.transformation import Transformation

        if isinstance(other, Element):
            if other.basis is not self.basis:
                raise ValueError("Element basis must match AlgebraProduct basis")
            T = contract_product_element(self.C, other.coeffs)
            return Transformation(
                old_basis=self.basis,
                new_basis=self.basis,
                T=T,
                allow_singular=True,
            )
        return NotImplemented


# ----------------------------------------------------------------------
# Compatibility helper for element-wise product
# ----------------------------------------------------------------------
def prod(a: Element, b: Element, product: AlgebraProduct) -> Element:
    """
    Multiply two Elements using an AlgebraProduct.

    Provided for backwards compatibility with older tests/helpers.
    """
    if not isinstance(product, AlgebraProduct):
        raise TypeError("product must be an AlgebraProduct")
    if not isinstance(a, Element) or not isinstance(b, Element):
        raise TypeError("a and b must be Element instances")
    if a.basis is not b.basis or a.basis is not product.basis:
        raise ValueError("Elements and product must share the same basis object")

    flat_a = a.coeffs.reshape(-1)
    flat_b = b.coeffs.reshape(-1)
    res_flat = np.einsum("i,j,ijk->k", flat_a, flat_b, product.C, optimize=True)
    res_shape = getattr(product.basis, "shape", (len(product.basis),))
    return Element(product.basis, res_flat.reshape(res_shape))


# ----------------------------------------------------------------------
# TensorProduct builder
# ----------------------------------------------------------------------
def TensorProduct(products: Iterable[AlgebraProduct]) -> AlgebraProduct:
    """
    Build a tensor-product AlgebraProduct from multiple factor products.

    Structure constants combine multiplicatively:
        (a1⊗...⊗ak) * (b1⊗...⊗bk) = (a1*b1) ⊗ ... ⊗ (ak*bk)
    """
    prods = list(products)
    if not prods:
        raise ValueError("TensorProduct requires at least one AlgebraProduct")
    for p in prods:
        if not isinstance(p, AlgebraProduct):
            raise TypeError("TensorProduct expects AlgebraProduct instances")

    sub_bases = [p.basis for p in prods]
    # Combine axes from sub-bases
    basis = TensorBasis(sub_bases)

    # Build combined structure constants via einsum over factor tensors
    Cs = [p.C for p in prods]
    num = len(Cs)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    needed = 3 * num
    if needed > len(letters):
        raise ValueError("Too many factors for einsum index generation")

    idx_a = [letters[3 * r] for r in range(num)]
    idx_b = [letters[3 * r + 1] for r in range(num)]
    idx_k = [letters[3 * r + 2] for r in range(num)]

    operands = []
    for ia, ib, ik, C in zip(idx_a, idx_b, idx_k, Cs):
        operands.append(C)

    op_strs = [f"{ia}{ib}{ik}" for ia, ib, ik in zip(idx_a, idx_b, idx_k)]
    out_str = "".join(idx_a + idx_b + idx_k)
    einsum_str = ",".join(op_strs) + "->" + out_str

    C_combined = np.einsum(einsum_str, *operands, optimize=True)

    n_sizes = [len(p.basis) for p in prods]
    N = int(np.prod(n_sizes, dtype=int))
    C_flat = C_combined.reshape(N, N, N)

    return AlgebraProduct(basis, C_flat)
