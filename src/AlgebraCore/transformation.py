from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING, Any, Tuple
import numpy as np

from AlgebraCore.contraction import generalized_matmul

from AlgebraCore.basis import Basis, TensorBasis

if TYPE_CHECKING:
    from AlgebraCore.element import Element
    from AlgebraCore.product import AlgebraProduct


@dataclass(frozen=True)
class Transformation:
    old_basis: Basis
    new_basis: Basis
    T: np.ndarray          # shape = old_basis.shape + new_basis.shape
    # Prefer .coeffs to access the underlying array; .T is kept for compatibility.
    allow_singular: bool = False

    def __post_init__(self):
        T_arr = np.asarray(self.T, dtype=float)
        old_shape = getattr(self.old_basis, "shape", (len(self.old_basis),))
        new_shape = getattr(self.new_basis, "shape", (len(self.new_basis),))
        expected_shape = tuple(old_shape) + tuple(new_shape)

        if T_arr.shape != expected_shape:
            raise ValueError(f"T must have shape {expected_shape}, got {T_arr.shape}")

        n_old = int(np.prod(old_shape, dtype=int))
        n_new = int(np.prod(new_shape, dtype=int))

        if n_old != n_new and not self.allow_singular:
            raise ValueError("Transformation supports only square (same flattened dimension) bases")

        if not self.allow_singular and n_old == n_new:
            _ = np.linalg.inv(T_arr.reshape(n_old, n_new))

        object.__setattr__(self, "T", T_arr)

    # Convenience to access flattened matrix
    @property
    def _T2d(self) -> np.ndarray:
        old_shape = getattr(self.old_basis, "shape", (len(self.old_basis),))
        new_shape = getattr(self.new_basis, "shape", (len(self.new_basis),))
        n_old = int(np.prod(old_shape, dtype=int))
        n_new = int(np.prod(new_shape, dtype=int))
        return self.T.reshape(n_old, n_new)

    @property
    def coeffs(self) -> np.ndarray:
        """Alias for the underlying array representation."""
        return self.T

    @property
    def _expected_shape(self) -> Tuple[int, ...]:
        return tuple(getattr(self.old_basis, "shape", (len(self.old_basis),))) + tuple(getattr(self.new_basis, "shape", (len(self.new_basis),)))

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_columns(
        cls,
        old_basis: Basis,
        new_names: Iterable[str],
        columns: np.ndarray,
        allow_singular: bool = False,
    ) -> Transformation:
        """
        Build a Transformation by specifying columns:

            f_a = sum_i e_i * T[i, a]

        columns has shape (n, n), its columns are the new basis in the old basis.
        """
        new_basis = Basis(list(new_names))
        return cls(
            old_basis=old_basis,
            new_basis=new_basis,
            T=np.asarray(columns, dtype=float),
            allow_singular=allow_singular,
        )

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------
    def __matmul__(self, other):
        """
        Compose with another Transformation or apply to an Element.

        Composition:
            self maps B0 -> B1, other maps B1 -> B2 => result maps B0 -> B2.

        Element application:
            self maps old_basis -> new_basis, and element is in old_basis,
            returning element expressed in new_basis.
        """
        # Compose transformations
        if isinstance(other, Transformation):
            def _compatible(b1: Basis, b2: Basis) -> bool:
                return getattr(b1, "names", None) == getattr(b2, "names", None) and getattr(b1, "shape", None) == getattr(b2, "shape", None)

            if self.new_basis is not other.old_basis and not _compatible(self.new_basis, other.old_basis):
                raise ValueError(
                    "Cannot compose transformations: Mismatching bases.\n"
                    f"Left output:  {self.new_basis.names}\n"
                    f"Right input:  {other.old_basis.names}"
                )

            T_combined = generalized_matmul(self.T, other.T)

            return Transformation(
                old_basis=self.old_basis,
                new_basis=other.new_basis,
                T=T_combined,
                allow_singular=self.allow_singular or other.allow_singular,
            )

        # Apply to element (lazy import to avoid circular import)
        try:
            from AlgebraCore.element import Element  # type: ignore
        except Exception:
            Element = None  # type: ignore

        if Element is not None and isinstance(other, Element):
            def _compatible_elem(b1: Basis, b2: Basis) -> bool:
                return getattr(b1, "names", None) == getattr(b2, "names", None) and getattr(b1, "shape", None) == getattr(b2, "shape", None)

            if other.basis is not self.old_basis and not _compatible_elem(other.basis, self.old_basis):
                raise ValueError("Element basis must match transformation old_basis (by identity or labels/shape)")
            old_shape = tuple(getattr(self.old_basis, "shape", (len(self.old_basis),)))
            axes_self = tuple(range(len(old_shape)))
            axes_other = tuple(range(len(old_shape)))
            coeffs_new = np.tensordot(self.T, other.coeffs, axes=(axes_self, axes_other))
            return Element(self.new_basis, coeffs_new)

        # Apply to product table
        try:
            from AlgebraCore.product import AlgebraProduct  # type: ignore
        except Exception:
            AlgebraProduct = None  # type: ignore

        if AlgebraProduct is not None and isinstance(other, AlgebraProduct):
            if other.basis is not self.old_basis:
                raise ValueError("AlgebraProduct basis must match transformation old_basis")
            return other.transform(self)

        return NotImplemented

    def __mul__(self, other):
        """Disallow composition with *; use @ instead."""
        if isinstance(other, Transformation):
            raise TypeError("Use the @ operator for Transformation composition")
        return NotImplemented

    # ------------------------------------------------------------------
    # Linear combinations
    # ------------------------------------------------------------------
    def __add__(self, other):
        if not isinstance(other, Transformation):
            return NotImplemented
        # Allow basis objects with matching labels/shapes even if not identical
        def _compatible(b1: Basis, b2: Basis) -> bool:
            return getattr(b1, "names", None) == getattr(b2, "names", None) and getattr(b1, "shape", None) == getattr(b2, "shape", None)

        if self.old_basis is not other.old_basis and not _compatible(self.old_basis, other.old_basis):
            raise ValueError("Cannot add Transformations with different old bases")
        if self.new_basis is not other.new_basis and not _compatible(self.new_basis, other.new_basis):
            raise ValueError("Cannot add Transformations with different new bases")
        T_sum = self.T + other.T
        return Transformation(
            old_basis=self.old_basis,
            new_basis=self.new_basis,
            T=T_sum,
            allow_singular=self.allow_singular or other.allow_singular,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    # ------------------------------------------------------------------
    # Indexing helper
    # ------------------------------------------------------------------
    def __getitem__(self, key: Any) -> float:
        """
        Access an entry using old-axis labels followed by new-axis labels.

        Example for basis axes [['i','j'], ['k','l']]:
            tf['i', 'k', 'j', 'l'] or tf[('i','k'), ('j','l')]
        """
        if not isinstance(key, (list, tuple)):
            raise TypeError("Transformation indices must be provided as a tuple or list")
        flat = []
        for k in key:
            if isinstance(k, (list, tuple)) and len(k) == 1:
                flat.append(k[0])
            else:
                flat.append(k)
        labels = tuple(flat)
        old_axes = len(getattr(self.old_basis, "axes", [self.old_basis.names]))
        new_axes = len(getattr(self.new_basis, "axes", [self.new_basis.names]))
        if len(labels) != old_axes + new_axes:
            raise IndexError(f"Expected {old_axes + new_axes} indices, got {len(labels)}")

        old_labels = labels[:old_axes]
        new_labels = labels[old_axes:]
        if hasattr(self.old_basis, "index_tuple"):
            old_idx = self.old_basis.index_tuple(*old_labels)
        else:
            old_idx = tuple(old_labels)  # type: ignore[assignment]
        if hasattr(self.new_basis, "index_tuple"):
            new_idx = self.new_basis.index_tuple(*new_labels)
        else:
            new_idx = tuple(new_labels)  # type: ignore[assignment]
        return float(self.T[old_idx + new_idx])

    # ------------------------------------------------------------------
    # Basis selection
    # ------------------------------------------------------------------
    def select(
        self,
        old_names: Iterable[str] | None = None,
        new_names: Iterable[str] | None = None,
    ) -> "Transformation":
        """
        Restrict the transformation to sub-bases.

        The provided name lists define the order of rows (old basis) and
        columns (new basis) in the resulting square transformation.
        """
        old_list = list(old_names) if old_names is not None else list(self.old_basis.names)
        new_list = list(new_names) if new_names is not None else list(self.new_basis.names)

        if len(old_list) != len(new_list):
            raise ValueError("Sub-bases must have the same size to form a square Transformation")

        sub_old = self.old_basis.subbasis(old_list) if hasattr(self.old_basis, "subbasis") else Basis(old_list)  # type: ignore[arg-type]
        sub_new = self.new_basis.subbasis(new_list) if hasattr(self.new_basis, "subbasis") else Basis(new_list)  # type: ignore[arg-type]

        # Build index arrays in flattened order
        def _indices(base: Basis, names: list[str]) -> list[int]:
            return [base.name_to_idx[n] for n in names]

        idx_old = _indices(self.old_basis, old_list)
        idx_new = _indices(self.new_basis, new_list)

        T_sub = self._T2d[np.ix_(idx_old, idx_new)].reshape(len(sub_old), len(sub_new))
        return Transformation(sub_old, sub_new, T_sub, allow_singular=self.allow_singular)

    # ------------------------------------------------------------------
    # Inverse
    # ------------------------------------------------------------------
    def invert(self) -> Transformation:
        """
        Return the inverse linear transformation.

        If this maps: old_basis -> new_basis with T,
        the inverse maps: new_basis -> old_basis with T^{-1}.
        """
        T_inv_flat = np.linalg.inv(self._T2d)
        new_shape = tuple(getattr(self.new_basis, "shape", (len(self.new_basis),))) + tuple(getattr(self.old_basis, "shape", (len(self.old_basis),)))
        T_inv = T_inv_flat.reshape(new_shape)
        return Transformation(
            old_basis=self.new_basis,
            new_basis=self.old_basis,
            T=T_inv,
            allow_singular=self.allow_singular,
        )

    # ------------------------------------------------------------------
    # Tensor product
    # ------------------------------------------------------------------
    def tensor_with(self, other: "Transformation") -> "Transformation":
        """
        Tensor two transformations factorwise:
        (A->B) ⊗ (C->D) = (A⊗C -> B⊗D).
        """
        if not isinstance(other, Transformation):
            raise TypeError("tensor_with expects another Transformation")

        basis_old = TensorBasis([self.old_basis, other.old_basis])
        basis_new = TensorBasis([self.new_basis, other.new_basis])

        # Kronecker product on flattened representations, then reshape back
        T_flat = np.kron(self._T2d, other._T2d)
        combined_shape = tuple(getattr(basis_old, "shape", (len(basis_old),))) + tuple(
            getattr(basis_new, "shape", (len(basis_new),))
        )
        T_combined = T_flat.reshape(combined_shape)

        return Transformation(
            old_basis=basis_old,
            new_basis=basis_new,
            T=T_combined,
            allow_singular=self.allow_singular or other.allow_singular,
        )

    def __and__(self, other: "Transformation") -> "Transformation":
        """Shorthand for tensor_with."""
        if not isinstance(other, Transformation):
            return NotImplemented
        return self.tensor_with(other)

    # ------------------------------------------------------------------
    # Powers
    # ------------------------------------------------------------------
    def __pow__(self, exponent: int) -> Transformation:
        """
        Compose the transformation with itself `exponent` times.

        Requires square (old_basis == new_basis).
        """
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        if exponent < 0:
            raise ValueError("Negative powers are not supported; use invert() explicitly")
        if self.old_basis is not self.new_basis:
            raise ValueError("Transformation powers require old_basis == new_basis")
        if exponent == 0:
            size = int(np.prod(getattr(self.old_basis, "shape", (len(self.old_basis),)), dtype=int))
            T_eye = np.eye(size, dtype=float).reshape(self._expected_shape)
            return Transformation(self.old_basis, self.new_basis, T_eye, allow_singular=False)
        if exponent == 1:
            return self
        result = self
        for _ in range(exponent - 1):
            result = result @ self
        return result

    # ------------------------------------------------------------------
    # Application to product
    # ------------------------------------------------------------------
    def apply_to_product(self, product: AlgebraProduct) -> AlgebraProduct:
        """
        Change basis of a AlgebraProduct using this transformation.

        Requires product.basis is self.old_basis.
        """
        if product.basis is not self.old_basis:
            raise ValueError("product.basis must match old_basis")
        
        # Updated to pass the Transformation object itself
        return product.transform(self)

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------
    def _matrix_to_str(self, M: np.ndarray) -> str:
        """Format matrix in aligned ASCII form."""
        rows = []
        for row in M:
            rows.append("  [" + "  ".join(f"{val:8.4g}" for val in row) + " ]")
        return "\n".join(rows)

    def __str__(self) -> str:
        header = (
            "Transformation:\n"
            f"  old_basis = {self.old_basis.names}\n"
            f"  new_basis = {self.new_basis.names}\n"
            f"  T (columns = new basis in old basis):\n"
        )
        T_str = self._matrix_to_str(self._T2d)
        return header + T_str

    def __repr__(self) -> str:
        return self.__str__()
