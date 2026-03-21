from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Union, TYPE_CHECKING, Callable, overload
import numpy as np

from AlgebraCore.contraction import contract_element_product

# We import Basis here. 
from AlgebraCore.basis import Basis, TensorBasis

# Use TYPE_CHECKING to avoid circular import at runtime
# Element -> Transformation -> Product -> Element
if TYPE_CHECKING:
    from AlgebraCore.transformation import Transformation
    from AlgebraCore.product import AlgebraProduct

Number = Union[int, float, np.number]



@dataclass
class Element:
    """
    Single algebra element in a fixed Basis.

    - basis: Basis (may have multiple axes)
    - coeffs: numpy array of shape basis.shape
    """
    basis: Basis
    coeffs: np.ndarray

    def __post_init__(self):
        arr = np.asarray(self.coeffs, dtype=float)
        expected_shape = getattr(self.basis, "shape", (len(self.basis),))
        if arr.shape != expected_shape:
            raise ValueError(
                f"Element coeffs must have shape {expected_shape}, got {arr.shape}"
            )
        self.coeffs = arr

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.basis)

    # --------------------------------------------------------------
    # Constructors
    # --------------------------------------------------------------
    @classmethod
    def zero(cls, basis: Basis) -> "Element":
        """Zero element in the given basis."""
        return cls(basis, np.zeros(getattr(basis, "shape", (len(basis),)), dtype=float))

    @classmethod
    def basis_vector(cls, basis: Basis, name: str) -> "Element":
        """
        Basis vector e_name in the given basis:
            Element.basis_vector(basis, "e1")
        """
        if hasattr(basis, "index_tuple"):
            try:
                idx_tuple = basis.index_tuple(name)
            except Exception:
                raise KeyError(f"Unknown basis name {name!r}")
        else:
            if name not in basis.name_to_idx:
                raise KeyError(f"Unknown basis name {name!r}")
            idx_tuple = basis.name_to_idx[name]
        coeffs = np.zeros(getattr(basis, "shape", (len(basis),)), dtype=float)
        coeffs[idx_tuple] = 1.0
        return cls(basis, coeffs)

    # --------------------------------------------------------------
    # Indexing
    # --------------------------------------------------------------
    def __getitem__(self, key: Any) -> float:
        if isinstance(key, str):
            idx_tuple = self.basis.index_tuple(key) if hasattr(self.basis, "index_tuple") else self.basis.name_to_idx[key]
            return float(self.coeffs[idx_tuple])
        if isinstance(key, (list, tuple)):
            idx_tuple = self.basis.index_tuple(*key) if hasattr(self.basis, "index_tuple") else tuple(key)
            return float(self.coeffs[idx_tuple])
        return float(self.coeffs[key])

    def __setitem__(self, key: Any, value: Number) -> None:
        if isinstance(key, str):
            idx_tuple = self.basis.index_tuple(key) if hasattr(self.basis, "index_tuple") else self.basis.name_to_idx[key]
            self.coeffs[idx_tuple] = float(value)
        elif isinstance(key, (list, tuple)):
            idx_tuple = self.basis.index_tuple(*key) if hasattr(self.basis, "index_tuple") else tuple(key)
            self.coeffs[idx_tuple] = float(value)
        else:
            self.coeffs[key] = float(value)

    # --------------------------------------------------------------
    # Linear operations
    # --------------------------------------------------------------
    def _binary(self, other: Any, op: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> "Element":
        """Helper for element-wise operations (add, sub)."""
        if isinstance(other, Element):
            if other.basis is not self.basis:
                raise TypeError("Elements belong to different bases")
            return Element(self.basis, op(self.coeffs, other.coeffs))
        return Element(self.basis, op(self.coeffs, float(other)))

    def __add__(self, other: Any) -> "Element":
        return self._binary(other, np.add)

    def __radd__(self, other: Any) -> "Element":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Element":
        return self._binary(other, np.subtract)

    def __rsub__(self, other: Any) -> "Element":
        if isinstance(other, Element):
            return other.__sub__(self)
        return Element(self.basis, float(other) - self.coeffs)

    def __neg__(self) -> "Element":
        return Element(self.basis, -self.coeffs)

    # --- Scalar Multiplication support ---

    def __mul__(self, other: Any) -> "Element":
        """
        Scalar multiplication: x * 2.0
        Note: x * y (Element * Element) is NOT supported here. 
        Use prod(x, y, P) for algebra multiplication.
        """
        if isinstance(other, (int, float, np.number)):
            return Element(self.basis, self.coeffs * float(other))
        return NotImplemented

    def __rmul__(self, other: Any) -> "Element":
        """
        Reverse scalar multiplication: 2.0 * x
        """
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Element":
        """
        Scalar division: x / 2.0
        """
        if isinstance(other, (int, float, np.number)):
            return Element(self.basis, self.coeffs / float(other))
        return NotImplemented

    def __matmul__(self, other: Any):
        """
        Interpret this element as:
        - left-multiplication linear map when paired with an AlgebraProduct.
        """
        # Local imports avoid circular dependency at module load
        from AlgebraCore.product import AlgebraProduct
        from AlgebraCore.transformation import Transformation

        if isinstance(other, AlgebraProduct):
            def _compatible(b1: Basis, b2: Basis) -> bool:
                return getattr(b1, "names", None) == getattr(b2, "names", None) and getattr(b1, "shape", None) == getattr(b2, "shape", None)

            if other.basis is not self.basis and not _compatible(other.basis, self.basis):
                raise ValueError("AlgebraProduct basis must match element basis")

            T = contract_element_product(self.coeffs, other.C)
            return Transformation(
                old_basis=self.basis,
                new_basis=self.basis,
                T=T,
                allow_singular=True,
            )

        return NotImplemented

    def tensor_with(self, other: "Element") -> "Element":
        """Tensor/outer product of two Elements."""
        if not isinstance(other, Element):
            raise TypeError("tensor_with expects another Element")
        new_basis = TensorBasis([self.basis, other.basis])
        coeffs = np.tensordot(self.coeffs, other.coeffs, axes=0)
        return Element(new_basis, coeffs)

    def __and__(self, other: Any) -> Any:
        """Shorthand for tensor/outer product of two Elements."""
        if isinstance(other, Element):
            return self.tensor_with(other)
        return NotImplemented

    # --------------------------------------------------------------
    # Transformations
    # --------------------------------------------------------------
    def transform(self, tf: "Transformation") -> "Element":
        """
        Transform this element to a new basis using a Transformation object.
        
        If [v]_old are coefficients in old basis and [v]_new in new basis:
        [v]_old = T @ [v]_new
        => [v]_new = T^{-1} @ [v]_old
        """
        # Local import prevents circular dependency cycle:
        # Element -> Transformation -> Product -> Element
        from AlgebraCore.transformation import Transformation

        if not isinstance(tf, Transformation):
             raise TypeError(f"Expected a Transformation, got {type(tf)}")

        if tf.old_basis is not self.basis:
             raise ValueError("Transformation source basis must match the element's current basis")

        T_inv = np.linalg.inv(tf._T2d)
        flat = self.coeffs.reshape(-1)
        new_flat = T_inv @ flat
        new_coeffs = new_flat.reshape(getattr(tf.new_basis, "shape", (len(tf.new_basis),)))
        return Element(tf.new_basis, new_coeffs)

    # --------------------------------------------------------------
    # Numpy interop
    # --------------------------------------------------------------
    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.coeffs, dtype=float)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.coeffs, dtype=dtype)

    # --------------------------------------------------------------
    # Pretty printing
    # --------------------------------------------------------------
    def _element_repr(self, vec: np.ndarray) -> str:
        parts = []
        for name, c in zip(self.basis.names, vec):
            if c == 0:
                continue
            # Always show an explicit coefficient
            s_val = f"{c:g}"
            parts.append(f"{s_val}*{name}")
        return " + ".join(parts) if parts else "0"

    def __repr__(self) -> str:
        flat = self.coeffs.reshape(-1)
        return self._element_repr(flat)

    def coeff(self, name: str) -> float:
        """Return the coefficient for the given basis name."""
        idx = self.basis.index_tuple(name) if hasattr(self.basis, "index_tuple") else self.basis.name_to_idx[name]
        return float(self.coeffs[idx])

    def select(self, names: Sequence[str]) -> "Element":
        """
        Project this element onto a subbasis defined by the provided names.
        Order of names is preserved.
        """
        sub_basis = self.basis.subbasis(names) if hasattr(self.basis, "subbasis") else Basis(names)
        coeffs = np.zeros(len(sub_basis), dtype=float)
        for i, n in enumerate(sub_basis.names):
            if hasattr(self.basis, "index_tuple"):
                idx = self.basis.index_tuple(n)
            else:
                idx = self.basis.name_to_idx[n]
            coeffs[i] = float(self.coeffs[idx])
        return Element(sub_basis, coeffs)


# --------------------------------------------------------------
# UnitElements Helper Classes
# --------------------------------------------------------------
@dataclass
class _UnitAccessor:
    basis: Basis
    labels: list[str]
    sep: str = "."

    def _validate_label(self, label: str, axis_idx: int) -> None:
        axis = self.basis.axes[axis_idx]
        if label not in axis:
            raise AttributeError(f"Basis axis {axis_idx} has no element named '{label}'")

    def __getattr__(self, attr: str):
        if attr.startswith("_"):
            label = attr[1:]
        else:
            label = attr

        axis_idx = len(self.labels)
        if axis_idx >= len(self.basis.axes):
            raise AttributeError(f"Too many indices for basis (got {axis_idx + 1})")

        self._validate_label(label, axis_idx)
        new_labels = self.labels + [label]

        if len(new_labels) == len(self.basis.axes):
            idx = self.basis.index_tuple(*new_labels)
            coeffs = np.zeros(self.basis.shape, dtype=float)
            coeffs[idx] = 1.0
            return Element(self.basis, coeffs)

        return _UnitAccessor(self.basis, new_labels, sep=self.sep)


@dataclass
class UnitElements:
    """
    Helper factory to access basis unit vectors as chained attributes.

    Example:
        u = UnitElements(Basis([['i', 'j'], ['k', 'l']]))
        u.i.k -> Element with coeff 1 at ('i', 'k')
        u.i_k -> flattened name if sep is set to "_"
    """
    basis: Basis
    sep: str = "."

    def __getattr__(self, attr: str):
        # Handle numeric-leading names via underscore prefix (_1 -> "1")
        if attr.startswith("_"):
            name = attr[1:]
        else:
            name = attr

        # Flattened name direct access
        if name in getattr(self.basis, "name_to_idx", {}):
            idx = self.basis.index_tuple(name) if hasattr(self.basis, "index_tuple") else self.basis.name_to_idx[name]
            coeffs = np.zeros(getattr(self.basis, "shape", (len(self.basis),)), dtype=float)
            coeffs[idx] = 1.0
            return Element(self.basis, coeffs)

        # Otherwise treat as first axis label in a chain
        accessor = _UnitAccessor(self.basis, [], sep=self.sep)
        return getattr(accessor, attr)
