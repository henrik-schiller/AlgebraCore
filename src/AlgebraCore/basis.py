from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Dict, Sequence, Tuple
import numpy as np
from AlgebraCore.indexing import BasisDim, Index

@dataclass
class Basis:

    axes: List[List[str]]
    names: List[str]
    name_to_idx: Dict[str, int]
    shape: BasisDim
    rank: int

    @staticmethod
    def _validate_name(n: str) -> None:
        if not isinstance(n, str):
            raise TypeError(f"Basis name must be a string, got {type(n)}")
        if len(n.strip()) == 0:
            raise ValueError("Basis names must not be empty")

    def __init__(self, axes: Iterable[Iterable[str] | str]):
        """
        Initialize a Basis with one or more axes of labels.

        - If a flat iterable of strings is provided, it is treated as a single axis.
        - If a sequence of iterables is provided, each iterable is an axis of labels.
        """
        axes_list: List[List[str]] = []

        # Allow backward-compatible single-axis usage: Basis(["a", "b"])
        candidate = list(axes)
        if candidate and all(isinstance(x, str) for x in candidate):
            candidate = [candidate]  # wrap single axis

        for axis in candidate:
            axis_names = list(axis)  # type: ignore[arg-type]
            if not axis_names:
                raise ValueError("Each axis must contain at least one name")
            for n in axis_names:
                self._validate_name(n)
            if len(set(axis_names)) != len(axis_names):
                raise ValueError(f"Duplicate labels detected within an axis: {axis_names}")
            axes_list.append(axis_names)

        if not axes_list:
            raise ValueError("Basis must have at least one axis of names")

        self.axes = axes_list
        self.shape = tuple(len(ax) for ax in self.axes)
        self.rank = len(self.axes)
        self.names = self.flatten()
        self.name_to_idx = {name: i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def subbasis(self, names: Sequence[str]) -> "Basis":
        """
        Return a Basis restricted to the given ordered list of names.
        """
        names = list(names)
        for n in names:
            if n not in self.name_to_idx:
                raise KeyError(f"Name {n!r} not in basis")
        return Basis(names)

    def __iter__(self):
        return iter(self.names)

    def __getitem__(self, idx):
        """
        Access by flat index or by multi-index (tuple of axis indices).
        """
        if isinstance(idx, tuple):
            return self.at(idx)
        if len(self.axes) != 1:
            raise IndexError("Flat indexing only supported for single-axis bases; use a tuple for multi-axis access")
        return self.axes[0][idx]

    def __repr__(self) -> str:
        return f"Basis(shape={self.shape}, axes={self.axes})"

    @property
    def n(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def index_tuple(self, *labels: int | str, sep: str = ".") -> tuple[int, ...]:
        """
        Resolve a mix of ints/strings into a multi-index tuple.

        Strings are matched against the corresponding axis labels.
        If a single string is provided and matches a flattened name, it is expanded.
        """
        if len(labels) == 1 and isinstance(labels[0], str) and labels[0] in self.name_to_idx and len(self.axes) > 1:
            flat_idx = self.name_to_idx[labels[0]]
            return tuple(np.unravel_index(flat_idx, self.shape))  # type: ignore[name-defined]

        if len(labels) == 1 and isinstance(labels[0], tuple):
            labels = labels[0]  # type: ignore[assignment]

        if len(labels) != len(self.axes):
            raise IndexError(f"Expected {len(self.axes)} labels, got {len(labels)}")

        idxs: list[int] = []
        for label, axis in zip(labels, self.axes):
            if isinstance(label, int):
                idxs.append(label)
                continue
            if label not in axis:
                raise KeyError(f"Label '{label}' not found in axis {axis}")
            idxs.append(axis.index(label))
        return tuple(idxs)

    # ------------------------------------------------------------------
    # Multi-axis helpers
    # ------------------------------------------------------------------
    def at(self, *idxs: int | Tuple[int, ...], sep: str = ".") -> str:
        """
        Access a basis label by multi-index. Example:
            Basis([['i', 'j'], ['k', 'l']]).at(0, 1) -> "i.l" (default sep=".")
        """
        if len(idxs) == 1 and isinstance(idxs[0], tuple):
            idx_tuple = idxs[0]
        else:
            idx_tuple = tuple(int(i) for i in idxs)
        if len(idx_tuple) != len(self.axes):
            raise IndexError(f"Expected {len(self.axes)} indices, got {len(idx_tuple)}")
        parts = []
        for axis_idx, axis in zip(idx_tuple, self.axes):
            parts.append(axis[axis_idx])
        if len(parts) == 1:
            return parts[0]
        return sep.join(parts)

    def flatten(self, sep: str = ".") -> List[str]:
        """
        Flatten the multi-axis labels into a single list by cartesian product.

        Example:
            axes = [['i', 'j'], ['k', 'l']] -> ["i.k", "i.l", "j.k", "j.l"]
        """
        combos = product(*self.axes)
        return [sep.join(names) if len(names) > 1 else names[0] for names in combos]


# ----------------------------------------------------------------------
# Tensor basis helper
# ----------------------------------------------------------------------
def TensorBasis(bases: Iterable[Basis]) -> Basis:
    """
    Build a tensor-product Basis from multiple Basis objects by concatenating axes.

    Example:
        TensorBasis([Basis(["a", "b"]), Basis(["x", "y"])]) -> Basis with axes [["a","b"], ["x","y"]]
    """
    bases_list = list(bases)
    if not bases_list:
        raise ValueError("TensorBasis requires at least one Basis")
    axes: list[list[str]] = []
    for b in bases_list:
        if not isinstance(b, Basis):
            raise TypeError("TensorBasis expects Basis instances")
        axes.extend(b.axes)
    return Basis(axes)
