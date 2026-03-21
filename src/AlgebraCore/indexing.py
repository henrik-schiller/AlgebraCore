from __future__ import annotations

from typing import Sequence, Union

DimSize = Union[int, float]
BasisDim = tuple[DimSize, ...]
Index = tuple[int, ...]
IndexPair = tuple[Index, Index]
IndexTriple = tuple[Index, Index, Index]


def normalize_index(idx: Index | Sequence[int] | int) -> Index:
    """
    Coerce an incoming index representation into a tuple of ints.
    """
    if isinstance(idx, tuple):
        return idx
    if isinstance(idx, int):
        return (idx,)
    if isinstance(idx, Sequence):
        return tuple(idx)
    return (idx,)
