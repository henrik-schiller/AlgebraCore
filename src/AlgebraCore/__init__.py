"""Public facade for the future AlgebraCore split."""
from .indexing import BasisDim, Index, IndexPair, IndexTriple, normalize_index
from .basis import Basis, TensorBasis
from .contraction import (
    contract_element_product,
    contract_product_element,
    generalized_matmul,
    is_product_ndarray,
    is_square_ndarray,
)
from .element import Element, UnitElements
from .product import AlgebraProduct, TensorProduct, prod
from .transformation import Transformation
from . import std

__all__ = [
    "BasisDim",
    "Index",
    "IndexPair",
    "IndexTriple",
    "normalize_index",
    "Basis",
    "TensorBasis",
    "contract_element_product",
    "contract_product_element",
    "generalized_matmul",
    "is_product_ndarray",
    "is_square_ndarray",
    "Element",
    "UnitElements",
    "AlgebraProduct",
    "TensorProduct",
    "prod",
    "Transformation",
    "std",
]
