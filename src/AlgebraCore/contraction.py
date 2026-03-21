from __future__ import annotations

import numpy as np


def is_square_ndarray(arr: np.ndarray) -> bool:
    """
    Return True if arr is a 2n-D array with matching first/second halves.
    """
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim % 2 != 0:
        return False
    n = arr.ndim // 2
    return tuple(arr.shape[:n]) == tuple(arr.shape[n:])


def is_product_ndarray(arr: np.ndarray) -> bool:
    """
    Return True if arr is a 3n-D array with matching three blocks of axes.
    """
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim % 3 != 0:
        return False
    n = arr.ndim // 3
    shape_a = tuple(arr.shape[:n])
    shape_b = tuple(arr.shape[n : 2 * n])
    shape_c = tuple(arr.shape[2 * n : 3 * n])
    return shape_a == shape_b == shape_c


def generalized_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Generalized matrix multiplication for 2n-D ndarrays.

    Contracts the last half of left with the first half of right.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != right.ndim or left.ndim % 2 != 0:
        raise ValueError("Inputs must have the same even rank for generalized matmul")
    if not is_square_ndarray(left) or not is_square_ndarray(right):
        raise ValueError("Inputs must have matching first/second halves for generalized matmul")
    n = left.ndim // 2
    if left.shape[n:] != right.shape[:n]:
        raise ValueError("Shapes are incompatible for generalized matmul")
    axes_left = tuple(range(n, 2 * n))
    axes_right = tuple(range(0, n))
    return np.tensordot(left, right, axes=(axes_left, axes_right))


def contract_element_product(elem: np.ndarray, product_array: np.ndarray) -> np.ndarray:
    """
    Contract a 1n-D element with a 3n-D product array to produce a 2n-D result.
    """
    coeffs = np.asarray(elem, dtype=float)
    C = np.asarray(product_array, dtype=float)
    if C.ndim != 3 * coeffs.ndim:
        raise ValueError("Product array must have 3n dimensions for an n-D element")
    if not is_product_ndarray(C):
        raise ValueError("Product array must have matching 3n axis blocks")
    n = coeffs.ndim
    if C.shape[:n] != coeffs.shape:
        raise ValueError("Element shape does not match product tensor input axes")

    axes_coeffs = tuple(range(n))
    axes_C = tuple(range(n))
    return np.tensordot(coeffs, C, axes=(axes_coeffs, axes_C))


def contract_product_element(product_array: np.ndarray, elem: np.ndarray) -> np.ndarray:
    """
    Contract a 3n-D product array with a 1n-D element over the middle block to produce a 2n-D result.
    """
    coeffs = np.asarray(elem, dtype=float)
    C = np.asarray(product_array, dtype=float)
    if C.ndim != 3 * coeffs.ndim:
        raise ValueError("Product array must have 3n dimensions for an n-D element")
    if not is_product_ndarray(C):
        raise ValueError("Product array must have matching 3n axis blocks")
    n = coeffs.ndim
    if C.shape[n:2 * n] != coeffs.shape:
        raise ValueError("Element shape does not match product tensor right axes")

    axes_C = tuple(range(n, 2 * n))
    axes_coeffs = tuple(range(n))
    return np.tensordot(C, coeffs, axes=(axes_C, axes_coeffs))
