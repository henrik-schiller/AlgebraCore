import numpy as np

from AlgebraCore.basis import Basis
from AlgebraCore.product import AlgebraProduct, TensorProduct, prod
from AlgebraCore.transformation import Transformation
from AlgebraCore.element import UnitElements


def test_algebra_product_identity_transform():
    b = Basis(["e0", "e1"])
    # e0 is identity, e1 squares to zero
    C = np.zeros((2, 2, 2), dtype=float)
    C[0, 0, 0] = 1.0
    C[0, 1, 1] = 1.0
    C[1, 0, 1] = 1.0

    P = AlgebraProduct(b, C)

    tf = Transformation(old_basis=b, new_basis=b, T=np.eye(2))
    P_new = P.transform(tf)

    assert np.allclose(P_new.C, P.C)
    table = P_new.to_table()
    assert table[("e0", "e0")] == "e0"
    assert table[("e0", "e1")] == "e1"
    assert table[("e1", "e0")] == "e1"


def test_algebra_product_multi_axis_basis_shape_and_transform():
    # 2D basis: axes [['i', 'j'], ['k']]
    b = Basis([["i", "j"], ["k"]])
    shape = b.shape  # (2,1)
    n = int(np.prod(shape))

    # Structure constants for a simple commutative product with identity at index 0
    C = np.zeros((n, n, n), dtype=float)
    C[0, :, :] = np.eye(n)  # e0 * eX = eX
    C[:, 0, :] = np.eye(n)  # eX * e0 = eX

    P = AlgebraProduct(b, C)

    # Transformation: identity (flattened) reshaped to tensor form
    T = np.eye(n).reshape(shape + shape)
    tf = Transformation(old_basis=b, new_basis=b, T=T)
    P_new = P.transform(tf)

    assert P_new.C.shape == (n, n, n)
    # Identity properties preserved
    assert np.allclose(P_new.C[0], np.eye(n))


def test_tensor_product_of_two_algebras():
    # Factor A: e0 identity, e1 nilpotent (e1^2 = 0)
    bA = Basis(["e0", "e1"])
    CA = np.zeros((2, 2, 2), dtype=float)
    CA[0, :, :] = np.eye(2)
    CA[:, 0, :] = np.eye(2)
    # e1 * e1 = 0 already encoded
    PA = AlgebraProduct(bA, CA)

    # Factor B: f0 identity, f1 idempotent (f1^2 = f1)
    bB = Basis(["f0", "f1"])
    CB = np.zeros((2, 2, 2), dtype=float)
    CB[0, :, :] = np.eye(2)
    CB[:, 0, :] = np.eye(2)
    CB[1, 1, 1] = 1.0
    PB = AlgebraProduct(bB, CB)

    TP = TensorProduct([PA, PB])
    u = UnitElements(TP.basis)

    e1f1 = u.e1.f1
    res = prod(e1f1, e1f1, TP)

    # (e1 ⊗ f1)^2 = (e1^2) ⊗ (f1^2) = 0 ⊗ f1 = 0
    assert np.allclose(res.coeffs, 0.0)

    e0f1 = u.e0.f1
    res_idem = prod(e0f1, e0f1, TP)
    assert np.allclose(res_idem.coeffs, e0f1.coeffs)
