from AlgebraCore.basis import Basis


def test_basis_accepts_varied_names():
    # Previously invalid names should now be accepted
    b = Basis(["id^2", "id.2", "1er", "i"])
    assert b.names == ["id^2", "id.2", "1er", "i"]
