import numpy as np

from fringes import Fringes
from fringes.filter import direct, indirect, visibility, exposure, curvature  # todo: height


f = Fringes(Y=100)
I = f.encode()


def test_direct_indirect():
    a, b, x = f.decode(I)

    d = direct(b)
    # d_max = np.max(np.abs(d - 255))
    assert np.allclose(d, f.Imax, rtol=0, atol=1.37), "Direct is off more than 1.37."

    g = indirect(a, b)
    # g_max = np.max(np.abs(g))
    assert np.all(g >= 0), "Global contains negative values."
    assert np.allclose(g, 0, rtol=0, atol=1.26), "Global is larger than 1.26."


def test_visibility_exposure():
    a, b, x = f.decode(I)

    V = visibility(a, b)
    # V_max = np.max(np.abs(V - 1))
    assert np.all(V >= 0), "Visibility contains negative values."
    assert np.allclose(V, 1, rtol=0, atol=0.005), "Visibility is off more than 0.005."

    E = exposure(a, I)
    # E_max = np.max(np.abs(E - 0.5))
    assert np.allclose(E, 0.5, rtol=0, atol=0.0005), "Exposure is off more than 0.0005."


def test_curvature():
    f = Fringes(Y=100)
    f.v = 13, 7, 89

    I = f.encode()
    a, b, x = f.decode(I)

    c = curvature(x, center=False, normalize=False)
    # c_max = np.max(np.abs(c - 2))
    assert np.allclose(c, 2, rtol=0, atol=0.05), "Curvature if off more than 0.05."


# def test_height():  # todo: test height
#     f = Fringes(Y=100)
#
#     dec = f.decode(f.encode())
#     c = curvature(dec.coordinate, center=False, normalize=False)
#     h = height(c)
#     assert np.allclose(h, 0, rtol=0, atol=0.1), "Height if off more than 0.1."
