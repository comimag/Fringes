import numpy as np

from fringes import Fringes
from fringes.filter import curvature, direct, exposure, indirect, visibility  # todo: height

f = Fringes(Y=100)
I = f.encode()
a, b, x = f.decode(I)


def test_direct_indirect():
    d = direct(b)
    # d_max = np.max(np.abs(d - 255))
    assert np.allclose(d, f.Imax, rtol=0, atol=1.37)


def test_indirect():
    g = indirect(a, b)
    # g_max = np.max(np.abs(g))
    assert np.all(g >= 0)
    assert np.allclose(g, 0, rtol=0, atol=1.26)


def test_visibility():
    V = visibility(a, b)
    # V_max = np.max(np.abs(V - 1))
    assert np.all(V >= 0)
    assert np.allclose(V, 1, rtol=0, atol=0.005)


def test_exposure():
    E = exposure(a, I)  # todo: test lessbits
    # E_max = np.max(np.abs(E - 0.5))
    assert np.allclose(E, 0.5, rtol=0, atol=0.0005)


def test_curvature():
    f = Fringes(Y=100)
    f.v = 13, 7, 89

    I = f.encode()
    a, b, x = f.decode(I)

    c = curvature(x, center=False, normalize=False)  # todo: test center, normalize
    # c_max = np.max(np.abs(c - 2))
    assert np.allclose(c, 2, rtol=0, atol=0.05)


# def test_height():  # todo: test height
#     f = Fringes(Y=100)
#
#     dec = f.decode(f.encode())
#     c = curvature(dec.coordinate, center=False, normalize=False)
#     h = height(c)
#     assert np.allclose(h, 0, rtol=0, atol=0.1)
