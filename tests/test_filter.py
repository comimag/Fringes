import numpy as np

from fringes import Fringes
from fringes.filter import direct, indirect, visibility, exposure, curvature  # todo: height


def test_direct_indirect():
    f = Fringes(Y=100)

    a, b, x = f.decode(f.encode())

    d = direct(b)
    assert np.allclose(d, f.Imax, rtol=0, atol=1.5), "Direct is off more than 1.5."  # todo: 0.1

    g = indirect(a, b)
    assert np.all(g >= 0), "Global contains negative values."
    assert np.allclose(g, 0, rtol=0, atol=1.5), "Global is larger than 1.5."  # todo: 0.1


def test_visibility_exposure():
    f = Fringes(Y=100)

    I = f.encode()
    a, b, x = f.decode(I)

    V = visibility(a, b)
    assert np.all(V >= 0), "Visibility contains negative values."
    assert np.allclose(V, 1, rtol=0, atol=0.01), "Visibility is off more than 0.01."

    E = exposure(a, I)
    assert np.allclose(E, 0.5, rtol=0, atol=0.01), "Exposure is off more than 0.01."


def test_curvature():
    f = Fringes(Y=100)

    dec = f.decode(f.encode())
    c = curvature(dec.registration, center=False, normalize=True)
    mn = c.min()
    mx = c.max()
    assert np.allclose(c, 0, rtol=0, atol=0.1), "Curvature if off more than 0.1."


# def test_height():  # todo: test height
#     f = Fringes(Y=100)
#
#     dec = f.decode(f.encode())
#     assert np.allclose(height(curvature(dec.registration)), 0, rtol=0, atol=0.1), "Height if off more than 0.1."
