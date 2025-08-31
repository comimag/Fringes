import numpy as np

from fringes import Fringes
from fringes.util import vshape, unzip, gamma_auto_correct, circular_distance


def test_vshape():
    data = np.ones(shape=(100))
    assert vshape(data).shape == (100, 1, 1, 1)

    data = np.ones(shape=(1200, 1920))
    assert vshape(data).shape == (1, 1200, 1920, 1)

    data = np.ones(shape=(1200, 1920, 3))
    assert vshape(data).shape == (1, 1200, 1920, 3)

    data = np.ones(shape=(100, 1200, 1920))
    assert vshape(data).shape == (100, 1200, 1920, 1)

    data = np.ones(shape=(100, 1200, 1920, 3))
    assert vshape(data).shape == (100, 1200, 1920, 3)

    data = np.ones(shape=(2, 3, 4, 1200, 1920))
    assert vshape(data).shape == (24, 1200, 1920, 1)


def test_deinterlace():
    f = Fringes(Y=100)

    I = f.encode()
    Irec = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # zip
    Irec = unzip(Irec, f.T)
    dec = f.decode(Irec)

    assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."


def test_gamma_auto_correct():
    f = Fringes(Y=100)

    f.dtype = "float32"
    I = f.encode()

    f.g = 2.2
    Irec = f.encode()
    Irec = gamma_auto_correct(Irec)

    assert np.allclose(I, Irec, rtol=0, atol=1e-6), f"Gamma correction is off more than {1e-6:.6f}."


def test_circular_distance():
    f = Fringes(Y=100)
    I = f.encode()
    dec = f.decode(I)
    x = dec.x

    x[0, :, 0, :] = f.X
    cdx = circular_distance(x[0], f.xc[0], f.X)
    # cdx_max = np.max(np.abs(cdx))
    assert np.allclose(cdx, 0, rtol=0, atol=0.13), "Circular distance is off more than 0.13."

    x[1, 0, :, :] = f.Y
    cdy = circular_distance(x[1], f.xc[1], f.Y)
    # cdy_max = np.max(np.abs(cdy))
    assert np.allclose(cdy[0], 0, rtol=0, atol=0.13), "Circular distance is off more than 0.13."
