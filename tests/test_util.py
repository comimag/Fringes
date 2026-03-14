import numpy as np
import pytest

from fringes import Fringes
from fringes.util import circular_distance, fade, gamma_auto_correct, unzip, vshape


@pytest.mark.parametrize(
    "shape,videoshape",
    [
        (100, (100, 1, 1, 1)),
        ((1200, 1920), (1, 1200, 1920, 1)),
        ((1200, 1920, 3), (1, 1200, 1920, 3)),
        ((10, 1200, 1920), (10, 1200, 1920, 1)),
        ((10, 1200, 1920, 3), (10, 1200, 1920, 3)),
        ((2, 3, 4, 1200, 1920), (24, 1200, 1920, 1)),
    ],
)
def test_vshape(shape, videoshape):
    data = np.empty(shape)
    assert vshape(data).shape == videoshape


def test_unzip():
    f = Fringes(Y=100)
    I = f.encode()

    Irec = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # zip
    Irec = unzip(Irec, f.T)
    dec = f.decode(Irec)
    assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)


def test_gamma_auto_correct():
    f = Fringes(Y=100)
    f.dtype = float
    I = f.encode()

    f.g = 2.2
    Iscreen = f.encode()
    Irec = gamma_auto_correct(Iscreen)
    assert np.allclose(I, Irec, rtol=0, atol=1e-6)


# @pytest.mark.parametrize("PSF", [0, 10])
def test_fade():
    f = Fringes()
    f.dtype = float
    f.V = 0.8

    I = f.encode()
    seed = 268664434431581513926327163960690138719
    Irec = fade(I, rng=seed)
    dec = f.decode(Irec)
    # da_max = np.max(np.abs(dec.a - f.A * 255))
    # db_max = np.max(np.abs(dec.b - f.B * 255))
    # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
    # i0 = np.unique(np.argwhere(np.abs(dec.x[0] - f.x[0]) > 2)[:, 1])
    # i1 = np.unique(np.argwhere(np.abs(dec.x[1] - f.x[1]) > 2)[:, 0])
    # dx_max = np.max(np.abs(dec.x[:, 4:, 2:-1, :] - f.x[:, 4:, 2:-1, :]))
    assert np.allclose(dec.a, f.A * 255, rtol=0, atol=4.3)
    assert np.allclose(dec.b, f.B * 255, rtol=0, atol=8.8)
    assert np.allclose(dec.x[:, 4:, 2:-1, :], f.x[:, 4:, 2:-1, :], rtol=0, atol=1.9)  # todo: index 0


def test_circular_distance():
    f = Fringes(Y=100)
    I = f.encode()
    dec = f.decode(I)
    x = dec.x

    x[0, :, 0, :] = f.X
    cdx = circular_distance(x[0], f.x[0], f.X)
    # cdx_max = np.max(np.abs(cdx))
    assert np.all(cdx < 0.13)

    x[1, 0, :, :] = f.Y
    cdy = circular_distance(x[1], f.x[1], f.Y)
    # cdy_max = np.max(np.abs(cdy))
    assert np.all(cdy < 0.13)
