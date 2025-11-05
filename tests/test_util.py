import cv2
import numpy as np

from fringes import Fringes
from fringes.util import vshape, unzip, gamma_auto_correct, circular_distance, fade


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


def test_unzip():
    f = Fringes(Y=100)
    I = f.encode()

    Irec = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # zip
    Irec = unzip(Irec, f.T)
    dec = f.decode(Irec)
    assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13)


def test_gamma_auto_correct():
    f = Fringes(Y=100)
    f.dtype = float
    I = f.encode()

    f.g = 2.2
    Iscreen = f.encode()
    Irec = gamma_auto_correct(Iscreen)
    assert np.allclose(I, Irec, rtol=0, atol=1e-6)


def test_fade():
    f = Fringes()
    f.Y = f.X = f.Lmax
    f.dtype = float
    f.V = 0.8
    I = f.encode()

    seed = 268664434431581513926327163960690138719

    Irec = fade(I, PSF=0, rng=seed)
    dec = f.decode(Irec)
    # da_max = np.max(np.abs(dec.a - f.A * 255))
    # db_max = np.max(np.abs(dec.b - f.B * 255))
    # i0 = np.unique(np.argwhere(np.abs(dec.x[0] - f.xc[0]) > f.L / 2)[:, 1])
    # i1 = np.unique(np.argwhere(np.abs(dec.x[1] - f.xc[1]) > f.L / 2)[:, 0])
    # dx_max = np.max(np.abs(dec.x[:, 1:-1, 2:, :] - f.xc[:, 1:-1, 2:, :]))
    assert np.allclose(dec.a, f.A * 255, rtol=0, atol=4.2)
    assert np.allclose(dec.b, f.B * 255, rtol=0, atol=8.8)
    assert np.allclose(dec.x[:, 1:-1, 2:, :], f.xc[:, 1:-1, 2:, :], rtol=0, atol=1.8)  # todo: index 0
    assert np.all(circular_distance(dec.x, f.xc, f.Lext) < 1.8)

    Irec = fade(I, PSF=10, rng=seed)
    dec = f.decode(Irec)
    # da_max = np.max(np.abs(dec.a - f.A * 255))
    # db_max = np.max(np.abs(dec.b - f.B * 255))
    # i0 = np.unique(np.argwhere(np.abs(dec.x[0] - f.xc[0]) > f.L / 2)[:, 1])
    # i1 = np.unique(np.argwhere(np.abs(dec.x[1] - f.xc[1]) > f.L / 2)[:, 0])
    # dx_max = np.max(np.abs(dec.x[:, 2:-1, 2:-1, :] - f.xc[:, 2:-1, 2:-1, :]))
    assert np.allclose(dec.a, f.A * 255, rtol=0, atol=4.7)
    assert np.allclose(dec.b, f.B * 255, rtol=0, atol=17.4)
    assert np.allclose(dec.x[:, 2:-1, 2:-1, :], f.xc[:, 2:-1, 2:-1, :], rtol=0, atol=2.0)  # todo: index 0
    assert np.all(circular_distance(dec.x, f.xc, f.Lext) < 2.0)


def test_distort():
    f = Fringes()
    I = f.encode()

    a = 0.15
    a = min(a, 0.5 / 3)
    x_map, y_map = f.x.astype(np.float32, copy=False)
    x_map[
        int(f.Y / 2 - f.Y * a + 0.5):int(f.Y / 2 + f.Y * a + 0.5),
        int(f.X / 2 - f.X * a + 0.5):int(f.X / 2 + f.X * a + 0.5)
    ] = x_map[
        int(f.Y / 2 + f.Y * a + 0.5):int(f.Y / 2 + 3 * f.Y * a + 0.5),
        int(f.X / 2 + f.X * a + 0.5):int(f.X / 2 + 3 * f.X * a + 0.5)
    ]
    y_map[
        int(f.Y / 2 - f.Y * a + 0.5):int(f.Y / 2 + f.Y * a + 0.5),
        int(f.X / 2 - f.X * a + 0.5):int(f.X / 2 + f.X * a + 0.5)
    ] = y_map[
        int(f.Y / 2 + f.Y * a + 0.5):int(f.Y / 2 + 3 * f.Y * a + 0.5),
        int(f.X / 2 + f.X * a + 0.5):int(f.X / 2 + 3 * f.X * a + 0.5)
    ]
    Irec = np.array([cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR) for frame in I])
    x = vshape(np.array([cv2.remap(xd, x_map, y_map, cv2.INTER_LINEAR) for xd in f.xc]))

    dec = f.decode(Irec)
    da_max = np.max(np.abs(dec.a - f.A))
    db_max = np.max(np.abs(dec.b - f.B))
    dx_max = np.max(np.abs(dec.x - x))
    assert np.allclose(dec.a, f.A, rtol=0, atol=0.13)
    assert np.allclose(dec.b, f.B, rtol=0, atol=0.69)
    assert np.allclose(dec.x, x, rtol=0, atol=0.13)


def test_circular_distance():
    f = Fringes(Y=100)
    I = f.encode()
    dec = f.decode(I)
    x = dec.x

    x[0, :, 0, :] = f.X
    cdx = circular_distance(x[0], f.xc[0], f.X)
    # cdx_max = np.max(np.abs(cdx))
    assert np.all(cdx < 0.13)

    x[1, 0, :, :] = f.Y
    cdy = circular_distance(x[1], f.xc[1], f.Y)
    # cdy_max = np.max(np.abs(cdy))
    assert np.all(cdy < 0.13)