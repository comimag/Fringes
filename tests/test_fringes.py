import glob
import os
import time
import tempfile

import numpy as np
import pytest
import subprocess

from fringes import Fringes, curvature, height, __version__


def test_version():
    assert __version__, "Version is not specified."


def test_properties():
    f = Fringes()

    for p in dir(f):
        if isinstance(getattr(type(f), p, None), property):
            assert getattr(type(f), p, None).__doc__ is not None, f"Property '{p}' has no docstring."

        assert f.params.keys() == f.defaults.keys(), "Property names differ from default names."


def test_init():
    f = Fringes()

    for k, v in f.params.items():
        if k in "Nlvf" and f.v.ndim == 1:
            assert np.array_equal(v, f.defaults[k][0]), \
                   f"'{k}' got overwritten by interdependencies. Choose consistent default values in initialization."
        else:
            assert np.array_equal(v, f.defaults[k]) or \
                   f"'{k}' got overwritten by interdependencies. Choose consistent default values in initialization."

    for k in "Nvf":
        assert getattr(f, f"_{k}").shape == (f.D, f.K), f"Set parameter {k} hasn't shape ({f.D}, {f.K})."


def test_set_T():
    f = Fringes()

    for T in range(1, 1001 + 1):  # f._Tmax + 1
        f.T = T
        assert f.T == T, f"Couldn't set 'T' to {T}."


def test_UMR_mutual_divisibility():
    f = Fringes()
    f.l = np.array([20.2, 60.6])

    assert np.array_equal(f.UMR, [60.6] * f.D), "'UMR' is not 60.6."


def test_save_load():
    f = Fringes()
    params = f.params

    with tempfile.TemporaryDirectory() as tempdir:
        for ext in f._loader.keys():
            fname = os.path.join(tempdir, f"params{ext}")

            f.save(fname)
            assert os.path.isfile(fname), "No params-file saved."

            params_loaded = f.load(fname)
            # assert len(params_loaded) == len(params), "A different number of attributes is loaded than saved."

            for k in params_loaded.keys():
                assert k in params, f"Fringes class has no attribute '{k}'"
                assert params_loaded[k] == params[k], \
                    f"Attribute '{k}' in file '{fname}' differs from its corresponding instance attribute."

            for k in params.keys():
                assert k in params_loaded or ext == ".toml" and params[k] is None, f"File '{fname}' has no attribute '{k}'"  # in toml there exists no None
                if k in params_loaded:
                    assert params[k] == params_loaded[k], \
                        f"Instance attribute '{k}' differs from its corresponding attribute in file '{fname}'."


def test_coordinates():
    f = Fringes()

    xi = f.coordinates()
    assert xi.ndim == 4, "Coordinates are not four-dimensional."
    assert xi.shape == (f.D, f.Y, f.X, 1), "Coordinates don't have shape (f.D, f.Y, f.X, 1)."

    assert np.array_equal(f.coordinates(), np.indices((f.Y, f.X))[::-1, :, :, None]), "XY-coordinates are wrong."

    f = Fringes(Y=1)
    assert np.array_equal(f.coordinates()[0], np.arange(f.X)[None, :, None]), "X-coordinates are wrong."

    f = Fringes(X=1)
    assert np.array_equal(f.coordinates()[0], np.arange(f.Y)[:, None, None]), "Y-coordinates are wrong."


def test_encoding():
    f = Fringes(Y=100)

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_encoding_getitem():
    f = Fringes(Y=100)

    I = f[0]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    I = f[-1]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    I = f[range(f.T)]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."
    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."

    I = f[np.arange(f.T)]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."
    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."

    I = f[:2]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    I = f[::]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."
    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_encoding_call():
    f = Fringes(Y=100)

    I = f()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_encoding_iter():
    f = Fringes(Y=100)

    for t, I in enumerate(f):
        assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."
    assert t + 1 == f.T, "Number of iterations does't equal number of frames."

    I = np.array(list(frame[0] for frame in f))
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."

    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_encoding_frames(fringes=Fringes(Y=100)):
    f = fringes
    
    I = np.array([f.encode(t)[0] for t in range(f.T)])
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."
    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."

    I = f.encode(tuple(range(f.T)))
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."
    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


# def test_decoding_numba():
#     f = Fringes(Y=100)
#     f.verbose = True
#
#     I = f.encode()
#     # test numba compile time
#     flist = glob.glob(os.path.join(os.path.dirname(__file__), "..", "fringes", "__pycache__", "decoder*decode*.nbc"))
#     for file in flist:
#         os.remove(file)
#
#     t0 = time.perf_counter()
#     dec = f.decode(I)
#     t1 = time.perf_counter()
#     assert t1 - t0 < 10 * 60, f"Numba compilation took longer than 10 minutes: {(t1 - t0) / 60} minutes."


def test_decoding():
    f = Fringes(Y=100)

    dec = f.decode(f.encode())
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert len(dec) == 3, f"Decode retuned {len(dec)} instead of 3 values."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
    assert np.allclose(dec.brightness, f.A, atol=0.1), "Brightness is off more than 1."
    assert np.allclose(dec.modulation, f.B, atol=1), "Modulation is off more than 1."  # todo: more precise?
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."

    f.mode = "precise"
    dec = f.decode(f.encode())
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert len(dec) == 3, f"Decode retuned {len(dec)} instead of 3 values."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
    assert np.allclose(dec.brightness, f.A, atol=0.1), "Brightness is off more than 1."
    assert np.allclose(dec.modulation, f.B, atol=1), "Modulation is off more than 1."  # todo: more precise?
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_decoding_verbose():
    f = Fringes(Y=100)

    dec = f.decode(f.encode(), verbose=True)
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert len(dec) == 9, f"Decode retuned {len(dec)} instead of 9 values."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."

    f.verbose = True
    dec = f.decode(f.encode())
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert len(dec) == 9, f"Decode retuned {len(dec)} instead of 9 values."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
    assert np.allclose(dec.brightness, f.A, atol=0.1), "Brightness is off more than 0.1."
    assert np.allclose(dec.modulation, f.B, atol=1), "Modulation is off more than 1."  # todo: more precise?
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."
    assert np.allclose(dec.phase, 0, atol=np.pi), "Phase values are not within [-PI, +PI]."
    assert np.allclose(dec.orders, f._orders(), atol=0), "Fringe orders are off."
    # assert np.allclose(dec.residuals, 0, atol=0.5), "Residuals are larger than 0.5."  # todo
    assert np.allclose(dec.uncertainty, 0, atol=0.5), "Uncertainty is larger than 0.5."
    assert np.allclose(dec.visibility, 1, atol=0.1), "Visibility is off more than 0.1."
    assert np.allclose(dec.exposure, 0.5, atol=0.1), "Visibility is off more than 0.1."

    f.mode = "precise"
    f.verbose = False

    dec = f.decode(f.encode(), verbose=True)
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert len(dec) == 9, f"Decode retuned {len(dec)} instead of 9 values."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."

    f.verbose = True
    dec = f.decode(f.encode())
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert len(dec) == 9, f"Decode retuned {len(dec)} instead of 9 values."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
    assert np.allclose(dec.brightness, f.A, atol=0.1), "Brightness is off more than 0.1."
    assert np.allclose(dec.modulation, f.B, atol=1), "Modulation is off more than 1."  # todo: more precise?
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."
    assert np.allclose(dec.phase, 0, atol=np.pi), "Phase values are not within [-PI, +PI]."
    assert np.allclose(dec.orders, f._orders(), atol=0), "Fringe orders are off."
    # assert np.allclose(dec.residuals, 0, atol=0.5), "Residuals are larger than 0.5."  # todo
    assert np.allclose(dec.uncertainty, 0, atol=0.5), "Uncertainty is larger than 0.5."
    assert np.allclose(dec.visibility, 1, atol=0.1), "Visibility is off more than 0.1."
    assert np.allclose(dec.exposure, 0.5, atol=0.1), "Visibility is off more than 0.1."


def test_decoding_despike():
    f = Fringes(Y=100)
    I = f.encode()
    I[:, 10, 5, :] += 127  #  = I[:, -5, -10, :]

    dec = f.decode(I, despike=True)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


# todo
# def test_decoding_denoise():
#     f = Fringes(Y=100)
#     I = f.encode()
#     I[:, 10, 5, :]
#
#     dec = f.decode(I, denoise=True)
#     assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


# def test_decoloring():  # todo
#     f = Fringes(Y=100)
#
#     f.h = "rgb"
#     I = f.encode()
#     dec = f.decode(I)
#     assert dec.registration.shape[-1] == 3, "Registration hasn't 3 color channels."
#     assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."
#     I = I.mean(axis=-1)
#     dec = f.decode(I)
#     assert dec.registration.shape[-1] == 3, "Registration hasn't 1 color channel."
#     assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."
#
#     f.h = "w"
#     f.h = (100, 100, 100)
#     I = f.encode()
#     dec = f.decode(I)
#     assert dec.registration.shape[-1] == 1, "Registration hasn't 3 color channels."
#     assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."
#     I = I.mean(axis=-1)
#     dec = f.decode(I)
#     assert dec.registration.shape[-1] == 1, "Registration hasn't 1 color channel."
#     assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_deinterlacing():
    f = Fringes(Y=100)

    I = f.encode()
    I = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # interlace

    I = f.deinterlace(I)
    dec = f.decode(I)
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_alpha():
    f = Fringes(Y=100)
    f.alpha = 1.1

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."  # todo: 0.1
    assert np.all(dec.registration < f.R[:, None, None, None]), "Registration values are larger than screen size."


# def test_grids():
#     f = Fringes(Y=100)
#
#     for g in f._grids:
#         f.grid = g
#
#         I = f.encode()
#         dec = f.decode(I)
#
#         d = dec.registration - f.coordinates()
#         assert np.allclose(d, 0, atol=0.1), f"Registration is off more than 0.1 for grid '{f.grid}'."  # todo: fix grids
#
#         # todo: test angles


def test_indexing_axis():
    f = Fringes(Y=100)

    for indexing in f._indexings:
        f.indexing = indexing
        for D in [1, 2]:
            f.D = D
            for axis in [0, 1]:
                f.axis = axis
                dec = f.decode(f.encode())
                assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_dtypes():
    f = Fringes(Y=100)

    for dtype in f._dtypes:
        f.dtype = dtype

        I = f.encode()

        assert I.dtype == f.dtype, "Wrong dtype."


def test_modes():
    f = Fringes(Y=100)

    I = f.encode()

    for mode in f._modes:
        f.mode = mode

        dec = f.decode(I)
        assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


# def test_scaling():
#     f = Fringes(Y=100)
#     f.K = 1
#
#     f.v = 1
#     I = f.encode()
#     dec = f.decode(I)
#
#     d = dec.registration - f.coordinates()
#     # std = np.std(d)
#     # a = 1
#     #assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."  # todo: 0.1


def test_unwrapping():
    f = Fringes()
    f.K = 1
    f.v = 13

    dec = f.decode(f.encode())
    for d in range(f.D):
        grad = np.gradient(dec.registration[d, :, :, 0], axis=1 - d)
        assert np.allclose(grad, 1, atol=0.1), "Gradient of unwrapped phase map isn't close to 1."

    # todo: func = "cv2"


def test_unwrapping_class_method():
    f = Fringes()
    f.K = 1
    f.v = 13
    f.verbose = True

    dec = f.decode(f.encode())
    Phi = Fringes.unwrap(dec.phase)  # todo: verbose -> reliability
    Phi /= 2 * np.pi * f.v / f.L
    for d in range(f.D):
        grad = np.gradient(Phi[d, :, :, 0], axis=1 - d)
        assert np.allclose(grad, 1, atol=0.1), "Gradient of unwrapped phase map isn't close to 1."

    # todo: cv2
    # Phi = Fringes.unwrap(dec.registration, func="cv2")  # todo: verbose -> reliability
    # for d in range(f.D):
    #     grad = np.gradient(Phi[d, :, :, 0], axis=1 - d)
    #     assert np.allclose(grad, 1, atol=0.1), "Gradient of unwrapped phase map isn't close to 1."


def test_remapping():
    f = Fringes(Y=100)

    dec = f.decode(f.encode())
    assert np.allclose(f.remap(dec.registration), 1, atol=0), "Source doesn't contain only ones."
    assert np.allclose(f.remap(dec.registration, dec.modulation), 1, atol=0.01), "Source doesn't contain only values close to one."
    assert np.allclose(f.remap(dec.registration, mode="precise"), 1, atol=0), "Source doesn't contain only ones."
    assert np.allclose(f.remap(dec.registration, dec.modulation, mode="precise"), 1, atol=0.01), "Source doesn't contain only values close to one."

    assert np.allclose(f.remap(dec.registration, mode="precise"), 1, atol=0), "Source doesn't contain only ones."
    assert np.allclose(f.remap(dec.registration, dec.modulation, mode="precise"), 1, atol=0.1), "Source doesn't contain only values close to one."


def test_curvature():
    f = Fringes(Y=100)

    dec = f.decode(f.encode())
    assert np.allclose(curvature(dec.registration)[1:, 1:], 0, atol=0.1), "Curvature if off more than 0.1."


def test_height():
    f = Fringes(Y=100)

    dec = f.decode(f.encode())
    h = height(curvature(dec.registration))
    assert np.allclose(height(curvature(dec.registration)), 0, atol=0.1), "Height if off more than 0.1."


def test_hues():
    f = Fringes(Y=100)
    f.h = "rggb"

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."

    f.H = 3  # todo: 2 and take care of M not being a scalar

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_averaging():
    f = Fringes(Y=100)
    f.M = 2

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


def test_WDM():
    f = Fringes(Y=100)
    f.N = 3
    f.WDM = True

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration, f.coordinates(), atol=0.1), "Registration is off more than 0.1."


# def test_SDM():
#     f = Fringes(Y=100)
#     f.SDM = True
#
#     I = f.encode()
#     dec = f.decode(I)
#
#     for d in range(f.D):
#         grad = np.gradient(dec.registration[d, 1:-1, 1:-1, 0], axis=1 - d)  # todo: boarder
#         #assert np.allclose(grad, 1, atol=0.1), "Gradient of registration isn't close to 1."
#
#     d = dec.registration[:, 1:-1, 1:-1, :] - f.coordinates()[:, 1:-1, 1:-1, :]  # todo: boarder
#     assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."


# def test_SDM_WDM():
#     f = Fringes(Y=100)
#     f.N = 3
#     f.SDM = True
#     f.WDM = True
#
#     I = f.encode()
#     dec = f.decode(I)
#
#     for d in range(f.D):
#         grad = np.gradient(dec.registration[d, 1:-1, 1:-1, 0], axis=1 - d)  # todo: boarder
#         #assert np.allclose(grad, 1, atol=0.1), "Gradient of registration isn't close to 1."
#
#     d = dec.registration[:, 1:-1, 1:-1, :] - f.coordinates()[:, 1:-1, 1:-1, :]  # todo: boarder
#     assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."  # todo: 0.1


def test_FDM():
    f = Fringes(Y=100)
    f.FDM = True

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, :], atol=0.1), "Registration is off more than 0.5."  # todo: boarder

    f.static = True
    f.N = 1

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, :], atol=0.1), "Registration is off more than 0.5."  # todo: boarder


def test_simulation():
    # todo:
    #  no low pass from PSF causing decrease in modulation
    #  no clipping
    #  no hit pixels -> add salt and pepper noise?

    f = Fringes()
    f.V = 0.8
    f.D = 1
    f.X = 832
    f.Y = 2003
    f.N = 4
    f.l = 9, 10, 11

    dec = f.decode(f.encode(simulate=True))
    d = dec.registration - f.coordinates()
    dabs = np.abs(d)
    dmed = np.nanmedian(dabs)
    davg = np.nanmean(dabs)
    dmax = np.nanmax(dabs)
    assert np.allclose(dmed, 0, atol=0.1), "Median of Registration is off more than 0.1."
    # assert np.allclose(dec.registration, f.coordinates(), atol=0.5), "Registration is off more than 0.5."  # todo: 0.1


if __name__ == "__main__":
    # todo: def test_...(): test_..., test..., ...

    # f = Fringes()
    # f.l = "1, 2, 3"  # testing argparse

    # pytest.main()
    subprocess.run(['pytest', '--tb=short', str(__file__)])
