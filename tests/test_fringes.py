import glob
import logging
import os
import tempfile
import time

import numpy as np
import pytest
import subprocess

from fringes import Fringes, __version__
from fringes.util import vshape, simulate, gamma_auto_correct, circular_distance
from fringes.filter import direct, indirect, visibility, exposure, curvature  # todo: height


# def test_compile_time():  # todo: test_numba_compile_time
#     # attention: this function must be the first within this module if running in debugging mode,
#     # because else the cache remains in RAM
#
#     flist = glob.glob(os.path.join(os.path.dirname(__file__), "..", "src", "fringes", "__pycache__", "decoder*decode*.nbc"))
#     for file in flist:
#         os.remove(file)
#
#     f = Fringes(Y=100)
#
#     I = f.encode()
#
#     t0 = time.perf_counter()
#     dec = f.decode(I)
#     t1 = time.perf_counter()
#     T = t1 - t0
#     assert T < 10 * 60, f"Numba compilation took longer than 10 minutes: {T / 60} minutes."


# def test_speed():  # todo: test_decoding_speed
#     f = Fringes()
#     f.X = 1920
#     f.Y = 1080
#     f.v = [1, 5]  # todo: [9, 10]
#
#     I = f.encode()
#
#     T = np.inf
#     for _ in range(10):
#         t0 = time.perf_counter()
#         f.decode(I)
#         t1 = time.perf_counter()
#
#         Tnew = t1 - t0
#         print(Tnew)
#
#         if Tnew < T:
#             T = Tnew
#
#     assert T <= 1, f"Decoding takes {int(np.round(T * 1000))}ms > 1000ms."


def test_logging():
    assert "fringes" in logging.Logger.manager.loggerDict, "Top level logger 'fringes' doesn't exist."


def test_version():
    assert __version__, "Version is not specified."
    # todo: check version is newer than latest on PyPi


def test_docstrings():
    f = Fringes()

    for p in dir(f):
        if isinstance(getattr(type(f), p, None), property):
            assert getattr(type(f), p, None).__doc__ is not None, f"Property '{p}' has no docstring."
        elif callable(p):
            assert p.__doc__ is not None, f"Method '{p}()' has no docstring."


def test_init():
    f = Fringes()

    assert list(f._params.keys()) == ["__version__"] + list(f._defaults.keys()),\
        "Property names differ from default names."

    for k, v in f._params.items():
        if k != "__version__":
            if k in "Nvf" and f.v.ndim == 1:
                assert np.array_equal(
                    v, f._defaults[k][0]
                ), f"'{k}' got overwritten by interdependencies. Choose consistent default values in '__init__()'."
            else:
                assert (
                    np.array_equal(v, f._defaults[k])
                    or f"'{k}' got overwritten by interdependencies. Choose consistent default values in '__init__()'."
                )

    for k in "Nvf":
        assert getattr(f, f"_{k}").shape == (f.D, f.K), f"'Set' parameter {k} does not have shape ({f.D}, {f.K})."


def test_equal():
    f1 = Fringes()
    f2 = Fringes()

    assert f1 == f2, "Classes are not identical."

    f2.N = 5
    assert f1 != f2, "Classes are identical."


def test_set_T():
    f = Fringes()

    for T in range(1, 1001 + 1):
        f.T = T

        assert f.T == T, f"Couldn't set 'T' to {T}."


def test_UMR():
    f = Fringes()
    f.l = 20.2, 60.6

    assert np.array_equal(f.UMR, [60.6] * f.D), "'UMR' is not 60.6."


def test_save_load():
    f = Fringes()
    params = f._params

    with tempfile.TemporaryDirectory() as tempdir:
        for ext in f._loader.keys():
            fname = os.path.join(tempdir, f"params{ext}")

            f.save(fname)
            assert os.path.isfile(fname), "No params-file saved."

            f.load(fname)
            params_loaded = f._params
            # assert len(params_loaded) == len(params), "A different number of attributes is loaded than saved."

            for k in params_loaded.keys():
                assert k in params, f"Fringes class has no attribute '{k}'"
                assert (
                    params_loaded[k] == params[k]
                ), f"Attribute '{k}' in file '{fname}' differs from its corresponding instance attribute."

            for k in params.keys():
                assert (
                    k in params_loaded or ext == ".toml" and params[k] is None
                ), f"File '{fname}' has no attribute '{k}'"  # in toml there exists no None
                if k in params_loaded:
                    assert (
                        params[k] == params_loaded[k]
                    ), f"Instance attribute '{k}' differs from its corresponding attribute in file '{fname}'."


def test_coordinates():
    f = Fringes()

    for D in range(1, f._Dmax + 1):
        f.D = D
        for ax in range(f._Dmax):
            f.axis = ax
            for idx in f._indexings:
                f.indexing = idx

                c = f.coordinates()
                i = np.indices((f.Y, f.X))
                if idx == "xy":
                    i = i[::-1]
                if D == 1:
                    i = i[ax][None, :, :]
                assert np.array_equal(
                    c, i
                ), f"Coordinates are wrong for D = {f.D}, axis = {f.axis}, indexing = {f.indexing}."


def test_encode():
    f = Fringes(Y=100)

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."

    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_getitem():
    f = Fringes(Y=100)

    I = f[0]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == (1, f.Y, f.X, f.C), f"Shape is not {(1, f.Y, f.X, f.C)}."

    I = f[-1]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == (1, f.Y, f.X, f.C), f"Shape is not {(1, f.Y, f.X, f.C)}."

    I = f[range(f.T)]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."
    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."

    I = f[np.arange(f.T)]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."
    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."

    I = f[:2]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == (2, f.Y, f.X, f.C), f"Shape is not {(2, f.Y, f.X, f.C)}."

    I = f[::]
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."
    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_call():
    f = Fringes(Y=100)

    I = f()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."

    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_iter():
    f = Fringes(Y=100)

    for t, I in enumerate(f):
        assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I.shape == (1, f.Y, f.X, f.C), f"Shape is not {(1, f.Y, f.X, f.C)}."
    assert t + 1 == f.T, "Number of iterations does't equal number of frames."

    I = np.array(list(frame[0] for frame in f))
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."

    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_frames():
    f = Fringes(Y=100)

    I = np.array([f.encode(frames=t)[0] for t in range(f.T)])
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."
    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."

    I = f.encode(frames=tuple(range(f.T)))
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.shape == f.shape, f"Shape is not {f.shape}."
    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_dtypes():
    f = Fringes(Y=100)

    for dtype in f._dtypes:

        f.dtype = dtype

        I = f.encode()
        assert I.dtype == f.dtype, f"dtype isn't {dtype}."

        if "uint" in dtype:  # todo: float, bool?
            dec = f.decode(I)
            assert np.allclose(
                dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
            ), f"Registration is off more than 0.1 with dtype = {dtype}."


def test_decode():
    f = Fringes(Y=100)

    I = f.encode()

    dec = f.decode(I)
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
    assert len(dec) == 3, f"Decode returned {len(dec)} instead of 3 values."
    assert hasattr(dec, "brightness")
    assert hasattr(dec, "modulation")
    assert hasattr(dec, "registration")
    assert np.allclose(dec.brightness, f.A, rtol=0, atol=1), "Brightness is off more than 1."  # todo: 0.1
    assert np.allclose(dec.modulation, f.B, rtol=0, atol=1), "Modulation is off more than 1."
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_verbose():
    f = Fringes(Y=100)

    dec = f.decode(f.encode(), verbose=True)
    assert isinstance(dec, tuple) and hasattr(dec, "_fields"), "Return value isn't a 'namedtuple'."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
    assert len(dec) == 7, f"Decode returned {len(dec)} instead of 7 values."
    assert hasattr(dec, "brightness")
    assert hasattr(dec, "modulation")
    assert hasattr(dec, "registration")
    assert hasattr(dec, "residuals")
    assert hasattr(dec, "order")
    assert hasattr(dec, "phase")

    assert np.allclose(dec.brightness, f.A, rtol=0, atol=1), "Brightness is off more than 1."  # todo: 0.1
    assert np.allclose(dec.modulation, f.B, rtol=0, atol=1), "Modulation is off more than 1."  # todo: 0.1
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."
    assert np.allclose(dec.phase, np.pi, rtol=0, atol=np.pi), "Phase values are not within [0, 2PI]."
    k = f.coordinates()[:, None, :, :, None] // f._l[:, :, None, None, None]
    k = k.reshape(f.D * f.K, f.Y, f.X, f.C)
    assert np.allclose(dec.order, k, rtol=0, atol=0), "Fringe orders are off."
    assert np.allclose(dec.residuals, 0, rtol=0, atol=0.1), "Residuals are larger than 0.5."  # todo: 0.1
    assert np.allclose(dec.uncertainty, 0, rtol=0, atol=0.5), "Uncertainty is larger than 0.5."  # todo: 0.1


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


def test_overexposure(caplog):
    f = Fringes(Y=100)

    # I = np.full_like(f, f.Imax)  # todo: full_like
    I = np.full(f.shape, f.Imax, f.dtype)
    f.decode(I, check_overexposure=True)

    assert "'I' is probably overexposed and decoding might yield unreliable results." in caplog.text
    assert caplog.records[0].levelname == "WARNING", "logging level of 'overexposure-warning' is not 'WARNING'."


def test_despike():
    f = Fringes(Y=100)
    I = f.encode()
    I[:, 10, 5, :] += int(f.Imax / 2)

    dec = f.decode(I, despike=True)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_denoise():
    f = Fringes(Y=100)
    # f.alpha = 1.1  # (f.R.min() + 2) / f.R.min()  # todo: use alpha

    dec = f.decode(simulate(f.encode()))
    assert np.allclose(
        dec.registration[:, 3:, 3:, :], f.coordinates()[:, 3:, 3:, None], rtol=0, atol=0.3
    ), "Registration is off more than 0.5."  # todo: index 0, 0.1


def test_decolorize():  # todo: decolorizing
    f = Fringes(Y=100)

    f.h = "rgb"
    I = f.encode()

    dec = f.decode(I)
    assert dec.registration.shape[-1] == 3, "Registration does not have 3 color channels."
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."
    I = I.mean(axis=-1)
    dec = f.decode(I)
    assert dec.registration.shape[-1] == 3, "Registration does not have 1 color channel."
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."

    f.h = "w"
    f.h = (100, 100, 100)

    dec = f.decode(f.encode())
    assert dec.registration.shape[-1] == 1, "Registration does not have 3 color channels."
    assert np.allclose(
        dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."  # todo: index 0


def test_deinterlacing():
    f = Fringes(Y=100)

    I = f.encode()
    I = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # interlace

    I = f._deinterlace(I)
    dec = f.decode(I)
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_alpha():
    f = Fringes(X=1000, Y=1)

    for alpha in [1.1, 2]:
        f.alpha = alpha

        dec = f.decode(f.encode())
        assert np.allclose(
            dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
        ), f"Registration is off more than 0.1 with alpha == {alpha}."


# def test_grids(): # todo: fix grids
#     f = Fringes(Y=100)
#
#     for g in f._grids:
#         f.grid = g
#
#         I = f.encode()
#         dec = f.decode(I)
#
#         d = dec.registration - f.coordinates()[..., None]
#         assert np.allclose(d, 0, rtol=0, atol=0.1), f"Registration is off more than 0.1 with grid == {f.grid}."
#
#         # todo: test angles(0, 90, 45)


def test_indexing_axis():  # todo: test setting D = 0 after setting axis
    f = Fringes(Y=100)
    f = Fringes()

    for indexing in f._indexings:
        f.indexing = indexing
        for D in range(1, f._Dmax + 1):
            f.D = D
            for axis in (0, 1):
                f.axis = axis

                dec = f.decode(f.encode())
                assert np.allclose(
                    dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
                ), f"Registration is off more than 0.2 with indexing = {f.indexing}, D = {f.D}, axis = {f.axis}."


def test_scaling():
    f = Fringes()
    f.K = 1
    f.v = 1
    f.N = 9

    dec = f.decode(f.encode())
    assert np.allclose(
        dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, None], rtol=0, atol=1
    ), "Registration is off more than 1."  # todo: 0.1, index 0


def test_unwrapping():
    f = Fringes()
    f.K = 1
    f.v = 13

    # todo: func = "cv2" (enable it first!)
    # todo: verbose -> reliability

    dec = f.decode(f.encode())
    for d in range(f.D):
        grad = np.gradient(dec.registration[d, :, :, 0], axis=0) + np.gradient(dec.registration[d, :, :, 0], axis=1)
        assert np.allclose(
            grad, 1, rtol=0, atol=0.1
        ), f"Gradient of unwrapped phase map isn't close to 1 at direction {d}."


# def test_unwrapping_class_method():  # todo: delete uwr class method
#     f = Fringes()
#     f.K = 1
#     f.v = 13
#
#     # todo: reliability
#
#     dec = f.decode(f.encode(), verbose=True)
#     Phi = Fringes.unwrap(dec.phase)
#     Phi[0] -= Phi[0].min()
#     Phi[1] -= Phi[1].min()
#     Phi *= f._l[:, 0, None, None, None] / (2 * np.pi)
#     for d in range(f.D):
#         grad = np.gradient(Phi[d, :, :, 0], axis=1 - d)
#         assert np.allclose(grad, 1, rtol=0, atol=0.1), \
#             f"Gradient of unwrapped phase map isn't close to 1 at direction {d}."
#
#     # todo: cv2
#     # Phi = Fringes.unwrap(dec.phase, func="cv2")
#     # Phi[0] -= Phi[0].min()
#     # Phi[1] -= Phi[1].min()
#     # Phi *= f._l[:, 0, None, None, None] / (2 * np.pi)
#     # for d in range(f.D):
#     #     grad = np.gradient(Phi[d, :, :, 0], axis=1 - d)
#     #     assert np.allclose(grad, 1, rtol=0, atol=0.1), \
#     #         f"Gradient of unwrapped phase map isn't close to 1 at direction {d}."


def test_curvature():
    f = Fringes(Y=100)

    dec = f.decode(f.encode())
    assert np.allclose(curvature(dec.registration)[1:, 1:], 0, rtol=0, atol=0.1), "Curvature if off more than 0.1."


# def test_height():  # todo: test height
#     f = Fringes(Y=100)
#
#     dec = f.decode(f.encode())
#     assert np.allclose(height(curvature(dec.registration)), 0, rtol=0, atol=0.1), "Height if off more than 0.1."


def test_hues():
    f = Fringes(Y=100)
    f.h = "rggb"

    dec = f.decode(f.encode())
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."

    f.H = 3

    dec = f.decode(f.encode())
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."

    # todo: H = 2 and take care of M not being a scalar


def test_averaging():
    f = Fringes(Y=100)
    f.M = 2

    dec = f.decode(f.encode())
    assert np.allclose(
        dec.registration, f.coordinates()[..., None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."


def test_WDM():
    f = Fringes(Y=100)
    f.N = 3
    f.WDM = True

    dec = f.decode(f.encode())
    assert np.allclose(
        dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, None], rtol=0, atol=0.1
    ), "Registration is off more than 0.1."  # todo: index 0


def test_SDM():
    f = Fringes()
    f.SDM = True
    # f.alpha = (f.R.min() + 2) / f.R.min()  # todo: use alpha

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, None],rtol=0, atol=0.5), \
        "Registration is off more than 0.5."  # todo: index 0, 0.1


def test_SDM_WDM():
    f = Fringes()
    f.N = 3
    f.SDM = True
    f.WDM = True
    # f.alpha = (f.R.min() + 2) / f.R.min()  # todo: use alpha

    dec = f.decode(f.encode())
    assert np.allclose(dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, None], rtol=0, atol=0.5), \
        "Registration is off more than 0.5."  # todo: index 0, 0.1


def test_FDM():
    f = Fringes(Y=100)
    f.FDM = True
    f.alpha = (f.R.min() + 2) / f.R.min()  # todo: remove alpha

    for static in (False, True):
        f.static = static
        f.N = 1
        dec = f.decode(f.encode())
        assert np.allclose(
            dec.registration[:, 1:, 1:, :], f.coordinates()[:, 1:, 1:, None], rtol=0, atol=0.2
        ), f"Registration is off more than 0.2 with static = {static}, N = {f.N}."  # todo: index 0, 0.1


def test_gamma_auto_correct():
    f = Fringes(Y=100)

    I = f.encode()

    f.gamma = 2.2
    f.dtype = "float32"

    I_gamma = f.encode() * 255
    I_ = gamma_auto_correct(I_gamma)

    assert np.allclose(I, I_, rtol=0, atol=0.5), "Gamma correction is off more than 0.5."


# todo: test degamma


def test_simulate():
    f = Fringes()
    # f.v = 9, 10, 11
    f.v = 13, 7, 89
    f.V = 0.8
    # f.alpha = 1.1  # (f.R.min() + 2) / f.R.min()  # todo: use alpha

    # x = np.abs((f.coordinates() + np.random.uniform(-0.5, +0.5, (f.D, f.Y, f.X))) % f.R[:, None, None])
    # I = f.encode(x=x)  # todo: no longer exists
    # I_ = simulate(I, PSF=0)

    PSF = 5

    dec = f.decode(simulate(f.encode(), PSF=PSF))
    assert np.allclose(
        dec.registration[:, PSF:-PSF+1, PSF:-PSF+1, :], f.coordinates()[:, PSF:-PSF+1, PSF:-PSF+1, None], rtol=0, atol=1
    ), "Registration is off more than 1."  # todo: index 0, 0.1


# todo: cdiff
# cdiff = circular_distance(dec.registration[:, :, :, :], f.coordinates()[..., None], f.R[:, None, None, None])
# assert np.all(cdiff < 1), "Registration is off more than 1."  # todo: index 0, 0.1


# def test_inside_out():  # todo: test inside_out
#     # decoding gaussian distro should be faster than uniform distro due to inside outwards decoding
#     ....


def test_mtf():
    # PSF
    ...  # todo: MTF from PSF

    # approx [Bothe2008]
    f = Fringes()
    f.lmin = f.L / 100

    v_new = 0, f.vmax / 2, f.vmax
    mtf_est = f.mtf(v_new)

    assert np.allclose([1, 0.5, 0], mtf_est, rtol=0, atol=1e-9), f"Estimated MTF if off more than {1e-9}."

    # measured
    f = Fringes()
    f.D = 2  # 1
    f.K = 11
    f.lmin = f.L / 100
    f.v = "linear"  # todo: "exponential"

    a, b, x = f.decode(f.encode())

    mtf = np.tile(np.linspace(1, 0, f.K), f.D)
    b *= mtf[:, None, None, None]
    f.set_mtf(b)

    mtf_est = f.mtf(f._v)  # get estimated modulation transfer values

    assert np.allclose(mtf.reshape(f.D, f.K), mtf_est, rtol=0, atol=0.01), "Estimated MTF if off more than 0.01."


if __name__ == "__main__":
    data = np.ones(100)
    videodata = vshape(data)
    videodata.shape

    # f = Fringes()
    # f.D = f.K = 1

    # test_source()
    # test_height()
    # test_grids()
    # test_unwrapping_class_method()

    pytest.main()
    # subprocess.run(['pytest', '--tb=short', str(__file__)])
