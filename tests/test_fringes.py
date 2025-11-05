import glob
from importlib.metadata import version
import logging
import os
import tempfile
import time

from numba import get_num_threads
import numpy as np
import pytest

import fringes
from fringes import Fringes, __version__
from fringes.fringes import _Decoded, _Decoded_verbose
from fringes.util import circular_distance

f = Fringes()
I = f.encode()
f10 = Fringes(Y=10)
I10 = f10.encode()


class TestPackage:
    def test_logging(self):
        assert (
            fringes.__package__ in logging.Logger.manager.loggerDict
        ), f"Top level logger '{fringes.__package__}' doesn't exist."

    def test_version(self):
        assert __version__, "Version is not specified."

        # https://packaging.python.org/en/latest/discussions/single-source-version/
        assert __version__ == version(fringes.__package__)

        # todo: check version is newer than latest on PyPi


class TestClass:
    def test_default_types(self):
        assert set(Fringes._defaults.keys()) == set(Fringes._types.keys()) == set(Fringes._types_str.keys())

    def test_setters_help(self):
        assert set(Fringes._setters) == set(Fringes._help.keys())

    def test_defaults_setters(self):
        assert set(Fringes._defaults.keys()) | {"A", "B", "l"} == set(Fringes._setters)

    def test_docstrings(self):
        for k in dir(Fringes):
            if k[0] != "_":
                v = getattr(Fringes, k)
                if isinstance(v, property):
                    assert v.__doc__ is not None, f"Property '{k}' has no docstring."
                elif callable(v):
                    assert v.__doc__ is not None, f"Method '{k}()' has no docstring."


class TestInstance:
    def test_init(self, caplog):
        # todo: the next line basically replaces the next for-loop
        assert "' got overwritten by interdependencies. Choose consistent initialization values." not in caplog.text

        for k, v in Fringes._defaults.items():
            if k in "N l v f".split() and getattr(f, k).ndim != np.array(v).ndim:
                assert np.array_equal(
                    getattr(f, k), v[0]
                ), f"'{k}' got overwritten by interdependencies. Choose consistent default values in '__init__()'."  # without contradiction
            else:
                assert np.array_equal(
                    getattr(f, k), v
                ), f"'{k}' got overwritten by interdependencies. Choose consistent default values in '__init__()'."  # without contradiction

        for k in "N v f".split():
            assert getattr(f, f"_{k}").shape == (
                f.D,
                f.K,
            ), f"'Set' parameter {k} does not have shape ({f.D}, {f.K})."

    def test_equal(self):
        f_ = Fringes()
        assert f_ == f, "Classes are not identical."

        f_.N += 1
        assert f_ != f, "Classes are identical."

    def test_reset(self):
        f_ = Fringes()
        f_.N += 1
        f_.reset()
        assert f_ == f, "Classes are not identical."

    # def test_optimize(self):  # todo: set T, ...
    #     f = Fringes()
    #
    #     for T in range(1, 1001 + 1):
    #         f.T = T
    #         assert f.T == T, f"Couldn't set 'T' to {T}."

    def test_save_load(self):
        f = Fringes()
        params = f._params

        with tempfile.TemporaryDirectory() as tempdir:
            for ext in {".json", ".yaml"}:
                fname = os.path.join(tempdir, f"params{ext}")

                f.save(fname)
                assert os.path.isfile(fname), "No params-file saved."

                f.load(fname)
                params_loaded = f._params

                # iterate over loaded params
                for k in params_loaded.keys():
                    assert k in params, f"Fringes class has no attribute '{k}'"
                    assert (
                        params_loaded[k] == params[k]
                    ), f"Attribute '{k}' in file '{fname}' differs from its corresponding instance attribute."

                # iterate over instance params
                for k in params.keys():
                    assert k in params_loaded, f"File '{fname}' has no attribute '{k}'"
                    if k in params_loaded:
                        assert (
                            params[k] == params_loaded[k]
                        ), f"Instance attribute '{k}' differs from its corresponding attribute in file '{fname}'."

    def test_UMR(self):
        f = Fringes()
        f.l = 20.2, 60.6
        assert np.array_equal(f.UMR, [60.6] * f.D), "'UMR' is not 60.6."

    def test_x(self):
        f = Fringes()

        for D in range(1, f._Dmax + 1):
            f.D = D
            for ax in range(f._Dmax):
                f.axis = ax
                for idx in Fringes._choices["indexing"]:
                    f.indexing = idx

                    x = f.x
                    i = np.indices((f.Y, f.X))
                    if idx == "xy":
                        i = i[::-1]
                    if D == 1:
                        i = i[ax][None, :, :]
                    assert np.array_equiv(
                        x, i
                    ), f"Coordinates are wrong for D = {f.D}, axis = {f.axis}, indexing = {f.indexing}."


class TestEncode:
    def test_shape(self):
        assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I.shape == f.shape, f"Shape is not {f.shape}."

    def test_dtypes(self):
        f = Fringes(Y=10)

        for dtype in Fringes._choices["dtype"]:
            f.dtype = dtype
            I = f.encode()
            assert I.dtype == f.dtype, f"dtype isn't {dtype}."

    def test_call(self):
        I10_ = f10()
        assert isinstance(I10_, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I10_.shape == f10.shape, f"Shape is not {f10.shape}."
        assert I10_.dtype == f10.dtype, f"Dtype is not {f10.dtype}."
        assert np.array_equal(I10_, I10)

    def test_iter(self):
        for t, frame in enumerate(f10):
            assert isinstance(frame, np.ndarray), "Return value isn't a 'Numpy array'."
            assert frame.shape == f10.shape[1:], f"Shape is not {f10.shape[1:]}."
            assert frame.dtype == f10.dtype, f"Dtype is not {f10.dtype}."
            assert np.array_equal(frame, I10[t])
        assert t + 1 == f10.T, "Number of iterations doesn't equal number of frames."

        I10_ = np.array(list(frame for frame in f10))
        assert isinstance(I10_, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I10_.shape == f10.shape, f"Shape is not {f10.shape}."
        assert I10_.dtype == f10.dtype, f"Dtype is not {f10.dtype}."
        assert np.array_equal(I10_, I10)

    @pytest.mark.skip("Test only on request.")
    def test_speed(self):
        f = Fringes()
        f.X = 3840
        f.Y = 2160

        T = np.empty(10)
        for t in range(len(T)):
            t0 = time.perf_counter()
            f.encode()
            t1 = time.perf_counter()
            T[t] = t1 - t0

        print(T)
        Tmin = np.min(T)
        Tmed = np.median(T)
        Tavg = np.mean(T)
        Tmax = np.max(T)
        assert Tmed <= 0.1, f"Encoding takes {Tmed * 1000:.0f}ms > 100ms."


class TestDecode:
    def test_x(self):
        f = Fringes()

        for D in range(1, f._Dmax + 1):
            f.D = D
            for ax in range(f._Dmax):  # if D == 1 else range(1):
                f.axis = ax
                for indexing in Fringes._choices["indexing"]:
                    f.indexing = indexing
                    I = f.encode()
                    dec = f.decode(I)
                    assert np.allclose(
                        dec.x, f.xc, rtol=0, atol=0.13
                    ), f"Coordinate is off more than 0.13 with {f.D = }, {f.axis = }, {f.indexing = }."

    def test_p0(self):
        f = Fringes()

        for p0 in {0, np.pi / 2, np.pi}:  # todo: 1.0
            f.p0 = p0
            I = f.encode()
            dec = f.decode(I)
            assert np.allclose(dec.x, f.xc, rtol=0, atol=0.5), f"Coordinate is off more than 0.13 with {f.mode = }."

    def test_modes(self):
        f = Fringes()

        for mode in Fringes._choices["mode"]:
            f.mode = mode
            I = f.encode()
            dec = f.decode(I)
            assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), f"Coordinate is off more than 0.13 with {f.mode = }."

    # def test_grids(): # todo: fix grids
    #     f = Fringes()
    #
    #     for g in Fringes._choices[grid"]:
    #         f.grid = g
    #         I = f.encode()
    #         dec = f.decode(I)
    #
    #         d = dec.x - f.xc
    #         assert np.allclose(d, 0, rtol=0, atol=0.13), f"Coordinate is off more than 0.13 with {f.grid = }."
    #
    #         # todo: test angles(0, 90, 45)

    def test_dtypes(self):
        f = Fringes(Y=10)

        for dtype in Fringes._choices["dtype"]:
            f.dtype = dtype
            I = f.encode()
            dec = f.decode(I)
            assert np.allclose(
                dec.x[:, 1:, :, :], f.xc[:, 1:, :, :], rtol=0, atol=0.13
            ), f"Coordinate is off more than 0.13 with {f.dtype = }."  # todo: index 0
            # xmax = np.max(np.abs(circular_distance(dec.x, f.xc, f.Lext)))
            # assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=0.13)  # todo: float32, float64

    def test_dtype_object(self):
        I10_ = np.empty((f10.T,), object)
        for t, frame in enumerate(I10):
            I10_[t] = frame
        dec = f10.decode(I10_)
        assert np.allclose(dec.x, f10.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13 with dtype = object."

    def test_check_overexposure(self, caplog):
        I10_ = np.full(f10.shape, f10.Imax, f10.dtype)
        f10.decode(I10_, check_overexposure=True)
        assert "'I' is probably overexposed and decoding might yield unreliable results." in caplog.messages
        assert caplog.records[0].levelname == "WARNING", "logging level of 'overexposure-warning' is not 'WARNING'."

        I10_ = np.full(f10.shape, 2**11 - 1, np.uint16)
        f10.decode(I10_, check_overexposure=True)
        assert "'I' is probably overexposed and decoding might yield unreliable results." in caplog.messages
        assert caplog.records[0].levelname == "WARNING", "logging level of 'overexposure-warning' is not 'WARNING'."

    def test_check_num_frames(self):
        I10_ = I10[:-1]
        with pytest.raises(ValueError) as excinfo:
            f10.decode(I10_)
        assert "Number of frames of data and parameters don't match." in str(excinfo.value)

    def test_check_color_channels(self):
        f = Fringes(Y=10)
        f.N = 3
        f.WDM = True
        I = f.encode()[..., 0]

        with pytest.raises(ValueError) as excinfo:
            f.decode(I)
        assert f"'I' must have 3 color channels because 'WDM' is active, but has only {1} color channels." in str(
            excinfo.value
        )

    def test_demodulate(self):
        dec = f.decode(I)
        assert isinstance(dec, tuple) and isinstance(dec, _Decoded), "Return value isn't a 'namedtuple'."
        assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - f.xc))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.13), "Brightness is off more than 0.13."
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.69), "Modulation is off more than 0.69."
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."

    def test_demodulate_verbose(self):
        dec = f.decode(I, verbose=True)
        assert isinstance(dec, tuple) and isinstance(dec, _Decoded_verbose), "Return value isn't a 'namedtuple'."
        assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."
        p = (f.xc[:, None, :, :, :] % f._l[:, :, None, None, None] / f._l[:, :, None, None, None]) * 2 * np.pi
        p = p.reshape(f.D * f.K, f.Y, f.X, f.C).astype(np.float32, copy=False)
        k = f.xc[:, None, :, :, :] // f._l[:, :, None, None, None]
        k = k.reshape(f.D * f.K, f.Y, f.X, f.C).astype(np.int_, copy=False)
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - f.xc))
        # dp_max = np.max(np.abs(dec.p - p))
        # r_max = np.max(np.abs(dec.r))
        # u_max = np.max(np.abs(dec.u))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.13), "Brightness is off more than 0.13."
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.69), "Modulation is off more than 0.69."
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."
        # assert np.allclose(dec.x, f.xc, rtol=0, atol=4 * dec.r)  # todo: x within factor * r
        assert np.allclose(dec.x, f.xc, rtol=0, atol=4 * dec.u)
        assert np.allclose(dec.p, np.pi, rtol=0, atol=np.pi), "Phase values are not within [0, 2PI]."
        assert np.allclose(dec.p, p, rtol=0, atol=0.0052), "Phase is off more than 0.0052."
        assert np.allclose(dec.k, k, rtol=0, atol=0), "Fringe orders are wrong."
        assert np.allclose(dec.r, 0, rtol=0, atol=0.09), "Residuals are larger than 0.09."
        assert np.allclose(dec.u, 0, rtol=0, atol=0.04), "Uncertainty is larger than 0.04."


    def test_threads(self):
        max_threads = get_num_threads()
        for threads in range(-max_threads, max_threads + 2):
            f10.decode(I10, threads=threads)

    def test_uncertainty(self):
        dec = f10.decode(I10)

        for ui in 3, np.full((f10.Y, f10.X, f10.C), 3):
            for b in 100, dec.b, None:
                for a in 127.5, dec.a, None:
                    for K in 0.038, None:
                        for dark_noise in 13.7, None:
                            u = f.uncertainty(ui, b, a, K, dark_noise)

        u = f.uncertainty(a=127.5, b=70, K=0.038, dark_noise=13.7)
        assert np.allclose(u, 0.5, rtol=0, atol=0.5), "Uncertainty is off more than 0.5."

    def test_single_fringe(self):
        f = Fringes()
        f.K = 1
        f.v = 1

        f.N = 9
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.17), "Brightness is off more than 0.17."
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.34), "Modulation is off more than 0.34."
        assert np.allclose(
            dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=0.92
        ), "Coordinate is off more than 0.92."  # todo: index 0
        assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=0.92)

        f.N = 23
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.20), "Brightness is off more than 0.20."
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.26), "Modulation is off more than 0.26."
        assert np.allclose(
            dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=0.48
        ), "Coordinate is off more than 0.48."  # todo: index 0
        assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=0.48)

    def test_alpha(self):
        f = Fringes(Y=10)

        for a in [1.1, 2]:
            f.a = a
            I = f.encode()
            dec = f.decode(I)
            assert np.allclose(dec.x, f.xc, rtol=0, atol=0.26), f"Coordinate is off more than 0.26 with {f.a = }."

    def test_spatial_unwrapping(self):
        f = Fringes()
        f.K = 1
        # todo: f.v = 7, 14
        I = f.encode()

        for uwr_func in {"ski", "cv2"}:  # todo: "cv2" is error-prone!
            dec = f.decode(I, uwr_func=uwr_func)

            for d in range(f.D):
                grad = np.gradient(dec.x[d], axis=0) + np.gradient(dec.x[d], axis=1)
                dg_max = np.max(np.abs(grad - 1))
                idx = np.argwhere(np.abs(grad - 1) > 0.10)
                assert np.allclose(
                    grad, 1, rtol=0, atol=0.10
                ), f"Gradient of unwrapped phase map isn't close to 1 at direction {d} for function {uwr_func}."

    def test_decolorize(self):
        f = Fringes(Y=10)

        f.h = "rgb"
        I = f.encode()
        dec = f.decode(I)
        assert dec.x.shape[-1] == 3, "Coordinate does not have 3 color channels."
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."
        I = I.mean(axis=-1)
        dec = f.decode(I)
        assert dec.x.shape[-1] == 3, "Coordinate does not have 1 color channel."
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."

        f.h = (100, 100, 100)
        I = f.encode()
        dec = f.decode(I)
        assert dec.x.shape[-1] == 1, "Coordinate does not have 1 color channel."
        assert np.allclose(
            dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=0.29
        ), "Coordinate is off more than 0.29."  # todo: index 0
        # xmax = np.max(np.abs(circular_distance(dec.x, f.xc, f.Lext)))
        # assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=0.29)  # todo

    def test_hues(self):
        f = Fringes(Y=10)

        f.h = "ww"
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."

        f.h = "rggb"
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."

        f.h = "www"
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.13), "Coordinate is off more than 0.13."

    def test_WDM(self):
        f = Fringes(Y=10)
        f.N = 3
        f.WDM = True
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(
            dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=0.09
        ), "Coordinate is off more than 0.09."  # todo: index 0
        # xmax = np.max(np.abs(circular_distance(dec.x, f.xc, f.Lext)))
        # assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=00)  # todo

    def test_SDM(self):
        f = Fringes()
        f.SDM = True
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(
            dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=1.19
        ), "Coordinate is off more than 1.19."  # todo: index 0
        # xmax = np.max(np.abs(circular_distance(dec.x, f.xc, f.Lext)))
        # assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=1.19)  # todo

    def test_SDM_WDM(self):
        f = Fringes()
        f.N = 3
        f.SDM = True
        f.WDM = True
        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(
            dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=1.13
        ), "Coordinate is off more than 1.13."  # todo: index 0
        # xmax = np.max(np.abs(circular_distance(dec.x, f.xc, f.Lext)))
        # assert np.allclose(circular_distance(dec.x, f.xc, f.Lext), 0, rtol=0, atol=1.13)  # todo

    def test_FDM(self):
        f = Fringes(Y=10)
        f.FDM = True

        for static in {False, True}:
            f.static = static
            f.N = 1
            I = f.encode()
            dec = f.decode(I)
            assert np.allclose(
                dec.x[:, 1:, 1:, :], f.xc[:, 1:, 1:, :], rtol=0, atol=0.31
            ), f"Coordinate is off more than 0.31 with {static = }, {f.N = }."  # todo: index 0
            # xmax = np.max(np.abs(circular_distance(dec.x, f.xc, f.Lext)))
            # assert np.all(
            #     circular_distance(dec.x, f.xc, f.Lext) < 0.31
            # ), f"Coordinate is off more than 0.31 with {static = }, {f.N = }."  # todo

    def test_8K(self):
        f = Fringes()
        f.X = 7680
        f.Y = 3840
        f.v = 13, 7, 89
        I = f.encode()
        dec = f.decode(I)
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - f.xc))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.09), "Brightness is off more than 0.13."
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.69), "Modulation is off more than 0.69."
        assert np.allclose(dec.x, f.xc, rtol=0, atol=0.08), "Coordinate is off more than 0.08."

    @pytest.mark.skip("Test only on request.")
    def test_speed(self):
        f = Fringes()

        # Tmed < 2s in battery-mode on 2025-08-04 and 2025-09-22
        # Tmed < 2s in AC-mode on 2025-08-04 and 2025-09-22
        f.X = 2048
        f.Y = 2048
        f.v = 13, 7

        I = f.encode()

        T = np.empty(10)
        for t in range(len(T)):
            t0 = time.perf_counter()
            f.decode(I)
            t1 = time.perf_counter()
            T[t] = t1 - t0
            time.sleep(4)  # time for CPU to cool down

        print(T)
        Tmin = np.min(T)
        Tmed = np.median(T)
        Tavg = np.mean(T)
        Tmax = np.max(T)
        assert Tmed <= 2.0, f"Decoding takes {Tmed * 1000:.0f}ms > 2000ms."  # todo: <= 1.0

    @pytest.mark.skip("Test only on request.")
    def test_compile_time(self):
        flist = glob.glob(
            os.path.join(os.path.dirname(__file__), "..", "src", "fringes", "__pycache__", "decoder*decode*.nbc")
        )
        for file in flist:
            os.remove(file)

        t0 = time.perf_counter()
        f.decode(I)
        t1 = time.perf_counter()
        T = t1 - t0

        print(T)
        assert T < 10 * 60, f"Numba compilation took {T / 60} minutes > 10 minutes."


# def test_mtf():
#     # PSF
#     ...  # todo: MTF from PSF
#
#     # approx [Bothe2008]
#     f = Fringes()
#     f.lmin = f.L / 100
#
#     v_new = 0, f.vmax / 2, f.vmax
#     mtf_est = f.mtf(v_new)
#
#     assert np.allclose([1, 0.5, 0], mtf_est, rtol=0, atol=1e-9), f"Estimated MTF if off more than {1e-9}."
#
#     # measured
#     f = Fringes()
#     f.D = 2  # 1
#     f.K = 11
#     f.lmin = f.L / 100
#     f.v = "linear"  # todo: "exponential"
#
#     a, b, x = f.decode(f.encode())
#
#     mtf = np.tile(np.linspace(1, 0, f.K), f.D)
#     b *= mtf[:, None, None, None]
#     f.set_mtf(b)
#
#     mtf_est = f.mtf(f._v)  # get estimated modulation transfer values
#
#     assert np.allclose(mtf.reshape(f.D, f.K), mtf_est, rtol=0, atol=0.01), "Estimated MTF if off more than 0.01."


# def test_inside_out():  # todo: test inside_out
#     # decoding gaussian distro should be faster than uniform distro due to inside outwards decoding
#     ....
