import glob
import logging
import os
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pytest
from numba import get_num_threads

import fringes
from fringes import Fringes, __version__
from fringes.fringes import _Decoded, _Decoded_verbose
from fringes.util import circular_distance, vshape

f = Fringes()
I = f.encode()
f10 = Fringes(Y=10)
I10 = f10.encode()


class TestPackage:
    def test_logging(self):
        assert fringes.__package__ in logging.Logger.manager.loggerDict, (
            f"Top level logger '{fringes.__package__}' is not available."
        )

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
        assert set(Fringes._defaults.keys()) | {"A", "B", "l", "D"} == set(Fringes._setters)

    @pytest.mark.parametrize("attr", [a for a in dir(Fringes) if a[0] != "_"])
    def test_docstrings(self, attr):
        v = getattr(Fringes, attr)
        if isinstance(v, property):
            assert v.__doc__ is not None, f"Property '{attr}' has no docstring."
        elif callable(v):
            assert v.__doc__ is not None, f"Method '{attr}()' has no docstring."


class TestInstance:
    def test_call(self):
        assert np.array_equal(I10, f10())

    # todo: parameterize
    def test_init_consistency(self, caplog):
        Fringes(dtype="uint16", bits=16)
        assert "' got overwritten by interdependencies. Choose consistent initialization values." not in caplog.text

        Fringes(dtype="uint16", bits=32)
        assert "' got overwritten by interdependencies. Choose consistent initialization values." in caplog.text
        assert caplog.records[0].levelname == "WARNING"

    @pytest.mark.parametrize("attr", Fringes._defaults.items())
    def test_init_defaults(self, attr):
        k, v = attr
        msg = f"'{k}' got overwritten by interdependencies. Choose consistent default values in '__init__()'."

        if k in "N l v f".split() and getattr(f, k).ndim != np.array(v).ndim:
            assert np.array_equal(getattr(f, k), v[0]), msg  # ...without contradiction
        else:
            assert np.array_equal(getattr(f, k), v), msg  # ...without contradiction

    @pytest.mark.parametrize("set_prop", [c for c in "Nlvf"])
    def test_set(self, set_prop):
        assert getattr(f, f"_{set_prop}").shape == (f.D, f.K)

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

    @pytest.mark.parametrize("ext", [".json", ".yaml"])
    def test_save_load(self, ext):
        f = Fringes()
        params = f._params

        with tempfile.TemporaryDirectory() as tempdir:
            fname = Path(tempdir) / f"params{ext}"

            f.save(fname)
            assert os.path.isfile(fname), "No params-file saved."

            f.load(fname)

            assert params == f._params


class TestCosys:
    @pytest.mark.parametrize("axes", [0, 1, (0, 1), (1, 0)])
    def test_x(self, axes):
        f = Fringes()
        f.axes = axes

        idx = np.indices((f.Y, f.X))[f.axes, ..., None]
        assert np.array_equal(f.x, idx)

    @pytest.mark.parametrize("axes", [0, 1, (0, 1), (1, 0)])
    def test_axes(self, axes):
        f = Fringes()
        f.axes = axes

        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)

    # def test_grids(self): # todo: fix grids
    #     f = Fringes()
    #
    #     for g in Fringes._choices["grid"]:
    #         f.grid = g
    #         f.Y = f.X = 1000
    #         f.l = 13, 7, 89
    #         I = f.encode()
    #         dec = f.decode(I)
    #         da_max = np.max(np.abs(dec.a - f.A))
    #         db_max = np.max(np.abs(dec.b - f.B))
    #         dx_max = np.max(np.abs(dec.x - f.x))
    #         # todo: mask in center if polar
    #         d = dec.x - f.x
    #         assert np.allclose(d, 0, rtol=0, atol=0.13)
    #
    #         # todo: test angles(0, 90, 45)

    @pytest.mark.parametrize("alpha", [1.1, 2])
    def test_alpha(self, alpha):
        f = Fringes(Y=10)
        f.a = alpha

        I = f.encode()
        dec = f.decode(I)
        assert np.allclose(dec.x, f.x, rtol=0, atol=0.26)

    def test_distort(self):
        a = 0.15
        a = min(a, 0.5 / 3)
        x_map, y_map = f.x.astype(np.float32, copy=False)
        x_map[
            int(f.Y / 2 - f.Y * a + 0.5) : int(f.Y / 2 + f.Y * a + 0.5),
            int(f.X / 2 - f.X * a + 0.5) : int(f.X / 2 + f.X * a + 0.5),
        ] = x_map[
            int(f.Y / 2 + f.Y * a + 0.5) : int(f.Y / 2 + 3 * f.Y * a + 0.5),
            int(f.X / 2 + f.X * a + 0.5) : int(f.X / 2 + 3 * f.X * a + 0.5),
        ]
        y_map[
            int(f.Y / 2 - f.Y * a + 0.5) : int(f.Y / 2 + f.Y * a + 0.5),
            int(f.X / 2 - f.X * a + 0.5) : int(f.X / 2 + f.X * a + 0.5),
        ] = y_map[
            int(f.Y / 2 + f.Y * a + 0.5) : int(f.Y / 2 + 3 * f.Y * a + 0.5),
            int(f.X / 2 + f.X * a + 0.5) : int(f.X / 2 + 3 * f.X * a + 0.5),
        ]
        Irec = np.array([cv2.remap(frame, x_map, y_map, cv2.INTER_LINEAR) for frame in I])
        x = vshape(np.array([cv2.remap(xd, x_map, y_map, cv2.INTER_LINEAR) for xd in f.x]))

        dec = f.decode(Irec)
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - x))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.13)
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.69)
        assert np.allclose(dec.x, x, rtol=0, atol=0.13)


class TestMux:
    def test_WDM(self):
        f = Fringes(Y=10)
        f.N = 3
        f.WDM = True

        I = f.encode()
        assert np.array_equal(I, np.array([frame for frame in f]))

        dec = f.decode(I)
        # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
        # dx_max = np.max(np.abs(dec.x[:, 1:, :, :] - f.x[:, 1:, :, :]))
        assert np.allclose(dec.x[:, 1:, :, :], f.x[:, 1:, :, :], rtol=0, atol=0.09)  # todo: index 0

    def test_SDM(self):
        f = Fringes()
        f.SDM = True

        I = f.encode()
        assert np.array_equal(I, np.array([frame for frame in f]))

        dec = f.decode(I)
        # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
        # dx_max = np.max(np.abs(dec.x[:, 1:, :, :] - f.x[:, 1:, :, :]))
        assert np.allclose(dec.x[:, 1:, :, :], f.x[:, 1:, :, :], rtol=0, atol=1.19)  # todo: index 0

    def test_SDM_WDM(self):
        f = Fringes()
        f.N = 3
        f.SDM = True
        f.WDM = True

        I = f.encode()
        assert np.array_equal(I, np.array([frame for frame in f]))

        dec = f.decode(I)
        # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
        # dx_max = np.max(np.abs(dec.x[:, 1:, :, :] - f.x[:, 1:, :, :]))
        assert np.allclose(dec.x[:, 1:, :, :], f.x[:, 1:, :, :], rtol=0, atol=1.13)  # todo: index 0

    @pytest.mark.parametrize("static", [False, True])
    def test_FDM(self, static):
        f = Fringes(Y=10)
        f.FDM = True
        f.static = static
        f.N = 1

        I = f.encode()
        assert np.array_equal(I, np.array([frame for frame in f]))

        dec = f.decode(I)
        # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
        # dx_max = np.max(np.abs(dec.x[:, 1:, 1:, :] - f.x[:, 1:, 1:, :]))
        assert np.allclose(dec.x[:, 1:, 1:, :], f.x[:, 1:, 1:, :], rtol=0, atol=0.31)  # todo: index 0


@pytest.mark.skip()
class TestSpeed:
    def test_encode(self):
        f = Fringes()
        f.X = 3840
        f.Y = 2160

        count = 10

        T_enc_ = np.empty(count)
        for t in range(len(T_enc_)):
            t0 = time.perf_counter()
            f.encode()
            t1 = time.perf_counter()
            T_enc_[t] = t1 - t0
        T_enc_med = np.median(T_enc_)
        assert T_enc_med <= 0.1, f"Encoding takes {T_enc_med * 1000:.0f}ms > 100ms."

        T_iter_ = np.empty(count)
        for t in range(len(T_iter_)):
            t0 = time.perf_counter()
            for frame in f:
                pass
            t1 = time.perf_counter()
            T_iter_[t] = t1 - t0
        T_iter_med = np.median(T_iter_)
        assert T_iter_med <= 0.1 * 1.2, f"Encoding takes {T_iter_med * 1000:.0f}ms > 100ms."

    def test_decode(self):
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

        Tmed = np.median(T)
        assert Tmed <= 2.0, f"Decoding takes {Tmed * 1000:.0f}ms > 2000ms."  # todo: <= 1.0

    def test_compile(self):
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


class TestDecode:
    def test_dtype_object(self):
        I10_ = np.empty((f10.T,), object)
        for t, frame in enumerate(I10):
            I10_[t] = frame
        dec = f10.decode(I10_)
        assert np.allclose(dec.x, f10.x, rtol=0, atol=0.13)

    @pytest.mark.parametrize("I10_", [np.full(f10.shape, f10.Imax, f10.dtype), np.full(f10.shape, 2**10 - 1, "uint16")])
    def test_check_overexposure(self, I10_, caplog):
        f10.decode(I10_, check_overexposure=True)
        assert "'I' is probably overexposed and decoding might yield unreliable results." in caplog.messages
        assert caplog.records[0].levelname == "WARNING"

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

    @pytest.mark.parametrize("threads", range(-get_num_threads(), get_num_threads() + 2))
    def test_threads(self, threads):
        f10.decode(I10, threads=threads)

    @pytest.mark.parametrize("verbose,rt", [(False, _Decoded), (True, _Decoded_verbose)])
    def test_named_tuple(self, verbose, rt):
        dec = f.decode(I, verbose=verbose)
        assert isinstance(dec, tuple) and isinstance(dec, rt), "Return value isn't a 'namedtuple'."
        assert all(isinstance(item, np.ndarray) for item in dec)


class TestPrecision:
    def test_verbose(self):
        dec = f.decode(I, verbose=True)
        p = (f.x[:, None, :, :, :] % f._l[:, :, None, None, None] / f._l[:, :, None, None, None]) * 2 * np.pi
        p = p.reshape(f.D * f.K, f.Y, f.X, f.C).astype(np.float32, copy=False)
        k = f.x[:, None, :, :, :] // f._l[:, :, None, None, None]
        k = k.reshape(f.D * f.K, f.Y, f.X, f.C).astype(np.int_, copy=False)
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - f.x))
        # dp_max = np.max(np.abs(dec.p - p))
        # r_max = np.max(np.abs(dec.r))
        # u_max = np.max(np.abs(dec.u))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.13)
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.69)
        assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)
        # assert np.allclose(dec.x, f.x, rtol=0, atol=4 * dec.r)  # todo: x within factor * r
        assert np.allclose(dec.x, f.x, rtol=0, atol=4 * dec.u)
        assert np.allclose(dec.p, np.pi, rtol=0, atol=np.pi), "Phase values are not within [0, 2PI]."
        assert np.allclose(dec.p, p, rtol=0, atol=0.0052)
        assert np.allclose(dec.k, k, rtol=0, atol=0), "Fringe orders are wrong."
        assert np.allclose(dec.r, 0, rtol=0, atol=0.09)
        assert np.allclose(dec.u, 0, rtol=0, atol=0.04)

    def test_8K(self):
        f = Fringes()
        f.X = 7680
        f.Y = 3840
        f.v = 13, 7, 89
        I = f.encode()
        dec = f.decode(I)
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - f.x))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.09)
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.69)
        assert np.allclose(dec.x, f.x, rtol=0, atol=0.08)

    def test_single_fringe(self):
        f = Fringes()
        f.K = 1
        f.v = 1
        f.N = 9  # with N = 23: atol=0.48

        I = f.encode()
        dec = f.decode(I)
        # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
        # dx_max = np.max(np.abs(dec.x[:, 1:, 1:, :] - f.x[:, 1:, 1:, :]))
        assert np.allclose(dec.a, f.A, rtol=0, atol=0.17)
        assert np.allclose(dec.b, f.B, rtol=0, atol=0.34)
        assert np.allclose(dec.x[:, 1:, 1:, :], f.x[:, 1:, 1:, :], rtol=0, atol=0.92)  # todo: index 0
        assert np.allclose(circular_distance(dec.x, f.x, f.Lext), 0, rtol=0, atol=0.92)

    @pytest.mark.parametrize("dark_noise", [None, 13.7])
    @pytest.mark.parametrize("K", [None, 0.038])
    @pytest.mark.parametrize("a", [None, 127.5, np.full((f10.D, f10.Y, f10.X, f10.C), 127.5)])
    @pytest.mark.parametrize("b", [None, 88, np.full((f10.D * f10.K, f10.Y, f10.X, f10.C), 88)])
    @pytest.mark.parametrize("ui", [3, np.full((f10.Y, f10.X, f10.C), 3)])
    def test_uncertainty(self, ui, b, a, K, dark_noise):
        u = f.uncertainty(ui, b, a, K, dark_noise)
        assert np.allclose(u, 0, rtol=0, atol=0.5)

    @pytest.mark.parametrize("spu_func", ["ski", "cv2"])  # todo: "cv2" is error-prone!
    def test_spatial_unwrapping(self, spu_func):
        f = Fringes()
        f.K = 1
        # todo: f.v = 7, 14

        I = f.encode()
        dec = f.decode(I, spu_func=spu_func)

        for d, ax in enumerate(f.axes):
            grad = np.gradient(dec.x[d], axis=ax)
            # dg_max = np.max(np.abs(grad - 1))
            # idx = np.argwhere(np.abs(grad - 1) > 0.10)
            assert np.allclose(grad, 1, rtol=0, atol=0.10), (
                f"Gradient of unwrapped phase map isn't close to 1 at axis {ax}."
            )


class TestMisc:
    def test_UMR(self):
        f = Fringes()
        f.l = 20.2, 60.6
        assert np.array_equal(f.UMR, [60.6] * f.D)

    @pytest.mark.parametrize("h", ["ww", "rb", "rgb", "www", "rggb"])
    def test_hues(self, h):
        f = Fringes(Y=10)
        f.h = h

        I = f.encode()
        assert np.array_equal(I, np.array([frame for frame in f]))

        dec = f.decode(I)
        assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)

    @pytest.mark.parametrize("p0", [0, np.pi / 2, np.pi, 3 * np.pi / 2, 1])
    def test_p0(self, p0):
        f = Fringes()
        f.p0 = p0

        I = f.encode()
        dec = f.decode(I)
        if p0 == 1:
            assert np.allclose(dec.x[:, 1:, 1:, :], f.x[:, 1:, 1:, :], rtol=0, atol=0.13)
        else:
            assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)

    @pytest.mark.parametrize("mode", Fringes._choices["mode"])
    def test_modes(self, mode):
        if mode == "sRGB":
            return  # not bijective

        f = Fringes()
        f.mode = mode

        I = f.encode()

        # if mode == "sRGB":
        #     I /= f.Imax
        #     I = np.where(I <= 0.04045, I / 12.92, ((I + 0.055) / 1.055) ** 2.4)
        #     I *= f.Imax

        dec = f.decode(I)
        assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)

    @pytest.mark.parametrize("dtype", Fringes._choices["dtype"])
    def test_dtypes(self, dtype):
        f = Fringes(Y=10)
        f.dtype = dtype

        I = f.encode()
        assert I.dtype == f.dtype

        dec = f.decode(I)
        # da_max = np.max(np.abs(dec.a - f.A))
        # db_max = np.max(np.abs(dec.b - f.B))
        # dx_max = np.max(np.abs(dec.x - f.x))
        if dtype in ["float32", "float64"]:
            # idx = np.argwhere(np.abs(dec.x - f.x) > 2)
            # dx_max = np.max(np.abs(dec.x[:, 1:, :, :] - f.x[:, 1:, :, :]))
            assert np.allclose(dec.x[:, 1:, :, :], f.x[:, 1:, :, :], rtol=0, atol=0.13)  # todo: index 0
        else:
            assert np.allclose(dec.x, f.x, rtol=0, atol=0.13)


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
