from collections import namedtuple
import glob
import os
import time
import tempfile

import toml
import numpy as np
import pytest
import subprocess

from fringes import __version__
from fringes import Fringes


def test_version():
    assert __version__ == toml.load("..\\pyproject.toml")["tool"]["poetry"]["version"]


def test_init():
    f = Fringes()
    for k, v in f.defaults.items():
        assert np.array_equal(v, getattr(f, "_" + k)),\
            f"'{k}' got overwritten by interdependencies. Choose consistent default values."

    assert np.array_equal(f.l, f.L / f.v), "l != L / v"  # todo: above assert should already handle this


def test_property_docs():
    f = Fringes()
    for p in dir(f):
        if isinstance(getattr(type(f), p, None), property):
            assert getattr(type(f), p, None).__doc__ is not None, f"Property '{p}' has no docstring defined."


def test_init_doc():
    f = Fringes()
    assert f.__init__.__doc__.count("\n") == len(f.defaults),\
        "Not all init parameters have an associated property with a defined docstring."


def test_UMR_mutual_divisibility():
    f = Fringes()
    f.l = np.array([20.2, 60.6])

    assert np.array_equal(f.UMR, [60.6] * f.D), "'UMR' is not 60.6."


def test_encoding():
    f = Fringes(Y=100)

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_one_direction():
    f = Fringes(Y=100)
    f.D = 1
    f.axis = 0

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."

    f.axis = 1

    I = f.encode()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_call():
    f = Fringes(Y=100)

    I = f()
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_iter():
    f = Fringes(Y=100)

    for t, I in enumerate(f):
        assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
        assert I.ndim == 4, "Fringe pattern sequence is not 4-dimensional."
    assert t + 1 == f.T, "Number of iterations does't eual number of frames."

    I = np.array(list(frame[0] for frame in f))
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_encoding_frames():
    f = Fringes(Y=100)
    
    I = np.array([f.encode(t)[0] for t in range(f.T)])
    assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."
    assert I.ndim == 4, "Fringe pattern sequence isn't 4-dimensional."

    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_decoding(rm: bool = False):  # todo: rm = True i.e. test decoding time
    f = Fringes(Y=100)
    f.verbose = True

    I = f.encode()

    # test numba compile time
    if rm:
        flist = glob.glob(os.path.join(os.path.dirname(__file__), "..", "fringes", "__pycache__", "decoder*decode*.nbc"))
        for file in flist:
            os.remove(file)
        t0 = time.perf_counter()
        dec = f.decode(I)
        t1 = time.perf_counter()
        assert t1 - t0 < 10 * 60, f"Numba compilation takes longer than 7 minutes: {(t1 - t0) / 60} minutes."

    dec = f.decode(I)

    # assert isinstance(dec, namedtuple), "Return value isn't a 'namedtuple'."
    assert all(isinstance(item, np.ndarray) for item in dec), "Return values aren't 'Numpy arrays'."

    assert np.max(dec.residuals) < 1, "Residuals are larger than 1."

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."

    if f.mode == "precise":  # todo: decide for each pixel individually i.e. multiply with B later on
        w = f._N * f._v ** 2  # weights of phase measurements are their inverse variances (must be multiplied with m later on)
        idx = [np.argmax(w[d] * (f._N[d] > 2) * (f._v[d] > 0)) for d in range(f.D)]  # the set with most precise phases
    # elif mode == "robust":  # todo: "exhaustive"?
    #     NotImplemented  # todo: try neighboring fringe orders / all permutations = exhaustive search?
    else:  # fast (fallback)
        w = np.ones((f.D, f.K), dtype=np.int_)
        idx = [np.argmax(1 / f._v[d] * (f._N[d] > 2) * (f._v[d] > 0)) for d in range(f.D)]  # the set with the least number of periods
    fid = f.coordinates() // np.array([f._l[d, idx[d]] for d in range(f.D)])[:, None, None, None]
    #assert np.allclose(dec.orders[[d * f.K + idx[d] for d in range(f.D)]], fid, atol=0), "Errors in fringe orders."


def test_decoding_despike():
    f = Fringes(Y=100)
    I = f.encode()
    I[:, 10, 5, :] = I[:, -5, -10, :]

    dec = f.decode(I, despike=True)
    d = dec.registration - f.coordinates()
    print(d.max())
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_deinterlacing():
    f = Fringes(Y=100)

    I = f.encode()
    I = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # interlace
    I = f.deinterlace(I)
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_dtypes():
    f = Fringes(Y=100)

    for dtype in f._dtypes:
        f.dtype = dtype

        I = f.encode()

        assert isinstance(I, np.ndarray), "Return value isn't a 'Numpy array'."


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
#     #assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."


def test_unwrapping():
    f = Fringes()
    f.K = 1
    f.v = 13

    I = f.encode()
    dec = f.decode(I)

    for d in range(f.D):
        grad = np.gradient(dec.registration[d, :, :, 0], axis=1 - d)
        assert np.allclose(grad, 1, atol=0.1), "Gradient of unwrapped phase map isn't close to 1."


def test_unwrapping_class_method():
    f = Fringes()
    f.K = 1
    f.v = 13

    I = f.encode()
    dec = f.decode(I, verbose=True)
    x = Fringes.unwrap(dec.phase)

    for d in range(f.D):
        grad = np.gradient(x[d, :, :, 0], axis=1 - d)
        assert np.allclose(grad, 0, atol=0.2), "Gradient of unwrapped phase map isn't close to 0."


def test_remapping(Y=100):
    f = Fringes()
    f.Y /= 2
    f.verbose = True

    I = f.encode()
    dec = f.decode(I)

    source = f.remap(dec.registration, normalize=True)
    assert np.all(source == 1), "Source doesn't contain only ones."

    source = f.remap(dec.registration, dec.modulation, normalize=True)
    assert np.allclose(source, 1, atol=0.1), "Source doesn't contain only values close to one."

    source = f.remap(dec.registration, scale=2, normalize=True)
    assert np.all(source[0::2, 0::2] == 1), "Source doesn't contain only ones at 'unscaled' coordinates."
    assert np.all(source[0::2, 1::2] == 0), "Source doesn't contain only zeros at 'added' coordinates."
    assert np.all(source[1::2, 0::2] == 0), "Source doesn't contain only zeros at 'added' coordinates."
    assert np.all(source[1::2, 1::2] == 0), "Source doesn't contain only zeros at 'added' coordinates."


def test_hues():
    f = Fringes(Y=100)
    f.h = "rggb"

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."

    f.H = 3  # todo: 2 and take care of M not being a scalar

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_averaging():
    f = Fringes(Y=100)
    f.M = 2

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_WDM():
    f = Fringes(Y=100)
    f.N = 3
    f.WDM = True

    I = f.encode()
    dec = f.decode(I)

    d = dec.registration - f.coordinates()
    assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."


def test_SDM():
    f = Fringes(Y=100)
    f.SDM = True

    I = f.encode()
    dec = f.decode(I)

    for d in range(f.D):
        grad = np.gradient(dec.registration[d, 1:-1, 1:-1, 0], axis=1 - d)  # todo: boarder
        #assert np.allclose(grad, 1, atol=0.1), "Gradient of registration isn't close to 1."

    # d = dec.registration - f.coordinates()
    # assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."  # todo: boarder
    d = dec.registration[:, 1:-1, 1:-1, :] - f.coordinates()[:, 1:-1, 1:-1, :]
    #assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."


def test_FDM():
    f = Fringes(Y=100)
    f.FDM = True

    I = f.encode()
    dec = f.decode(I)

    # d = dec.registration - f.coordinates()
    # assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."  # todo: boarder
    d = dec.registration[:, 1:, 1:, :] - f.coordinates()[:, 1:, 1:, :]
    assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."

    f.static = True
    f.N = 1
    I = f.encode()
    dec = f.decode(I)

    # d = dec.registration - f.coordinates()
    # assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."  # todo: boarder
    d = dec.registration[:, 1:, 1:, :] - f.coordinates()[:, 1:, 1:, :]
    assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."


def test_SDM_WDM():
    f = Fringes(Y=100)
    f.N = 3
    f.SDM = True
    f.WDM = True

    I = f.encode()
    dec = f.decode(I)

    for d in range(f.D):
        grad = np.gradient(dec.registration[d, 1:-1, 1:-1, 0], axis=1 - d)  # todo: boarder
        #assert np.allclose(grad, 1, atol=0.1), "Gradient of registration isn't close to 1."

    # d = dec.registration - f.coordinates()
    # assert np.allclose(d, 0, atol=0.1), "Registration is off more than 0.1."  # todo: boarders
    d = dec.registration[:, 1:-1, 1:-1, :] - f.coordinates()[:, 1:-1, 1:-1, :]
    #assert np.allclose(d, 0, atol=0.5), "Registration is off more than 0.5."


def test_save_load():
    f = Fringes()
    params = f.params

    with tempfile.TemporaryDirectory() as tempdir:
        for ext in [".json", ".yaml", ".asdf"]:  # todo: ".toml"
            fname = os.path.join(tempdir, f"params{ext}")

            f.save(fname)
            assert os.path.isfile(fname), "No params-file saved."

            params_loaded = f.load(fname)
            # assert len(params_loaded) == len(params), "A different number of attributes is loaded than saved."

            for k in params_loaded.keys():
                assert k in params, f"Fringes class has no attribute {k}"
                assert params_loaded[k] == params[k], \
                    f"Attribute {k} in params-file differs from its corresponding class attribute."

            for k in params.keys():
                assert k in params_loaded, f"params-file has no attribute {k}"
                assert params[k] == params_loaded[k], \
                    f"Class attribute {k} differs from its corresponding attribute in params-file."


if __name__ == "__main__":
    # pytest.main()
    subprocess.run(['pytest', '--tb=short', str(__file__)])
