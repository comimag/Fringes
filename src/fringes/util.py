import importlib
import logging
import os
import time

import cv2
import numba as nb
import numpy as np
import skimage as ski
import scipy as sp
import toml

# import sympy.ntheory.generate

logger = logging.getLogger(__name__)


def _version():
    """Version of package.

    Use version string in 'pyproject.toml' as the single source of truth."""

    try:
        # in order not to confuse an installed version of a package with a local one,
        # first try the local one (not being installed)
        meta = toml.load(os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml"))
        version = meta["project"]["version"]  # Python Packaging User Guide expects version here
    except KeyError:
        version = meta["tool"]["poetry"]["version"]  # Poetry expects version here
    except FileNotFoundError:
        version = importlib.metadata.version("fringes")  # installed version

    return version


def vshape(data: np.ndarray, channels: list | tuple = (1, 3, 4)) -> np.ndarray:
    """Standardizes the input data shape.

    Transforms video data into the standardized shape (T, Y, X, C), where
    T is number of frames, Y is height, X is width, and C is number of color channels.

    Inspired from `scikit-video <http://www.scikit-video.org/stable/modules/generated/skvideo.utils.vshape.html>`_.

    Parameters
    ----------
    data : ndarray
        Input data of arbitrary shape.
    channels : list | tuple
        Allowed number of color channels.

    Returns
    -------
    videodata : ndarray
        Standardized version of data, in shape (T, Y, X, C), where
        T is number of frames, Y is height, X is width, and C is number of color channels.

    Notes
    -----
    Ensures that the array becomes 4-dimensional
    and that the length of the last dimension is in 'channels').
    To do this, leading dimensions may be flattened.

    Examples
    --------
    >>> from fringes import vshape

    >>> data = np.ones(100)
    >>> videodata = vshape(data)
    >>> videodata.shape
    (100, 1, 1, 1)

    >>> data = np.ones((1200, 1920))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (1, 1200, 1920, 1)

    >>> data = np.ones((1200, 1920, 3))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (1, 1200, 1920, 3)

    >>> data = np.ones((100, 1200, 1920))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (100, 1200, 1920, 1)

    >>> data = np.ones((100, 1200, 1920, 3))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (100, 1200, 1920, 3)

    >>> data = np.ones((2, 3, 4, 1200, 1920))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (24, 1200, 1920, 1)
    """
    if data.ndim == 0:
        data = data.reshape(1)  # returns a view

    if data.ndim > 4:
        if data.shape[-1] in channels:
            T = np.prod(data.shape[:-3])
            Y, X, C = data.shape[-3:]
        else:
            T = np.prod(data.shape[:-2])
            Y, X = data.shape[-2:]
            C = 1
    elif data.ndim == 4:
        if data.shape[-1] in channels:
            T, Y, X, C = data.shape
        else:
            T = np.prod(data.shape[:2])
            X, Y = data.shape[2:]
            C = 1
    elif data.ndim == 3:
        if data.shape[-1] in channels:
            T = 1
            Y, X, C = data.shape
        else:
            T, Y, X = data.shape
            C = 1
    elif data.ndim == 2:
        T = 1
        Y, X = data.shape
        C = 1
    elif data.ndim == 1:
        T = data.shape[0]
        Y = 1
        X = 1
        C = 1

    return data.reshape(T, Y, X, C)  # returns a view


def deinterlace(self, I: np.ndarray, T_: int) -> np.ndarray:
    """Deinterlace pattern sequences.

    This applies for pattern sequences
    recorded with a line scan camera,
    where each frame has been displayed and captured
    as the object moved one pixel.

    Parameters
    ----------
    I : np.ndarray
        Pattern sequence.
        It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
    T_ : int
        Number of frames of the pattern sequence.

    Returns
    -------
    I : np.ndarray
        Deinterlaced pattern sequence.

    Raises
    ------
    ValueError
        If the number of frames of `I` and the number of frames of the pattern sequence 'T_' don't match.

    Examples
    --------
    >>> from fringes import Fringes
    >>> from fringes.util import deinterlace
    >>> f = Fringes()
    >>> I = f.encode()
    >>> I_rec = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # interlace; this is how a line camera would record
    >>> I_rec = deinterlace(I_rec)
    """
    t0 = time.perf_counter()

    T, Y, X, C = vshape(I).shape
    if T * Y % T_ != 0:
        raise ValueError("Number of frames of data and keyword parameter 'T_' don't match.")

    # I = I.reshape((T * Y, X, C))  # concatenate
    I = I.reshape((-1, T_, X, C)).swapaxes(0, 1)  # returns a view

    logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

    return I


def simulate(
    I: np.ndarray,
    # M: float = 1,
    PSF: float = 0,
    system_gain: float = 0.038,
    dark_current: float = 3.64 / 0.038,  # [electrons]  # some cameras feature a dark current compensation
    dark_noise: float = 13.7,  # [electrons]
    seed: int = 268664434431581513926327163960690138719,  # secrets.randbits(128)
) -> np.ndarray:
    """Simulate the recording, i.e. the transmission channel.

    This includes the modulation transfer function (computed from the imaging system's point spread function)
    and intensity noise added by the camera.

    Parameters
    ----------
    I : np.ndarray
        Fringe pattern sequence.

    PSF : float, optional
        Standard deviation of the Point Spread Function, in pixel units.
        The default is 0.

    system_gain : float, optional
        System gain of the digital camera.
        The default is 0.038.

    dark_current : float, optional
        Dark current of the digital camera, in unit electrons.
        The default is ~100.

    dark_noise : float, optional
        Dark noise of the digital camera, in units electrons.
        The default is 13.7.

    seed : int, optional
        A seed to initialize the Random Number Generator.
        It makes the random numbers predictable.
        See `Seeding and Entropy <https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy>`_ for more information about seeding.

    Returns
    -------
    I : np.ndarray
        Simulated fringe pattern sequence.
    """

    # M : float
    #         Optical magnification of the imaging system.

    t0 = time.perf_counter()

    I.shape = vshape(I).shape
    dtype = I.dtype
    I = I.astype(float, copy=False)

    # # magnification
    # if magnification != 1:  # attention: magnification must be an integer
    #     I = sp.ndimage.uniform_filter(I, size=magnification, mode="reflect", axes=(1, 2))

    # # magnification
    # if M != 1:  # todo: float magnification
    #     I = sp.ndimage.uniform_filter(I, size=M, mode="nearest", axes=(1, 2))

    # PSF (e.g. defocus)
    if PSF != 0:
        I = sp.ndimage.gaussian_filter(I, sigma=PSF, order=0, mode="nearest", axes=(1, 2))

    if system_gain > 0:
        # random number generator
        rng = np.random.default_rng(seed)

        # add shot noise
        shot = (rng.poisson(I) - I) * np.sqrt(system_gain)
        shot_new = rng.poisson(I / system_gain) * system_gain - I
        I_shot = rng.poisson(I / system_gain) * system_gain
        I += shot
        # todo: int
        # s_ = np.std(shot)

        if dark_current > 0 or dark_noise > 0:
            # add dark signal and dark noise
            dark_current_y = dark_current * system_gain
            dark_noise_y = dark_noise * system_gain
            dark = rng.normal(dark_current_y, dark_noise_y, I.shape)
            I += dark
            # d_ = np.std(dark)

    DSNU = 0  # todo: spatial non-uniformity
    if DSNU:
        # add spatial non-uniformity
        I += DSNU

    # clip values
    if system_gain > 0 or np.any(DSNU > 0):
        Imax = np.iinfo(dtype).max if dtype.kind in "ui" else 1
        np.clip(I, 0, Imax, out=I)

    # quantization noise is added by converting to integer
    I = I.astype(dtype, copy=False)

    logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

    return I


def gamma_auto_correct(I: np.ndarray) -> np.ndarray:
    """Automatically estimate and apply the gamma correction factor
    to linearize the display/camera response curve.

    Parameters
    ----------
    I : np.ndarray
        Recorded data.

    Returns
    -------
    I_lin : np.ndarray
        Linearized data.
    """

    # normalize to [0, 1]
    Imax = np.iinfo(I.dtype).max if I.dtype.kind in "ui" else 1 if I.max() < 1 else I.max()
    I = I / Imax

    # estimate gamma correction factor
    med = np.nanmedian(I)  # Median is a robust estimator for the mean.
    gamma = np.log(med) / np.log(0.5)
    inv_gamma = 1 / gamma

    # apply inverse gamma
    # table = np.array([((g / self.Imax) ** invGamma) * self.Imax for g in range(self.Imax + 1)], self.dtype)
    # I = cv2.LUT(I, table)
    I **= inv_gamma
    I *= Imax

    return I


# def degamma(I):  # todo degamma
#     """Gamma correction.
#
#     Assumes equally distributed histogram."""
#     NotImplemented


def circular_distance(a: float | np.ndarray, b: float | np.ndarray, c: float) -> np.ndarray:
    """Circular distance.

    Parameters
    ----------
    a : float or np.ndarray
        Start points.
    b : float or np.ndarray
        End points.
    c : float
        Circumference (distance) after which wrapping occurs.

    Returns
    -------
    d : float or np.ndarray
        Circular distance from a to b.

    Notes
    -----
    For more details, see https://ieeexplore.ieee.org/document/9771407.
    """

    # dmax = c / 2
    # d = b - a
    # if d > dmax:
    #     d -= c
    # elif d < -dmax:
    #     d += c

    # cd = np.minimum(np.abs(a - b), c - np.abs(a - b))
    cd = c / 2 - np.abs(c / 2 - np.abs(a - b))

    return cd


# @nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
# def circ_dist(a, b, c) -> float:
#     d = b - a
#     dmax = c / 2
#
#     if d > dmax:
#         d -= c
#     elif d < -dmax:
#         d += c
#     return d


def bilateral(I: np.ndarray, k: int = 7) -> np.ndarray:  # todo: test
    """Bilateral filter.

    Edge-preserving, denoising filter.

    Parameters
    ----------
    I : np.ndarray
        Input data.
        It is reshaped to video-shape (frames 'T', height 'Y', width 'X', color channels 'C') before processing.
    k : int, optional
        Size of the filter kernel.
        Default is 7.

    Returns
    -------
    out : np.ndarray
        Filtered data.
    """

    t0 = time.perf_counter()

    T, Y, X, C = vshape(I).shape
    I = I.reshape(T, Y, X, C)
    out = np.empty_like(I)

    for t in range(T):
        if C in [1, 3]:
            rv = ski.restoration.denoise_bilateral(I[t], win_size=k, sigma_spatial=1, channel_axis=-1)
            out[t] = rv if C == 3 else rv[..., None]
            # out[t] = cv2.bilateralFilter(I[t], d=k, sigmaColor=np.std(I[t]), sigmaSpace=1)
        else:
            for c in range(C):
                out[t, :, :, c] = ski.restoration.denoise_bilateral(I[t, :, :, c], win_size=k, sigma_spatial=1)
                # out[t, :, :, c] = cv2.bilateralFilter(I[t], d=k, sigmaColor=np.std(I[t, :, :, c]), sigmaSpace=1)

    logging.debug(f"{1000 * (time.perf_counter() - t0)}ms")

    return out


# # @nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)  # todo
# def filter(combos, K, L, lmin):
#     kroot = L ** (1 / K)
#     if lmin <= kroot:
#         lcombos = np.array([l for l in combos if np.any(l > kroot) and np.lcm.reduce(l) >= L])
#     else:
#         lcombos = np.array([l for l in combos if np.lcm.reduce(l) >= L])
#
#     return lcombos
#
#
# def coprime(n: list[int] | tuple[int] | np.ndarray) -> bool:  # n: iterable  # todo: extend to rational numbers
#     """Test whether numbers are pairwise co-prime.
#
#     Parameters
#     ----------
#     n : list, tuple, np.ndarray
#         Integer numbers.
#
#     Returns
#     -------
#     iscoprime : bool
#         True if numbers are pairwise co-prime, else False.
#     """
#
#     # convert to array and flatten
#     n = np.array(n).ravel()  # returns a view
#
#     # check whether iterable has entries
#     if n.size == 0:
#         return False
#
#     # check whether numbers are integers
#     if not np.all([i % 1 == 0 for i in n]):
#         return False
#
#     # ensure numbers are integers
#     if n.dtype != int:
#         n = n.astype(int, copy=False)
#
#     # check pairwise for coprimality, i.e. gcd(a, b) == 1
#     for i in range(n.size):  # each combination; number of combinations = n.size * (n.size - 1) / 2
#         for j in range(i + 1, n.size):
#             if np.gcd(n[i], n[j]) != 1:
#                 return False
#
#     # alternatively: np.lcm.reduce(n) == np.prod(n)
#
#     return True
#
#
# def extgcd(a, b):
#     """Erweiterter euklidischer Algorithmus.
#
#     https://hwlang.de/krypto/algo/euklid-erweitert.htm
#
#     Parameters
#     ----------
#     a : int
#         Ganzzahl.
#
#     b : int
#         Ganzzahl.
#
#     Returns
#     -------
#     a : int
#         Größter gemeinsamer Teiler von 'a' und 'b'.
#
#     u : int
#         Koeffizienten 'u' und 'v' einer Darstellung von 'a' als ganzzahlige Linearmbination.
#
#     v : int
#         Koeffizienten 'u' und 'v' einer Darstellung von 'a' als ganzzahlige Linearmbination.
#     """
#
#     u, v, s, t = 1, 0, 0, 1
#     while b != 0:
#         q = a // b
#         a, b = b, a - q * b
#         u, s = s, u - q * s
#         v, t = t, v - q * t
#     return a, u, v
#
#
# def modinverse(a, n):
#     """Berechnet das multiplikativ inverse Element von a modulo n.
#
#     https://hwlang.de/krypto/grund/inverses-element.htm
#
#     Parameters
#     ----------
#     a : int
#         Ganzzahl.
#
#     n : int
#         Modul.
#
#     Returns
#     -------
#     mi : int
#         Multiplikativ inverse Element von 'a' modulo 'n'.
#     """
#
#     g, u, v = extgcd(a, n)
#     return u % n
#
#
# def chineseRemainder(nn, rr):
#     """Chinesischer-Restsatz-Algorithmus.
#
#     Der Vorteil dieser Implementierung nach dem Divide-and-Conquer-Prinzip besteht darin,
#     dass in den unteren Rekursionsebenen viele Berechnungen mit kleinen Zahlen stattfinden
#     und erst in den oberen Rekursionsebenen wenige Berechnungen mit großen Zahlen.
#
#     https://hwlang.de/krypto/algo/chinese-remainder.htm
#
#     Parameters
#     ----------
#     nn : np.ndarray, list
#         Liste paarweise teilerfremder Moduln.
#
#     rr : np.ndarray, list
#         Liste der Reste.
#
#     Returns
#     -------
#     mn : int
#         Produkt der Moduln.
#
#     x : int
#         Zahl x nach dem chinesischen Restsatz.
#     """
#
#     if len(nn) == 1:
#         return nn[0], rr[0]
#     else:
#         k = len(nn) // 2
#         m, a = chineseRemainder(nn[:k], rr[:k])
#         n, b = chineseRemainder(nn[k:], rr[k:])
#         g, u, v = extgcd(m, n)
#         x = (b - a) * u % n * m + a
#         return m * n, x
#
#
# def modinv(x, p):
#     """Modular multiplicative inverse.
#
#     y = invmod(x, p) such that x*y == 1 (mod p)
#
#     https://bugs.python.org/issue36027
#
#     Parameters
#     ----------
#
#     x : int
#         Integer.
#
#     p : int
#         Modul.
#
#     Returns
#     -------
#     y : Modular multiplicative inverse.
#     """
#     return pow(x, -1, p)
