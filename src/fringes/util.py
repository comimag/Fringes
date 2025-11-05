from collections.abc import Sequence
import logging
import time

import cv2
import numba as nb
import numpy as np
import scipy as sp

# import sympy.ntheory.generate

logger = logging.getLogger(__name__)


def vshape(data: np.ndarray, channels: set[int] = (1, 3, 4)) -> np.ndarray:
    """Standardizes the input data shape.

    Transforms video data into the standardized shape (T, Y, X, C), where
    T is number of frames, Y is height, X is width, and C is number of color channels.

    Inspired from `scikit-video <https://www.scikit-video.org/stable/modules/generated/skvideo.utils.vshape.html>`_.

    Parameters
    ----------
    data : ndarray
        Input data in arbitrary shape.
    channels : set of ints, default = {1, 3, 4}
        Allowed number of color channels.

    Returns
    -------
    videodata : ndarray
        Standardized version of data, in shape (T, Y, X, C), where
        T is number of frames, Y is height, X is width, and C is number of color channels.

    Notes
    -----
    Ensures that the array becomes 4-dimensional
    and that the length of the last dimension is in `channels`.
    To do this, leading dimensions may be flattened.

    Examples
    --------
    >>> from fringes import vshape

    >>> data = np.ones(shape=(100))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (100, 1, 1, 1)

    >>> data = np.ones(shape=(1200, 1920))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (1, 1200, 1920, 1)

    >>> data = np.ones(shape=(1200, 1920, 3))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (1, 1200, 1920, 3)

    >>> data = np.ones(shape=(100, 1200, 1920))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (100, 1200, 1920, 1)

    >>> data = np.ones(shape=(100, 1200, 1920, 3))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (100, 1200, 1920, 3)

    >>> data = np.ones(shape=(2, 3, 4, 1200, 1920))
    >>> videodata = vshape(data)
    >>> videodata.shape
    (24, 1200, 1920, 1)
    """
    data = np.array(data)

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


def unzip(I: np.ndarray, T: int) -> np.ndarray:
    """Unzip pattern sequences.

    This applies for pattern sequences
    recorded with a line scan camera,
    where each frame has been displayed and captured
    as the object moved by one pixel.

    Parameters
    ----------
    I : np.ndarray
        Pattern sequence.
        It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
    T : int
        Number of frames of the pattern sequence.

    Returns
    -------
    I : np.ndarray
        Unzipped pattern sequence.

    Raises
    ------
    ValueError
        If the number of frames of `I` and the number of frames of the pattern sequence `T` don't match.

    Examples
    --------
    >>> from fringes import Fringes
    >>> from fringes.util import unzip
    >>> f = Fringes()
    >>> I = f.encode()
    >>> Irec = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # zip; this is how a line camera would record
    >>> Irec = unzip(Irec, f.T)
    """
    t0 = time.perf_counter()

    I = vshape(I)
    T_, Y, X, C = I.shape
    if T_ * Y % T != 0:
        raise ValueError("Number of frames of data and keyword parameter 'T' don't match.")

    # I = I.reshape((T_ * Y, X, C))  # concatenate
    I = I.reshape((-1, T, X, C)).swapaxes(0, 1)  # returns a view

    logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")
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
    Ilin : np.ndarray
        Linearized data.
    """

    # normalize to [0, 1]
    Imax = np.iinfo(I.dtype).max if I.dtype.kind in "ui" else 1 if I.max() < 1 else I.max()
    Ilin = I / Imax

    # estimate gamma correction factor
    med = np.nanmedian(Ilin)  # Median is a robust estimator for the mean.
    g = np.log(med) / np.log(0.5)
    inv_g = 1 / g

    # apply inverse gamma
    Ilin **= inv_g
    Ilin *= Imax

    return Ilin


def fade(
    I: np.ndarray,
    I0: float | np.ndarray = 0.0,
    T: float | np.ndarray = 1.0,
    M: int = 1,
    PSF: float = 0.0,
    K: float = 0.038,  # [DN/electron]
    dark_noise: float = 13.7,  # [electrons]
    dark_current: float = 14.2,  # [electrons]  # some cameras feature a dark current compensation
    bits: int = 8,
    rng: np.random.Generator | int | None = None,
    clip: bool = True,
):
    """Fade image (sequence).

    This includes the modulation transfer function
    (computed from the imaging system's point spread function),
    and intensity noise added by the camera.

    Parameters
    -----------
    I: np.ndarray
        Image (sequence) with values within [0, 1].
    I0 : float | np.ndarray, default=0.0
        Offset with values within [0, 2**`b`-1].
    T : float | np.ndarray, default=1.0
        Transmission factor with values within [0, 1].
    M : int, default=1
        Optical magnification >= 1.
    PSF : float, default=0.0
        Standard deviation of the point spread function, in pixel units.
    K : float, default=0.038 (@ `bits` = 8).
        System gain of the digital camera, in units of DN (digital numbers) / electron.
    dark_noise : float, default=13.7
        Dark noise of the digital camera, in units of electrons.
    dark_current : float, default=14.2
        Dark current of the digital camera, in units of electrons.
    bits : int, default=8
        Number of bits to store information of one pixel.
        This is used to determine the output dtype.
        If e.g. bits equals 8, `Imax` will be 2 ** 8 - 1 = 255 and `dtype` will be `uint8`.
        If e.g. bits equals 0, `Imax` will be 1 and `dtype` will be `float`.
    clip : bool, default=True
        If True (default), the output will be clipped after noise is applied.
        This is needed to maintain the proper image data range.
    rng : {numpy.random.Generator, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see numpy.random.default_rng()).
        If rng is an int, it is used to seed the generator.

    Returns
    -------
    In : np.ndarray
        Faded image (sequence).
    """
    t0 = time.perf_counter()

    # check input
    assert np.issubdtype(I.dtype, np.floating)
    assert I.min() >= 0
    assert I.max() <= 1

    # scale values
    Imax = 1 if bits == 0 else 2**bits - 1
    In = vshape(I) * Imax  # this makes a copy

    # transmission
    if isinstance(T, np.ndarray) or T != 1.0:
        assert np.all(0 <= T <= 0)
        In *= vshape(T)

    # offset
    if isinstance(I0, np.ndarray) or I0 > 0.0:
        assert np.all(I0 >= 0)
        In += vshape(I0)

    # magnification
    if M > 1:  # attention: magnification must be an integer
        In = sp.ndimage.uniform_filter(In, size=M, mode="wrap", axes=(1, 2))

    # blur (e.g. defocus)
    if PSF > 0:
        In = sp.ndimage.gaussian_filter(In, sigma=PSF, order=0, mode="wrap", axes=(1, 2))

    # camera noise
    if K > 0:
        # random number generator
        if rng is None:
            rng = np.random.default_rng()
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)

        # shot noise
        # note: In has to be scaled to final value range first, else the SNR is too bad;
        # then dividing by K yields number of electrons -> calc poisson -> multiplying by K yields DN
        In = rng.poisson(In / K) * K

        # dark signal and dark noise
        if dark_current > 0 or dark_noise > 0:
            In += rng.normal(dark_current * K, dark_noise * K, In.shape)
    # # spatial non-uniformities
    # DSNU = 0  # dark signal non-uniformity
    # # In += DSNU  # todo: dark signal non-uniformity
    # PRNU = 0  # photoresponse non-uniformity
    # # In += PRNU  # todo: photoresponse non-uniformity

    # clip values
    if clip or bits != 0:
        In.clip(0, Imax, out=In)

    # quantization noise
    # In += rng.uniform(0, 1, In.shape)
    if bits >= 1:
        bits += -bits % 8  # next power of two divisible by 8
        dtype = f"uint{bits}"
        In = In.astype(dtype, copy=False)

    logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")
    return In[0] if In.shape[0] == 1 else In


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
        Circular distance between a and b.

    References
    ----------
    .. [#] `Uhlig et al.,
           “A Probabilistic Approach for Spatio-Temporal Phase Unwrapping in Multi-Frequency Phase-Shift Coding”,
           IEEE,
           2022.
           <https://doi.org/10.1109/ACCESS.2022.3174121>`_

    .. [#] `"Circular Statistics in Python: An Intuitive Intro",
           www.insideainews.com,
           2021.
           <https://insideainews.com/2021/02/12/circular-statistics-in-python-an-intuitive-intro/>`_
    """
    # return np.minimum(np.abs(a - b), c - np.abs(a - b))
    return c / 2 - np.abs(c / 2 - np.abs(a - b))


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
#         Größter gemeinsamer Teiler von `a` und `b`.
#
#     u : int
#         Koeffizienten `u` und `v` einer Darstellung von `a` als ganzzahlige Linearmbination.
#
#     v : int
#         Koeffizienten `u` und `v` einer Darstellung von `a` als ganzzahlige Linearmbination.
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
#         Multiplikativ inverse Element von `a` modulo `n`.
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
