import logging
import time

import cv2
import numpy as np

from fringes.util import vshape

logger = logging.getLogger(__name__)


def direct(b: np.ndarray):
    """Direct illumination component.

    Parameters
    ----------
    b : np.ndarray
        Modulation

    Returns
    -------
    d : np.ndarray
        Direct illumination component.

    References
    ----------
    ..  [#] `Nayar et al.,
            “Fast separation of direct and global components of a scene using high frequency illumination”,
            SIGGRAPH,
            2006.
            <https://dl.acm.org/doi/abs/10.1145/1179352.1141977>`_
    """
    return 2 * b


def indirect(a: np.ndarray, b: np.ndarray):
    """Indirect (global) illumination component.

    Parameters
    ----------
    a : np.ndarray
        Brightness.
    b : np.ndarray
        Modulation

    Returns
    -------
    g : np.ndarray
        Indirect (global) illumination component.

    References
    ----------
    ..  [#] `Nayar et al.,
            “Fast separation of direct and global components of a scene using high frequency illumination”,
            SIGGRAPH,
            2006.
            <https://dl.acm.org/doi/abs/10.1145/1179352.1141977>`_
    """
    # todo: assert videoshape of a and b

    D = a.shape[0]
    K = int(b.shape[0] / D)

    g = 2 * (a.reshape(D, 1, -1) - b.reshape(D, K, -1)).reshape(b.shape).clip(0, None)

    return g


def visibility(a: np.ndarray, b: np.ndarray):
    """Visibility.

    Parameters
    ----------
    a : np.ndarray
        Brightness.
    b : np.ndarray
        Modulation

    Returns
    -------
    V : np.ndarray
        Visibility.
    """
    # todo: assert videoshape of a and b

    D, Y, X, C = a.shape
    K = int(b.shape[0] / D)

    V = np.minimum(
        1, b.reshape(D, K, Y, X, C) / np.maximum(a[:, None, :, :, :], np.finfo(np.float_).eps)
    )  # avoid division by zero

    return V.astype(np.float32, copy=False).reshape(D * K, Y, X, C)


def exposure(a: np.ndarray, I_rec: np.ndarray, lessbits: bool = True):
    """Exposure.

    Parameters
    ----------
    a : np.ndarray
        Brightness.
    I_rec : np.ndarray
        Fringe pattern sequence.
    lessbits: bool, optional
        The camera recorded `I_rec` may contain fewer bits of information than its data type can hold,
        e.g. 12 bits for dtype `uint16`.
        If this flag is activated, it looks for the maximal value in `I`
        and sets `Imax` to the same or next power of two which is divisible by two.
        Example: If `I.max()` is 3500, `Imax` is set to 4095 (the maximal value a 12bit camera can deliver).

    Returns
    -------
    E : np.ndarray
        Exposure.
    """

    if I_rec.dtype.kind in "ui":
        if np.iinfo(I_rec.dtype).bits > 8 and lessbits:  # data may contain fewer bits of information
            B = int(np.ceil(np.log2(I_rec.max())))  # same or next power of two
            B += -B % 2  # same or next power of two which is divisible by two
            Imax = 2**B - 1
        else:
            Imax = np.iinfo(I_rec.dtype).max
    else:  # float
        Imax = 1  # assume

    E = a / Imax

    return E.astype(np.float32, copy=False)


def curvature(s: np.ndarray, center: bool = False, normalize: bool = False) -> np.ndarray:  # todo: test
    """Mean curvature map.

    Computed by differentiating a slope map.

    Parameters
    ----------
    s : np.ndarray
        Slope map.
        It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
    center : bool, optional
        If this flag is set to True, the curvature values are centered to zero using the median.
        Default is False.
    normalize : bool
        Flag indicating whether to use the acr-tangent function
        to non-linearly map the codomain from [-inf, inf] to [-1, 1].
        Default is False.

    Returns
    -------
    c : np.ndarray
        Curvature map.
    """

    t0 = time.perf_counter()

    T, Y, X, C = vshape(s).shape
    s = s.reshape(T, Y, X, C)  # returns a view

    assert T == 2, "Number of direction doesn't equal 2."
    assert X >= 2 and Y >= 2, "Shape too small to calculate numerical gradient."

    # Gy = np.gradient(s[0], axis=0) + np.gradient(s[1], axis=0)
    # Gx = np.gradient(s[0], axis=1) + np.gradient(s[1], axis=1)
    # c = np.sqrt(Gx**2 + Gy**2)  # here only positive values!
    c = np.gradient(s[0], axis=0) + np.gradient(s[1], axis=0) + np.gradient(s[0], axis=1) + np.gradient(s[1], axis=1)

    if center:
        # c -= np.mean(c, axis=(0, 1))
        c -= np.median(c, axis=(0, 1))  # Median is a robust estimator for the mean.

    if normalize:
        c = np.arctan(c) * 2 / np.pi  # scale [-inf, inf] to [-1, 1]

    logging.debug(f"{1000 * (time.perf_counter() - t0)}ms")

    return c.reshape(-1, Y, X, C)


# def height(curv: np.ndarray, iterations: int = 3) -> np.ndarray:  # todo: test
#     """Local height map.
#
#     It is computed by iterative local integration via an inverse laplace filter.
#     Think of it as a relief, where height is only relative to the local neighborhood.
#
#     Parameters
#     ----------
#     curv : np.ndarray
#         Curvature map.
#         It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
#     iterations : int, optional
#         Number of iterations of the inverse Laplace filter kernel.
#         Default is 3.
#
#     Returns
#     -------
#     z : np.ndarray
#         Local height map.
#     """
#
#     t0 = time.perf_counter()
#
#     kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
#     # kernel *= iterations  # todo
#
#     T, Y, X, C = vshape(curv).shape
#     curv = curv.reshape(T, Y, X, C)  # returns a view
#
#     if T == 1:
#         curv = np.squeeze(curv, axis=0)  # returns a view
#
#     if curv.min() == curv.max():
#         return np.zeros_like(curv)
#
#     z = np.zeros_like(curv)
#     for c in range(C):
#         for i in range(iterations):
#             z[..., c] = (cv2.filter2D(z[..., c], -1, kernel) - curv[..., c]) / 4
#
#     # todo: residuals
#     # filter2(kernel_laplace, z) - curvature;
#
#     logging.debug(f"{1000 * (time.perf_counter() - t0)}ms")
#
#     return z.reshape(-1, Y, X, C)
