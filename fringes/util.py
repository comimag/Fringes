import time
import typing as tp
import itertools as it
import logging as lg

import numpy as np
import numba as nb
import scipy as sp

# import sympy.ntheory.generate
import skimage as ski
import cv2


def vshape(I: np.ndarray) -> np.ndarray:
    """Standardizes the input data shape.
    Transforms video data of arbitrary shape and dimensionality into the standardized shape (T, Y, X, C), where
    T is number of frames, Y is height, X is width, and C is number of color channels.
    Ensures that the array becomes 4-dimensional and that the size of the last dimension is in {1, 3, 4}.
    To do this, leading dimensions may be flattened."""

    assert isinstance(I, np.ndarray)
    assert I.ndim > 0

    T = Y = X = C = 1  # init values
    channels = [1, 3, 4]  # possible number of color channels

    if I.ndim > 4:
        if I.shape[-1] in channels:
            T = np.prod(I.shape[:-3])
            Y, X, C = I.shape[-3:]
        else:
            T = np.prod(I.shape[:-2])
            Y, X = I.shape[-2:]
    elif I.ndim == 4:
        if I.shape[-1] in channels:
            T, Y, X, C = I.shape
        else:
            T = np.prod(I.shape[:2])
            X, Y = I.shape[2:]
            C = 1
    elif I.ndim == 3:
        if I.shape[-1] in channels:
            Y, X, C = I.shape
        else:
            T, Y, X = I.shape
    elif I.ndim == 2:
        Y, X = I.shape
    elif I.ndim == 1:
        T = I.shape

    return I.reshape(T, Y, X, C)


def circular_distance(a: np.ndarray, b: np.ndarray, c: float) -> np.ndarray:
    """Shortest circular distance from a to b.
    param a: start point
    param b: end point
    param c: circumference (distance) after which wrapping occurs
    from: https://ieeexplore.ieee.org/document/9771407"""

    d = c / 2 - np.abs(c / 2 - (a - b))  # todo: check

    return d


def curvature(reg: np.ndarray, calibrated: bool = False) -> np.ndarray:  # todo: test
    """Local curvature map by differentiating a slope map."""

    T, Y, X, C = vshape(reg).shape
    reg = reg.reshape(T, Y, X, C)  # returns a view

    assert T == 2, "More than 2 directions."
    assert X >= 2 and Y >= 2, "Shape too small to calculate numerical gradient."

    curv = (
        np.gradient(reg[0], axis=0)
        + np.gradient(reg[0], axis=1)
        + np.gradient(reg[1], axis=0)
        + np.gradient(reg[1], axis=1)
    )

    if not calibrated:
        # curv -= np.mean(curv, axis=(0, 1))
        curv -= np.median(curv, axis=(0, 1))

    curv = np.arctan(curv) * 2 / np.pi  # scale [-inf, inf] to [-1, 1]

    return curv


def height(curv: np.ndarray, iterations: int = 3) -> np.ndarray:  # todo: test
    """Local height map by iterative local integration via an inverse laplace filter.
    Think of it as a relief, where height is only relative to the local neighborhood."""

    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    # k *= iterations  # todo

    T, Y, X, C = vshape(curv).shape
    curv = curv.reshape(T, Y, X, C)  # returns a view

    if T == 1:
        curv = np.squeeze(curv, axis=0)  # returns a view

    assert curv.ndim <= 3
    assert curv.max() != curv.min()

    z = np.zeros_like(curv)
    for c in range(C):
        for i in range(iterations):
            z[..., c] = (cv2.filter2D(z[..., c], -1, k) - curv[..., c]) / 4

    # todo: residuals
    # filter2(kernel_laplace, z) - curvature;

    return z


def bilateral(img: np.ndarray, k: int = 7) -> np.ndarray:
    """Bilateral filter.
    Edge-preserving, denoising filter.
    https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_bilateral
    https://docs.opencv.org/4.5.2/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    """
    T, Y, X, C = vshape(img).shape
    img = img.reshape(T, Y, X, C)
    out = np.empty_like(img)

    for t in range(T):
        if C in [1, 3]:
            rv = ski.restoration.denoise_bilateral(img[t], win_size=k, sigma_spatial=1, channel_axis=-1)
            out[t] = rv if C == 3 else rv[..., None]
            # out[t] = cv2.bilateralFilter(img[t], d=k, sigmaColor=np.std(img[t]), sigmaSpace=1)
        else:
            for c in range(C):
                out[t, :, :, c] = ski.restoration.denoise_bilateral(img[t, :, :, c], win_size=k, sigma_spatial=1)
                # out[t, :, :, c] = cv2.bilateralFilter(img[t], d=k, sigmaColor=np.std(img[t, :, :, c]), sigmaSpace=1)

    return out


def median(img: np.ndarray, k: int = 3) -> np.ndarray:  # todo: test
    """Median filter.
    Removes salt and pepper noise.
    https://docs.opencv.org/4.5.2/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html
    """
    T, Y, X, C = vshape(img).shape
    img = img.reshape(T, Y, X, C)
    out = np.empty_like(img)

    for t in range(T):
        for c in range(C):
            if 1 < k <= 5 and k % 2 == 1 and img.dtype == np.uint8:  # use opencv for faster performance
                out[t, :, :, c] = cv2.medianBlur(img[t, :, :, c], k=k)
            else:
                out[t, :, :, c] = sp.ndimage.median_filter(img[t, :, :, c], size=(k, k), mode="nearest")

    return out


@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def remap(
    reg: np.ndarray,
    mod: np.ndarray = np.ones(1),
    scale: float = 1,
    Y: int = 0,
    X: int = 0,
    C: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Mapping registration points (having sub-pixel accuracy) from camera grid
    to (integer) positions on screen grid,
    with weights from modulation.
    This yields a grid representing the screen (light source)
    with the pixel values being a relative measure
    of how much a screen (light source) pixel contributed
    to the exposure of the camera sensor.
    """

    if mod.ndim > 1:
        assert reg.shape[1:] == mod.shape[1:]

    if reg.shape[0] == 1:
        # mod = np.vstack(mod, np.zeros_like(mod))
        reg = np.vstack((reg, np.zeros_like(reg)))  # todo: axis

    if X is None:
        X = 0

    if Y is None:
        Y = 0

    X = int(X)
    Y = int(Y)

    if scale <= 0:
        scale = 1

    if Y <= 0:
        Y = max(1, int(np.nanmax(reg[1]) * scale + 0.5))
    else:
        Y = int(Y * scale + 0.5)

    if X <= 0:
        X = max(1, int(np.nanmax(reg[0]) * scale + 0.5))
    else:
        X = int(X * scale + 0.5)

    if C not in [1, 3, 4]:
        if reg.shape[-1] in [3, 4]:
            C = reg.shape[-1]
        else:
            C = 1
            # reg = reg.reshape([s for s in reg.shape] + [C])  # todo: how to get reg[..., 1] if C-axis doesn't exist?

    src = np.zeros((Y, X, C), np.float32)

    Xc = reg.shape[2]
    Yc = reg.shape[1]
    K = mod.shape[0]
    for xc in nb.prange(Xc):
        for yc in nb.prange(Yc):
            for c in nb.prange(C):
                if not np.isnan(reg[0, yc, xc, c]):
                    xs = int(reg[0, yc, xc, c] * scale + 0.5)  # i.e. rint()
                    if xs < X:
                        if not np.isnan(reg[1, yc, xc, c]):
                            ys = int(reg[1, yc, xc, c] * scale + 0.5)  # i.e. rint()
                            if ys < Y:
                                for k in nb.prange(K):
                                    if mod.ndim > 1:
                                        m = mod[k, yc, xc, c]
                                        if not np.isnan(m):
                                            src[ys, xs, c] += m
                                    else:
                                        src[ys, xs, c] += 1

    if normalize:
        mx = src.max()
        if mx > 0:
            src /= mx

    return src


# @nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def filter(combos, K, L, lmin):
    kroot = L ** (1 / K)
    if lmin <= kroot:
        lcombos = np.array([l for l in combos if np.any(l > kroot) and np.lcm.reduce(l) >= L])
    else:
        lcombos = np.array([l for l in combos if np.lcm.reduce(l) >= L])

    return lcombos


def coprime(n: list[int] | tuple[int] | np.ndarray) -> bool:  # n: iterable  # todo: extend to rational numbers
    """Test whether numbers are pairwise co-prime."""
    n = np.array(n).ravel()  # return view

    if n.size == 0:  # check whether iterable has entries
        return False

    if not np.all([i % 1 == 0 for i in n]):  # check whether numbers are integers
        return False

    if n.dtype != int:  # convert numbers to integers
        n = n.astype(int, copy=True)  # return copy

    for i in range(len(n)):  # each combination; number of combinations = len(n) * (len(n) - 1) / 2
        for j in range(i + 1, len(n)):
            if n[j] == 1 or np.gcd(n[i], n[j]) != 1:
                return False

    return True


def coprime2(n: list[int] | tuple[int]) -> bool:
    """Test whether numbers in list are pairwise co-prime."""
    if n:  # check whether list has entries
        if all(i % 1 == 0 for i in n):  # check whether numbers are integers
            if not all(isinstance(int, i) for i in n):  # ensure numbers are integers
                n = [int(i) for i in n]

            if np.lcm.reduce(n) == np.prod(n):
                return True
    return False


if __name__ == "__main__":
    l = np.array([29, 31])
    # l = np.array([9, 10, 11])
    # l = np.array([12, 17, 19])
    # l = np.array([4, 5, 7])
    X = 1920
    v = X // l

    # l = v
    m = coefficients(l)  # todo: numba, LUT?
    x = 35
    p = x % l
    crtOK = np.lcm.reduce(l) == np.prod(l)
    i, f = np.divmod(p, 1)  # integer and fractional part
    umr = np.prod(l)  # np.lcm.reduce(l)
    lcm = np.lcm.reduce(l)
    x2 = np.sum(m * i) % umr + np.mean(f)
    x3 = np.sum(m[::-1] * i) % umr + np.mean(f)
    idx0 = x2 // l

    a = 1
