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
    Ensures that the array becomes 4-dimensional and that the size of the last dimension is one of the allowed ones.
    """

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


def curvature(reg: np.ndarray, calibrated: bool = False) -> np.ndarray:  # todo: test
    """Local curvature map by deriving a slope map."""

    T, Y, X, C = vshape(reg).shape
    reg = reg.reshape(T, Y, X, C)  # returns a view

    assert T == 2, "More than 2 directions."
    assert X >= 2 and Y >= 2, "Shape too small to calculate numerical gradient."

    curv = np.gradient(reg[0], axis=0) + np.gradient(reg[0], axis=1) + \
           np.gradient(reg[1], axis=0) + np.gradient(reg[1], axis=1)

    if not calibrated:
        # curv -= np.mean(curv, axis=(0, 1))
        curv -= np.median(curv, axis=(0, 1))

    curv = np.arctan(curv) * 2 / np.pi  # scale [-inf, inf] to [-1, 1]

    return curv


def height(curv: np.ndarray, iterations: int = 3) -> np.ndarray:  # todo: test
    """
    Local height map by dual local integration via an inverse laplace filter [[19]](#19).\
    Think of it as a relief, where height is only relative to the local neighborhood.
    """

    k = np.array([[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]], np.float32)
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


# def none2ndarray(func):
#     def call(*args, **kwargs):
#         if "mod" in kwargs and kwargs["mod"] is None:
#             kwargs["mod"] = np.ones(1)
#         elif len(args) > 1 and args[1] is None:
#             args_new = list(args)
#             args_new[1] = np.ones(1)
#             args = tuple(args_new)
#         return func(*args, **kwargs)
#     return call
#
#
# @none2ndarray
# @nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
# def remap(
#         reg: np.ndarray,
#         mod: np.ndarray = np.ones(1),
#         modmin: float = 0,
#         scale: float = 1,
#         Y: int = 0,
#         X: int = 0,
#         C: int = 0,
#         normalize: bool = True,
# ) -> np.ndarray:
#     """
#     Mapping registration points (having sub-pixel accuracy) from camera grid
#     to (integer) positions on screen grid,
#     with weights from modulation.
#     This yields a grid representing the screen (light source)
#     with the pixel values being a relative measure
#     of how much a screen (light source) pixel contributed
#     to the exposure of the camera sensor.
#     """
#
#     if mod.ndim > 1:
#         assert reg.shape[1:] == mod.shape[1:]
#
#     if reg.shape[0] == 1:
#         # mod = np.vstack(mod, np.zeros_like(mod))
#         reg = np.vstack((reg, np.zeros_like(reg)))  # todo: axis
#
#     if X is None:
#         X = 0
#
#     if Y is None:
#         Y = 0
#
#     X = int(X)
#     Y = int(Y)
#
#     if scale <= 0:
#         scale = 1
#
#     if Y <= 0:
#         Y = max(1, int(np.nanmax(reg[1]) * scale + .5))  # todo: mod > modmin
#     else:
#         Y = int(Y * scale + 0.5)
#
#     if X <= 0:
#         X = max(1, int(np.nanmax(reg[0]) * scale + .5))  # todo: mod > modmin
#     else:
#         X = int(X * scale + 0.5)
#
#     if C not in [1, 3, 4]:
#         if reg.shape[-1] in [3, 4]:
#             C = reg.shape[-1]
#         else:
#             C = 1
#             # reg = reg.reshape([s for s in reg.shape] + [C])  # todo: how to get reg[..., 1] if C-axis doesn't exist?
#
#     src = np.zeros((Y, X, C), np.float32)
#
#     Xc = reg.shape[2]
#     Yc = reg.shape[1]
#     K = mod.shape[0]
#     for xc in nb.prange(Xc):
#         for yc in nb.prange(Yc):
#             for c in nb.prange(C):
#                 if not np.isnan(reg[0, yc, xc, c]):
#                     xs = int(reg[0, yc, xc, c] * scale + .5)  # i.e. rint()
#                     if xs < X:
#                         if not np.isnan(reg[1, yc, xc, c]):
#                             ys = int(reg[1, yc, xc, c] * scale + .5)  # i.e. rint()
#                             if ys < Y:
#                                 for k in nb.prange(K):
#                                     if mod.ndim > 1:
#                                         m = mod[k, yc, xc, c]
#                                         if not np.isnan(m) and m >= modmin:
#                                             src[ys, xs, c] += m
#                                     else:
#                                         src[ys, xs, c] += 1
#
#     if normalize:
#         mx = src.max()
#         if mx > 0:
#             src /= mx
#
#     return src


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


def close(K: int, L: int, lmin: int) -> np.ndarray:
    if K == 1:
        l = np.array([max(lmin, L)])
    elif K == 2:  # 2 consecutive integers are always coprime
        r = max(lmin, int(np.sqrt(L) + 0.5))  # round(root)
        l = np.array([r, r + 1])
    else:
        l = lmin + np.arange(K)
        while (np.lcm.reduce(l)) < L:
            l += 1

        # lmax = int(L ** (1 / K) + 0.5)  # wavelengths are around self.L ** (1 / self.K)
        # p = np.array([lmax])
        # if K >= 2:
        #     lmax += 1
        #     p = np.append(p, p + 1)
        #     if K > 2:
        #         ith = K - 2
        #         lmax = sympy.ntheory.generate.nextprime(lmax, ith)
        #         p = np.append(p, -1)
        #
        # r = lmax - lmin - Kclip
        # l2 = lmin + np.arange(Kclip)
        # for _ in range(r):
        #     umr = np.lcm.reduce(l2)
        #     if umr < L:
        #         l2 += 1
        #     else:
        #         break
    return l


def small(K: int, L: int, lmin: int) -> np.ndarray:
    lclose = close(K, L, lmin)
    summax = np.sum(lclose)
    # lmax = max(lmin + K, summax - (K - 1) * lmin - (K - 1) * (K - 2) // 2)
    lmax = max(lmin + K - 1, summax - (K - 1) * lmin - (K - 1) * (K - 2) // 2)  # max() because if K == 2 -> lmax = 0
    n = lmax - lmin + 1  # number of things, is always > K
    C = sp.special.comb(n, K, exact=True, repetition=False)  # number of unique combinations

    combos = np.array([c for c in it.combinations(range(lmin, lmax + 1), K) if np.sum(c) <= summax and np.lcm.reduce(c) >= L])

    l = combos[0]
    mn = np.sum(l)
    summin = K * lmin + K * (K - 1) // 2
    for c in combos[1:]:
        sum = np.sum(c)  # np.max(c)

        if sum == mn:
            if np.sum(c ** 2) < np.sum(l ** 2):  # take set with the smallest wavelength; todo: necessary? smallest squared wavelength?
                l = c
            elif np.sum(c ** 2) == np.sum(l ** 2):
                # print("==1")
                pass
        elif sum < mn:
            mn = sum
            l = c

        if sum <= summin:
            break
    return l


def exponential(L: int, lmin: int, F: int = 2) -> np.ndarray:
    # log(L) - log(lmin) = log(L / lmin) = log(vmax)
    Kexp = int(np.ceil(np.log(L / lmin) / np.log(F))) + 1  # Number of sets required for hierarchical (temporal) phase unwrapping using the reversed exponential sequence approach.
    l = np.ones(Kexp) * lmin * F ** np.arange(0, Kexp)

    # todo: do upper and lower formula generate the same results?

    # geometrical progression factor F should be an irrational number to minimize the number of sets required
    l2 = np.array([lmin])
    while umr(tuple(l2), tuple(L / l2)) < L:
        l2.append(l2[-1] * F)
    return l


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