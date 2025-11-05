import logging

import cv2  # only for spu()
import numpy as np
import scipy as sp  # only for ftm()
import skimage as ski  # only for spu()

from fringes.decoder_numba import decode

# from fringes.decoder_ import decode

logger = logging.getLogger(__name__)


_2PI: float = 2 * np.pi


def temp_demod_numpy_unknown_frequencies(I, N, p0: float = np.pi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal demodulation using `numpy`.

    Parameters
    ----------
    I : np.ndarray
        Fringe pattern sequence.
        Must be in `vshape` (frames `T`, height `Y`, width `X`, color channels `C`).
    N : np.ndarray
        Number of phase shifts.
        Must be in shape (number of directions `D`, number of sets `K`).
    p0 : float, default=np.pi
        Phase offset.

    Returns
    -------
    a : np.ndarray
        Brightness: average signal.
    b : np.ndarray
        Modulation: amplitude of the cosine signal.
    p : np.ndarray, optional
        Local phase.
    """
    T, Y, X, C = I.shape
    D, K = N.shape
    a = np.empty(shape=(D, Y, X, C), dtype=np.float32)
    b = np.empty(shape=(D, K, Y, X, C), dtype=np.float32)
    p = np.empty(shape=(D, K, Y, X, C), dtype=np.float32)
    t0 = 0
    for d in range(D):
        for i in range(K):
            I_ = I[t0 : t0 + N[d, i]]  # real signal -> spectrum is Hermitian ("conjugate symmetric")
            # c = np.fft.fft(I_, axis=0)
            c = np.fft.rfft(I_, axis=0)
            avg = np.abs(c)
            # a[d] += np.sum(I_)
            a[d] += avg[0]
            idx = np.argmax(avg[1:], axis=0)
            idx = int(np.median(idx)) + 1  # median is a robust estimator for the mean
            cidx = c[idx]  # usually frequency '1'
            b[d, i] = np.abs(cidx) / N[d, i]  # todo: * 2  # * 2: also add amplitudes of frequencies with opposite sign
            # p[d, i] = -np.angle(cidx * np.exp(-1j * (p0 - np.pi))) % _2PI  # todo: why p0 - PI???
            cidx *= np.exp(1j * p0)  # shift back by p0
            p[d, i] = np.angle(cidx) % _2PI  # todo: test p0
            t0 += N[d, i]
    return a, b, p


def temp_demod_numpy(I, N, f, p0: float = np.pi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal demodulation using `numpy`.

    Parameters
    ----------
    I : np.ndarray
        Fringe pattern sequence.
        Must be in `vshape` (frames `T`, height `Y`, width `X`, color channels `C`).
    N : np.ndarray
        Number of phase shifts.
        Must be in shape (number of directions `D`, number of sets `K`).
    f : np.ndarray
        Temporal frequencies.
        Must be in shape (number of directions `D`, number of sets `K`).
    p0 : float, default=np.pi
        Phase offset.

    Returns
    -------
    a : np.ndarray
        Brightness: average signal.
    b : np.ndarray
        Modulation: amplitude of the cosine signal.
    p : np.ndarray, optional
        Local phase.
    """
    T, Y, X, C = I.shape
    D, K = N.shape
    a = np.empty(shape=(D, Y, X, C), dtype=np.float32)
    b = np.empty(shape=(D, K, Y, X, C), dtype=np.float32)
    p = np.empty(shape=(D, K, Y, X, C), dtype=np.float32)
    t0 = 0
    for d in range(D):
        for i in range(K):
            I_ = I[t0 : t0 + N[d, i]]
            a[d] += np.sum(I_)
            t_ = np.arange(N[d, i]) / N[d, i]  # temporal sampling points
            s = np.exp(1j * (_2PI * f[d, i] * t_ + p0))  # complex filter i.e. sampling points on unit circle
            # z = np.sum(I_ * c[:, None, None, None], axis=0)  # weighted sum -> complex phasor
            z = np.dot(
                np.moveaxis(I_, 0, -1), s
            )  # 'If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.'
            b[d, i] = z / N[d, i] * 2  # * 2: also add amplitudes of frequencies with opposite sign
            p[d, i] = np.angle(z) % _2PI  # arctan2 maps to [-PI, PI], but we need [0, 2PI)  # todo: test p0
            t0 += N[d, i]
    return a, b, p


def spu(p: np.ndarray, verbose: bool = True, uwr_func: str = "ski") -> np.ndarray:
    """Unwrap phase maps spatially.

    Parameters
    ----------
    p : np.ndarray
        Phase maps to unwrap spatially, stacked along the first dimension.
        Must be in image-shape (height `Y`, width `X`, color channels `C`).
        The frames (first dimension) as well the color channels (last dimension)
        are unwrapped separately.
    verbose : bool, default=False
        Flag for computing InverseReliabilityMap if `uwr_func` is 'cv2'.
    uwr_func : {'ski', 'cv2'}, default='ski'
        Unwrapping function to use.

        - 'ski': `Scikit-image[1]_ <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_

        - 'cv2': `OpenCV[2]_ <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_

    Returns
    -------
    puwr : np.ndarray
        Unwrapped phase map.

    References
    ----------
    .. [1] `Herráez et al.,
            "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path",
            Applied Optics,
            2002.
            <https://doi.org/10.1364/AO.41.007437>`_

    .. [2] `Lei et al.,
            “A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping”,
            Optik - International Journal for Light and Electron Optics,
            2015.
            <https://doi.org/10.1016/j.ijleo.2015.04.070>`_
    """
    Y, X, C = p.shape

    if uwr_func in "cv2":  # OpenCV unwrapping
        params = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
        params.height = Y
        params.width = X
        unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)

    puwr = np.empty((Y, X, C), np.float32)
    if verbose:
        r = np.empty((Y, X, C), np.float32)

    for c in range(C):
        if uwr_func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
            # dtype must be np.float32  # todo: test this
            puwr[:, :, c] = unwrapping_instance.unwrapPhaseMap(p[:, :, c])

            if verbose:
                r[:, :, c] = unwrapping_instance.getInverseReliabilityMap()
        else:  # Scikit-image algorithm is slower but delivers better results on edges
            puwr[:, :, c] = ski.restoration.unwrap_phase(p[:, :, c])

            if verbose:
                r[:, :, c] = np.nan

    puwrmin = puwr.min()
    if puwrmin < 0:
        puwr -= puwrmin

    return puwr  # todo: return r


def ftm(I, D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fourier-transform method.

    Parameters
    ----------
    I : np.ndarray
        Fringe pattern.
        Must be in image shape: (height `Y`, width `X`) or (height `Y`, width `X`, color channels `C`).

    Returns
    -------
    a : np.ndarray
        Brightness: average signal.
    b : np.ndarray
        Modulation: amplitude of the cosine signal.
    p : np.ndarray, optional
        Local phase.

    Raises
    ------
    ValueError
        If number of dimensions of `I` is neither '2' nor '3'.

    References
    ----------
    .. [1] `Takeda et al.,
            "Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry",
            J. Opt. Soc. Am.,
            1982.
            <https://doi.org/10.1364/JOSA.72.000156>`_

    .. [2] `Massig et al.,
            "Fringe-pattern analysis with high accuracy by use of the Fourier-transform method: theory and experimental tests",
            Appl. Opt.,
            2001.
            <https://doi.org/10.1364/AO.40.002081>`_
    """
    if I.ndim == 2:
        Y, X = I.shape
        C = 1
    elif I.ndim == 3:
        Y, X, C = I.shape
    else:
        raise ValueError("Number of dimensions must be '2' or '3'.")
    I.shape = Y, X, C

    a = np.empty(shape=(D, Y, X, C), dtype=np.float32)
    b = np.empty(shape=(D, Y, X, C), dtype=np.float32)
    x = np.empty(shape=(D, Y, X, C), dtype=np.float32)

    # todo: make passband symmetrical around carrier frequency?
    if D == 2:
        fx = np.fft.fftshift(np.fft.fftfreq(X))  # todo: hfft
        fy = np.fft.fftshift(np.fft.fftfreq(Y))
        fxx, fyy = np.meshgrid(fx, fy)
        mx = np.abs(fxx) > np.abs(fyy)  # mask for x-frequencies
        # todo: make left and right borders round (symmetrical around base band)
        my = np.abs(fxx) < np.abs(fyy)  # mask for y-frequencies
        # todo: make lower and upper borders round (symmetrical around base band)

        W = 100  # assume window width for filtering out baseband
        W = min(max(3, W), min(X, Y) / 20)  # clip to ensure plausible value
        a_pos = int(min(max(0, W), X / 4) + 0.5)  # todo: find good upper cut off frequency
        # a_pos = X // 4
        mx[:, :a_pos] = 0  # remove high frequencies
        b_pos = int(X / 2 - W / 2 + 0.5)
        mx[:, b_pos:] = 0  # remove baseband and positive frequencies

        H = 100  # assume window height for filtering out baseband
        H = min(max(3, H), min(X, Y) / 20)  # clip to ensure plausible value
        c = int(min(max(0, H), Y / 4) + 0.5)  # todo: find good upper cut off frequency
        # c = Y // 4
        my[:c, :] = 0  # remove high frequencies
        d = int(Y / 2 - H / 2 + 0.5)
        my[d:, :] = 0  # remove baseband and positive frequencies

        # todo: smooth edges of filter masks, i.e. make them Hann-Windows

        for c in range(C):
            # todo: hfft
            I_FFT = np.fft.fftshift(np.fft.fft2(I[:, :, c]))

            I_FFT_x = I_FFT * mx
            ixy, ixx = np.unravel_index(I_FFT_x.argmax(), I_FFT_x.shape)  # get indices of carrier frequency
            I_FFT_x = np.roll(I_FFT_x, X // 2 - ixx, 1)  # move to center

            I_FFT_y = I_FFT * my
            iyy, iyx = np.unravel_index(I_FFT_y.argmax(), I_FFT_y.shape)  # get indices of carrier frequency
            I_FFT_y = np.roll(I_FFT_y, Y // 2 - iyy, 0)  # move to center

            Jx = np.fft.ifft2(np.fft.ifftshift(I_FFT_x))
            Jy = np.fft.ifft2(np.fft.ifftshift(I_FFT_y))

            x[0, :, :, c] = np.angle(Jx)
            x[1, :, :, c] = np.angle(Jy)
            # todo: a: local average of I ?!
            b[0, :, :, c] = np.abs(Jx) * 2  # factor 2 because one sideband is filtered out
            b[1, :, :, c] = np.abs(Jy) * 2  # factor 2 because one sideband is filtered out
            # if verbose:
            #     r[0, ..., c] = np.log(np.abs(I_FFT))  # J  # todo: hfft
            #     r[1, ..., c] = np.log(np.abs(I_FFT))  # J
            # todo: I - J
    elif D == 1:
        Lmax = max(X, Y)
        fx = np.fft.fftshift(np.fft.fftfreq(X)) * X * Y / Lmax  # todo: hfft
        fy = np.fft.fftshift(np.fft.fftfreq(Y)) * Y * X / Lmax
        fxx, fyy = np.meshgrid(fx, fy)
        frr = np.sqrt(fxx**2 + fyy**2)  # todo: normalization of both directions

        mr = frr <= Lmax / 2  # ensure same sampling in all directions
        W = 10
        W = min(max(1, W / 2), Lmax / 20)
        mr[frr < W] = 0  # remove baseband
        mr[frr > Lmax / 4] = 0  # remove too high frequencies

        mh = np.empty([Y, X])
        mh[:, : X // 2] = 1
        mh[:, X // 2 :] = 0

        for c in range(C):
            # todo: hfft
            I_FFT = np.fft.fftshift(np.fft.fft2(I[:, :, c]))

            I_FFT_r = I_FFT * mr
            iy, ix = np.unravel_index(I_FFT_r.nanargmax(), I_FFT_r.shape)  # get indices of carrier frequency
            y_, x_ = Y / 2 - iy, X / 2 - ix
            angle = np.degrees(np.arctan2(y_, x_))
            mhr = sp.ndimage.rotate(mh, angle, reshape=False, order=0, mode="nearest")

            I_FFT_r *= mhr  # remove one sideband
            I_FFT_r = np.roll(I_FFT_r, X // 2 - ix, 1)  # move to center
            I_FFT_r = np.roll(I_FFT_r, Y // 2 - iy, 0)  # move to center

            J = np.fft.ifft2(np.fft.ifftshift(I_FFT_r))

            x[0, :, :, c] = np.angle(J)
            # todo: a
            b[0, :, :, c] = np.abs(J) * 2  # factor 2 because one sideband is filtered out
            # if verbose:
            #     r[0, ..., c] = np.log(np.abs(I_FFT))  # J
            # todo: I - J

    return a, b, x
