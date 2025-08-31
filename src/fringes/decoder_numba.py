import glob
import logging
import os

import numba as nb
import numpy as np

logger = logging.getLogger(__name__)

flist = glob.glob(os.path.join(os.path.dirname(__file__), "__pycache__", "decoder*decode*.nbc"))
if not flist or os.path.getmtime(__file__) > max(os.path.getmtime(file) for file in flist):
    logger.warning(
        "The 'decode()'-function has not been compiled yet. "
        "At its first execution, this will take a few minutes "
        "(depending on your CPU and energy settings)."
    )


PI2: float = 2 * np.pi


def tpu():
    NotImplemented  # todo: tpu() with numba


@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def decode(
    I: np.ndarray,
    N: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    L: np.ndarray,
    UMR: np.ndarray,
    crt: np.ndarray,
    gcd: np.ndarray,
    x0: float = 0,
    p0: float = np.pi,
    bmin: float = 0.0,
    Vmin: float = 0.0,
    mode: str = "fast",
    unwrap: bool = True,  # todo: fuse unwrap into mode?
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporal demodulation and spatial demodulation
    by virtue of generalized temporal phase unwrapping
    using directional statistics.

    Parameters
    ----------
    I : np.ndarray
        Fringe pattern sequence.
        Must be in `vshape` (frames `T`, height `Y`, width `X`, color channels `C`).
    N : np.ndarray
        Number of phase shifts.
        Must be in shape (number of directions 'D', number of sets 'K').
    v : np.ndarray
        Spatial frequencies.
        Must be in shape (number of directions 'D', number of sets 'K').
    f : np.ndarray
        Temporal frequencies.
        Must be in shape (number of directions 'D', number of sets 'K').
    L : np.ndarray
        Coding lengths.
        Must be of length 'D'.
    UMR : np.ndarray
        Unambiguous measurement range.
    crt : np.ndarray
        Coefficients for Chinese remainder theorem.
    gcd : np.ndarray
        Greatest common divisor of moduli.
    x0 : float, default=0
        Coordinate offset.
    p0 : float, default=np.pi
        Phase offset.
    bmin : float
        Minimum modulation for measurement to be valid.
        If 'bmin' isn't reached at a pixel, spatial unwrapping is skipped for this very pixel.
        This can accelerate decoding.
    Vmin : float, default=0
        Minimum visibility for measurement to be valid.
        If 'Vmin' isn't reached at a pixel, spatial unwrapping is skipped for this very pixel.
        This can accelerate decoding.
    mode : str, default="fast"
        Mode for decoding.
    unwrap : bool, default=True
        Flag for unwrapping.
    verbose : bool, default=False
        Flag for returning intermediate and verbose results.

    Returns
    -------
    A : np.ndarray
        Brightness.
    B : np.ndarray
        Modulation.
    P : np.ndarray
        Phase.
    O : np.ndarray
        Fringe order.
    Xi : np.ndarray
        Coordinate.
    R : np.ndarray
        Residuals.
    """
    T, Y, X, C = I.shape
    # I = I.reshape(Y * X * C)  # only possible for continuous arrays, but we have multiplexing and deinterlacing
    D, K = N.shape

    Lext = np.max(L) + 2 * x0  # extended coding length
    l = Lext / v  # lambda i.e. period lengths in [px]

    t0: int = 0  # time start index

    # allocate return values
    A = np.empty((D, Y, X, C), np.float32)  # offset is identical for all sets, therefore we average them
    B = np.empty((D, K, Y, X, C), np.float32)  # modulation
    P = np.empty((D, K, Y, X, C), np.float32)  # phase
    O = np.empty((D, K, Y, X, C), np.int_)  # fringe order
    Xi = np.empty((D, Y, X, C), np.float32)  # position
    R = np.empty((D, Y, X, C), np.float32)  # residuals

    for d in range(D):
        # complex filter coefficients i.e. sampling points on unit circle
        s = np.empty((K, np.max(N[d])), np.complex128)
        for i in range(K):
            for n in range(N[d, i]):
                t_ = n / N[d, i]  # temporal sampling points
                s[i, n] = np.exp(1j * (PI2 * f[d, i] * t_ + p0))

        Nsum = np.sum(N[d])

        # initial weights of phase averaging (are their )inverse variances)
        # (must be multiplied with b**2 later on)
        w0 = N[d] * v[d] ** 2

        # choose reference phase, i.e. that v from which the other fringe orders `k` of the remaining sets are derived
        if "precise" in mode:
            i0 = np.argmax(w0)  # precise
        else:
            i0 = np.argmin(v[d])  # fast

        # usually, camera sees only the central part of the screen
        # -> try central fringe orders first and move outwards
        # (this only accelerates if a break criterion is used and reached)
        Lx0 = L[d] + x0
        L2x0 = Lx0 + x0
        vmax = np.ceil(v[d] * L2x0 / Lext).astype(np.int_)
        kout = np.empty((K, np.max(vmax)), np.int_)  # indices for traversing v from the center outwards
        for i in range(K):
            kc = (vmax[i] - 1) // 2  # central fringe order
            for k in range(vmax[i]):
                if k % 2 == 0:
                    kout[i, k] = kc - (k + 1) // 2
                else:
                    kout[i, k] = kc + (k + 1) // 2

        # looping
        for y in nb.prange(Y):  # numba's prange affects only outer prange-loop
            for x in nb.prange(X):
                for c in nb.prange(C):
                    # temporal demodulation
                    a: float = 0.0  # offset
                    # z_ = np.empty(K, np.complex128)  # complex phasor
                    b = np.empty(K)  # modulation
                    p = np.empty(K)  # phase
                    t = t0  # time index
                    for i in range(K):
                        z: complex = 0j  # complex phasor
                        for n in range(N[d, i]):
                            a += I[t, y, x, c]
                            z += I[t, y, x, c] * s[i, n]
                            t += 1
                        # z_[i] = z  # todo
                        b[i] = np.abs(z) / N[d, i] * 2
                        p[i] = np.angle(z) % PI2  # arctan2 maps to [-PI, PI], but we need [0, 2PI)
                    # a /= t
                    a /= Nsum

                    A[d, y, x, c] = a
                    B[d, :, y, x, c] = b

                    if verbose:
                        P[d, :, y, x, c] = p

                    if bmin > 0:
                        for i in range(K):
                            if b[i] < bmin:
                                Xi[d, y, x, c] = np.nan
                                if verbose:
                                    O[d, :, y, x, c] = -1
                                    R[d, y, x, c] = np.nan
                                continue  # skip spatial demodulation (signal is too weak to yield a reliable result)

                    if Vmin > 0 and a > 0:  # avoid division by zero
                        for i in range(K):
                            V = b / a  # note: a >= b
                            if V < Vmin:
                                Xi[d, y, x, c] = np.nan
                                if verbose:
                                    O[d, :, y, x, c] = -1
                                    R[d, y, x, c] = np.nan
                                continue  # skip spatial demodulation (signal is too weak to yield a reliable result)

                    # spatial demodulation i.e. unwrapping
                    if K == 1:
                        if v[d, 0] == 0:  # no spatial modulation
                            if R[d] == 1:  # the only possible value (but there is no point in encoding it at all)
                                Xi[d, y, x, c] = 0

                                if verbose:
                                    O[d, :, y, x, c] = 0
                                    R[d, y, x, c] = 0
                            else:  # no spatial modulation, therefore we can't compute value
                                Xi[d, y, x, c] = np.nan

                                if verbose:
                                    O[d, :, y, x, c] = -1
                                    R[d, y, x, c] = np.nan
                        elif v[d, 0] <= 1:  # one period covers whole screen: no unwrapping required
                            Xi[d, y, x, c] = p[0] / PI2 * l[d, 0] - x0  # change codomain from [0, PI2) to [0, Lext)

                            if verbose:
                                O[d, :, y, x, c] = 0
                                R[d, y, x, c] = 0
                        else:  # spatial phase unwrapping (to be done in a later step)
                            Xi[d, y, x, c] = p[0] - x0 * PI2 / l[d, 0]

                            if verbose:
                                # attention: residuals are to be received from SPU
                                O[d, :, y, x, c] = -1
                    elif unwrap:  # generalized temporal phase unwrapping
                        # weights of phase measurements
                        w = w0 * b**2  # weights for inverse variance weighting
                        w /= np.sum(w)  # normalize weights

                        # criterion for when correct solution is found
                        # which is when the phasor is large enough
                        # i.e. the circular variance is small enough
                        rmin = 1

                        # maximal phasor length: initialize with minimal value
                        rmax = 0

                        # derive fringe orders from the reference set
                        # max. time complexity O(ceil(v[i0]))
                        # max. time complexity O(ceil(vmin))
                        r: float = 0
                        Z: float = np.nan
                        o = np.empty(K, np.int_)
                        for k0 in kout[i0, : vmax[i0]]:  # fringe orders of reference set 'i0'
                            a0 = (k0 * PI2 + p[i0]) / v[d, i0]  # reference angle
                            zi = w[i0] * np.exp(1j * a0)
                            o[i0] = k0

                            for i in range(i0):
                                ki = np.rint((a0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                                ai = (ki * PI2 + p[i]) / v[d, i]
                                zi += w[i] * np.exp(1j * ai)
                                o[i] = ki

                            # skip reference set 'i0'

                            for i in range(i0 + 1, K):
                                ki = np.rint((a0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                                ai = (ki * PI2 + p[i]) / v[d, i]
                                zi += w[i] * np.exp(1j * ai)
                                o[i] = ki

                            r = np.abs(zi)
                            if r >= rmax:
                                rmax = r
                                Z = zi

                                if verbose:
                                    O[d, :, y, x, c] = o

                                # if r >= rmin:
                                #     break

                        xi = np.angle(Z) % PI2 / PI2 * Lext - x0  # change codomain from [-PI, PI] to [0, Lext)
                        Xi[d, y, x, c] = xi

                        if verbose:
                            R[d, y, x, c] = np.sqrt(-2 * np.log(r))  # circular standard deviation
        # t0 += t
        # t0 += np.sum(N[d])
        t0 += Nsum

    return A, B.reshape(-1, Y, X, C), P.reshape(-1, Y, X, C), O.reshape(-1, Y, X, C), Xi, R
