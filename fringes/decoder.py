import numpy as np
import numba as nb

# todo: fast_unwrap(CRT + ssdlim)


# @nb.jit(
#     [
#         nb.types.UniTuple(nb.float32[:, :, :, :], 5)(
#             nb.uint8[:, :, :, :],
#             nb.int_[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.int_[:],
#             nb.float64,
#             nb.float64,
#             nb.float64,
#             nb.types.unicode_type,
#             nb.float64,
#             nb.bool_,
#         ),
#         nb.types.UniTuple(nb.float32[:, :, :, :], 5)(
#             nb.uint16[:, :, :, :],
#             nb.int_[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.int_[:],
#             nb.float64,
#             nb.float64,
#             nb.float64,
#             nb.types.unicode_type,
#             nb.float64,
#             nb.bool_,
#         ),
#         nb.types.UniTuple(nb.float32[:, :, :, :], 5)(
#             nb.float_[:, :, :, :],
#             nb.int_[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.int_[:],
#             nb.float64,
#             nb.float64,
#             nb.float64,
#             nb.types.unicode_type,
#             nb.float64,
#             nb.bool_,
#         ),
#     ],
#     cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def decode(
    I: np.ndarray,
    N: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    R: np.ndarray,  # range
    a: float = 1,  # alpha
    o: float = np.pi,
    rmax: float = 0,  # max residual, i.e. max circular standard deviation (can accelerate unwrapping)
    mode: bool = True,
    Vmin: float = 0.0,  # 10 / 255  # min fringe contrast i.e. min visibility (can accelerate unwrapping)
    verbose: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    PI2 = 2 * np.pi

    T, Y, X, C = I.shape
    D, K = N.shape
    L = R.max() * a
    l = L / v  # lambda i.e. period lengths in [px]

    # precomputations for later use in for-loops
    Nmax = np.max(N)
    cf = np.empty((D, K, Nmax), np.complex_)  # discrete complex filter
    for d in range(D):
        for i in range(K):
            for n in range(N[d, i]):
                t = n / N[d, i]  # todo: variable/individual t as in Uni-PSA-Gen
                cf[d, i, n] = np.exp(1j * PI2 * f[d, i] * t)  # complex filter

    # time/frame indices (for when decoding shifts of each set)
    Nacc = np.cumsum(N.ravel()).reshape(D, K)
    N0 = np.array([[0], [Nacc[0, -1]]])[:D]
    ti = np.concatenate((N0, Nacc), axis=1)  # indices for traversing t

    w0 = N * v**2  # weights of phase measurements are their inverse variances (must be multiplied with B**2 later on)
    # choose reference phase, i.e. that v from which the other fringe orders of the remaining sets are derived
    if mode == "precise":  # todo: decide for each pixel individually i.e. multiply with B later on
        iref = [np.argmax(w0[d]) for d in range(D)]  # set with most precise phases
    else:  # fast (fallback)
        iref = [np.argmax(1 / v[d] * (v[d] > 0)) for d in range(D)]  # set with the least number of periods
    vmaxref = [int(np.ceil(v[d, iref[d]] * R[d] / L)) for d in range(D)]  # max number of periods for each direction

    # # usually, camera sees only part of/center of the screen -> try central fringe orders first and move outwards
    # # (this only accelerates if a break criterion is used and reached)
    # # indices for traversing v from the center outwards
    # kout = [[(vmax[d] - 1) // 2 + [-1, +1][i % 2] * ((i + 1) // 2) for i in range(vmax[d])] for d in range(D)]
    # kout = [[(vmax[d] + [~i, i][i % 2]) // 2 for i in range(vmax[d])] for d in range(D)]
    # kout = [np.array([(vmax[d] + [~i, i][i % 2]) // 2 for i in range(vmax[d])]) for d in range(D)]
    # k2PIout = [np.array([(vmax[d] + [~i, i][i % 2]) // 2 * PI2 for i in range(vmax[d])]) for d in range(D)]

    # # starting value for TPU solver based on maximum length R[d] and Popoviciu's inequality on variances
    # varmax = (R / 2) ** 2  # Popoviciu's inequality on variances
    # ssdmax = varmax * K

    # # convergence limit for TPU solver based on smallest allowed residual rmax (based on estimated SNR)
    # rmax = l[range(D), i0] / 2  # todo: ...[d]
    # rmax = 1  # range of interval
    # varlim = (rmax / 2) ** 2

    # # allocate
    # dt = np.float32  # float32's precision is usually better than quantization noise in phase shifting sequence
    # bri = np.empty((D,  Y * X * C), dt)  # brightness must be identical for all sets, therefore we arverage over them
    # mod = np.empty((D,  K * Y * X * C), dt)
    # phi = np.empty((D,  K * Y * X * C), dt)
    # reg = np.empty((D,  Y * X * C), dt)
    # res = np.empty((D,  Y * X * C), dt)
    #
    # for j in nb.prange(Y * X * C):  # todo: test speed improvement
    #     for d in nb.prange(D):  # todo: or first d, then j?

    dt = np.float32  # float32's precision is usually better than quantization noise in phase shifting sequence
    bri = np.empty((D, Y, X, C), dt)  # brightness must be identical for all sets, therefore we arverage over them
    mod = np.empty((D, K, Y, X, C), dt)
    phi = np.empty((D, K, Y, X, C), dt)
    reg = np.empty((D, Y, X, C), dt)
    res = np.empty((D, Y, X, C), dt)

    for x in nb.prange(X):  # usually X > Y -> put X first, because parallelization only affects outer for-loop
        for y in nb.prange(Y):
            for c in nb.prange(C):
                for d in nb.prange(D):
                    # temporal demodulation
                    A = 0
                    B = np.empty(K)
                    p = np.empty(K)

                    for i in nb.prange(K):
                        z = 0  # complex phasor
                        for n, t in enumerate(range(ti[d, i], ti[d, i + 1])):
                            A += I[t, y, x, c]
                            z += I[t, y, x, c] * cf[d, i, n]

                        B[i] = np.abs(z) / N[d, i] * 2  # * 2: also add amplitudes of frequencies with opposite sign
                        p[i] = np.arctan2(z.imag, z.real)  # arctan maps to [-PI, PI]
                    A /= np.sum(N[d])

                    bri[d, y, x, c] = A
                    mod[d, :, y, x, c] = B

                    if verbose:
                        phi[d, :, y, x, c] = p

                    if Vmin > 0 and np.any(B / A < Vmin):
                        reg[d, y, x, c] = np.nan
                        if verbose:
                            res[d, y, x, c] = np.nan

                        continue

                    # spatial demodulation i.e. unwrapping
                    if K == 1:
                        if v[d, 0] == 0:  # no spatial modulation
                            if R[d] == 1:
                                # the only possible value; however it makes no senso to encode one single coordinate
                                reg[d, y, x, c] = 0

                                if verbose:
                                    res[d, y, x, c] = 0
                            else:
                                # no spatial modulation, therefore we can't compute value
                                reg[d, y, x, c] = np.nan

                                if verbose:
                                    res[d, y, x, c] = np.nan
                        elif v[d, 0] <= 1:
                            # one period covers whole screen: no unwrapping required
                            pn = (p[0] + o) / PI2 % 1  # revert offset and change codomain from [-PI, PI] to [0, 1)
                            reg[d, y, x, c] = pn * l[d, 0]

                            if verbose:
                                res[d, y, x, c] = 0
                        else:
                            # spatial phase unwrapping (to be done in a later step)
                            reg[d, y, x, c] = p[0]

                            if verbose:
                                # todo: residuals are to be received from SPU
                                pass
                    else:
                        w = w0[d] * B**2  # weights for inverse variance weighted phasor lengths
                        w /= np.sum(w)  # normalize weights

                        # varlim = 0  # todo

                        # choose reference set, from which the other fringe orders of the remaining sets are derived
                        if mode == "precise":
                            i0 = np.argmax(w)  # reference set
                            vmax = int(np.ceil(v[d, i0] * R[d] / L))  # max number of periods
                        else:
                            i0 = iref[d]
                            vmax = vmaxref[d]

                        p += o  # revert offset

                        r_max = 0  # minimal phasor length
                        for k in range(vmax):  # fringe orders of set 'i0'  # todo: kout (if varlim > 0)
                            a0 = (p[i0] + PI2 * k) / v[d, i0]  # reference angle
                            zi = w[i0] * np.exp(1j * a0)

                            for i in range(i0):
                                ki = np.rint((a0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                                ai = (p[i] + PI2 * ki) / v[d, i]
                                zi += w[i] * np.exp(1j * ai)

                            # leaving out reference set 'i0'

                            for i in range(i0 + 1, K):
                                ki = np.rint((a0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                                ai = (p[i] + PI2 * ki) / v[d, i]
                                zi += w[i] * np.exp(1j * ai)

                            r = np.abs(zi)  # length of phasor

                            if r > r_max:
                                z = zi

                                # var = np.sqrt(- 2 * np.log(r))
                                # if var <= varlim:
                                #     break

                                r_max = r

                        xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L  # arctan maps to [-PI, PI]
                        reg[d, y, x, c] = xi

                        if verbose:
                            res[d, y, x, c] = np.sqrt(-2 * np.log(r_max))  # circular standard deviation

    return bri, mod.reshape(-1, Y, X, C), phi.reshape(-1, Y, X, C), reg, res
