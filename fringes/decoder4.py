import numpy as np
import numba as nb

PI2 = 2 * np.pi


@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def circ_dist(a, b, c) -> float:
    d = b - a
    dmax = c / 2

    # return dmax - np.abs(dmax - d)

    if d > dmax:
        d -= c
    elif d < -dmax:
        d += c
    return d


# @nb.jit(cache=True, nopython=True, nogil=True, fastmath=True)
@nb.jit(cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
# @nb.jit(
#     nb.types.UniTuple(  # output types
#         nb.float32[:, :, :, :],
#         6
#     )
#     (  # input types
#         nb.uint8[:, :, :, :],
#         nb.int_[:, :],
#         nb.float64[:, :],
#         nb.float64[:, :],
#         nb.int_[:],
#         nb.float64,
#         nb.float64,
#         nb.float64,
#         nb.types.unicode_type,
#         nb.float64,
#         nb.bool_,
#     ),
#     cache=True, nopython=True, nogil=True, parallel=True, fastmath=True)
def decode(
    I: np.ndarray,
    N: np.ndarray,
    v: np.ndarray,
    f: np.ndarray,
    R: np.ndarray,  # range
    a: float = 1,  # alpha
    o: float = np.pi,
    r: float = 0,
    mode: bool = True,
    Vmin: float = 0.0,  # 10 / 255  # min fringe contrast i.e. min visibility (can accelerate unwrapping)
    verbose: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):

    # assertions
    assert I.ndim == 4, "Image sequence must be in video shape (T, Y, X, C)."
    assert N.ndim == v.ndim == f.ndim == 2, "N, v, f each must have two dimensions."
    assert N.shape == v.shape == f.shape, "N, v, f each must have same shape (D, K)."
    assert False not in [np.any(Nd >= 3) for Nd in N], "Each direction must have at least one set with >= 3 shifts."
    T, Y, X, C = I.shape
    assert np.sum(N) == T, "Sum of shifts must equal number of frames."
    D, K = N.shape
    assert R.size == D, "Screen length must be given for each direction."

    L = R.max() * a
    l = L / v  # lambda i.e. period lengths in [px]

    # precomputations for later use in for-loops
    Nmax = np.max(N)
    cf = np.empty((D, K, Nmax), np.complex_)  # discrete complex filter
    for d in range(D):
        for i in range(K):
            for n in range(N[d, i]):
                t = n / 4 if N[d, i] == 2 else n / N[d, i]  # todo: variable/individual t as in Uni-PSA-Gen
                cf[d, i, n] = np.exp(1j * 2 * np.pi * f[d, i] * t)

    # time/frame indices for when decoding shifts of each set
    Nacc = np.cumsum(N.ravel()).reshape(D, K)
    N0 = np.array([[0], [Nacc[0, -1]]])[:D]
    ti = np.concatenate((N0, Nacc), axis=1).astype(np.int_)  # indices for traversing t

    # check if N contains ones or twos
    KN1 = [[i for i in range(K) if N[d, i] == 1] for d in range(D)]  # i-indices where N == 1
    KN2 = [[i for i in range(K) if N[d, i] == 2] for d in range(D)]  # i-indices where N == 2
    KN3 = [[i for i in range(K) if N[d, i] >= 3] for d in range(D)]  # i-indices where N >= 3
    EN1 = [np.sum(Nd[Nd == 1]) for Nd in N]  # number of sets where N == 1
    EN11 = [
        np.sum(Nd[Nd == 1]) + 1 for Nd in N
    ]  # EN1 plus one, since we compare all shifts with N == 1 to one with N >= 3
    EN3 = [np.sum(Nd[Nd >= 3]) for Nd in N]  # number of sets where N >= 3

    # choose that v from which the other fringe orders of the remaining sets are derived
    if mode == "precise":  # todo: decide for each pixel individually i.e. multiply with B later on
        w = N * v**2  # weights of phase measurements are their inverse variances (must be multiplied with B later on)
        idx = [np.argmax(w[d] * (N[d] > 2) * (v[d] > 0)) for d in range(D)]  # the set with most precise phases
    # elif mode == "robust":  # todo: "exhaustive"?
    #     NotImplemented  # todo: try neighboring fringe orders / all permutations = exhaustive search?
    else:  # fast (fallback)
        w = np.ones((D, K), dtype=np.int_)
        idx = [
            np.argmax(1 / v[d] * (N[d] > 2) * (v[d] > 0)) for d in range(D)
        ]  # the set with the least number of periods
    i_not_idx = [[i for i in range(K) if i != idx[d]] for d in range(D)]  # indices of v without idx

    # usually, camera sees only part of/center of the screen
    # -> try central fringe orders first and move outwards
    # (this only accelerates if break criterion is used and reached)
    scale = R / L
    vmax = [
        int(np.ceil(v[d, idx[d]] * scale[d])) for d in range(D)
    ]  # max number periods of v[:, idx] in each direction
    # kout = [[(vmax[d] - 1) // 2 + [-1, +1][i % 2] * ((i + 1) // 2) for i in range(vmax[d])] for d in range(D)]  # indices for traversing v from the center outwards
    # kout = [[(vmax[d] + [~i, i][i % 2]) // 2 for i in range(vmax[d])] for d in range(D)]  # indices for traversing v from the center outwards
    kout = [np.array([(vmax[d] + [~i, i][i % 2]) // 2 for i in range(vmax[d])]) for d in range(D)]

    # starting value for TPU solver based on maximum length R[d] and Popoviciu's inequality on variances
    varmax = (R / 2) ** 2  # Popoviciu's inequality on variances
    ssdmax = varmax * K

    # convergence limit for TPU solver based on smallest allowed residual r (based on estimated SNR)
    # r = l[range(D), idx] / 2
    # r = 1  # range of interval

    varlim = (r / 2) ** 2
    ssdlim = varlim * K

    # allocate
    dt = np.float32  # float32's precision is usually better than quantization noise in phase shifting sequence
    bri = np.empty((D, Y, X, C), dt)  # brightness must be identical for all sets, therefore we arverage over them
    mod = np.empty((D, K, Y, X, C), dt)
    phi = np.empty((D, K, Y, X, C), dt)
    reg = np.empty((D, Y, X, C), dt)
    res = np.empty((D, Y, X, C), dt)
    fid = np.empty((D, K, Y, X, C), dt)

    for x in nb.prange(X):  # usually X > Y -> put X first, because parallelization only affects outer for-loop
        for y in nb.prange(Y):
            for c in range(C):
                for d in range(D):
                    # temporal demodulation
                    A = 0
                    B = np.empty(K)
                    p = np.empty(K)

                    for i in KN3[d]:  # where N >= 3
                        z = 0  # complex phasor
                        for n, t in enumerate(range(ti[d, i], ti[d, i + 1])):
                            A += I[t, y, x, c]
                            z += I[t, y, x, c] * cf[d, i, n]
                        B[i] = (
                            np.abs(z) / N[d, i] * 2
                        )  # factor of two because we also have to add the amplitudes of the frequencies with opposite sign
                        p[i] = np.angle(z)  # arctan maps to [-PI, PI]
                        p[i] = np.arctan2(z.imag, z.real)
                    A /= EN3[d]

                    for i in KN2[d]:  # where N == 2
                        t1 = ti[d, i]
                        t2 = ti[d, i] + 1
                        re = I[t1, y, x, c] - A
                        im = I[t2, y, x, c] - A
                        B[i] = np.sqrt(re**2 + im**2)  # / N[d, i] * 2 is obsolete because == 1
                        p[i] = np.arctan2(im, re)  # arctan maps to [-PI, PI]

                    if EN1[d]:  # if 1 in N
                        m_avg = np.mean(B[N[d] >= 2])
                        for i in KN1[d]:  # where N == 1
                            t = ti[d, i]
                            # arg = min(max(-1, (I[t, y, x, c] - A) / m_avg), 1)  # todo: clipping necessary?
                            arg = (I[t, y, x, c] - A) / m_avg
                            B[i] = np.nan
                            p[i] = np.arccos(arg)  # arccos maps [-1, 1] to [0, PI]

                    bri[d, y, x, c] = A
                    mod[d, :, y, x, c] = B
                    if verbose:
                        phi[d, :, y, x, c] = p

                    # spatial demodulation i.e. unwrapping
                    # todo: reg, res = unwrap()
                    if K == 1:
                        if v[d, 0] == 0:  # no spatial modulation
                            if R[d] == 1:
                                reg[
                                    d, y, x, c
                                ] = 0  # the only possible value; however this is obsolete as it makes no senso to encode one single coordinate
                                if verbose:
                                    res[d, y, x, c] = 0
                                    fid[d, 0, y, x, c] = 0
                            else:
                                reg[d, y, x, c] = np.nan  # no spatial modulation, therefore we can't compute value
                                if verbose:
                                    res[d, y, x, c] = np.nan
                                    fid[d, 0, y, x, c] = np.nan
                        elif v[d, 0] <= 1:  # one period covers whole screen: no unwrapping required
                            pn = (
                                (p[0] + o) / PI2 % 1
                            )  # revert offset and change codomain from [-PI, PI] to [0, 1) -> normalized phi
                            reg[d, y, x, c] = pn * l[d, 0]
                            if verbose:
                                res[d, y, x, c] = 0  # todo: uncertainty
                                fid[d, 0, y, x, c] = 0
                        else:  # spatial phase unwrapping (to be done in a later step)
                            reg[d, y, x, c] = p[0]
                            if verbose:
                                # todo: residuals are to be received from SPU (spatial phase unwrapping)
                                fid[d, 0, y, x, c] = np.nan  # unknown
                    else:  # K > 1: temporal phase unwrapping
                        if False:  # todo
                            continue
                        else:
                            mn = ssdmax[d] + 0  # adding zero creates a copy
                            pn = (
                                p % PI2 + o
                            ) / PI2  # change codomain from [-PI, PI] to [-0.5, 0.5) and revert offset -> normalized phi
                            # pn = (p + o) / PI2  # change codomain from [-PI, PI] to [-0.5, 0.5] and revert offset -> normalized phi
                            pn = (p % PI2 + o) / PI2 % 1
                            for k in kout[
                                d
                            ]:  # fringe orders of set 'idx'  # todo: start from fringe order given by CRT: move those to the beginning of kout
                                xi = (k + pn[idx[d]]) * l[d, idx[d]]  # position for phase and k-th fringe order
                                sum = xi
                                ssd = xi**2  # sum of squared distances

                                if N[d, idx[d]] == 1:
                                    sum2 = sum + 0  # adding zero creates a copy
                                    ssd2 = ssd + 0  # adding zero creates a copy

                                for i in i_not_idx[d]:  # indices of sets without idx
                                    ki = int(
                                        xi / l[d, i] - pn[i] + 0.5
                                    )  # fringe order of i-th set  # todo: also try neighboring fringe orders (if phase is close to them ~ phimax / 2)
                                    xj = (ki + pn[i]) * l[
                                        d, i
                                    ]  # position for phase and ki-th fringe order  # todo: Verschiebungssatz AND weighted average compatible?
                                    sum += xj
                                    ssd += xj**2

                                    if N[d, i] == 1:
                                        xj2 = (ki + (1 - pn[i])) * l[
                                            d, i
                                        ]  # position for phase and j2-th fringe order #  todo: Verschiebungssatz AND weighted average compatible?
                                        sum2 += xj2
                                        ssd2 += xj2**2

                                ssd -= sum**2 / K  # Verschiebungssatz i.e. Steiner's theorem

                                if ssd < mn:
                                    mn = ssd
                                    avg = sum / K
                                    if verbose:
                                        fid[d, idx[d], y, x, c] = k
                                        for i in i_not_idx[d]:  # indices of v without idx
                                            fid[d, i, y, x, c] = int(
                                                xi / l[d, i] - pn[i] + 0.5
                                            )  # fringe order of i-th set  # todo: also try neighboring fringe orders (if phase is close to them ~ phimax / 2)

                                    if mn <= ssdlim:
                                        break

                                if EN1[d]:  # 1 in N[d]
                                    ssd2 -= sum2**2 / EN11[d]  # Verschiebungssatz i.e. Steiner's theorem

                                    if ssd2 < mn:
                                        mn = ssd2
                                        avg = sum2 / EN11[d]
                                        if verbose:
                                            fid[d, idx[d], y, x, c] = k
                                            for i in i_not_idx[d]:  # indices of v without idx
                                                fid[d, i, y, x, c] = int(
                                                    xi / l[d, i] - pn[i] + 0.5
                                                )  # fringe order of i-th set  # todo: also try neighboring fringe orders (if phase is close to them ~ phimax / 2)

                                        if mn <= ssdlim:
                                            break

                            if mn >= ssdmax[d]:
                                reg[d, y, x, c] = np.nan  # np.nan  # no suitable fringe orders found
                                if verbose:
                                    res[d, y, x, c] = (
                                        L / 2
                                    )  # no suitable fringe orders found -> set max value possible due to Popoviciu's inequality on variances
                                    fid[d, :, y, x, c] = np.nan  # no suitable fringe orders found
                            else:
                                reg[d, y, x, c] = avg
                                if verbose:
                                    res[d, y, x, c] = np.sqrt(mn / K)  # ssd (sum of squared distances) -> std

                            # if verbose:
                            # # weighted averaging with weight being inverse variances
                            #     wk = w * B ** 2  # weights are inverse variances
                            #     wk /= np.sum(wk)
                            #     xk = p + fid[d, :, y, x, c] * l[d]
                            #     x = np.dot(wk, xk)
                            #     reg[d, y, x, c] = x

    # q = w[0] * B**2

    return (
        phi.reshape(-1, Y, X, C),
        bri,
        mod.reshape(-1, Y, X, C),
        reg,
        res,
        fid.reshape(-1, Y, X, C),
    )  # todo: uncertainty
