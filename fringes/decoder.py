import numpy as np
import numba as nb


# @nb.jit(
#     [
#         nb.types.UniTuple(nb.float32[:, :, :, :], 5)(
#             nb.uint8[:, :, :, :],
#             nb.int_[:, :],
#             nb.float64[:, :],
#             nb.float64[:, :],
#             nb.int_[:],
#             nb.int_[:],
#             nb.float64[:],
#             nb.float64[:],
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
#             nb.int_[:],
#             nb.float64[:],
#             nb.float64[:],
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
#             nb.int_[:],
#             nb.float64[:],
#             nb.float64[:],
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
    R: np.ndarray,
    UMR: np.ndarray,
    x0: float = 0,
    p0: float = np.pi,
    Vmin: float = 0.0,
    verbose: bool = False,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Temporal demodulation and spatial demodulation
    by virtue of generalized temporal phase uwrapping
    using directional statistics.

    Parameters
    ----------
    I : np.ndarray
        Fringe pattern sequence.
        Must be in videoshape (frames `T`, height `Y`, width `X`, color channels `C`).

    N : np.ndarray
        Number of phase shifts.
        Must be in shape (number of directions 'D', number of sets 'K').

    v : np.ndarray
        Spatial frequencies.
        Must be in shape (number of directions 'D', number of sets 'K').

    f : np.ndarray
        Temporal frequencies.
        Must be in shape (number of directions 'D', number of sets 'K').

    R : np.ndarray
        Decoding range, i.e. length of fringe patterns for each direction.
        Must be of length 'D'.

    UMR : np.ndarray
        Unambiguous measurement range.

    x0 : float, default=0
        Coordinate offset.

    p0 : np.ndarray, default=np.pi
        Phase offset.

    Vmin : float, default=0
        Minimum visibility for measurement to be valid.
        If 'Vmin' isn't reached at a pixel, spatial unwrapping is skipped for this very pixel.

    verbose : bool, default=False
        Flag for additionally returning intermediate and verbose results: Phase maps 'phi' and residuals 'res'.

    Returns
    -------
    bri : np.ndarray
        Brightness.

    mod : np.ndarray
        Modulation.

    phi : np.ndarray
        Phase.

    reg : np.ndarray
        Registration.

    res : np.ndarray
        Residuals.
    """

    PI2 = 2 * np.pi

    T, Y, X, C = I.shape
    # I = I.reshape(Y * X * C)  # only possible for continuous arrays, but we have multiplexing and deinterlacing
    D, K = N.shape

    L = np.max(R) + 2 * x0  # coding range
    l = L / v  # lambda i.e. period lengths in [px]

    # allocate return values
    dt = np.float32  # float32's precision is usually better than quantization noise in the phase shifting sequence
    bri = np.empty((D, Y, X, C), dt)  # brightness should be identical for all sets, therefore we arverage them
    mod = np.empty((D, K, Y, X, C), dt)
    phi = np.empty((D, K, Y, X, C), dt)
    reg = np.empty((D, Y, X, C), dt)
    res = np.empty((D, Y, X, C), dt)

    # looping
    for d in range(D):
        # time/frame indices (for when decoding shifts of each set)
        t_end = np.cumsum(N)[d * K : (d + 1) * K]
        t_start = t_end - N[d]

        # complex filter coefficients
        cf = np.empty((K, np.max(N[d])), np.complex_)  # discrete complex filter
        for i in range(K):
            for n in range(N[d, i]):
                t = n / N[d, i]  # temporal sampling points
                cf[i, n] = np.exp(1j * (PI2 * f[d, i] * t + p0))  # complex filter

        # initial weights of phase averaging are their inverse variances
        # (must be multiplied with b**2 later on)
        w0 = N[d] * v[d] ** 2

        # choose reference phase, i.e. that v from which the other fringe orders of the remaining sets are derived
        iref = np.argmin(v[d])  # fast
        # iref = np.argmax(w0)  # precise  # todo: iref

        # usually, camera sees only the central part of the screen
        # -> try central fringe orders first and move outwards
        # (this only accelerates if a break criterion is used and reached)
        Rx0 = R[d] + x0
        R2x0 = Rx0 + x0
        vmax = np.ceil(v[d] * R2x0 / L).astype(np.int_)
        kout = np.empty((K, np.max(vmax)), np.int_)  # indices for traversing v from the center outwards
        for i in range(K):
            kc = (vmax[i] - 1) // 2  # central fringe order
            for k in range(vmax[i]):
                if k % 2 == 0:
                    kout[i, k] = kc - (k + 1) // 2
                else:
                    kout[i, k] = kc + (k + 1) // 2

        # # coefficients for CRT
        # precision = 13
        # atol = 10**-precision
        #
        # m = np.empty((D, K), np.int_)  # correction factor for when extending CRT to rational numbers
        # lcm = np.empty(D, np.int_)
        # CRT = np.empty(D, np.bool_)  # flag indicating whether CRT is applicable
        # # MM_ = np.empty((D, K), np.int_)  # Solution of multiple congruences, derived with the Chinese Remainder Theorem.
        # MM_ = np.zeros((D, K), np.int_)
        # for d in range(D):
        #     # find moduli
        #     if np.all(l[d] % 1 == 0):  # all l are integers
        #         m_ = l[d].astype(np.int_)
        #         gcd = np.gcd.reduce(m_)
        #         m[d] = m_ / gcd
        #         lcm[d] = np.lcm.reduce(m[d])
        #     elif np.all(v[d] % 1 == 0):  # all v are integers
        #         m_ = v[d].astype(np.int_)
        #         lcm[d] = np.lcm.reduce(m_)
        #         m[d] = lcm[d] / m_  # changes order of m_
        #     elif np.allclose(l[d], np.rint(l[d]), rtol=0, atol=atol):
        #         m_ = np.rint(l[d]).astype(np.int_)
        #         gcd = np.gcd.reduce(m_)
        #         m[d] = m_ / gcd
        #         lcm[d] = np.lcm.reduce(m[d])
        #     elif np.allclose(v[d], np.rint(v[d]), rtol=0, atol=atol):
        #         m_ = v[d].astype(np.int_)
        #         lcm[d] = np.lcm.reduce(m_)
        #         m[d] = lcm[d] / m_  # changes order of m_
        #     else:
        #         ml = l[d]
        #         mv = v[d]
        #
        #         # # mutual divisibility test
        #         # for i in range(K - 1):
        #         #     for j in range(i + 1, K):
        #         #         if l[d, i] % l[d, j] < atol or 1 - atol < l[d, i] % l[d, j] < 1:
        #         #             ml[j] = 1
        #         #         elif l[d, j] % l[d, i] < atol or 1 - atol < l[d, j] % l[d, i] < 1:
        #         #             ml[i] = 1
        #         # ml = ml[ml != 1]
        #         # mv = mv[ml != 1]
        #
        #         # estimate whether elements are rational or irrational
        #         Dl = max(str(i)[::-1].find(".") for i in ml)
        #         Dv = max(str(i)[::-1].find(".") for i in mv)
        #         # Dl = max(len(str(i)) - len(str(int(i))) - 1 for i in ml)  # -1: dot
        #         # Dv = max(len(str(i)) - len(str(int(i))) - 1 for i in mv)  # -1: dot
        #
        #         if (
        #             Dl < precision or Dv < precision
        #         ):  # rational numbers without integers (i.e. not covered by isclose(atol))
        #             # extend lcm/gcd to rational numbers
        #
        #             if Dl <= Dv:
        #                 m_ = (ml * 10**Dl).astype(np.int_)  # wavelengths scaled
        #                 gcd = np.gcd.reduce(m_)
        #                 m[d] = m_ / gcd
        #                 lcm[d] = np.lcm.reduce(m[d])
        #             else:
        #                 m_ = (mv * 10**Dv).astype(np.int_)  # spatial frequencies scaled
        #                 lcm[d] = np.lcm.reduce(m_)
        #                 m[d] = lcm[d] / m_
        #         else:  # irrational numbers or rational numbers with more digits than "precision"
        #             # round and extend lcm to rational numbers
        #             m[d] = np.nan  # todo: int?
        #             lcm[d] = np.prod(l)  # todo: int?
        #
        #     # find coefficients
        #     # todo: equal m -> weighted average of p
        #     CRT[d] = np.gcd.reduce(m[d]) == 1
        #     if CRT[d]:
        #         # todo: v = 9, 10, 11
        #         for i in range(K):
        #             M = int(lcm[d] / m[d, i])
        #             for M_ in range(1, m[d, i]):
        #                 MM = M * M_
        #                 if MM % m[d, i] == 1:  # M_ is modular multiplicative inverse element of M
        #                     if not np.any(
        #                         MM_[d, :i] % m[d, i] == 1
        #                     ):  # assure no existing coefficient also equals 1 modulo m[i]
        #                         MM_[d, i] = MM
        #             #         else:
        #             #             MM_[d, i] = 0
        #             #
        #             #         break
        #             # else:  # else clause executes only after the loop completes normally, i.e. did not encounter a break, i.e. no MMIE found
        #             #     MM_[d, i] = 0  # no solution found
        #
        # gcd = np.mean(l / m[d, :], axis=1)
        # S = UMR / gcd  # number of steps (intervals) for unwrapping
        # dev = np.exp(1j * np.pi / S)  # unit deviation vector

        # # fringe order combinations
        # gcd = np.ones(D, np.int_)  # todo: from CRT?!
        # # xi = np.arange(R[d], step=gcd[d])  # todo: x0
        # xi = np.arange(x0, Rx0[d], gcd[d])
        # kp = (xi[None, :] // l[d, :, None]).astype(np.int_)
        # Kp = set(tuple(k) for k in kp.T)  # matching combinations
        #
        # Kn = set()  # neighbouring combinations (to account for noise)
        # for i in range(K):
        #     for j in (-1, +1):
        #         kn = kp.copy()
        #         kn[i] = np.roll(kn[i], j)
        #         Kn.update(tuple(k) for k in kn.T)
        # Kn -= Kp  # neighbouring without matching combinations
        #
        # args = (range(int(np.ceil(vmax[i]))) for i in range(K))
        # pools = (tuple(pool) for pool in args)
        # Ka = [()]
        # for pool in pools:  # combinatorial product
        #     Ka = [x + (y,) for x in Ka for y in pool]
        # Kr = set(Ka) - Kn - Kp  # remaining combinations, i.e. all without neighbouring and also without matching one
        #
        # Kp = np.array(sorted(Kp))
        # Kn = np.array(sorted(Kn))
        # Kr = np.array(sorted(Kr))

        for x in nb.prange(X):  # numba's prange affects only outer prange-loop, so we put largest direction first
            for y in nb.prange(Y):
                # aa01_crt_tried = 0
                # aa02_der_tried = 0
                # aa03_der_tried = 0
                # aa04_nei_tried = 0
                # aa05_nei_tried = 0
                # aa06_exh_tried = 0
                #
                # aa01_crt_corr = 0
                # aa02_der_corr = 0
                # aa03_der_corr = 0
                # aa04_nei_corr = 0
                # aa05_nei_corr = 0
                # aa06_exh_corr = 0
                #
                # aa0_all_tried = X
                # aa0_all_corr = 0
                #
                # false = []
                for c in nb.prange(C):
                    # temporal demodulation
                    zp = np.zeros(K, np.complex_)  # complex phasor
                    for i in range(K):
                        I_ = I[t_start[i] : t_end[i], y, x, c]
                        zp[i] = np.sum(I_ * cf[i])

                    a = np.mean(I_)
                    b = np.abs(zp) / N[d] * 2  # * 2: also add amplitudes of frequencies with opposite sign
                    p = np.arctan2(zp.imag, zp.real) % PI2  # arctan2 maps to [-PI, PI], but we need [0, 2PI)
                    # p = np.angle(zp) % PI2  # arctan2 maps to [-PI, PI], but we need [0, 2PI)

                    bri[d, y, x, c] = a
                    mod[d, :, y, x, c] = b

                    if verbose:
                        phi[d, :, y, x, c] = p

                    V = np.minimum(1, b / np.maximum(np.finfo(np.float_).eps, a))  # avoid division by zero
                    if Vmin > 0 and np.any(V < Vmin):
                        reg[d, y, x, c] = np.nan
                        if verbose:
                            res[d, y, x, c] = np.nan

                        continue  # skip spatial demodulation because signal is too weak for a reliable result

                    # spatial demodulation i.e. unwrapping
                    if K == 1:
                        if v[d, 0] == 0:  # no spatial modulation
                            if R[d] == 1:
                                # the only possible value; however it makes no senso to encode a single coordinate only
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
                            reg[d, y, x, c] = p[0] / PI2 * l[d, 0] - x0  # change codomain from [0, PI2) to [0, L)
                            # xi = np.clip(xi, 0, R[d])  # todo: clip

                            if verbose:
                                res[d, y, x, c] = 0
                        else:
                            # spatial phase unwrapping (to be done in a later step)
                            reg[d, y, x, c] = p[0] - PI2 / l[d, 0] * x0
                            # xi = np.clip(xi, 0, R[d])  # todo: clip

                            if verbose:
                                # attention: residuals are to be received from SPU
                                pass  # todo
                    else:
                        # generalized temporal phase unwrapping

                        # weights of phase measurements
                        w = w0 * b**2  # weights for inverse variance weighting
                        w /= np.sum(w)  # normalize weights

                        # criterion for when correct solution is found,
                        # which is when the phasor is large enough
                        # i.e. the circular variance is small enough
                        # todo: other criterion than all phasors aligned and smallest one inbetween two fringe orders?
                        # rmin = 0  # minimal phasor length for unwrapping to be successful:
                        # for i in range(K):
                        #     zmin = 0
                        #     for j in range(K):
                        #         if i == j:
                        #             zmin += w[j] * dev[d]
                        #         else:
                        #             zmin += w[j]
                        #
                        #     r = np.abs(zmin)
                        #     if r > rmin:
                        #         rmin = r
                        rmin = 1

                        # maximal phasor length: initialize with minimal value
                        rmax = 0

                        # try in this order:
                        # 1.) CRT
                        # new: KDTree
                        # 2.) derive(vmin), derive(wmax)
                        # 3.) matching, their neighbors
                        # 4.) exhaustive

                        # # KDTree
                        # anker = (np.arange(lcm[d]) + 1 / 2) * gcd[d]

                        # if CRT[d]:  # CRT is applicable
                        #     aa01_crt_tried += 1
                        #
                        #     # apply Chinese Remainder Theorem
                        #     # time complexity O(1)
                        #
                        #     p_int, p_fract = np.divmod(p / PI2 * m[d], 1)
                        #     p_int3, p_fract3 = np.divmod(p / PI2 * l[d], gcd[d])
                        #
                        #     # refine
                        #     #
                        #     # The validity of the absolute phase measurement
                        #     # depends on the correct outcome of the INT operations,
                        #     # which means that the measurement noise should not cause any p_ to cross an INT boundary.
                        #     # It is very interesting and useful to realize
                        #     # that p_fract is theoretically the same for all i,
                        #     # so that we can average them to suppress random errors
                        #     # and to obtain a more reliable xi_est.
                        #
                        #     p_fract_circ_mean = (
                        #         np.angle(np.sum(w * np.exp(1j * PI2 * p_fract))) % PI2 / PI2
                        #     )  # weighted circular mean
                        #     p_fract_circ_mean3 = (
                        #         np.angle(np.sum(w * np.exp(1j * PI2 * p_fract3))) % PI2 / PI2
                        #     )  # weighted circular mean
                        #
                        #     d_p = p_fract - p_fract_circ_mean > 0.5
                        #     d_m = p_fract_circ_mean - p_fract > 0.5
                        #     p_int[p_fract - p_fract_circ_mean > 0.5] += 1
                        #     # p_int[p_fract - p_fract_circ_mean < - 0.5] -= 1
                        #     p_int[p_fract_circ_mean - p_fract > 0.5] -= 1
                        #
                        #     a000 = np.sum(MM_[d] * p_int.astype(np.int_))
                        #     b000 = p_fract_circ_mean
                        #     c000 = a000 + b000
                        #     d000 = c000 % lcm[d]
                        #     e000 = d000 * gcd[d]
                        #
                        #     xi_crt_ = np.sum(MM_[d] * p_int.astype(np.int_))
                        #     xi_crt = (np.sum(MM_[d] * p_int.astype(np.int_)) % lcm[d] + p_fract_circ_mean) * gcd[
                        #         d
                        #     ]  # lcm * gcd = UMR
                        #     xi_crt3 = (
                        #         np.sum(MM_[d] * p_int3.astype(np.int_)) % lcm[d] * gcd[d] + p_fract_circ_mean3
                        #     )  # lcm * gcd = UMR
                        #
                        #     xi_uc = np.angle(w * np.exp(1j * p / v[d])) % PI2 / PI2 * L - x0
                        #     xi_uc3 = np.angle(np.exp(1j * np.sum(p / v[d] * MM_[d]) / lcm[d])) % PI2 / PI2 * L - x0
                        #
                        #     x_ = x
                        #     xi_ = xi_crt
                        #     xi = xi_crt
                        #     z = np.exp(1j * PI2 * xi / L)
                        #
                        #     # xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0  # todo:L or UMR[d]?
                        #     if np.round(xi) == x:
                        #         aa01_crt_corr += 1
                        # else:
                        #     r = 0
                        r = 0

                        if r < rmin:
                            # aa02_der_tried += 1

                            # derive fringe orders from the reference set,
                            # i.e. the one with the least number of periods
                            # max. time complexity O(ceil(vmin))

                            for k0 in kout[iref, : vmax[iref]]:  # fringe orders of reference set 'iref'
                                arg0 = (k0 * PI2 + p[iref]) / v[d, iref]  # reference angle
                                zi = w[iref] * np.exp(1j * arg0)

                                for i in range(iref):
                                    ki = np.rint((arg0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                                    ai = (ki * PI2 + p[i]) / v[d, i]
                                    zi += w[i] * np.exp(1j * ai)

                                # leaving out reference set 'iref'

                                for i in range(iref + 1, K):
                                    ki = np.rint((arg0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                                    ai = (ki * PI2 + p[i]) / v[d, i]
                                    zi += w[i] * np.exp(1j * ai)

                                r = np.abs(zi)
                                if r >= rmax:
                                    rmax = r
                                    z = zi

                        #             if r > rmin:
                        #                 xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0
                        #                 if np.round(xi) == x:
                        #                     aa02_der_corr += 1
                        #                 break  # optimal solution found, stop loop
                        #
                        #         if r < rmin and i0 != np.argmax(w):
                        #             aa03_der_tried += 1
                        #
                        #             # derive fringe orders from the reference set,
                        #             # i.e. the one with the most precise phases
                        #             # max. time complexity O(ceil(vmax[d]))
                        #
                        #             i0 = np.argmax(w)
                        #
                        #             for k0 in kout[d, i0, : vmax[d, i0]]:  # fringe orders of set 'i0'
                        #                 arg0 = (k0 * PI2 + p[i0]) / v[d, i0]  # reference angle
                        #                 zi = w[i0] * np.exp(1j * arg0)
                        #
                        #                 for i in range(i0):
                        #                     ki = np.rint((arg0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                        #                     ai = (ki * PI2 + p[i]) / v[d, i]
                        #                     zi += w[i] * np.exp(1j * ai)
                        #
                        #                 # leaving out reference set 'i0'
                        #
                        #                 for i in range(i0 + 1, K):
                        #                     ki = np.rint((arg0 * v[d, i] - p[i]) / PI2)  # fringe order of i-th set
                        #                     ai = (ki * PI2 + p[i]) / v[d, i]
                        #                     zi += w[i] * np.exp(1j * ai)
                        #
                        #                 r = np.abs(zi)
                        #
                        #                 if r >= rmax:
                        #                     rmax = r
                        #                     z = zi
                        #
                        #                     if r > rmin:
                        #                         xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0
                        #                         if np.round(xi) == x:
                        #                             aa03_der_corr += 1
                        #                         break  # optimal solution found, stop loop
                        #
                        #             if r < rmin:
                        #                 aa04_nei_tried += 1
                        #
                        #                 # try natching fringe order combinations
                        #                 # max. time complexity O(ceil(R[d] / gcd[d]))
                        #
                        #                 for k in Kp[d]:
                        #                     arg = (k * PI2 + p) / v[d]
                        #                     zm = np.sum(w * np.exp(1j * arg))
                        #                     r = np.abs(zm)
                        #
                        #                     if r >= rmax:
                        #                         rmax = r
                        #                         z = zm
                        #
                        #                         if r > rmin:
                        #                             xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0
                        #                             if np.round(xi) == x:
                        #                                 aa04_nei_corr += 1
                        #                             break  # optimal solution found, stop loop
                        #
                        #                 if r < rmin:
                        #                     aa05_nei_tried += 1
                        #
                        #                     # try neighbors of matching fringe order combinations to account for noise
                        #                     # max. time complexity O(ceil(R[d] / gcd[d]) * K * 2 - ceil(R[d] / gcd[d]))
                        #
                        #                     for k in Kn[d]:
                        #                         arg = (k * PI2 + p) / v[d]
                        #                         zn = np.sum(w * np.exp(1j * arg))
                        #                         r = np.abs(zn)
                        #
                        #                         if r >= rmax:
                        #                             rmax = r
                        #                             z = zn
                        #
                        #                             if r > rmin:
                        #                                 xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0
                        #                                 if np.round(xi) == x:
                        #                                     aa05_nei_corr += 1
                        #                                 break  # optimal solution found, stop loop
                        #
                        #                     if r < rmin:
                        #                         aa06_exh_tried += 1
                        #
                        #                         # exhaustive search (without matching and their neighborng fringe order combinations)
                        #                         # max. time complexity O(prod(v[d] - ceil(R[d] / gcd[d]) * K * 2 - ceil(R[d] / gcd[d]))
                        #
                        #                         for k in Ka[d]:
                        #                             arg = (k * PI2 + p) / v[d]
                        #                             ze = np.sum(w * np.exp(1j * arg))
                        #                             r = np.abs(ze)
                        #
                        #                             if r >= rmax:
                        #                                 rmax = r
                        #                                 z = ze
                        #
                        #                                 if r > rmin:
                        #                                     break  # optimal solution found, stop loop
                        #
                        #                         xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0
                        #                         if np.round(xi) == x:
                        #                             aa06_exh_corr += 1
                        #
                        # if np.round(xi) == x:
                        #     aa0_all_corr += 1
                        # else:
                        #     false.append(x)

                        xi = np.arctan2(z.imag, z.real) % PI2 / PI2 * L - x0  # change codomain from [-PI, PI] to [0, L)
                        # xi = np.angle(z) % PI2 / PI2 * L - x0  # change codomain from [-PI, PI] to [0, L)
                        # xi = np.clip(xi, 0, R[d])  # todo: clip

                        reg[d, y, x, c] = xi

                        if verbose:
                            res[d, y, x, c] = np.sqrt(-2 * np.log(r.item()))  # circular standard deviation

    return bri, mod.reshape(-1, Y, X, C), phi.reshape(-1, Y, X, C), reg, res