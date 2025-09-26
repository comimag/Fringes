import sys
from collections import namedtuple
from collections.abc import Sequence
from importlib.metadata import version
import itertools as it
import json
import logging
import os
import time

from numba import set_num_threads, get_num_threads
import numpy as np
import scipy as sp
import sympy
import yaml

from fringes.decoder import decode, ftm, spu, temp_demod_numpy
from fringes import grid
from fringes.util import vshape

logger = logging.getLogger(__name__)

_2PI = 2 * np.pi
Decoded = namedtuple("decoded", ("a", "b", "x"))
Decoded_verbose = namedtuple("decoded_verbose", ("a", "b", "x", "p", "k", "r", "u"))


class Fringes:
    """Configure, encode and decode fringe patterns using phase shifting algorithms.

    Note
    ----
    All parameters are implemented as properties
    (managed attributes) and are parsed when set.
    Note that some attributes have sub-dependencies,
    hence dependent attributes might change as well.
    Circular dependencies are resolved automatically.
    """

    # note: the class docstring is continued at the end of the class in an automated manner

    # value limits
    _Dmax: int = 2
    _amax: float = 2.0
    _gmax: float = 4.0  # most screens have a gamma value of ~2.2
    # _lminmin: float = 2.0  # l == 2 only if p0 != pi / 2 + 2pi*k, best if p0 == pi + 2pi*k with k being a positive integer
    #            also l <= 2 yields errors in SPU: phase jumps = 2PI / lmin >= np.pi
    _lminmin: float = 3.0  # l >= 3 yields sufficient modulation theoretically
    # _lminmin: float = 8.0  # l >= 8 yields sufficient modulation practically [Liu2014]

    # choices/allowed values; take care to only use immutable types!  # todo: is that so?
    _choices = {
        "grid": {
            "image",
        },  # todo: ("image", "Cartesian", "polar", "log-polar")
        "indexing": {"xy", "ij"},  # todo: Literal["xy", "ij"]
        "dtype": {
            # "bool",  # results are too unprecise
            "uint8",
            "uint16",
            # "uint32",  # integer overflow in pyqtgraph -> replace line 528 of ImageItem.py with:
            # "uint64",  #  bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype="uint64")
            # "float16",  # numba doesn't handle float16, also most algorithms convert float16 to float32 anyway
            "float32",
            "float64",
        },
        "mode": {"fast", "precise", "MRD"},  # , "numpy", "numba"),
        "h": {"w", "r", "g", "b", "c", "m", "y"},
    }

    # default values are defined here; take care to only use immutable types (numpy arrays ARE mutable)!
    def __init__(
        self,
        *args,  # bundles all args, what follows are only kwargs
        # **_defaults,  # todo: kwargs
        X: int = 1920,
        Y: int = 1200,
        D: int = 2,
        K: int = 2,
        N: int | Sequence[int] = ((4, 4), (4, 4)),
        # l: int | float | Sequence[int | float] | str = ((1920 / 13., 1920 / 7.), (1920 / 13., 1920 / 7.)),  # inferred from v
        v: int | float | Sequence[int | float] | str = ((13.0, 7.0), (13.0, 7.0)),
        f: int | float | Sequence[int | float] = ((1.0, 1.0), (1.0, 1.0)),
        h: Sequence[int] | str = ((255, 255, 255),),
        p0: float = np.pi,
        g: float = 1.0,
        # A: float = 255 / 2,  # i.e. Imax / 2 @ uint8; inferred from Imax and E
        # B: float = 255 / 2,  # i.e. Imax / 2 @ uint8; inferred from Imax and E and V
        E: float = 0.5,
        V: float = 1.0,
        a: float = 1.0,
        bits: int = 8,
        dtype: str | np.dtype = "uint8",  # inferred from bits
        grid: str = "image",
        # angle: float = 0.0,
        axis: int = 0,
        SDM: bool = False,
        WDM: bool = False,
        FDM: bool = False,
        static: bool = False,
        lmin: float = 8.0,
        indexing: str = "xy",
        reverse: bool = False,
        mode: str = "fast",
        # **kwargs,  # bundles all further (hence undefined) kwargs;
        #            # todo: raise error if not empty:
        #            # __init__() got an unexpected keyword argument
    ):
        # given values which are in defaults but are not identical to them
        given = {
            k: v
            for k, v in sorted(locals().items())
            if k in self._defaults and not np.array_equal(v, self._defaults[k])
        }  # sorted() ensures setting _params in the right order

        # set defaults values to private attributes used by properties
        self._mtf = None  # empty cache
        self._UMR = None  # empty cache
        self._m = None  # empty cache
        self._crt = None  # empty cache
        for k, v in self._defaults.items():
            setattr(
                self, f"_{k}", np.array(v) if isinstance(v, tuple) else v
            )  # define private variables from which the properties get their values from

        # set given values (check is run in '_params.setter')
        self._params = given

        # precompute
        self.UMR  # compute and cache; logs warning if necessary
        self.crt  # compute and cache (also computes _m)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.encode(*args, **kwargs)

    def __iter__(self):
        self._t: int = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._t < self.T:
            I = self.encode(frames=self._t)[0]
            self._t += 1
            return I
        else:
            del self._t
            raise StopIteration()

    def __len__(self) -> int:
        """Number of frames."""
        return self.T

    def __eq__(self, other) -> bool:
        return other.__class__ is self.__class__ and self._params == other._params

    def __contains__(self, item):
        return hasattr(self, item)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self._params}"

    def load(self, fname: str | None = None) -> None:
        """Load parameters from a config file to the `Fringes` instance.

        .. warning:: The parameters are only loaded if the config file provides the section `fringes`.

        Parameters
        ----------
        fname : str, optional
            File name of the file to load.
            Supported file formats are: *.json, *.yaml.
            If `fname` is not provided, the file `.fringes.yaml` within the user home directory is loaded.

        Examples
        --------
        >>> import os
        >>> fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        >>> from fringes import Fringes
        >>> f = Fringes()
        >>> f.N = f.N + 1
        >>> f.save(fname)

        >>> f_ = Fringes()
        >>> f_.load(fname)

        >>> f_ == f
        True
        """
        pname = self.__class__.__name__.lower()

        if fname is None:
            fname = os.path.join(os.path.expanduser("~"), f".{pname}.yaml")

        if not os.path.isfile(fname):
            logger.error(f"File '{fname}' does not exist.")
            return

        with open(fname, "r") as f:
            ext = os.path.splitext(fname)[-1]

            if ext == ".json":
                p = json.load(f)
            elif ext == ".yaml":
                p = yaml.safe_load(f)
            else:
                logger.error(f"Unknown file type '{ext}'.")
                return

        if pname in p:
            params = p[pname]
            self._params = params

            logger.info(f"Loaded parameters from '{fname}'.")
        else:
            logger.error(f"No '{pname}' section in file '{fname}'.")

    def save(self, fname: str | None = None) -> None:
        """Save the parameters of the `Fringes` instance to a config file.

        Within the file, the parameters are written to the section `fringes`.

        Parameters
        ----------
        fname : str, optional
            File name of the file to save.
            Supported file formats are: *.json, *.yaml.
            If `fname` is not provided, the parameters are saved to
            the file `.fringes.yaml` within the user home directory.

        Examples
        --------
        >>> import os
        >>> fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        >>> from fringes import Fringes
        >>> f = Fringes()

        >>> f.save(fname)
        """
        pname = self.__class__.__name__.lower()

        if fname is None:
            fname = os.path.join(os.path.expanduser("~"), f".{pname}.yaml")

        if not os.path.isdir(os.path.dirname(fname)):
            logger.error(f"File directory does not exist.")
            return

        name, ext = os.path.splitext(fname)
        if not ext:
            name, ext = ext, name

        if ext not in {".json", ".yaml"}:  # toml does not allow the type 'None'
            logger.error(f"File extension is unknown. Must be '.json' or '.yaml'.")
            return

        with open(fname, "w") as f:  # todo: file gets overwritten or only section 'fringes'?
            if ext == ".json":
                json.dump({f"{pname}": self._params}, f, indent=4)
            elif ext == ".yaml":
                yaml.dump({f"{pname}": self._params}, f)

        logger.info(f"Saved parameters to {fname}.")

    def reset(self) -> None:
        """Reset parameters of the `Fringes` instance to default values."""
        self._params = self._defaults
        logger.info("Reset parameters to defaults.")

    # def optimize(self, T: int | None = None, umax: float | None = None) -> None:  # todo: optimize
    #     """Optimize the parameters of the `Fringes` instance.
    #
    #     Parameters
    #     ----------
    #      T : int, optional
    #         Number of frames.
    #         If `T` is not provided, the number of frames from the `Fringes` instance is used.
    #         Then, the `Fringes` instance's number of shifts `N` is distributed optimally over the directions and sets.
    #      umax : float, optional
    #         Maximum allowable uncertainty.
    #         Must be greater than zero.
    #         Standard deviation of maximum uncertainty for measurement to be valid.
    #         [umax] = px.
    #
    #     Notes
    #     -----
    #     todo: update
    #
    #     If `umax` is specified, the parameters are determined
    #     that allow a maximal uncertainty of `umax`
    #     with a minimum number of frames.
    #
    #     Else, the parameters of the `Fringes` instance are optimized to yield the minimal uncertainty
    #     using the given number of frames `T`.
    #     """
    #     # todo: compute based on given ui -> upi -> ux
    #
    #     if T is not None:
    #         _T = int(max(1, T))
    #
    #         if _T == 1:  # WDM + SDM
    #             # todo: FTM?
    #             if self.grid not in self._choices["grid"][:2]:
    #                 logger.error(f"Couldn't set 'T = 1': grid not in {self._choices["grid"][:2]}'.")
    #                 return
    #
    #             if self.H > 1:
    #                 self.h = self.h[0]
    #             self.K = 1
    #             self.FDM = False  # reset FDM before setting N
    #             self.N = 3  # set N before WDM
    #             self.WDM = True
    #             if self.D == 2:
    #                 self.SDM = True
    #             self.v = 1
    #         elif _T == 2:  # WDM
    #             if self.grid not in self._choices["grid"][:2]:
    #                 logger.error(f"Couldn't set 'T = 1': grid not in {self._choices["grid"][:2]}'.")
    #                 return
    #
    #             if self.H > 1:
    #                 self.h = self.h[0]
    #             self.D = 2
    #             self.K = 1
    #             self.FDM = False  # reset FDM before setting N
    #             self.SDM = False
    #             self.N = 3  # set N before WDM
    #             self.WDM = True
    #             self.v = 1
    #         else:
    #             # set boundaries
    #             self.FDM = False
    #             self.SDM = False
    #             self.WDM = False
    #
    #             # todo: T == 4 -> no modulation
    #             #  T == 5 -> FDM if _T >= self._Nmin?
    #
    #             # try D == 2  # todo: mux
    #             if _T < 2 * self._Nmin:
    #                 self.D = 1
    #             else:
    #                 self.D = 2
    #
    #             # try to keep 'H'
    #             Hmax = _T // (self.D * self._Nmin)
    #             if self.H > Hmax:
    #                 while _T % Hmax != 0:
    #                     Hmax -= 1
    #                 self.h = self.h[:Hmax]
    #                 _T //= self.H
    #
    #             # try to keep 'K'
    #             Kmin = _T // (self.D * self._Nmin)
    #             if self.K > Kmin:
    #                 self.K = Kmin
    #
    #             # ensure UMR >= L
    #             if self._ambiguous:
    #                 imin = np.argmin(self._v, axis=0)
    #                 self._v[imin] = 1
    #
    #             # set 'N'
    #             N = np.full((self.D, self.K), _T // (self.D * self.K))
    #             dT = _T - np.sum(N)
    #             if dT > 0:
    #                 k = int(dT // self.D)
    #                 N[:, :k] += 1
    #                 if dT % self.D != 0:
    #                     d = dT % self.D
    #                     N[:d, k] += 1
    #             self.N = N
    #     elif umax is not None:
    #         ...
    #     else:
    #         ...
    #
    #     vopt = self.vmax  # todo: optimal v
    #     K = np.log(self.Lext) / np.log(self.Lext / vopt)  # lopt ** K = Lext
    #     K = np.ceil(max(2, K))
    #     self.K = K
    #     self.v = "optimal"
    #
    #     if umax is not None:  # umax -> T
    #         self.N = int(np.median(self.N))  # make N const.
    #         a = self.u.max() / umax
    #         N = self.N * a**2
    #         self.N = np.maximum(3, np.ceil(N))
    #
    #         if self.u > umax:
    #             ...  # todo: check if umax is reached
    #     else:  # T -> u
    #         if T is None:
    #             T = self.T
    #
    #         # distribute frames optimally (evenly) on shifts
    #
    #     logger.info("Optimized parameters.")

    def _modulate(
        self, x: np.ndarray | tuple[np.ndarray] | tuple[np.ndarray, np.ndarray], frames: int | Sequence[int]
    ) -> np.ndarray:
        """Encode base fringe patterns by spatio-temporal modulation.

        Parameters
        ----------
        x : np.ndarray | tuple[np.ndarray, ...]
            Coordinates.
            Might be generated by `numpy.indices <https://numpy.org/doc/stable/reference/generated/numpy.indices.html>`_.
            Must have three dimensions at most.
            The first dimension must match `D`.
        frames : np.ndarray
            Indices of the frames to be encoded.

        Returns
        -------
        I : np.ndarray
            Base fringe patterns.
        """
        t0 = time.perf_counter()

        frames = np.array(list(set(t % np.sum(self._N) for t in frames)))  # numpy.unique would return sorted
        T = len(frames)
        is_mixed_color = np.any((self.h != 0) * (self.h != 255))
        dtype = np.dtype("float64") if self.SDM or self.FDM or is_mixed_color or self.uwr == "FTM" else self.dtype
        I = np.empty((T, self.Y, self.X), dtype)

        # Ncum = np.cumsum(self._N).reshape(self.D, self.K)
        # for t in frames:
        #     d, i = np.argwhere(t < Ncum)[0]
        #     n = t - Ncum[d, i] + self._N[0, 0]
        #     ...

        idx = 0
        frame = 0
        for d in range(self.D):
            x_ = (x[d] + self.x0) / self.Lext  # normalize x to be within interval [0, 1)

            for i in range(self.K):
                k = _2PI * self._v[d, i]
                w = _2PI * self._f[d, i]

                if "mrd" in self.mode:
                    k *= self.Lext / self.L[d]

                if self.reverse:
                    w *= -1

                for n in range(self._N[d, i]):  # todo: allow any n: for n in self.n[d, i]:
                    if frame in frames:
                        t = n / 4 if self._N[d, i] == 2 else n / self._N[d, i]

                        Iidx = self.Imax * (self.E * (1 + self.V * np.cos(k * x_ - w * t - self.p0))) ** self.g

                        if dtype.kind in "ui":  # todo: and rint:
                            np.rint(Iidx, out=Iidx)
                        # elif dtype.kind in "b":
                        #     val = val >= 0.5

                        I[idx] = Iidx

                        idx += 1
                    frame += 1

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return I.reshape(-1, self.Y, self.X, 1)

    def _demodulate(
        self,
        I: np.ndarray,
        bmin: float = 0,
        Vmin: float = 0,
        unwrap: bool = True,
        verbose: bool | str = False,
        threads: int | None = None,
        uwr_func: str = "ski",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode base fringe patterns by spatio-temporal demodulation.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to `vshape` (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
        bmin : float
            Minimum modulation for measurement to be valid.
            Can accelerate decoding.
        Vmin : float
            Minimum visibility for measurement to be valid.
            Can accelerate decoding.
        unwrap : bool, default=True
            Flag for unwrapping.
        verbose : bool or str, default=False
            Flag for returning intermediate and verbose results.
        threads : int, optional
            Number of threads to use.
            Default is all threads.
        uwr_func : {'ski', 'cv2'}, optional
            Unwrapping function to use.

            - 'ski': `Scikit-image <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_ [1]_

            - 'cv2': `OpenCV <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_ [2]_

        Returns
        -------
        a : np.ndarray
            Brightness: average signal.
        b : np.ndarray
            Modulation: amplitude of the cosine signal.
        p : np.ndarray
            Local phase.
        k : np.ndarray
            Fringe order.
        x : np.ndarray
            Coordinate: decoded screen coordinates.
        r : np.ndarray
            Residuals from the optimization-based unwrapping process.

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
        t0 = time.perf_counter()

        # parse
        I = vshape(I)
        T, Y, X, C = I.shape  # extract Y, X, C from data as these parameters depend on used camera

        if self.uwr == "FTM":
            a, b, x = ftm(I, self.D)
            # todo: p, k, r
        elif "numpy" in self.mode:
            a, b, p = temp_demod_numpy(I, self._N, self._f, self.p0)
            # a, b, p = temp_demod_numpy_unknown_frequencies(I, self._N, self.p0)
            if self.uwr == "temporal":
                NotImplemented  # todo: call tpu() from decoder (as ufunc?)

            # todo: x, k, r

            b.shape = (-1, Y, X, C)
            p.shape = (-1, Y, X, C)
        else:
            try:
                if threads is not None:
                    set_num_threads(threads)

                a, b, p, k, x, r = decode(
                    I,
                    self._N,
                    self._v * self.Lext / self.L[:, None] if "mrd" in self.mode else self._v,
                    -self._f if self.reverse else self._f,
                    self.L,
                    self.UMR,
                    self.crt,
                    self.gcd,
                    self.x0,
                    self.p0,
                    bmin,
                    Vmin,
                    self.mode,
                    verbose,
                )
            finally:
                set_num_threads(get_num_threads())

        # spatial unwrapping
        for d in range(self.D):
            if self.UMR[d] < self.L[d] * self.a:
                logger.warning("Unwrapping is not spatially independent and only yields a relative phase map.")

                # todo: self.K > 1
                # if "precise" in self.mode:
                #     w0 = self.N[d] * self.v[d] ** 2 * b.reshape(self.D, self.K, -1)[d].mean(axis=-1)**2
                #     i0 = np.argmax(w0)  # precise
                # else:
                #     i0 = np.argmin(self.v[d])  # fast
                # todo: unwrap phases via TPU as far as possible in order to reduce the number of phase transitions

                x[d] = spu(x[d], verbose=verbose, uwr_func=uwr_func)

                if "precise" in self.mode:
                    w0 = self._N[d] * self._v[d] ** 2
                    i0 = np.argmax(w0)  # precise
                else:
                    i0 = np.argmin(self._v[d])  # fast
                x[d] *= self._l[d, i0] / _2PI

                if verbose:
                    r[d, :] = np.nan  # todo: from spu() if possible

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return a, b, p, k, x, r

    def _multiplex(self, I: np.ndarray) -> np.ndarray:
        """Multiplex fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Base fringe patterns.

        Returns
        -------
        Imux : np.ndarray
            Multiplexed fringe patterns.
        """
        t0 = time.perf_counter()

        Imux = I

        if self.WDM:
            assert not self.FDM
            assert self._monochrome
            assert np.all(self.N == 3)

            Imux = Imux.reshape((-1, 3, self.Y, self.X, 1))  # returns a view
            Imux = Imux.swapaxes(1, -1)  # returns a view
            Imux = Imux.reshape((-1, self.Y, self.X, self.C))  # returns a view

        if self.SDM:
            assert not self.FDM
            assert self.grid in {"image", "Cartesian"}  # self._choices["grid"][:2]
            assert self.D == 2
            assert I.dtype.kind == "f"

            Imux = Imux.reshape((self.D, -1))  # returns a view
            Imux -= self.A
            Imux = np.sum(Imux, axis=0)
            Imux += self.A
            Imux = Imux.reshape((-1, self.Y, self.X, self.C if self.WDM else 1))  # returns a view
            if self.dtype.kind in "uib":
                Imux = Imux.astype(self.dtype, copy=False)  # returns a view

        if self.FDM:
            assert not self.WDM
            assert not self.SDM
            assert len(np.unique(self.N)) == 1
            assert I.dtype.kind == "f"

            if np.any(self._N < 2 * np.abs(self._f).max() + 1):  # todo: fractional periods
                logger.warning("Decoding might be disturbed.")

            Imux = Imux.reshape((self.D * self.K, -1))  # returns a view
            Imux -= self.A
            Imux = np.sum(Imux, axis=0)
            Imux += self.A
            Imux = Imux.reshape((-1, self.Y, self.X, 1))  # returns a view
            if self.dtype.kind in "uib":
                Imux = Imux.astype(self.dtype, copy=False)  # returns a view

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return Imux

    def _demultiplex(self, I: np.ndarray) -> np.ndarray:
        """Demultiplex fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Multiplexed fringe patterns.

        Returns
        -------
        I : np.ndarray
            Demultiplexed fringe patterns.
        """
        t0 = time.perf_counter()

        I = vshape(I)
        T, Y, X, C = I.shape  # extract Y, X, C from data as these parameters depend on used camera

        if self.SDM:
            assert not self.FDM
            assert self.grid in {"image", "Cartesian"}  # self._choices["grid"][:2]
            assert self.D == 2

            if X % 2 == 0:  # todo: revert padding by simply taking I[:-1] ?
                fx = np.fft.fftshift(np.fft.rfftfreq(X))
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) >= np.abs(fyy)
                my = np.abs(fxx) <= np.abs(fyy)
                J = I
                I = np.empty((2 * T, Y, X, C))
                for t in range(T):
                    for c in range(C):
                        I_FFT = np.fft.fftshift(np.fft.rfft2(J[t, ..., c]))
                        I[t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * mx))
                        I[T + t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * my))
            elif Y % 2 == 0:
                fx = np.fft.fftshift(np.fft.rfftfreq(Y))
                fy = np.fft.fftshift(np.fft.fftfreq(X))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) >= np.abs(fyy)
                my = np.abs(fxx) <= np.abs(fyy)
                J = I.transpose(0, 2, 1, 3)
                I = np.empty((2 * T, X, Y, C))
                for t in range(T):
                    for c in range(C):
                        I_FFT = np.fft.fftshift(np.fft.rfft2(J[t, ..., c]))
                        I[t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * my))
                        I[T + t, ..., c] = np.fft.irfft2(np.fft.ifftshift(I_FFT * mx))
                I = I.transpose(0, 2, 1, 3)
            else:
                fx = np.fft.fftshift(np.fft.fftfreq(X))
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) >= np.abs(fyy)
                my = np.abs(fxx) <= np.abs(fyy)
                J = I
                I = np.empty((2 * T, Y, X, C))
                for t in range(T):
                    for c in range(C):
                        I_FFT = np.fft.fftshift(np.fft.fft2(J[t, ..., c]))
                        I[t, ..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(I_FFT * mx)))
                        I[T + t, ..., c] = np.abs(np.fft.ifft2(np.fft.ifftshift(I_FFT * my)))

        if self.WDM:
            assert not self.FDM
            assert C == 3
            I = I.reshape((-1, 1, Y, X, C))  # returns a view
            I = I.swapaxes(-1, 1)  # returns a view
            I = I.reshape((-1, Y, X, 1))  # returns a view

        if self.FDM:
            assert not self.WDM
            assert not self.SDM  # todo: allow self.SDM?
            assert len(np.unique(self.N)) == 1
            I = np.tile(I, (self.D * self.K, 1, 1, 1))

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return I

    def _colorize(self, I: np.ndarray, frames: Sequence[int]) -> np.ndarray:
        """Colorize fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Base fringe patterns,
            possibly multiplexed.
        frames : np.ndarray
            Indices of the frames to be encoded.

        Returns
        -------
        Icol : np.ndarray
            Colorized fringe patterns.
        """
        t0 = time.perf_counter()

        T_ = I.shape[0]  # number of frames for each hue
        T = len(frames)
        Icol = np.empty((T, self.Y, self.X, self.C), self.dtype)

        for t in frames:
            tb = t % I.shape[0]  # indices from base fringe pattern I

            for c in range(self.C):
                h = int(t // T_)  # hue index
                cb = c if self.WDM else 0  # color index of base fringe pattern I

                if self.h[h, c] == 0:  # uibf -> uibf
                    Icol[t, ..., c] = 0
                elif self.h[h, c] == 255 and Icol.dtype == self.dtype:  # uibf -> uibf
                    Icol[t, ..., c] = I[tb, ..., cb]
                elif self.dtype.kind in "uib":  # f -> uib
                    Icol[t, ..., c] = np.rint(I[tb, ..., cb] * (self.h[h, c] / 255))
                else:  # f -> f
                    Icol[t, ..., c] = I[tb, ..., cb] * (self.h[h, c] / 255)

        # for h in range(self.H):
        #     if frames is None:
        #         for c in range(self.C):
        #             cj = c if self.WDM else 0  # todo: ???
        #             if self.h[h, c] == 0:  # uib -> uib, f -> f
        #                 Icol[h * Th : (h + 1) * Th, ..., c] = 0
        #             elif self.h[h, c] == 255 and Icol.dtype == self.dtype:  # uib -> uib, f -> f
        #                 Icol[h * Th : (h + 1) * Th, ..., c] = I[..., cj]
        #             elif self.dtype.kind in "uib":  # f -> uib
        #                 Icol[h * Th : (h + 1) * Th, ..., c] = np.rint(I[..., cj] * (self.h[h, c] / 255)).astype(
        #                     self.dtype, copy=False
        #                 )
        #             else:  # f -> f
        #                 Icol[h * Th : (h + 1) * Th, ..., c] = I[..., cj] * (self.h[h, c] / 255)
        #     elif h in hues:  # i.e. frames is not None and h in hues
        #         for c in range(self.C):
        #             cj = c if self.WDM else 0  # todo: ???
        #             if self.h[h, c] == 0:  # uib -> uib, f -> f
        #                 Icol[i, ..., c] = 0
        #             elif self.h[h, c] == 255 and Icol.dtype == self.dtype:  # uib -> uib, f -> f
        #                 Icol[i, ..., c] = I[i, ..., cj]
        #             elif self.dtype.kind in "uib":  # f -> uib
        #                 Icol[i, ..., c] = np.rint(I[i, ..., cj] * (self.h[h, c] / 255)).astype(self.dtype, copy=False)
        #             else:  # f -> f
        #                 Icol[i, ..., c] = I[i, ..., cj] * (self.h[h, c] / 255)
        #         i += 1

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return Icol

    def _decolorize(self, I: np.ndarray) -> np.ndarray:
        """Decolorize fringe patterns by weighted averaging of hues.

        Parameters
        ----------
        I : np.ndarray
            Colorized fringe patterns.

        Returns
        -------
        I : np.ndarray
            Decolorized fringe patterns.
        """
        t0 = time.perf_counter()

        I = vshape(I)
        T, Y, X, C = I.shape  # extract Y, X, C from data as these parameters depend on used camera
        I = I.reshape((self.H, T // self.H, Y, X, C))  # returns a view

        solo = np.all(np.count_nonzero(self.h, axis=0) == 1)  # each RGB component exists exactly once
        base = np.all(np.count_nonzero(self.h, axis=1) == 1)  # each hue consists of only one RGB base color
        same = len(set(self.h[self.h != 0])) == 1  # all color channels have the same value
        # if solo and base: no averaging necessary
        # if same: all weights are the same

        if self.H == 3 and C in [1, 3] and solo and base and same:
            I = np.moveaxis(I, 0, -2)  # returns a view

            # basic slicing returns a view
            idx = np.argmax(self.h, axis=1)
            if np.array_equal(idx, [0, 2, 1]):  # RBG
                I = I[..., 0::-1, :]
            elif np.array_equal(idx, [1, 2, 0]):  # GBR
                I = I[..., 1:1:, :]
            elif np.array_equal(idx, [1, 0, 2]):  # GRB
                I = I[..., 1:1:-1, :]
            elif np.array_equal(idx, [2, 1, 0]):  # BGR
                I = I[..., 2::-1, :]
            elif np.array_equal(idx, [2, 0, 1]):  # BRG
                I = I[..., 2:2:-1, :]
            # elif np.array_equal(idx, [0, 1, 2]):  # RGB
            #     I = I[..., :, :]

            if C == 1:
                I = I[..., 0]  # returns a view
            elif C == 3:
                I = np.diagonal(I, axis1=-2, axis2=-1)  # returns a view
        elif self.H == 2 and C in [1, 3] and solo and same:
            I = np.moveaxis(I, 0, -2)  # returns a view

            # advanced indexing returns a copy, not a view
            if C == 1:
                idx = np.argmax(self.h, axis=0)
                I = I[..., idx, :]
                I = I[..., 0]  # returns a view
            elif C == 3:
                idx = self.h != 0
                I = I[..., idx]
        else:
            # fuse colors by weighted averaging,
            # this assumes constant camera exposure settings for each set of hues

            w = self.h / np.sum(self.h, axis=0)  # normalized weights
            w[np.isnan(w)] = 0
            if C == 1 and same:
                w = w[:, 0][:, None]  # ensures that the result has only one color channel

            # if np.all((w == 0) | (w == 1)):  # doesn't happen due to testing for 'solo' and 'base'
            #     w = w.astype(bool, copy=False)  # multiplying with bool preserves dtype
            #     dtype = I.dtype  # without this, np.sum chooses a dtype which can hold the theoretical maximal sum
            # else:

            dtype = float  # without this, np.sum chooses a dtype which can hold the theoretical maximal sum, i.e. uint32 or uint64 with problems in pyqtgraph's
            I = np.sum(I * w[:, None, None, None, :], axis=0, dtype=dtype)
            # todo: np.dot or np.tensordot ?

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return I

    def encode(self, frames: int | Sequence[int] | None = None) -> np.ndarray:
        """Encode fringe patterns.

        Parameters
        ----------
        frames : int or Sequence of ints, optional
            Indices of the frames to be encoded.
            The default, frames=None, will encode all frames at once.
            Else, `frames` is used as an index into the fringe pattern sequence
            (negative indexes are allowed; indexes with magnitude larger than `T` are wrapped around).

        Returns
        -------
        I : np.ndarray
            Fringe pattern sequence.

        Notes
        -----
        To receive the frames iteratively (i.e. in a lazy manner),
        simply iterate over the Fringes instance.

        Examples
        --------
        >>> from fringes import Fringes
        >>> f = Fringes()

        Encode the complete fringe pattern sequence.

        >>> I = f.encode()

        Create a generator to receive the frames iteratively, i.e. in a lazy manner.

        >>> I = (frame for frame in f)

        Call the instance to encode the complete fringe pattern sequence.

        >>> I = f()
        """
        t0 = time.perf_counter()

        # check frames
        if frames is None:
            frames = np.arange(self.H * np.sum(self._N))
        else:
            if not hasattr(frames, "__iter__"):
                frames = [frames]

            frames = np.array(frames, int) % self.T  # t=0:T:1

            if self.FDM:
                frames = np.array(
                    [np.arange(i * self.D * self.K, (i + 1) * self.D * self.K) for i in frames]
                ).ravel()  # t*D*K:(t+1)*D*K:1

            if self.WDM:  # WDM before SDM
                N = 3
                frames = np.array([np.arange(i * N, (i + 1) * N) for i in frames]).ravel()  # t*3:(t+1)*3:1

            if self.SDM or self.uwr == "FTM":  # WDM before SDM
                len_D0 = np.sum(self._N[0])
                frames = np.array([np.arange(i, i + len_D0 + 1, len_D0) for i in frames]).ravel()  # t:t+len_D0:len_D0

        # modulate
        I = self._modulate(self._x, frames)

        # multiplex (reduce number of frames)
        if self.SDM or self.WDM or self.FDM or self.uwr == "FTM":
            I = self._multiplex(I)

        # apply inscribed circle
        if self.grid in ["polar", "log-polar"]:
            I *= grid.inner_circ(self.Y, self.X)[None, :, :, None]

        # colorize (extended averaging)
        if self.H > 1 or np.any(self.h != 255):  # can be used for extended averaging
            I = self._colorize(I, frames)

        logger.info(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return I

    def decode(
        self,
        I: np.ndarray,
        bmin: float = 0,
        Vmin: float = 0,
        unwrap: bool = True,
        verbose: bool | str = False,
        check_overexposure: bool = False,
        threads: int | None = None,
        uwr_func: str = "ski",
    ) -> Decoded | Decoded_verbose:
        r"""Decode fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to `vshape` (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

            .. note:: `I` must have been encoded with the same parameters set to the Fringes instance as the current one.

        bmin : float, default=0
            Minimum modulation for measurement to be valid.
            Can accelerate decoding.
        Vmin : float, default=0
            Minimum visibility for measurement to be valid.
            Can accelerate decoding.
        unwrap : bool, default=True
            Flag for unwrapping.
        verbose : bool or str, default=False
            Flag for returning intermediate and verbose results.
        check_overexposure: bool, default=False
            Flag for checking if 'I' is overexposed.
            Logs a warning message if so.
        threads : int, optional
            Number of threads to use.
            Default is all threads.
        uwr_func : {'ski', 'cv2'}, optional
            Spatial unwrapping function to use.

            - 'ski': `Scikit-image <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_ [1]_

            - 'cv2': `OpenCV <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_ [2]_

        Returns
        -------
        a : np.ndarray
            Brightness: average signal.
        b : np.ndarray
            Modulation: amplitude of the cosine signal.
        x : np.ndarray
            Registration: decoded screen coordinates.
        p : np.ndarray, optional
            Local phase.
        k : np.ndarray, optional
            Fringe order.
        r : np.ndarray, optional
            Residuals from the optimization-based unwrapping process.
        u : np.ndarray, optional
            Uncertainty of positional decoding in pixel units

        Raises
        ------
        ValueError
            If the number of frames of `I` and the attribute `T` of the `Fringes` instance don't match.
            If `WDM` is active but `I` does not have 3 color channels.

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

        Examples
        --------
        >>> from fringes import Fringes
        >>> f = Fringes()
        >>> I = f.encode()
        >>> Irec = I  # todo: replace this line with the recorded data as in 'record.py'

        >>> a, b, x = f.decode(Irec)

        Also return intermediate and verbose results:

        >>> a, b, x, p, k, r, u = f.decode(Irec, verbose=True)
        """
        t0 = time.perf_counter()

        # parse dtype 'object'
        if I.dtype == "O":
            I = np.array([frame for frame in I])  # creates a copy

            # todo: without a copy
            # dtype = I[0].dtype
            # dtype = float
            # I = I.astype(dtype)
            # I = I.astype(dtype, copy=False)

        # get and apply video-shape
        I = vshape(I)
        T, Y, X, C = I.shape  # extract Y, X, C from data as these parameters depend on used camera

        # check for overexposure
        if check_overexposure:
            if I.dtype.kind in "ui":
                Imax = I.max()
                lessbits = True  # todo: lessbits
                if lessbits and np.iinfo(I.dtype).bits > 8:  # data may contain fewer bits of information
                    bits = int(np.ceil(np.log2(Imax + 1)))  # same or next power of two
                    bits += -bits % 2  # same or next power of two which is divisible by two
                    maxval = 2**bits - 1
                    if maxval < Imax:  # if
                        maxval = Imax
                else:
                    maxval = np.iinfo(I.dtype).max

                if np.count_nonzero(I == maxval) > 0.1 * I.size:
                    logger.warning("'I' is probably overexposed and decoding might yield unreliable results.")

        # check number of frames
        if len(I) != self.T:
            raise ValueError("Number of frames of data and parameters don't match.")

        # check color channels
        if self.WDM and C != 3:
            raise ValueError(
                f"'I' must have 3 color channels because 'WDM' is active, but has only {C} color channels."
            )

        # decolorize (fuse hues/colors); for gray fringes, color fusion is not performed, but extended averaging is
        if self.H > 1 or not self._monochrome:
            I = self._decolorize(I)

        # demultiplex
        if self.SDM and 1 not in self.N or self.WDM or self.FDM:
            I = self._demultiplex(I)

        # demodulate
        bmin = float(max(0, bmin))
        Vmin = float(min(max(0, Vmin), 1))
        a, b, p, k, x, r = self._demodulate(I, bmin, Vmin, unwrap, verbose, threads)
        if verbose:
            u = self.uncertainty(b=b)

        # # blacken where color value of hue was black
        # if self.H > 1 and C == 3:
        #     idx = np.sum(self.h, axis=0) == 0
        #     if np.any(idx):  # blacken where color value of hue was black
        #         a[..., idx] = 0
        #         b[..., idx] = 0
        #         x[..., idx] = np.nan
        #         if verbose:
        #             r[..., idx] = np.nan
        #             u[..., idx] = np.nan  # self.L / np.sqrt(12)  # todo: circular distribution
        #             p[..., idx] = np.nan

        # # coordinate retransformation  # todo: test
        # if self.D == 2:
        #     # todo: swapaxes
        #     if self.grid == "Cartesian":
        #         if self.X >= self.Y:
        #             x[0] += self.X / 2 - 0.5
        #             x[0] %= self.X
        #             x[1] *= -1
        #             x[1] += self.Y / 2 - 0.5
        #             x[1] %= self.X
        #         else:
        #             x[0] += self.X / 2 - 0.5
        #             x[0] %= self.Y
        #             x[1] *= -1
        #             x[1] += self.Y / 2 - 0.5
        #             x[1] %= self.Y
        #
        #     # todo: polar, logpolar, spiral
        #     # if self.angle != 0:
        #     #     t = np.deg2rad(-self.angle)
        #     #
        #     #     if self.angle % 90 == 0:
        #     #         c = np.cos(t)
        #     #         s = np.sin(t)
        #     #         L = np.array([[c, -s], [s, c]])
        #     #         # L = np.matrix([[c, -s], [s, c]])
        #     #         ur = L[0, 0] * x[0] + L[0, 1] * x[1]
        #     #         vr = L[1, 0] * x[0] + L[1, 1] * x[2]
        #     #         # u = np.dot(uu, L)  # todo: matrix multiplication
        #     #         # v = np.dot(vv, L)
        #     #     else:
        #     #         tan = np.tan(t)
        #     #         ur = x[0] - x[1] * np.tan(t)
        #     #         vr = x[0] + x[1] / np.tan(t)
        #     #
        #     #     vv = (x[1] - x[0]) / (1 / tan - tan)
        #     #     uu = x[0] + vv * tan
        #     #     x = np.stack((uu, vv), axis=0)
        #     #     x = np.stack((ur, vr), axis=0)

        logger.info(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        # return namedtuple
        if verbose:
            return Decoded_verbose(a, b, x, p, k, r, u)
        else:
            return Decoded(a, b, x)

    def uncertainty(
        self,
        ui: float | np.ndarray = np.sqrt(1 / 12),
        b: float | np.ndarray | None = None,
        a: float | np.ndarray | None = None,
        K: float | None = None,
        dark_noise: float | None = None,
    ) -> np.ndarray:
        """Uncertainty of positional decoding, in pixel units.

        Using inverse variance weighting in unwrapping
        and assuming no fringe order errors.

        Parameters
        ----------
        ui: float | np.ndarray, default=np.sqrt(1 / 12)
            Standard deviation of intensity noise.
            The default assumes only quantization noise.
            If np.ndarray, it must have image shape (Y, X, C).
        b : float | np.ndarray, optional
            Modulation.
            If np.ndarray, it must have video shape (T, Y, X, C).
        a : float | np.ndarray, optional
            Offset (mean number of gray values).
            If np.ndarray, it must have video shape (T, Y, X, C).
        K : float, optional
            System gain of used camera, in units DN/electron.
        dark_noise : float, optional
            Dark noise of used camera, in units electrons.

        Returns
        -------
        u : np.ndarray
            Uncertainty of positional decoding, in pixel units.

        Note
        ----
        If `a`, `K` and `dark_noise` are given, `ui` is ignored and calculated from them.
        In any case, if either `ui`, `b` or `a` is a numpy.ndarray,
        they must be broadcastable and hence their shape must end in the same (Y, X, C) values
        (the function takes care of those who are in front).

        Examples
        --------
        >>> from fringes import Fringes
        >>> f = Fringes()

        Assuming only quantization noise and using `f.B` as `b` (both by default):

        >>> u = f.uncertainty()

        Setting `ui` and `b` with floats:

        >>> u = f.uncertainty(ui=2.28, b=100)

        Using decoded values (numpy.ndarrays) and camera parameters:

        >>> I = f.encode()
        >>> Irec = I  # todo: replace this line with the recorded data as in 'record.py'
        >>> a, b, x = f.decode(Irec)

        >>> u = f.uncertainty(b=b, a=a, K=0.038, dark_noise=13.7)
        """
        t0 = time.perf_counter()

        if b is None:
            b = self.B
        elif isinstance(b, np.ndarray):
            # make b broadcastable
            Tb, Yb, Xb, Cb = vshape(b).shape
            assert Tb == self.D * self.K
            b = b.reshape(self.D, self.K, Yb, Xb, Cb)

            # check if b and (a or ui) are broadcastable
            if isinstance(a, np.ndarray):
                assert vshape(a).shape == (self.D, Yb, Xb, Cb)
            elif isinstance(ui, np.ndarray):
                assert vshape(ui).shape == (1, Yb, Xb, Cb)

        if a is not None and K is not None and dark_noise is not None:
            if isinstance(a, np.ndarray):
                # make a broadcastable
                Ta, Ya, Xa, Ca = vshape(a).shape
                assert Ta == self.D
                a = a.reshape(Ta, 1, Ya, Xa, Ca)

            # compute ui
            d = dark_noise * K
            q = np.sqrt(1 / 12)
            p = np.sqrt(a / K) * K
            ui = np.sqrt(d**2 + q**2 + p**2)
        elif isinstance(ui, np.ndarray):
            # make ui broadcastable
            Tui, Yui, Xui, Cui = vshape(ui).shape
            assert Tui == 1
            ui = ui.reshape(1, 1, Yui, Xui, Cui)

        if isinstance(ui, np.ndarray) or isinstance(b, np.ndarray) or isinstance(a, np.ndarray):
            # make N and l broadcastable
            N = self._N[:, :, None, None, None]
            l = self._l[:, :, None, None, None]
        else:
            N = self._N
            l = self._l

        up = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(N) * ui / b  # phase uncertainties  # todo: M
        ux = up / _2PI * l  # positional uncertainties
        u = np.sqrt(1 / np.sum(1 / ux**2, axis=1))  # positional uncertainty
        u = u.astype(np.float32, copy=False)

        logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")

        return u

    def _trim(self, a: np.ndarray) -> np.ndarray:
        """Change ndim of `a` to 2 and limit its shape."""
        # D, K = self._v.shape  # todo: if K is read only
        if a.ndim == 0:
            a = np.full((self.D, self.K), a)
        elif a.ndim == 1:
            a = np.vstack([a[: self._Kmax] for d in range(self.D)])
        elif a.ndim == 2:
            a = a[: self._Dmax, : self._Kmax]
        else:
            a = a[: self._Dmax, : self._Kmax, 0]

        return a

    def set_mtf(self, b: np.ndarray):
        """Set the modulation transfer function (MTF).

        Parameters
        ----------
        b : np.ndarray
            Measured modulation.

        Raises
        ------
        ValueError
            If 'b' contains negative values.

        Notes
        -----
        The normalized modulation transfer function is a cubic spline interpolator.
        It is set to the private attribute '_mtf' and used when the method 'mtf()' is called.
        """
        if b is None:
            self._mtf = None
            logger.debug(f"Reset modulation transfer function.")
            return

        if np.any(b < 0):
            raise ValueError("Negative values for 'b' are not allowed.")

        b = vshape(b)
        T, Y, X, C = b.shape  # extract Y, X, C from data as these parameters depend on used camera

        assert T == self.D * self.K  # todo: description of AssertionError

        b = b.reshape(self.D, self.K, Y, X, C)

        # filter
        if C > 1:
            b = np.nanmedian(b, axis=-1)  # filter along color axis
        # b = np.nanmedian(b, axis=(2, 3))  # filter along spatial axes
        b = np.nanquantile(b, 0.9, axis=(2, 3))  # filter along spatial axes
        b[np.isnan(b)] = 0
        # v_ = np.unique(self._v, axis=1)
        v_ = self._v

        self._mtf = [sp.interpolate.CubicSpline(v_[d], b[d], extrapolate=True) for d in range(self.D)]

        logger.debug(f"Set modulation transfer function.")

    # @property
    # def MTF_(self): ...
    #
    # def MTF_2(self):
    #     ...

    def mtf(self, v: Sequence[float] | None = None, PSF: float = 0.0) -> np.ndarray:
        """Modulation Transfer Function,

        normalized to values ∈ [0, 1].

        Parameters
        ----------
        v: array_like
            Spatial frequencies at which to determine the normalized modulation.
        PSF: float, default=0
            Standard deviation of the Point Spread Function for defocus.

        Returns
        -------
        b : np.ndarray
            Relative modulation, at spatial frequencies 'v'.
            'b' is in the same shape as 'v'.

        Raises
        ______
        ValueError
            If 'v' contains negative numbers.

        Notes
        -----
        - If the private attribute `_mtf` of the Fringes instance is not None, the MTF is interpolated from previous measurements.\n
        - Else, if the attribute `PSF` of the Fringes instance is larger than zero, the MTF is computed from the optical transfer function of the optical system, i.e. as the magnitude of the Fourier-transformed 'Point Spread Function' (PSF).\n
        - Else, it returns ones.
        """
        # todo: test

        if v is None:
            v = self.v

        v_ = np.array(v, float, copy=False, ndmin=2)

        if np.any(v_ < 0):
            raise ValueError("Negative numbers are not allowed.")

        D, K = v_.shape

        if self._mtf is not None:  # interpolate from (previous) measurement
            b = np.array([self._mtf[d](v_[d]) for d in range(D)]).reshape(D, K)
            bmax = np.array([self._mtf[d](0) for d in range(D)])
            b /= bmax
            np.clip(b, 0, 1, out=b)
            # todo: self._mtf = None when v or l or D changes or (D == 1 and axis changes) or indexing changes
        elif PSF > 0:  # determine MTF from PSF
            b = self.B * np.exp(-2 * (np.pi * PSF * v_) ** 2)  # todo: fix
            # todo: what is smaller: dl or lv?
        else:
            b = 1 - v_ / self.vmax  # approximation of [Bothe2008]
        # else:
        #     b = np.ones(v_.shape)

        return b

    @property
    def _Td(self) -> np.ndarray:
        """Number of frames per direction."""
        if self.uwr == "FTM":
            return np.array([0.5, 0.5])  # todo: D = 1 ?

        Td = self.H * np.sum(self._N, axis=1, dtype=int)

        if self.FDM:  # todo: fractional periods
            Td = Td / (self.D * self.K)

        if self.SDM:
            Td = Td / self.D

        if self.WDM:
            if np.all(self.N == 3):  # WDM along shifts
                Td = Td / 3
            # elif self.K > self.D:  # WDM along sets todo
            #     a = np.sum(self._N, axis=1)
            #     b = np.max(a)
            #     c = np.ceil(b / 3)
            #     d = int(c)
            #
            #     a2 = np.sum(self._N, axis=0)
            #     b2 = np.max(a2)
            #     c2 = np.ceil(b2 / 2)
            #     d2 = int(c2)
            #
            #     if d < d2:
            #         T = int(np.ceil(np.max(np.sum(self._N, axis=1)) / 3))
            #     else:  # use red and blue
            #         T = int(np.ceil(np.max(np.sum(self._N, axis=0)) / 2))
            # else:  # WDM along directions, use red and blue todo
            #     a = np.sum(self._N, axis=0)
            #     b = np.max(a)
            #     c = np.ceil(b / 2)
            #     d = int(c)
            #     T = int(np.ceil(np.max(np.sum(self._N, axis=1)) / 2))

        return Td

    @property
    def T(self) -> int:
        """Number of frames."""
        return int(np.sum(self._Td))  # use int() to ensure type is "int" instead of "numpy.core.multiarray.scalar"

    @property
    def Y(self) -> int:
        """Height of fringe patterns.
        [Y] = px."""
        return self._Y

    @Y.setter
    def Y(self, Y: int):
        _Y = int(max(1, Y))

        if self._Y != _Y:
            self._Y = _Y
            logger.debug(f"{self._Y=}")

            if self._Y >= 2**20:
                logger.warning(f"Height 'Y' exceeds 'OPENCV_IO_MAX_IMAGE_HEIGHT': {self._Y} > 2**20.")

            if self._Y * self._X >= 2**30:
                logger.warning(
                    f"Frame size 'Y' * 'X' exceeds 'OPENCV_IO_MAX_IMAGE_PIXELS': {self._X * self._Y} > 2**30."
                )

            if self._Y == 1:
                self.D = 1
                self.axis = 0

            self._UMR = None  # empty cache
            self.UMR  # compute and cache

    @property
    def X(self: int) -> int:
        """Width of fringe patterns.
        [X] = px."""
        return self._X

    @X.setter
    def X(self, X: int):
        _X = int(max(1, X))

        if self._X != _X:
            self._X = _X
            logger.debug(f"{self._X=}")

            if self._Y >= 2**20:
                logger.warning(f"Width 'X' exceeds 'OPENCV_IO_MAX_IMAGE_WIDTH': {self._X} > 2**20.")

            if self._Y * self._X >= 2**30:
                logger.warning(
                    f"Frame size 'Y' * 'X' exceeds 'OPENCV_IO_MAX_IMAGE_PIXELS': {self._X * self._Y} > 2**30."
                )

            if self._Y == 1:
                self.D = 1
                self.axis = 0

            self._UMR = None  # empty cache
            self.UMR  # compute and cache

    @property
    def C(self) -> int:
        """Number of color channels."""
        return 3 if self.WDM or not self._monochrome else 1

    @property
    def L(self) -> np.ndarray:
        """Coding lengths, i.e. lengths of fringe patterns for each direction.
        [L] = px."""
        L = np.array([self.X, self.Y])

        if self.indexing == "ij":
            L = L[::-1]

        if self.D == 1:
            L = np.atleast_1d(L[self.axis])

        return L

    @property
    def Lmax(self) -> int:
        """Maximum coding length.
        [Lmax] = px"""
        return self.L.max()

    @property
    def Lext(self) -> float:
        """Extended coding length.
        [Lext] = px."""
        return self.a * self.Lmax

    @property
    def a(self) -> float:
        """Factor for extending the coding length."""
        # a = 1 + 2 * x0 / self.Lmax
        return self._a

    @a.setter
    def a(self, a: float):
        _a = float(min(max(1, a), self._amax))

        if self._a != _a:
            self._a = _a
            logger.debug(f"{self._a=}")
            logger.debug(f"{self.x0=}")
            logger.debug(f"self._l = {str(self._l.round(3)).replace(chr(10), ',')}")
            self._UMR = None  # empty cache
            self.UMR  # compute and cache

    @property
    def _x(self) -> tuple[np.ndarray] | np.ndarray:
        """Coordinate matrices of the coordinate system defined in `grid`.

        Indexing order is defined in `indexing`.
        If possible, a spase representation of the grid is returned.
        """
        if self.grid == "image":  # todo: and self.angle == 0:
            bits = int(np.ceil(np.log2(self.Lmax - 1)))  # next power of two
            bits += -bits % 8  # next power of two divisible by 8

            x = np.indices((self.Y, self.X), dtype=f"uint{bits}", sparse=True)

            if self.indexing == "xy":
                x = x[::-1]
        else:
            if self.grid == "image":
                sys = "img"
            elif self.grid == "Cartesian":
                sys = "cart"
            elif self.grid == "polar":
                sys = "polar"
            else:
                sys = "logpol"

            # x = np.array(getattr(grid, sys)(self.Y, self.X, self.angle))
            x = np.array(getattr(grid, sys)(self.Y, self.X, 0))  # todo: spiral angle 45

            if self.indexing == "ij":
                x = x[::-1]  # returns a view

            if self.grid in ["polar", "log-polar"]:
                x *= self.Lext

        if self.D == 1:
            x = x[self.axis][None, ...]

        return x

    @property
    def x(self) -> np.ndarray:
        """Coordinate matrices of the coordinate system defined in `grid`.

        Indexing order is defined in `indexing`.
        This is always a fleshed out representation.
        """
        return np.array(np.broadcast_arrays(*self._x)) if isinstance(self._x, tuple) else self._x

    @property
    def xc(self) -> np.ndarray:
        """Coordinate matrices as in `x` but with an additional axis for the color channel.
        This is useful when running tests.
        """
        return self.x[..., None]  # add axis for color channel

    @property
    def x0(self) -> float:
        """Coordinate offset."""
        # return self.Lmax * (self.a - 1) / 2
        return (self.Lext - self.Lmax) / 2

    @property
    def grid(self) -> str:
        """Coordinate system of the fringe patterns.

        The following values can be set:\n
        'image':     The top left corner pixel of the grid is the origin and positive directions are right- resp. downwards.\n
        'Cartesian': The center of grid is the origin and positive directions are right- resp. upwards.\n
        'polar':     The center of grid is the origin and positive directions are clockwise resp. outwards.\n
        'log-polar': The center of grid is the origin and positive directions are clockwise resp. outwards.
        """
        return self._grid

    @grid.setter
    def grid(self, grid: str):
        _grid = str(grid)

        if (self.SDM or self.uwr == "FTM") and self.grid not in {"image", "Cartesian"}:  # self._choices["grid"][:2]:
            logger.error(f"Couldn't set 'grid': 'grid' not in {{{"image", "Cartesian"}}}.")  # self._choices["grid"][:2]
            return

        if self._grid != _grid and _grid in self._choices["grid"]:
            self._grid = _grid
            logger.debug(f"{self._grid=}")
            self.SDM = self.SDM

    # @property
    # def angle(self) -> float:
    #     """Angle of the coordinate system's principal axis."""
    #     return self._angle
    #
    # @angle.setter
    # def angle(self, angle: float):
    #     _angle = float(np.remainder(angle, 360))  # todo: +- 45
    #
    #     if self._angle != _angle:
    #         self._angle = _angle
    #         logger.debug(f"{self._angle=}")

    @property
    def D(self) -> int:
        """Number of directions."""
        return self._D

    @D.setter
    def D(self, D: int):
        _D = int(min(max(1, D), self._Dmax))

        if self._D > _D:  # this means that _D == 1
            self._D = _D
            logger.debug(f"{self._D=}")

            self.N = self._N[self.axis, :]
            self.v = self._v[self.axis, :]
            self.f = self._f[self.axis, :]

            self.SDM = False  # adjusts B
            if self._K == 1:
                self.FDM = False
        elif self._D < _D:  # this means that D_ == 2
            self._D = _D
            logger.debug(f"{self._D=}")

            self.N = np.append(self._N, np.tile(self._N[-1, :], (_D - self._N.shape[0], 1)), axis=0)
            self.v = np.append(self._v, np.tile(self._v[-1, :], (_D - self._v.shape[0], 1)), axis=0)
            self.f = np.append(self._f, np.tile(self._f[-1, :], (_D - self._f.shape[0], 1)), axis=0)

            self.B = self.B  # checks clipping

    @property
    def axis(self) -> int:
        """Axis along which to shift if number of directions equals one."""
        return self._axis

    @axis.setter
    def axis(self, axis: int):
        _axis = int(min(max(0, axis), self._Dmax - 1))

        if self._axis != _axis:
            self._axis = _axis
            logger.debug(f"{self._axis=}")

    @property
    def M(self) -> int:  # float, np.ndarray
        """Number of averaged intensity samples."""
        # M = np.sum(self.h, axis=0) / 255  # todo: might be a fraction
        M = np.count_nonzero(self.h, axis=0)  # todo: always is an integer

        # M = np.atleast_1d(M[0]) if self._monochrome else M
        M = M[0] if self._monochrome else M
        M = M.min()  # todo: array of M
        return int(M)  # use int() or float() to ensure type is "int" instead of "numpy.core.multiarray.scalar"

    # @M.setter
    # def M(self, M: int):
    #     _M = int(max(1, M))
    #     self.h = "w" * _M
    #     # todo: float

    @property
    def H(self) -> int:
        """Number of hues."""
        return self.h.shape[0]

    # @H.setter
    # def H(self, H: int):
    #     _H = int(max(1, H))
    #
    #     if self.H != _H:
    #         if self.WDM:
    #             logger.error("Couldn't set 'H': WDM is active.")
    #             return
    #             self.h = "w" * _H
    #         elif _H == 1:
    #             self.h = "w"
    #         elif _H == 2:
    #             self.h = "rb"
    #         else:
    #             rgb = "rgb" * int(np.ceil(_H / 3))  # repeat "rgb" this many times
    #             self.h = rgb[:_H]

    @property
    def h(self) -> np.ndarray:
        """Hues i.e. colors of fringe patterns.

        Possible values are any sequence of RGB color triples ∈ [0, 255].
        However, black (0, 0, 0) is not allowed.

        The hue values can also be set by assigning any combination of the following characters as a string:\n
        - 'r': red \n
        - 'g': green\n
        - 'b': blue\n
        - 'c': cyan\n
        - 'm': magenta\n
        - 'y': yellow\n
        - 'w': white\n

        Before decoding, repeating hues will be fused by averaging."""
        return self._h

    @h.setter
    def h(self, h: Sequence[int] | str):
        if isinstance(h, str):
            LUT = {
                "r": (255, 0, 0),
                "g": (0, 255, 0),
                "b": (0, 0, 255),
                "c": (0, 255, 255),
                "m": (255, 0, 255),
                "y": (255, 255, 0),
                "w": (255, 255, 255),
            }
            if all(c in LUT.keys() for c in h.lower()):
                # set(h.lower()).issubset(LUT.keys())
                # set(h.lower()) <= LUT.keys()):  # subset  # todo: notation since which python version?
                h = [LUT[c] for c in h.lower()]
            elif h == "default":
                h = self._defaults["h"]
            else:
                return

        # make array, clip first and then cast to dtype to avoid integer under-/overflow
        _h = np.array(h).clip(0, 255).astype("uint8", copy=False)

        if not _h.size:  # empty array
            return

        # trim: change shape to (H, 3) or limit shape
        if _h.ndim == 0:
            _h = np.full((self.H, 3), _h)
        elif _h.ndim == 1:
            _h = np.vstack([_h[:3] for h in range(self.H)])
        elif _h.ndim == 2:
            _h = _h[:, :3]
        else:
            _h = _h[:, :3, ..., -1]

        if _h.shape[1] != 3:  # length of color-axis must equal 3
            logger.error("Couldn't set 'h': 3 color channels must be provided.")
            return

        if np.any(np.max(_h, axis=1) == 0):
            logger.error("Didn't set 'h': Black color is not allowed.")
            return

        if self.WDM and not self._monochrome:
            logger.error("Couldn't set 'h': 'WDM' is active, but not all hues are monochromatic.")
            return

        if not np.array_equal(self._h, _h):
            Hold = self.H
            self._h = _h
            logger.debug(f"self._h = {str(self._h).replace(chr(10), ',')}")
            if Hold != _h.shape[0]:
                logger.debug(f"self.H = {_h.shape[0]}")  # computed upon call
            logger.debug(f"{self.M=}")  # computed upon call

    @property
    def SDM(self) -> bool:
        """Spatial division multiplexing.

        The directions 'D' are multiplexed, resulting in a crossed fringe pattern.
        The amplitude 'B' is halved.
        It can only be activated if we have two directions, i.e. 'D' ≡ 2.
        The number of frames 'T' is reduced by the factor 2."""
        return self._SDM

    @SDM.setter
    def SDM(self, SDM: bool):
        _SDM = bool(SDM)

        if _SDM:
            if self.D != 2:
                _SDM = False
                logger.error("Didn't set 'SDM': Pointless as only one dimension exist.")

            if self.grid not in {"image", "Cartesian"}:  # self._choices["grid"][:2]:
                _SDM = False
                logger.error(
                    f"Couldn't set 'SDM': 'grid' not in {{{"image", "Cartesian"}}}."
                )  # self._choices["grid"][:2]

            if self.FDM:
                _SDM = False
                logger.error("Couldn't set 'SDM': FDM is active.")

        if self._SDM != _SDM:
            self._SDM = _SDM
            logger.debug(f"{self._SDM=}")
            logger.debug(f"{self.T=}")  # computed upon call

            if self.SDM:
                self.B /= self.D
            else:
                self.B *= self.D

    @property
    def WDM(self) -> bool:
        """Wavelength division multiplexing.

        The shifts are multiplexed into the color channel, resulting in an RGB fringe pattern.
        It can only be activated if all shifts equal 3, i.e. 'N' ≡ 3.
        The number of frames 'T' is reduced by the factor 3."""
        return self._WDM

    @WDM.setter
    def WDM(self, WDM: bool):
        _WDM = bool(WDM)

        if _WDM:
            if not np.all(self.N == 3):
                _WDM = False
                logger.error("Couldn't set 'WDM': At least one Shift != 3.")

            if not self._monochrome:
                _WDM = False
                logger.error("Couldn't set 'WDM': Not all hues are monochromatic.")

            if self.FDM:  # todo: remove this, already covered by N
                _WDM = False
                logger.error("Couldn't set 'WDM': FDM is active.")

        if self._WDM != _WDM:
            self._WDM = _WDM
            logger.debug(f"{self._WDM=}")
            logger.debug(f"{self.T=}")  # computed upon call

    @property
    def FDM(self) -> bool:
        """Frequency division multiplexing.

        The directions 'D' and the sets K are multiplexed, resulting in a crossed fringe pattern if 'D' ≡ 2.
        It can only be activated if 'D' ∨ 'K' > 1 i.e. 'D' * 'K' > 1.
        The amplitude 'b' is reduced by the factor 'D' * 'K'.
        Usually 'f' equals 1 and is essentially only changed if frequency division multiplexing ('FDM') is activated:
        Each set per direction receives an individual temporal frequency 'f',
        which is used in temporal demodulation to distinguish the individual sets.
        A minimal number of shifts Nmin ≥ ⌈ 2 * fmax + 1 ⌉ is required
        to satisfy the sampling theorem and N is updated automatically if necessary.
        If one wants a static pattern, i.e. one that remains congruent when shifted, set 'static' to True.
        """
        return self._FDM

    @FDM.setter
    def FDM(self, FDM: bool):
        _FDM = bool(FDM)

        if _FDM:
            if self.D == self.K == 1:
                _FDM = False
                logger.error("Didn't set 'FDM': Dimensions * Sets = 1, so nothing to multiplex.")

            if self.SDM:
                _FDM = False
                logger.error("Couldn't set 'FDM': SDM is active.")

            if self.WDM:  # todo: remove, already covered by N
                _FDM = False
                logger.error("Couldn't set 'FDM': WDM is active.")

        if self._FDM != _FDM:
            self._FDM = _FDM
            logger.debug(f"{self._FDM=}")
            # self.K = self._K
            self.N = self._N
            self.v = self._v
            if self.FDM:
                self.f = self._f
            else:
                self.f = np.ones((self.D, self.K))

            # keep maximum possible visibility constant
            if self.FDM:
                self.B /= self.D * self.K
            else:
                self.B *= self.D * self.K

    @property
    def static(self) -> bool:
        """Flag for creating static fringes (so they remain congruent when shifted)."""
        return self._static

    @static.setter
    def static(self, static: bool):
        _static = bool(static)

        if self._static != _static:
            self._static = _static
            logger.debug(f"{self._static=}")
            self.v = self._v
            self.f = self._f

    @property
    def _Kmax(self) -> int:
        """Maximum number of sets."""
        return (
            int(self.Imax / 2) if (self.SDM or self.FDM) and self.dtype != float else sys.maxsize
        )  # np.inf instead of sys.maxsize but then it's float

    @property
    def K(self) -> int:  # todo: int | Sequence[int]:
        """Number of sets (number of fringe patterns with different spatial frequencies)."""
        return self._K

    @K.setter
    def K(self, K: int):
        # todo: different K for each D: use array of arrays; see above
        # a = np.ones(2)
        # b = np.ones(5)
        # c = np.array([a, b], dtype=object)

        _K = int(min(max(1, K), self._Kmax))

        isBmax = self.B == self._Bmax

        if self._K > _K:  # remove elements
            self._K = _K
            logger.debug(f"{self._K=}")

            self.N = self._N[:, : self.K]
            self.v = self._v[:, : self.K]
            self.f = self._f[:, : self.K]

            if self._D == self._K == 1:
                self.FDM = False

            if isBmax:
                self.B = self._Bmax  # maximizes B
        elif self._K < _K:  # add elements
            self._K = _K
            logger.debug(f"{self._K=}")

            self.N = np.append(
                self._N, np.tile(self._N[0, 0], (self.D, _K - self._N.shape[1])), axis=1
            )  # don't append N from _defaults, this might be in conflict with WDM!
            v = self.Lext ** (1 / np.arange(self._v.shape[1] + 1, _K + 1))
            self.v = np.append(self._v, np.tile(v, (self.D, 1)), axis=1)
            self.f = np.append(
                self._f,
                np.tile(np.array(self._defaults["f"]).ravel()[0], (self.D, _K - self._f.shape[1])),
                axis=1,
            )

            self.B = self.B  # checks clipping

    @property
    def _Nmin(self) -> int:
        """Minimum number of shifts to (uniformly) sample temporal frequencies.

        Per direction at least one set with N ≥ 3 is necessary
        to solve for the three unknowns brightness 'a', modulation 'b' and coordinate x."""
        if self.FDM:
            Nmin = int(np.ceil(2 * self.f.max() + 1))  # sampling theorem
            # todo: 2 * D * K + 1 -> fractional periods if static
        else:
            Nmin = 3  # todo: 1 -> use old decoder
        return Nmin

    @property
    def N(self) -> np.ndarray:
        """Number of (equally distributed) phase shifts."""
        if self.D == 1 or len(np.unique(self._N, axis=0)) == 1:  # sets along directions are identical
            N = self._N[0]  # 1D
        else:
            N = self._N  # 2D
        return N

    @N.setter
    def N(self, N: Sequence[int]):
        if self.K == 1 and np.all(np.array(N) == 1):  # FTM
            _N = np.array([[1], [1]], int)
            self.SDM = True
        else:
            _N = np.maximum(np.array(N, int), self._Nmin)  # make array, cast to dtype, clip
            if np.all(self.N == 1):
                self.SDM = False  # go back from FTM

        if not _N.size:  # empty array
            return

        _N = self._trim(_N)

        if np.all(_N == 1) and _N.shape[1] == 1:  # any
            pass  # FTM
        elif np.any(_N <= 2):
            for d in range(self.D):
                if not any(_N[d] >= 3):
                    i = np.argmax(_N[d])  # np.argmin(_N[d])
                    _N[d, i] = 3

        if self.WDM and not np.all(_N == 3):  # WDM locks N
            logger.error("Couldn't set 'N': At least one Shift != 3.")
            return

        if self.FDM and not np.all(_N == _N[0, 0]):
            _N = np.tile(_N[0, 0], _N.shape)  # all N must be equal

        if not np.array_equal(self._N, _N):
            self._N = _N
            logger.debug(f"self._N = {str(self._N).replace(chr(10), ',')}")
            self.D, self.K = self._N.shape
            logger.debug(f"{self.T=}")
            self._UMR = None  # FDM: N -> fmax -> f -> v  # empty cache
            self.UMR  # compute and cache

    @property
    def lmin(self) -> float:
        """Minimum resolvable wavelength.
        [lmin] = px."""

        # don't use self._fmax, else circular loop
        if self.FDM and self.static:
            fmax = min((self._Nmin - 1) / 2, self.Lext / self._lmin)
            return min(self._lmin, self.Lext / fmax)
        else:
            #     fmax = (self._Nmin - 1) / 2
            return self._lmin

    @lmin.setter
    def lmin(self, lmin: float):
        _lmin = float(max(self._lminmin, lmin))

        if self._lmin != _lmin:
            self._lmin = _lmin
            logger.debug(f"{self._lmin=}")
            logger.debug(f"{self.vmax=}")  # computed upon call
            self.l = self.l  # l triggers v
            if self._mtf is not None:
                self._mtf = None

    @property
    def l(self) -> np.ndarray:
        """Wavelengths of fringe periods.
        [l] = px.

        When Lext changes, v is kept constant and only l is changed."""
        return self.Lext / self.v

    @l.setter
    def l(self, l: Sequence[int | float] | str):
        if isinstance(l, str):
            if "," in l:
                l = np.fromstring(l, sep=",")
            # elif l == "optimal":  # todo: optimal l
            #     lmin = int(np.ceil(self.lmin))
            #     lmax = int(
            #         np.ceil(
            #             max(
            #                 self.Lext / lmin,  # todo: only if b differ slightly
            #                 self.lmin,
            #                 min(self.Lext / self.vopt, self.Lext),
            #                 np.sqrt(self.Lext),
            #             )
            #         )
            #     )
            #
            #     if lmin == lmax and lmax < self.Lext:
            #         lmax += 1
            #
            #     if lmax < self.Lext and not sympy.isprime(lmax):
            #         lmax = sympy.ntheory.generate.nextprime(lmax, 1)  # ensures lcm(a, lmax) >= Lext for all a >= lmin
            #
            #     n = lmax - lmin + 1
            #
            #     l_ = np.array([lmin])
            #     l_max = lmin + 1
            #     lcm = l_
            #     while lcm < self.Lext:
            #         lcm_new = np.lcm(lcm, l_max)
            #         if lcm_new > lcm:
            #             l_ = np.append(l_, l_max)
            #             lcm = lcm_new
            #         l_max += 1
            #     K = min(len(l_), self.K)
            #
            #     C = sp.special.comb(n, K, exact=True, repetition=True)  # number of unique combinations
            #     combos = it.combinations_with_replacement(range(lmin, lmax + 1), K)
            #
            #     # B = int(np.ceil(np.log2(lmax - 1) / 8)) * 8  # number of bytes required to store integers up to lmax
            #     # combos = np.fromiter(combos, np.dtype((f"uint{B}", K)), C)
            #
            #     kroot = self.Lext ** (1 / K)
            #     if self.lmin <= kroot:
            #         lcombos = np.array(
            #             [l for l in combos if np.any(np.array([l]) > kroot) and np.lcm.reduce(l) >= self.Lext]
            #         )
            #     else:
            #         lcombos = np.array([l for l in combos if np.lcm.reduce(l) >= self.Lext])
            #
            #     # lcombos = filter(lcombos, K, self.Lext, lmin)
            #
            #     # idx = np.argmax(np.sum(1 / lcombos**2, axis=1))
            #     # l = lcombos[idx]
            #
            #     v = self.Lext / lcombos
            #     b = self.mtf(v)
            #     var = 1 / self.M / self.N * lcombos**2 / b**2  # todo: D, M
            #     idx = np.argmax(np.sum(1 / var, axis=1))
            #
            #     l = lcombos[idx]
            #
            #     if K < self.K:
            #         l = np.concatenate((np.full(self.K - K, lmin), l))
            #
            #     # while lmax < self.Lext and np.gcd(lmin, lmax) != 1:
            #     #     lmax += 1  # maximum number of iterations? = min(next prime after lmax - lmax, max(0, Lext - lmax, ))
            #     # l = np.array([lmin] * (self.K - 1) + [lmax])
            #
            #     # vmax = int(max(1 if self.K == 1 else 2, self.vmax))  # todo: ripples from int()
            #     # v = np.array([vmax] * (self.K - 1) + [vmax - 1])  # two consecutive numbers are always coprime
            #     # lv = self.Lext / v
            #     # lv = np.maximum(self._lmin, np.minimum(lv, self.Lext))
            #     #
            #     # idx = np.argmax((np.sum(1 / (l ** 2), axis=0), np.sum(1 / (lv ** 2), axis=0)))
            #     # l = l if idx == 0 else lv
            #     # print("l" if idx == 0 else "v")
            elif l == "close":
                lmin = int(max(np.ceil(self.lmin), self.Lext ** (1 / self.K) - self.K))
                l = lmin + np.arange(self.K)
                while np.lcm.reduce(l) < self.Lext:
                    l += 1
            elif l == "small":
                lmin = int(np.ceil(self.lmin))
                lmax = int(np.ceil(self.Lext ** (1 / self.K)))  # wavelengths are around kth root of self.Lext

                if self.K >= 2:
                    lmax += 1

                    if self.K >= 3:
                        lmax += 1

                        if lmax % 2 == 0:  # kth root was even
                            lmax += 1

                        if self.K > 3:
                            ith = self.K - 3
                            lmax = sympy.ntheory.generate.nextprime(lmax, ith)

                if lmin > lmax or lmax - lmin + 1 <= self.K:
                    l = lmin + np.arange(self.K)
                else:
                    lmax = max(
                        lmin, min(lmax, int(np.ceil(self.Lext)))
                    )  # max in outer condition ensures lmax >= lmin even if Lext < lmin
                    if lmin == lmax and lmax < self.Lext:
                        lmax += 1  # ensures lmin and lmax differ so that lcm(l) >= Lext

                    n = lmax - lmin + 1
                    K = min(self.K, n)  # ensures K <= n
                    C = sp.special.comb(n, K, exact=True, repetition=False)  # number of unique combinations
                    combos = np.array(
                        [
                            c
                            for c in it.combinations(range(lmin, lmax + 1), K)
                            if np.any(np.array([c]) > self.Lext ** (1 / self.K)) and np.lcm.reduce(c) >= self.Lext
                        ]
                    )

                    idx = np.argmax(np.sum(1 / combos**2, axis=1))
                    l = combos[idx]

                if K < self.K:
                    l = np.concatenate((l, np.arange(l.max() + 1, l.max() + 1 + self.K - K)))
            elif l == "exponential":
                l = np.concatenate(([np.inf], np.geomspace(self.Lext, self.lmin, self.K - 1)))
            elif l == "linear":
                l = np.concatenate(([np.inf], np.linspace(self.Lext, self.lmin, self.K - 1)))
            else:
                return

        self.v = self.Lext / np.array(l, float)

    @property
    def _l(self) -> np.ndarray:  # kept for backwards compatibility with fringes-GUI
        """Wavelengths of fringe periods.
        [l] = px.

        When Lext changes, v is kept constant and only l is changed."""
        return self.Lext / self._v

    @property
    def vmax(self) -> float:
        """Maximum resolvable spatial frequency."""
        return self.Lext / self.lmin

    @property
    def v(self) -> np.ndarray:
        """Spatial frequencies (number of fringe periods across extended coding length `Lext`)."""
        if self.D == 1 or len(np.unique(self._v, axis=0)) == 1:  # sets along directions are identical
            v = self._v[0]  # 1D
        else:
            v = self._v  # 2D
        return v  # todo: round to number of digits?

    @v.setter
    def v(self, v: Sequence[int | float] | str):
        if isinstance(v, str):
            if "," in v:
                v = np.fromstring(v, sep=",")
            # if v == "optimal":  # todo: optimal v
            # def vopt(self) -> float:
            #     """Optimal spatial frequency for minimal decoding uncertainty."""
            #     # todo: test
            #     if self._mtf is not None:  # interpolate from measurement
            #         v = np.arange(1, self.vmax + 1)
            #         b = self.mtf(v)
            #         N = self._N.ravel()[
            #             np.argpartition(self._N.ravel(), int(self._N.size // 2))[int(self._N.size // 2)]]
            #         var = 1 / self.M / N / (v ** 2) / b ** 2  # todo: D, M
            #         idx = np.argmax(np.sum(1 / var, axis=1))
            #         vopt = v[idx]
            #     elif self.PSF > 0:  # determine from PSF
            #         vopt_ = 1 / (_2PI * self.PSF)  # todo
            #         lopt = 1 / vopt_
            #         vopt = self.Lext / lopt
            #         vopt = self.vmax / 2  # approximation [Bothe2008]
            #     else:
            #         vopt = int(self.vmax)
            #
            #     return vopt
            #     # |{v}| = 2
            #     vmax = int(max(1 if self.K == 1 else 2, self.vopt))
            #     v = np.array([vmax] * (self.K - 1) + [vmax - 1])  # two consecutive numbers are always coprime
            #
            #     # # # |{v}| = K
            #     # vmax = int(max(self.K, self.vopt))
            #     # v = vmax - np.arange(self.K)
            elif v == "exponential":
                # K = int(np.ceil(np.log2(self.vmax))) + 1  # + 1: 2 ** 0 = 1
                v = np.concatenate(([0], np.geomspace(1, self.vmax, self.K - 1)))
            elif v == "linear":
                v = np.concatenate(([0], np.linspace(1, self.vmax, self.K - 1)))
            else:
                return

        _v = np.array(v, float).clip(0, self.vmax)  # make array, cast to dtype, clip
        # todo: allow negative values (effect: inverting amplitude?)

        if not _v.size:  # empty array
            return

        _v = self._trim(_v)

        if self.FDM:
            if self.static:
                if (
                    _v.size != self.D * self.K  # todo
                    or not np.all(_v % 1 == 0)
                    or not np.lcm.reduce(_v.astype(int, copy=False).ravel()) == np.prod(_v)  # todo: equal ggt = 1 ?
                ):  # todo: allow coprimes?!
                    n = min(10, self.vmax // 2)
                    ith = self.D * self.K
                    pmax = sympy.ntheory.generate.nextprime(n, ith + 1)
                    p = np.array(list(sympy.ntheory.generate.primerange(n, pmax + 1)))[:ith]  # primes
                    p = [p[-i // 2] if i % 2 else p[i // 2] for i in range(len(p))]  # resort primes
                    _v = np.sort(np.array(p, float).reshape((self.D, self.K)), axis=1)  # resort primes
                    logger.warning(
                        f"Periods are not coprime. " f"Changing values to {str(_v.round(3)).replace(chr(10), ',')}."
                    )
            # else:
            #     vmax = (self._Nmax - 1) / 2 > _v
            #     _v = np.minimum(_v, vmax)

        if not np.array_equal(self._v, _v):
            self._v = _v
            logger.debug(f"self._v = {str(self._v.round(3)).replace(chr(10), ',')}")
            logger.debug(f"self._l = {str(self._l.round(3)).replace(chr(10), ',')}")
            self.D, self.K = self._v.shape
            self.f = self._f
            self._UMR = None  # empty cache
            self.UMR  # compute and cache
            self._m = None  # empty cache
            self._crt = None  # empty cache
            self.crt  # compute and cache (also computes _m)

    @property
    def _fmax(self):
        """Maximum temporal frequency (maximum number of periods to shift over)."""
        return (
            min((self._Nmin - 1) / 2, self.vmax) if self.FDM and self.static else np.inf  # (self._Nmax - 1) / 2
        )  # todo: Nmin vs. Nmax

    @property
    def f(self) -> np.ndarray:
        """Temporal frequency (number of periods to shift over)."""
        if self.D == 1 or len(np.unique(self._f, axis=0)) == 1:  # sets along directions are identical
            f = self._f[0]  # 1D
        else:
            f = self._f  # 2D
        return f

    @f.setter
    def f(self, f: Sequence[int | float]):
        _f = np.array(f, float).clip(-self._fmax, self._fmax)  # make array, cast to dtype, clip

        if not _f.size:  # empty array
            return

        _f = self._trim(_f)

        # 'f' must not be divisible by 'N'  # todo: test this
        D = min(_f.shape[0], self._N.shape[0])
        K = min(_f.shape[1], self._N.shape[1])
        if np.any(_f[:D, :K] % self._N[:D, :K] == 0):
            # _f = np.ones(_f.shape)
            _f[:D, :K][_f[:D, :K] % self._N[:D, :K] == 0] = 1

        if self.FDM:
            if self.static:
                _f = self._v  # periods to shift over = one full revolution
            else:
                if (
                    _f.shape != (self.D, self.K)
                    or not np.all(i % 1 == 0 for i in _f)
                    or len(np.unique(np.abs(_f))) < _f.size
                ):  # assure _f are int and absolute values of _f differ
                    _f = np.arange(1, self.D * self.K + 1, dtype=float).reshape((self.D, self.K))

        if 0 not in _f and not np.array_equal(self._f, _f):
            self._f = _f
            logger.debug(f"self._f = {str((-self._f if self.reverse else self._f).round(3)).replace(chr(10), ',')}")
            self.D, self.K = self._f.shape
            self.N = self._N  # todo: remove if fractional periods is implemented, log warning

    @property
    def p0(self) -> float:
        """Phase offset ∈ (-2pi, +2pi).

        It can be used to e.g. let the fringe patterns start (at the origin) with a gray value of zero.
        """
        return 0 if "mrd" in self.mode else self._p0

    @p0.setter
    def p0(self, p0: float):
        _p0 = float(np.sign(p0) * np.abs(p0) % _2PI)

        if self._p0 != _p0:
            self._p0 = _p0
            logger.debug(f"self._p0 = {self._p0 / np.pi} \u03c0")  # π

    @property
    def _monochrome(self) -> bool:
        """True if all hues are monochromatic, i.e. the RGB values are identical for each hue."""
        return all(len(set(h)) == 1 for h in self.h)

    @property
    def _ambiguous(self) -> bool:
        """True if unambiguous measurement range is smaller than the coding range."""
        return bool(np.any(self.UMR < self.L * self.a))

    @property
    def indexing(self) -> str:
        """Indexing convention.

        Cartesian indexing `xy` (the default) will index the row first,
        while matrix indexing `ij` will index the colum first.
        This equivalent to the argument 'indexing' in numpy.meshgrid().
        """
        return "ij" if "mrd" in self.mode else self._indexing

    @indexing.setter
    def indexing(self, indexing):
        _indexing = str(indexing).lower()

        if self._indexing != _indexing and _indexing in self._choices["indexing"]:
            self._indexing = _indexing
            logger.debug(f"{self._indexing=}")
            if self.D == 2:
                self._UMR = None  # empty cache
                self.UMR  # compute and cache

    @property
    def reverse(self) -> bool:
        """Flag for shifting fringes in reverse direction."""
        return self._reverse

    @reverse.setter
    def reverse(self, reverse: bool):
        _reverse = bool(reverse)

        if self._reverse != _reverse:
            self._reverse = _reverse
            logger.debug(f"{self._reverse=}")
            logger.debug(f"self._f = {str((-self._f if self.reverse else self._f).round(3)).replace(chr(10), ',')}")

    @property
    def mode(self) -> str:
        """Mode for decoding.

        The following values can be set:\n
        - 'fast'\n
        - 'precise'
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        _mode = str(mode).lower()

        if self._mode != _mode and any(m in _mode for m in self._choices["mode"]):  # _mode in self._choices["mode"]:
            self._mode = _mode
            logger.debug(f"{self._mode=}")

    @property
    def uwr(self) -> str:
        """Phase unwrapping method."""

        if self.K == 1 and np.all(self._N == 1) and self.grid in {"image", "Cartesian"}:  # self._choices["grid"][:2]:
            # todo: v >> 1, i.e. l ~ 8
            uwr = "ftm"  # Fourier-transform method
        elif np.all(self.v <= 1):
            uwr = "none"  # only averaging
        elif self._ambiguous:
            uwr = "spatial"
        else:
            uwr = "temporal"

        return uwr

    @property
    def g(self) -> float:
        """Gamma correction factor used to compensate nonlinearities of the display response curve."""
        return self._g

    @g.setter
    def g(self, g: float):
        _g = float(min(max(0, g), self._gmax))

        if self._g != _g and _g != 0:
            self._g = _g
            logger.debug(f"{self._g=}")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of fringe pattern sequence in video shape (frames, height, with, color channels)."""
        return self.T, self.Y, self.X, self.C

    @property
    def bits(self) -> int:
        """Number of bits."""
        return self._bits

    @bits.setter
    def bits(self, bits: int | float):
        self._bits = min(max(0, bits), 64)  # max uint64
        logger.debug(f"{self.bits = }")
        logger.debug(f"{self.dtype = }")
        logger.debug(f"{self.Imax = }")

    @property
    def dtype(self) -> np.dtype:
        """Data type which can hold 2**`bits` of information."""
        if self.bits == 0:
            dtype = np.dtype(float)
        else:
            bits = int(np.ceil(np.log2(2**self.bits - 1)))  # next power of two
            bits += -bits % 8  # next power of two divisible by 8
            dtype = np.dtype(f"uint{bits}")
        return dtype

    @dtype.setter
    def dtype(self, dtype: np.dtype | str):
        _dtype = np.dtype(dtype)

        if self._dtype != _dtype and str(_dtype) in self._choices["dtype"]:
            self._dtype = _dtype
            logger.debug(f"{self._dtype=}")
            self._bits = np.iinfo(_dtype).bits if _dtype.kind in "ui" else 0
            logger.debug(f"{self._bits=}")
            logger.debug(f"{self.Imax=}")
            logger.debug(f"self.A = {self.A}")
            logger.debug(f"self.B = {self.B}")

    @property
    def Imax(self) -> int:
        """Maximum gray value."""
        # return np.iinfo(self.dtype).max if self.dtype.kind in "ui" else 1
        return 2**self.bits - 1 if self.dtype.kind in "ui" else 1

    @property
    def _Amin(self):
        """Minimum offset."""
        return self.B / self._Vmax

    @property
    def _Amax(self):
        """Maximum offset."""
        return self.Imax - self._Amin

    @property
    def A(self) -> float:
        """Offset."""
        return self.Imax * self.E

    @A.setter
    def A(self, A: float):
        _A = float(min(max(self._Amin, A), self._Amax))

        if self.A != _A:
            self.E = _A / self.Imax

    @property
    def _Bmax(self):
        """Maximum amplitude."""
        return min(self.A, self.Imax - self.A) * self._Vmax

    @property
    def B(self) -> float:
        """Amplitude."""
        return self.A * self.V

    @B.setter
    def B(self, B: float):
        _B = float(min(max(0, B), self._Bmax))

        if self.B != _B:  # and _B != 0:
            self.V = _B / self.A

    @property
    def _Emax(self):
        """Maximum relative offset (exposure)."""
        return 1 / (1 + self.V)

    @property
    def E(self) -> float:
        """Relative exposure, i.e. relative mean intensity ∈ [0, 1]."""
        return self._E

    @E.setter
    def E(self, E) -> float:
        _E = float(min(max(0, E), self._Emax))

        if self._E != _E:
            self._E = _E
            logger.debug(f"{self.E=}")
            logger.debug(f"{self.A=}")

    @property
    def _Vmax(self):
        """Maximum visibility."""
        if self.FDM:
            return 1 / (self.D * self.K)
        elif self.SDM:
            return 1 / self.D
        else:
            return 1

    @property
    def V(self) -> float:
        """Fringe visibility (fringe contrast) ∈ [0, 1].

        Screens are extremely nonlinear near the minimum and maximum values,
        we can avoid them by setting V to e.g. 0.95.
        """
        return self._V

    @V.setter
    def V(self, V: float):
        _V = float(min(max(0, V), self._Vmax))

        if self._V != _V:
            self._V = _V
            logger.debug(f"{self.V=}")
            logger.debug(f"{self.B=}")

    @property
    def UMR(self) -> np.ndarray:
        """Unambiguous measurement range.
        [UMR] = px

        The coding is only unique within the interval [0, UMR); after that it repeats itself.

        The UMR is derived from 'l' and 'v':\n
        - If 'l' ∈ ℕ, UMR = lcm('l'), with lcm() being the least common multiple.\n
        - Else, if 'v' ∈ ℕ, UMR = 'Lext' / gcd('v'), with gcd() being the greatest common divisor.\n
        - Else, if 'l' ∨ 'v' ∈ ℚ, lcm() resp. gcd() are extended to rational numbers.\n
        - Else, if 'l' ∧ 'v' ∈ ℝ ∖ ℚ, UMR = prod('l'), with prod() being the product operator.
        """
        # cache
        if self._UMR is not None:
            return self._UMR
        elif not self._N.shape == self._v.shape:
            return np.zeros(self.D)

        precision = np.finfo(float).precision - 2  # todo = 13
        atol = 10**-precision

        UMR = np.empty(self.D, float)
        for d in range(self.D):
            l = self._l[d].copy()
            v = self._v[d].copy()

            # if 1 in self._N[d] and self.uwr != "FTM":  # here, in TPU twice the combinations have to be tried
            #     # todo: test if valid
            #     l[self._N[d] == 1] /= 2
            #     v[self._N[d] == 1] *= 2

            if 0 in v:  # equivalently: np.inf in l
                l = l[v != 0]
                v = v[v != 0]

            if len(l) == 0 or len(v) == 0:
                UMR[d] = 1  # one since we can only control discrete pixels
                continue

            if np.all(l % 1 == 0):  # all l are integers
                UMR[d] = np.lcm.reduce(l.astype(int, copy=False))
            elif np.all(v % 1 == 0):  # all v are integers
                UMR[d] = self.Lext / np.gcd.reduce(v.astype(int, copy=False))
            elif np.allclose(l, np.rint(l), rtol=0, atol=atol):  # all l are integers within tolerance
                UMR[d] = np.lcm.reduce(np.rint(l).astype(int, copy=False))
            elif np.allclose(v, np.rint(v), rtol=0, atol=atol):  # all v are integers within tolerance
                UMR[d] = self.Lext / np.gcd.reduce(np.rint(v).astype(int, copy=False))
            else:
                # mutual divisibility test
                K = len(v)  # zeros may have been filtered out above
                for i in range(K - 1):
                    for j in range(i + 1, K):
                        if 0 <= l[i] % l[j] < atol or 1 - atol < l[i] % l[j] < 1:
                            l[j] = 1
                        elif 0 <= l[j] % l[i] < atol or 1 - atol < l[j] % l[i] < 1:
                            l[i] = 1
                mask = l != 1
                v = v[mask]
                l = l[mask]

                # number of decimals
                Dl = max(str(i)[::-1].find(".") for i in l)
                Dv = max(str(i)[::-1].find(".") for i in v)

                # estimate whether elements are rational or irrational
                if Dl < precision or Dv < precision:  # rational numbers without integers
                    if Dl <= Dv:
                        ls = l * 10**Dl  # wavelengths scaled
                        UMR[d] = np.lcm.reduce(ls.astype(int, copy=False)) / 10**Dl
                        logger.debug("Extended 'lcm' to rational numbers.")
                    else:
                        vs = v * 10**Dv  # spatial frequencies scaled
                        UMR[d] = self.Lext / (np.gcd.reduce(vs.astype(int, copy=False)) / 10**Dv)
                        logger.debug("Extended 'gcd' to rational numbers.")
                else:  # irrational numbers or rational numbers with more digits than "precision"
                    UMR[d] = np.prod(l)

        self._UMR = UMR
        logger.debug(f"self.UMR = {str(self._UMR)}")

        if self._ambiguous:
            logger.warning(
                "Unwrapping will not be spatially independent and only yield a relative phase map (UMR < L)."
            )

        return self._UMR

    @property
    def m(self) -> np.ndarray:
        """Moduli for Chinese remainder theorem."""
        # todo: """coprime moduli"""

        if self._m is not None:  # cache
            return self._m

        precision = np.finfo(float).precision - 2  # todo: 13
        atol = 10**-precision

        m = np.empty((self.D, self.K), int)  # moduli, i.e. integer ratios
        for d in range(self.D):
            # if all(l.is_integer() for l in self._l[d]):
            #     ...

            if np.all(self._l[d] % 1 == 0):  # all l are integers
                m_ = self._l[d].astype(int)
                gcd = np.gcd.reduce(m_)
                m[d] = m_ / gcd
                # lcm = np.lcm.reduce(m[d])
            elif np.all(self._v[d] % 1 == 0):  # all v are integers
                m_ = self._v[d].astype(int)
                lcm = np.lcm.reduce(m_)
                m[d] = lcm / m_  # changes succession of m_
            elif np.allclose(self._l[d], np.rint(self._l[d]), rtol=0, atol=atol):
                m_ = np.rint(self._l[d]).astype(int)
                gcd = np.gcd.reduce(m_)
                m[d] = m_ / gcd
                # lcm = np.lcm.reduce(m[d])
            elif np.allclose(self._v[d], np.rint(self._v[d]), rtol=0, atol=atol):
                m_ = self._v[d].astype(int)
                lcm = np.lcm.reduce(m_)
                m[d] = lcm / m_  # changes succession of m_
            else:  # extend lcm/gcd to non (ir)rational numbers
                l = self._l[d]
                v = self._v[d]

                # # mutual divisibility test
                # for i in range(self.K - 1):
                #     for j in range(i + 1, self.K):
                #         if 0 <= self._l[d, i] % self._l[d, j] < atol or 1 - atol < self._l[d, i] % self._l[d, j] < 1:
                #             l[j] = 1
                #         elif 0 <= self._l[d, j] % self._l[d, i] < atol or 1 - atol < self._l[d, j] % self._l[d, i] < 1:
                #             l[i] = 1
                # mask = l != 1  # todo: WAVG
                # l = l[mask]
                # v = v[mask]

                # number of decimals
                Dl = max(str(i)[::-1].find(".") for i in l)
                Dv = max(str(i)[::-1].find(".") for i in v)

                # estimate whether elements are rational or irrational
                if Dl < precision or Dv < precision:  # rational numbers without integers
                    if Dl <= Dv:
                        m_ = (l * 10**Dl).astype(int)  # wavelengths scaled
                        gcd = np.gcd.reduce(m_)
                        m[d] = m_ / gcd
                    else:
                        m_ = (v * 10**Dv).astype(int)  # spatial frequencies scaled
                        lcm = np.lcm.reduce(m_)
                        m[d] = lcm / m_
                else:  # irrational numbers or rational numbers with more digits than 'precision'
                    m[d] = 0
                    # lcm = np.prod(l)

        self._m = m
        logger.debug(f"self._m = {str(self._m)}")

        return m

    @property
    def gcd(self) -> np.ndarray:
        """Greatest common divisor of moduli.

        Must be smaller than noise."""
        gcd = self._l / self.m
        return np.mean(gcd, axis=1)  # each 'gcd' should be identical; we average to get more precision

    @property
    def crt(self) -> np.ndarray:
        """Coefficients for Chinese remainder theorem."""

        if self._crt is not None:  # cache
            return self._crt

        crt = np.empty((self.D, self.K), int)  # Solution of multiple congruences.
        for d in range(self.D):
            lcm = np.lcm.reduce(self.m[d])

            if lcm > 0:
                for i in range(self.K):
                    # try:
                    #     M = int(lcm / self.m[d, i])
                    # except ZeroDivisionError:
                    #     crt[d, i] = 0  # no solution
                    #     continue
                    M = int(lcm / self.m[d, i])

                    # if np.any(crt[d, :i] % self.m[d, i] == 1):  # one element already equals 1 modulo m[i]
                    #     crt[d, i] = -1  # 0  # todo: WAVG them?
                    #     continue

                    try:
                        # M_ = pow(M, -1, self.m[d, i])  # only for Python 3.8+
                        M_ = sympy.core.numbers.mod_inverse(M, self.m[d, i])
                        crt[d, i] = M * M_
                    except ValueError:
                        crt[d, i] = 0  # no solution found

                    if np.any(crt[d, :i] % self.m[d, i] == 1):  # one element already equals 1 modulo m[i]
                        crt[d, i] *= -1  # todo: WAVG them?
                        continue
            else:
                crt[d, :] = 0  # no solution found

            # coprimality test
            j0 = -1
            for i in range(self.K):
                for j in range(i + 1, self.K):
                    if np.gcd(self.m[d, i], self.m[d, j]) != 1:
                        j0 = j
            if j0 > 0:
                crt[d, j0] = 0

        self._crt = crt
        logger.debug(f"self._crt = {str(self._crt)}")

        return self._crt

    # @property
    # def ui(self) -> float:
    #     """Intensity noise."""
    #     return self._ui
    #
    # @ui.setter
    # def ui(self, ui):
    #     _ui = float(max(0, ui))
    #
    #     if self._ui != _ui:
    #         self._ui = _ui
    #         logger.debug(f"{self._ui=}")
    #
    # @property
    # def up(self) -> float:
    #     """Phase uncertainties.
    #     [upi] = rad"""
    #     b = self.B * self.mtf(self._v)
    #     SNR = b / self.ui
    #     upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR
    #     return upi
    #
    # @property
    # def ux(self) -> np.ndarray:
    #     """Positional uncertainties."""
    #     return self.up / _2PI * self._l  # local positional uncertainties
    #
    # @property
    # def u(self) -> np.ndarray:
    #     """Measurement uncertainty.
    #     [u] = px.
    #
    #     It is based on the phase noise model
    #     and propagated through the unwrapping process and the phase fusion.
    #     """
    #     return np.sqrt(1 / np.sum(1 / self.ux**2, axis=1))  # global positional uncertainty
    #
    # @property
    # def SNR(self) -> np.ndarray:
    #     """Signal-to-noise ratio of the phase shift coding.
    #
    #     It is a measure of how many points can be distinguished within the screen length [0, L).
    #     """
    #     return self.L / self.u
    #
    # @property
    # def SNRdB(self) -> np.ndarray:
    #     """Signal-to-noise ratio.
    #     [SNRdB] = dB."""
    #     return 20 * np.log10(self.SNR)
    #
    # @property
    # def DR(self) -> np.ndarray:
    #     """Dynamic range of the phase shift coding.
    #
    #     It is a measure of how many points can be distinguished within the unambiguous measurement range [0, UMR).
    #     """
    #     return self.UMR / self.u
    #
    # @property
    # def DRdB(self) -> np.ndarray:
    #     """Dynamic range. [DRdB] = dB."""
    #     return 20 * np.log10(self.DR)

    @property
    def eta(self) -> float:
        """Spatial coding efficiency."""
        eta = self.L / self.UMR
        eta[self.UMR < self.L] = 0
        return eta

    @property
    def _params(self) -> dict:  # todo: get_params
        """Base parameters required for en- & decoding fringe patterns.

        This contains all property objects of the class which have a setter method,
        i.e. are (usually) not derived from others.
        """
        params = {"__version__": version(__package__)}
        for k in sorted(self._setters):
            # for k in sorted(dir(self)):  # sorted() ensures setting params in the right order in __init__()
            #     if not k.startswith("_"):  # avoid e.g. '_params'
            #         if isinstance(getattr(type(self), k, None), property) and getattr(type(self), k, None).fset is not None:
            v = getattr(self, k, None)
            if isinstance(v, np.ndarray):
                params[k] = v.tolist()
            elif isinstance(v, np.dtype):
                params[k] = str(v)
            else:
                params[k] = v

        return params

    @_params.setter
    def _params(self, params: dict):  # todo: set_params
        params_old = self._params.copy()  # copy current params; they become old params

        # set params
        for k in self._setters:  # iterate over `_setters`, which are sorted!
            if k in params:
                setattr(self, k, params[k])

        # check params
        for k in self._setters:  # iterate over `_setters`, which are sorted!
            if k in params:
                if k in "N l v f".split() and getattr(self, k).ndim != np.array(params[k]).ndim:
                    if not np.array_equal(getattr(self, k), params[k][0]):
                        break
                else:
                    if not np.array_equal(getattr(self, k), params[k]):
                        break
        else:  # else clause executes only after the loop completes normally, i.e. did not encounter a break
            return

        logger.warning(
            f"Parameter '{k}' got overwritten by interdependencies. Choose consistent initialization values."
        )
        self._params = params_old  # restore old params

    # note: class attributes are not available in comprehensions
    _glossary: dict = {
        k: v.__doc__.splitlines()[0] for k, v in sorted(vars().items()) if k[0] != "_" and isinstance(v, property)
    }  # vars() is a workaround for class attributes, which are not accessible in comprehensions
    _setters: tuple = sorted(
        tuple(
            k for k, v in vars().items() if k[0] != "_" and isinstance(v, property) and v.fset is not None
        )  # attention: exclude `_params`!
    )  # note: sorted() ensures setting params in the same/right order, e.g. in __init__() and _params.setter  # todo: does order matter?
    _help: dict = {
        k: v.__doc__.splitlines()[0]
        for k, v in vars().items()
        if k[0] != "_" and isinstance(v, property) and v.fset is not None
    }
    _defaults: dict = __init__.__kwdefaults__
    # _types: dict = __init__.__annotations__
    _types: dict = {k: type(v) for k, v in _defaults.items()}
    _types_str: dict = {k: v.__name__ for k, v in _types.items()}

    # restrict instance attributes to the ones listed here
    # (commend the next line out or add '__dict__' to prevent this)
    __slots__ = tuple("_" + k for k in _defaults.keys()) + (
        "_mtf",
        "_UMR",
        "_crt",
        "_m",
        "_t",
    )

    # automated continuation of the class docstring following the NumPy style guide:
    # https://numpydoc.readthedocs.io/en/latest/format.html#documenting-classes
    # https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
    # __doc__ += "\n"
    __doc__ += "\nParameters\n----------"  # implemented as properties
    __doc__ += "\n*args : iterable\n    Non-keyword arguments are explicitly discarded."
    __doc__ += "\n    Only keyword arguments are considered."
    for __k, __v in _defaults.items():
        # do this in for loop instead of a comprehension,
        # because for the latter, names in class scope are not accessible
        __doc__ += f"\n{__k} : {'array_like' if __k in "N l v f h".split() else _types_str[__k]}, default = {__v}"
        __doc__ += f"\n    {_help[__k]}"
    # __doc__ += "\n\nAttributes\n----------"  # implemented as properties and derived from Parameters
    # for __k, __v in sorted(vars().items()):
    #     # do this in for loop instead of a comprehension,
    #     # because for the latter, names in class scope are not accessible
    #     if __k not in _defaults and __k in _glossary:
    #         # __doc__ += f"\n{__k} : {'array_like' if __k in "N l v f h".split() else _types_str[__k]}\n    {_glossary[__k]}"
    #         __doc__ += f"\n{__k}\n    {_glossary[__k]}"
    __doc__ += "\n"
    __doc__ += "\nMethods\n-------"
    __doc__ += f"\nencode\n    {encode.__doc__.splitlines()[0]}"
    __doc__ += "\n    See `encode`."
    __doc__ += f"\ndecode\n    {decode.__doc__.splitlines()[0]}"
    __doc__ += "\n    See `decode`."
    # todo: add all public methods? which should be public?
    # for __k, __v in sorted(vars().items()):
    #     # do this in for loop instead of a comprehension,
    #     # because for the latter, names in class scope are not accessible
    #     if __k not __k.startswith("_") and callable(__v):
    #         __doc__ += f"\n{__k}\n : {__v.__doc__.splitlines()[0]}"

    del __k, __v
