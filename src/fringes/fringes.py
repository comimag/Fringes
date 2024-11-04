"""This module contains the handy `Fringes`class.

It handles all the required parameters
for configuring fringe pattern sequences
and provides methods for creating and analyzing them.
"""

from collections import namedtuple
import itertools as it
import json
import logging
import os
import time

import cv2  # todo: make optional dependency for unwrapping, darkfield, brightfield
import numpy as np
import scipy as sp
import skimage as ski  # todo: make optional dependency for unwrapping
import sympy
import toml
import yaml

from fringes import grid
from fringes.decoder import decode
from fringes.util import _version, vshape, bilateral, _remap

logger = logging.getLogger(__name__)


class Fringes:
    """Easy-to-use class to configure, encode and decode customized fringe patterns using phase shifting algorithms."""

    # note: the class docstring is continued at the end of the class

    # value limits
    _Hmax = 101  # this is arbitrary
    _Dmax = 2
    _Kmax = 101  # this is arbitrary, but must be < 128 when deploying spatial or frequency multiplexing @ uint8
    _Nmax = 1001  # this is arbitrary; more is better but the improvement scales with sqrt(N); @FDM: > 2 * _fmax + 1
    _Mmax = 101  # this is arbitrary; more is better but the improvement scales with sqrt(M)
    # _Pmax: int = 35651584  # ~8K i.e. max luma picture size of h264, h265, h266 video codecs as of 2022
    _Pmax = 2**30  # 2^30 = 1,073,741,824 i.e. default size   limit of imread() in OpenCV
    _Xmax = 2**20  # 2^20 = 1,048,576     i.e. default width  limit of imread() in OpenCV
    _Ymax = 2**20  # 2^20 = 1,048,576     i.e. default height limit of imread() in OpenCV
    _Lmax = 2**20  # 2^20 = 1,048,576     i.e. default height limit of imread() in OpenCV
    _Tmax = _Hmax * _Dmax * _Kmax * _Nmax  # 20.422.402
    _alphamax = 2
    _gammamax = 3  # most screens have a gamma of ~2.2
    # _lminmin = 2  # l == 2 only if p0 != pi / 2 + 2pi*k, best if p0 == pi + 2pi*k with k being a positive integer
    #            also l <= 2 yields errors in SPU: phase jumps = 2PI / lmin >= np.pi
    _lminmin = 3  # l >= 3 yields sufficient modulation theoretically
    # _lminmin = 8  # l >= 8 yields sufficient modulation practically [Liu2014]

    # allowed values; take care to only use immutable types!  # todo: is that so?
    values = {
        "grid": ("image"),  # todo: ("image", "Cartesian", "polar", "log-polar")
        "indexing": ("xy", "ij"),
        "dtype": (
            # "bool",  # results are too unprecise
            "uint8",
            "uint16",
            # "uint32",  # integer overflow in pyqtgraph -> replace line 528 of ImageItem.py with:
            # "uint64",  #  bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype="uint64")
            # "float16",  # numba doesn't handle float16, also most algorithms convert float16 to float32 anyway
            "float32",
            "float64",
        ),
        "mode": ("fast", "precise"),
    }

    # allowed values; take care to only use immutable types!
    _grids = (("image"),)  # todo: ("image", "Cartesian", "polar", "log-polar")
    _indexings = ("xy", "ij")
    _dtypes = (
        # "bool",  # results are too imprecise
        "uint8",
        "uint16",
        # "uint32",  # integer overflow in pyqtgraph -> replace line 528 of ImageItem.py with:
        # "uint64",  # bins = self._xp.arange(mn, mx + 1.01 * step, step, dtype="uint64")
        # "float16",  # numba doesn't handle float16, also most algorithms convert float16 to float32 anyway
        "float32",
        "float64",
    )
    _modes = ("fast", "precise", "MRD")

    _loader = {
        ".json": json.load,
        ".yaml": yaml.safe_load,
        ".toml": toml.load,
    }

    _verbose_output = (
        "brightness",
        "modulation",
        "registration",
        "phase",
        "order",
        "residuals",
        "uncertainty",
        "exposure",
        "visibility",
    )

    # default values are defined here; take care to only use immutable types!
    def __init__(
        self,
        *args,  # bundles all args, what follows are only kwargs
        Y: int = 1200,
        X: int = 1920,
        H: int = 1,  # inferred from h
        M: float = 1.0,  # inferred from h
        D: int = 2,
        K: int = 3,
        T: int = 24,  # T is inferred
        N: tuple | list | np.ndarray = np.array([[4, 4, 4], [4, 4, 4]], int),
        l: tuple | list | np.ndarray = 1920 / np.array([[13, 7, 89], [13, 7, 89]], float),  # inferred from v
        v: tuple | list | np.ndarray = np.array([[13, 7, 89], [13, 7, 89]], float),
        f: tuple | list | np.ndarray = np.array([[1, 1, 1], [1, 1, 1]], float),
        h: tuple | list | np.ndarray = np.array([[255, 255, 255]], int),
        p0: float = np.pi,
        gamma: float = 1.0,
        A: float = 255 / 2,  # i.e. Imax / 2 @ uint8; inferred from Imax and E
        B: float = 255 / 2,  # i.e. Imax / 2 @ uint8; inferred from Imax and E and V
        E: float = 0.5,
        V: float = 1.0,
        alpha: float = 1.0,
        dtype: str | np.dtype = "uint8",
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
        ui: float = float(
            np.sqrt(1 / 12)
        ),  # quantization noise  # np.sqrt(0.038 ** 2 * 13.7 ** 2 + 1 / 12 + 0.038 * 120),  # ~2.22
        PSF: float = 0.0,
        mode: str = "fast",
        #  **kwargs,  # bundles all undefined kwargs, else error: __init__() got an unexpected keyword argument
    ):
        # given values which are in defaults but are not identical to them
        given = {
            k: v
            for k, v in sorted(locals().items())
            if k in self._defaults and not np.array_equal(v, self._defaults[k])
        }  # sorted() ensures setting _params in the right order

        # set default values
        self._mtf = None
        self._UMR = None  # empty cache
        self._MM_ = None  # empty cache
        for k, v in self._defaults.items():
            if k not in "HMTlAB":  # these properties are inferred from others
                setattr(self, f"_{k}", v)  # define private variables from which the properties get their values from

        # set given values
        self._params = given

        # precompute
        self.UMR  # compute and cache; logs warning if necessary
        self.MM_  # compute and cache

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.encode(*args, **kwargs)

    def __getitem__(self, frames: int | tuple | slice) -> np.ndarray:
        if isinstance(frames, slice):
            frames = np.arange(self.T)[frames]

        return self.encode(frames=frames)

    def __iter__(self):
        self._t = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._t < self.T:
            I = self.encode(frames=self._t)
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
        return self.__name__

    def __repr__(self) -> str:
        return f"{self._params}"

    def load(self, fname: str | None = None) -> None:
        """Load parameters from a config file to the `Fringes` instance.

        .. warning:: The parameters are only loaded if the config file provides the section `fringes`.

        Parametersf
        ----------
        fname : str, optional
            File name of the file to load.
            Supported file formats are: *.json, *.yaml, *.toml.
            If `fname` is not provided, the file `.fringes.yaml` within the user home directory is loaded.

        Examples
        --------
        >>> import os
        >>> fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> f.N += 1
        >>> f.save(fname)

        >>> f2 = frng.Fringes()
        >>> f2.load(fname)

        >>> f2 == f
        True
        """
        if fname is None:
            fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        if not os.path.isfile(fname):
            logger.error(f"File '{fname}' does not exist.")
            return

        with open(fname, "r") as f:
            ext = os.path.splitext(fname)[-1]

            if ext == ".json":
                p = json.load(f)
            elif ext == ".yaml":
                p = yaml.safe_load(f)
            elif ext == ".toml":
                p = toml.load(f)
            else:
                logger.error(f"Unknown file type '{ext}'.")
                return

        if "fringes" in p:
            params = p["fringes"]
            self._params = params

            logger.info(f"Loaded parameters from '{fname}'.")
        else:
            logger.error(f"No 'fringes' section in file '{fname}'.")

    def save(self, fname: str | None = None) -> None:
        """Save the parameters of the `Fringes` instance to a config file.

        Within the file, the parameters are written to the section `fringes`.

        Parameters
        ----------
        fname : str, optional
            File name of the file to save.
            Supported file formats are: *.json, *.yaml, *.toml.
            If `fname` is not provided, the parameters are saved to
            the file `.fringes.yaml` within the user home directory.

        Examples
        --------
        >>> import os
        >>> fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        >>> import fringes as frng
        >>> f = frng.Fringes()

        >>> f.save(fname)
        """
        if fname is None:
            fname = os.path.join(os.path.expanduser("~"), ".fringes.yaml")

        if not os.path.isdir(os.path.dirname(fname)):
            logger.warning(f"File directory does not exist.")
            return

        name, ext = os.path.splitext(fname)
        if not ext:
            name, ext = ext, name

        if ext not in self._loader.keys():
            logger.warning(f"File extension is unknown. Must be one of {self._loaders.keys()}")
            return

        with open(fname, "w") as f:
            if ext == ".json":
                json.dump({"fringes": self._params}, f, indent=4)
            elif ext == ".yaml":
                yaml.dump({"fringes": self._params}, f)
            elif ext == ".toml":
                toml.dump({"fringes": self._params}, f)

        logger.debug(f"Saved parameters to {fname}.")  # todo: info?

    def reset(self) -> None:
        """Reset parameters of the `Fringes` instance to default values."""
        self._params = self._defaults
        logger.info("Reset parameters to defaults.")

    def optimize(self, T: int | None = None, umax: float | None = None) -> None:
        """Optimize the parameters of the `Fringes` instance.

        Parameters
        ----------
         T : int, optional
            Number of frames.
            If `T` is not provided, the number of frames from the `Fringes` instance is used.
            Then, the `Fringes` instance's number of shifts `N` is distributed optimally over the directions and sets.

         umax : float, optional
            Maximum allowable uncertainty.
            Must be greater than zero.
            Standard deviation of maximum uncertainty for measurement to be valid.
            [umax] = px.

        Notes
        -----
        todo: update

        If `umax` is specified, the parameters are determined
        that allow a maximal uncertainty of `umax`
        with a minimum number of frames.

        Else, the parameters of the `Fringes` instance are optimized to yield the minimal uncertainty
        using the given number of frames `T`.
        """
        # todo: compute based on given ui -> upi -> ux

        if T is not None:
            ...
        elif umax is not None:
            ...
        else:
            self.T = self.T
            ...

        K = np.log(self.L) / np.log(self.L / self.vopt)  # lopt ** K = L
        K = np.ceil(max(2, K))
        self.K = K
        self.v = "optimal"

        if umax is not None:  # umax -> T
            self.N = int(np.median(self.N))  # make N const.
            a = self.u.max() / umax
            N = self.N * a**2
            self.N = np.maximum(3, np.ceil(N))

            if self.u > umax:
                ...  # todo: check if umax is reached
        else:  # T -> u
            if T is None:
                T = self.T

            self.T = T  # distribute frames optimally (evenly) on shifts

        logger.info("Optimized parameters.")

    def _deinterlace(self, I: np.ndarray) -> np.ndarray:
        """Deinterlace fringe patterns.

        This for fringe patterns
        which were recorded with a line scan camera
        while each frame has been displayed and captured
        while the object was moving by one pixel.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        Returns
        -------
        I : np.ndarray
            Deinterlaced fringe pattern sequence.

        Raises
        ------
        AssertionError
            If the number of frames of `I` and the attribute `T` of the `Fringes` instance don't match.

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> I = f.encode()
        >>> I_rec = I.swapaxes(0, 1).reshape(-1, f.T, f.X, f.C)  # interlace; this is how a line camera would record  # todo: replace this line with recording data as in '../examples/record.py'
        >>> I_rec = f._deinterlace(I_rec)
        """
        t0 = time.perf_counter()

        T, Y, X, C = vshape(I).shape
        assert T * Y % self.T == 0, "Number of frames of parameters and data don't match."

        # I = I.reshape((T * Y, X, C))  # concatenate
        I = I.reshape((-1, self.T, X, C)).swapaxes(0, 1)  # returns a view

        logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

        return I

    def coordinates(self) -> np.ndarray:
        """Generate the coordinate matrices of the coordinate system defined in `grid`.

        Returns
        -------
        x : np.ndarray
            Coordinate matrices.
        """
        t0 = time.perf_counter()

        sys = (
            "img"
            if self.grid == "image"
            else "cart" if self.grid == "Cartesian" else "pol" if self.grid == "polar" else "logpol"
        )

        # x = np.array(getattr(grid, sys)(self.Y, self.X, self.angle))
        x = np.array(getattr(grid, sys)(self.Y, self.X, 0))  # todo: spiral angle 45

        if self.indexing == "ij":
            x = x[::-1]  # returns a view

        if self.D == 1:
            x = x[self.axis][None, :, :]  # returns a view

        if self.grid in ["polar", "log-polar"]:
            x *= self.L

        logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

        return x

    @property
    def x(self) -> np.ndarray:
        """Coordinate matrices of the coordinate system defined in `grid`."""

        if self.grid == "image":  # and self.angle == 0:
            B = int(np.ceil(np.log2(self.R.max() - 1)))  # next power of two
            B += -B % 8  # next power of two divisible by 8

            x = np.indices((self.Y, self.X), dtype=f"uint{B}", sparse=True)

            if self.indexing == "xy":
                x = x[::-1]

            if self.D == 1:
                x = x[self.axis][None, ...]
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

            if self.D == 1:
                x = x[self.axis][None, :, :]  # returns a view

            if self.grid in ["polar", "log-polar"]:
                x *= self.L

        return x

    def _modulate(self, frames: np.ndarray) -> np.ndarray:
        """Encode base fringe patterns by spatio-temporal modulation.

        Parameters
        ----------
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
        x = self.x

        I = np.empty((T, self.Y, self.X), dtype)

        # Ncum = np.cumsum(self._N).reshape(self.D, self.K)
        # for t in frames:
        #     d, i = np.argwhere(t < Ncum)[0]
        #     n = t - Ncum[d, i] + self._N[0, 0]
        #     ...

        idx = 0
        frame = 0
        for d in range(self.D):
            x_ = (x[d] + self.x0[d]) / self.L

            for i in range(self.K):
                k = 2 * np.pi * self._v[d, i]
                w = 2 * np.pi * self._f[d, i]

                if self.mode.lower() == "mrd":
                    k *= self.L / self.R[d]

                if self.reverse:
                    w *= -1

                for n in range(self._N[d, i]):
                    if frame in frames:
                        t = n / 4 if self._N[d, i] == 2 else n / self._N[d, i]

                        val = self.Imax * (self.E * (1 + self.V * np.cos(k * x_ - w * t - self.p0))) ** self.gamma

                        if dtype.kind in "ui":  # and rint:
                            np.rint(val, out=val)
                        # elif dtype.kind in "b":
                        #     val = val >= 0.5

                        I[idx] = val

                        idx += 1
                    frame += 1

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return I.reshape(-1, self.Y, self.X, 1)

    def _demodulate(
        self, I: np.ndarray, Vmin: float = 0, verbose: bool | str = False, func: str = "ski"
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Decode base fringe patterns by spatio-temporal demodulation.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        Vmin : float
            Minimum visibility for measurement to be valid
            (can accelerate decoding).

        verbose : bool, optional
            If this or the argument `verbose` of the Fringes instance is set to True,
            intermediate and verbose results are returned as well.

        func : str, optional
            Unwrapping function to use. The default is 'ski'.

            - 'ski': `scikit-image <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_

            - else: `OpenCV https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_

        Returns
        -------
        brightness : np.ndarray
            Local background signal.

        modulation : np.ndarray
            Local amplitude of the cosine signal.

        phase : np.ndarray, optional
            Local phase.

        registration : np.ndarray
            Decoded coordinates.

            .. note:: The registration is a mapping in the same pixel grid as the camera sensor
              and contains the information where each camera pixel, i.e. each camera sightray,
              was looking at during the fringe pattern recording.

        residuals : np.ndarray, optional
            Residuals from the optimization-based unwrapping process.
        """
        t0 = time.perf_counter()

        # parse
        T, Y, X, C = vshape(I).shape  # extract Y, X, C from data as these parameters depend on used camera
        I = I.reshape((T, Y, X, C))

        # if self.FDM:
        #    c = np.fft.rfft(I, axis=0) / T  # todo: hfft
        #
        #    # if np.any(self._N > 2 * self.D * self.K):  # 2 * np.abs(_f).max() + 1:
        #    #     i = np.append(np.zeros(1, int), self._f.flatten().astype(int, copy=False))  # add p0
        #    #     c = c[i]
        #
        #    a = abs(c)
        #    # i = np.argsort(a[1:], axis=0)[::-1]  # indices of frequencies, sorted by their magnitude
        #    phi = -np.angle(c * np.exp(-1j * (self.p0 - np.pi)))[_f.flatten().astype(int, copy=False)]  # todo: why p0 - PI???

        if self.uwr == "FTM":
            # todo: make passband symmetrical around carrier frequency?
            if self.D == 2:
                fx = np.fft.fftshift(np.fft.fftfreq(X))  # todo: hfft
                fy = np.fft.fftshift(np.fft.fftfreq(Y))
                fxx, fyy = np.meshgrid(fx, fy)
                mx = np.abs(fxx) > np.abs(
                    fyy
                )  # mask for x-frequencies  # todo: make left and right borders round (symmetrical around base band)
                my = np.abs(fxx) < np.abs(
                    fyy
                )  # mask for y-frequencies  # todo: make lower and upper borders round (symmetrical around base band)

                W = 100  # assume window width for filtering out baseband
                W = min(max(3, W), min(X, Y) / 20)  # clip to ensure plausible value
                a = int(min(max(0, W), X / 4) + 0.5)  # todo: find good upper cut off frequency
                # a = X // 4
                mx[:, :a] = 0  # remove high frequencies
                b = int(X / 2 - W / 2 + 0.5)
                mx[:, b:] = 0  # remove baseband and positive frequencies

                H = 100  # assume window height for filtering out baseband
                H = min(max(3, H), min(X, Y) / 20)  # clip to ensure plausible value
                c = int(min(max(0, H), Y / 4) + 0.5)  # todo: find good upper cut off frequency
                # c = Y // 4
                my[:c, :] = 0  # remove high frequencies
                d = int(Y / 2 - H / 2 + 0.5)
                my[d:, :] = 0  # remove baseband and positive frequencies

                # todo: smooth edges of filter masks, i.e. make them Hann Windows

                if C > 1:
                    I = I[..., 0]
                    C = 1
                    I = I.reshape((T, Y, X, C))

                phi = np.empty([self.D, Y, X, C], np.float32)
                res = np.empty([self.D, Y, X, C], np.float32)
                # if verbose:
                #     phi = np.empty([self.D, Y, X, C], np.float32)
                #     res = np.empty([self.D, Y, X, C], np.float32)
                reg = np.empty([self.D, Y, X, C], np.float32)
                bri = np.empty([self.D, Y, X, C], np.float32)
                mod = np.empty([self.D, Y, X, C], np.float32)
                fid = np.full([self.D, Y, X, C], np.nan, np.float32)
                for c in range(C):
                    # todo: hfft
                    I_FFT = np.fft.fftshift(np.fft.fft2(I[0, ..., c]))

                    I_FFT_x = I_FFT * mx
                    ixy, ixx = np.unravel_index(I_FFT_x.argmax(), I_FFT_x.shape)  # get indices of carrier frequency
                    I_FFT_x = np.roll(I_FFT_x, X // 2 - ixx, 1)  # move to center

                    I_FFT_y = I_FFT * my
                    iyy, iyx = np.unravel_index(I_FFT_y.argmax(), I_FFT_y.shape)  # get indices of carrier frequency
                    I_FFT_y = np.roll(I_FFT_y, Y // 2 - iyy, 0)  # move to center

                    Jx = np.fft.ifft2(np.fft.ifftshift(I_FFT_x))
                    Jy = np.fft.ifft2(np.fft.ifftshift(I_FFT_y))

                    reg[0, ..., c] = np.angle(Jx)
                    reg[1, ..., c] = np.angle(Jy)
                    # todo: bri: local average of I ?!
                    mod[0, ..., c] = np.abs(Jx) * 2  # factor 2 because one sideband is filtered out
                    mod[1, ..., c] = np.abs(Jy) * 2  # factor 2 because one sideband is filtered out
                    if verbose:
                        phi[0, ..., c] = reg[0, ..., c]
                        phi[1, ..., c] = reg[1, ..., c]
                        res[0, ..., c] = np.log(np.abs(I_FFT))  # J  # todo: hfft
                        res[1, ..., c] = np.log(np.abs(I_FFT))  # J
                        # todo: I - J
            elif self.D == 1:
                L = max(X, Y)
                fx = np.fft.fftshift(np.fft.fftfreq(X)) * X * Y / L  # todo: hfft
                fy = np.fft.fftshift(np.fft.fftfreq(Y)) * Y * X / L
                fxx, fyy = np.meshgrid(fx, fy)
                frr = np.sqrt(fxx**2 + fyy**2)  # todo: normalization of both directions

                mr = frr <= L / 2  # ensure same sampling in all directions
                W = 10
                W = min(max(1, W / 2), L / 20)
                mr[frr < W] = 0  # remove baseband
                mr[frr > L / 4] = 0  # remove too high frequencies

                mh = np.empty([Y, X])
                mh[:, : X // 2] = 1
                mh[:, X // 2 :] = 0

                if C > 1:
                    I = I[..., 0]
                    C = 1
                    I = I.reshape((T, Y, X, C))

                phi = np.empty([self.D, Y, X, C], np.float32)
                res = np.empty([self.D, Y, X, C], np.float32)
                fid = np.full([self.D, Y, X, C], np.nan, np.float32)
                reg = np.empty([self.D, Y, X, C], np.float32)
                bri = np.empty([self.D, Y, X, C], np.float32)
                mod = np.empty([self.D, Y, X, C], np.float32)
                for c in range(C):
                    # todo: hfft
                    I_FFT = np.fft.fftshift(np.fft.fft2(I[0, ..., c]))

                    I_FFT_r = I_FFT * mr
                    iy, ix = np.unravel_index(I_FFT_r.nanargmax(), I_FFT_r.shape)  # get indices of carrier frequency
                    y, x = Y / 2 - iy, X / 2 - ix
                    a = np.degrees(np.arctan2(y, x))
                    mhr = sp.ndimage.rotate(mh, a, reshape=False, order=0, mode="nearest")

                    I_FFT_r *= mhr  # remove one sideband
                    I_FFT_r = np.roll(I_FFT_r, X // 2 - ix, 1)  # move to center
                    I_FFT_r = np.roll(I_FFT_r, Y // 2 - iy, 0)  # move to center

                    J = np.fft.ifft2(np.fft.ifftshift(I_FFT_r))

                    reg[0, ..., c] = np.angle(J)
                    # todo: bri
                    mod[0, ..., c] = np.abs(J) * 2  # factor 2 because one sideband is filtered out
                    if verbose:
                        phi[0, ..., c] = reg[0, ..., c]
                        res[0, ..., c] = np.log(np.abs(I_FFT))  # J
                        # todo: I - J
        else:
            bri, mod, phi, reg, res = decode(
                I,
                self._N,
                self._v * self.L / self.R[:, None] if self.mode.lower() == "mrd" else self._v,
                -self._f if self.reverse else self._f,
                self.R,
                self.UMR,
                self.MM_,
                self.x0,
                self.p0,
                Vmin,
                verbose,
            )

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return bri, mod, phi, reg, res

    def _multiplex(self, I: np.ndarray) -> np.ndarray:
        """Multiplex fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Base fringe patterns.
        Returns
        -------
        I : np.ndarray
            Multiplexed fringe patterns.
        """
        t0 = time.perf_counter()

        if self.WDM:
            assert not self.FDM
            assert self._monochrome
            assert np.all(self.N == 3)

            I = I.reshape((-1, 3, self.Y, self.X, 1))  # returns a view
            I = I.swapaxes(1, -1)  # returns a view
            I = I.reshape((-1, self.Y, self.X, self.C))  # returns a view

        if self.SDM:
            assert not self.FDM
            assert self.grid in self._grids[:2]
            assert self.D == 2
            assert I.dtype.kind == "f"

            I = I.reshape((self.D, -1))  # returns a view
            I -= self.A
            I = np.sum(I, axis=0)
            I += self.A
            # I *= 1 / self.D
            I = I.reshape((-1, self.Y, self.X, self.C if self.WDM else 1))  # returns a view
            if self.dtype.kind in "uib":
                I = I.astype(self.dtype, copy=False)  # returns a view

        if self.FDM:
            assert not self.WDM
            assert not self.SDM
            assert len(np.unique(self.N)) == 1
            assert I.dtype.kind == "f"

            if np.any(self._N < 2 * np.abs(self._f).max() + 1):  # todo: fractional periods
                logger.warning("Decoding might be disturbed.")

            I = I.reshape((self.D * self.K, -1))  # returns a view
            I -= self.A
            I = np.sum(I, axis=0)
            I += self.A
            # I *= 1 / (self.D * self.K)
            I = I.reshape((-1, self.Y, self.X, 1))  # returns a view
            if self.dtype.kind in "uib":
                I = I.astype(self.dtype, copy=False)  # returns a view

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return I

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

        T, Y, X, C = vshape(I).shape

        if self.SDM:
            assert not self.FDM
            assert self.grid in self._grids[:2]
            assert self.D == 2

            if X % 2 == 0:
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

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return I

    def _colorize(self, I: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """Colorize fringe patterns.

        Parameters
        ----------
        I : np.ndarray
            Base fringe patterns,
            possibly multiplexed.
        frames : None or int or tuple of ints, optional
            Indices of the frames to be encoded.
            The default, frames=None, will encode all frames at once.
            If frames is negative, it counts from the last to the first frame.
            If frames contains numbers whose magnitude is larger than the total number of frames
            (as specified by the attribute `T` of the Fringes instance), it is wrapped around.
            If frames is a tuple of ints, only the frames specified in the tuple are encoded.

        Returns
        -------
        I : np.ndarray
            Colorized fringe patterns.
        """
        t0 = time.perf_counter()

        T_ = I.shape[0]  # number of frames for each hue
        T = len(frames)
        J = np.empty((T, self.Y, self.X, self.C), self.dtype)

        for t in frames:
            tb = t % I.shape[0]  # indices from base fringe pattern I

            for c in range(self.C):
                h = int(t // T_)  # hue index
                cb = c if self.WDM else 0  # color index of base fringe pattern I

                if self.h[h, c] == 0:  # uibf -> uibf
                    J[t, ..., c] = 0
                elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uibf -> uibf
                    J[t, ..., c] = I[tb, ..., cb]
                elif self.dtype.kind in "uib":  # f -> uib
                    J[t, ..., c] = np.rint(I[tb, ..., cb] * (self.h[h, c] / 255))  # .astype(self.dtype, copy=False)
                else:  # f -> f
                    J[t, ..., c] = I[tb, ..., cb] * (self.h[h, c] / 255)

        # for h in range(self.H):
        #     if frames is None:
        #         for c in range(self.C):
        #             cj = c if self.WDM else 0  # todo: ???
        #             if self.h[h, c] == 0:  # uib -> uib, f -> f
        #                 J[h * Th : (h + 1) * Th, ..., c] = 0
        #             elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uib -> uib, f -> f
        #                 J[h * Th : (h + 1) * Th, ..., c] = I[..., cj]
        #             elif self.dtype.kind in "uib":  # f -> uib
        #                 J[h * Th : (h + 1) * Th, ..., c] = np.rint(I[..., cj] * (self.h[h, c] / 255)).astype(
        #                     self.dtype, copy=False
        #                 )
        #             else:  # f -> f
        #                 J[h * Th : (h + 1) * Th, ..., c] = I[..., cj] * (self.h[h, c] / 255)
        #     elif h in hues:  # i.e. frames is not None and h in hues
        #         for c in range(self.C):
        #             cj = c if self.WDM else 0  # todo: ???
        #             if self.h[h, c] == 0:  # uib -> uib, f -> f
        #                 J[i, ..., c] = 0
        #             elif self.h[h, c] == 255 and J.dtype == self.dtype:  # uib -> uib, f -> f
        #                 J[i, ..., c] = I[i, ..., cj]
        #             elif self.dtype.kind in "uib":  # f -> uib
        #                 J[i, ..., c] = np.rint(I[i, ..., cj] * (self.h[h, c] / 255)).astype(self.dtype, copy=False)
        #             else:  # f -> f
        #                 J[i, ..., c] = I[i, ..., cj] * (self.h[h, c] / 255)
        #         i += 1

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return J

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

        T, Y, X, C = vshape(I).shape
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

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return I

    def encode(
        self,
        frames: int | tuple | list | np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode fringe patterns.

        Parameters
        ----------
        frames : array_like, optional
            Indices of the frames to be encoded.
            The default, frames=None, will encode all frames at once.
            If frames is negative, it counts from the last to the first frame.
            If frames contains numbers whose magnitude is larger than the total number of frames
            (as specified by the attribute `T` of the Fringes instance), it is wrapped around.
            If frames is a tuple of ints, only the frames specified in the tuple are encoded.
            If indices occur more than once, only the first occurence is encoded.

        Returns
        -------
        I : np.ndarray
            Fringe pattern sequence.

        Notes
        -----
        To receive the frames iteratively (i.e. in a lazy manner),
        simply iterate over the Fringes instance.
        Alternatively, to receive arbitrary frames,
        index the Fringes instance directly,
        either with an integer, a tuple or a slice.

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()

        Encode the complete fringe pattern sequence.

        >>> I = f.encode()

        Encode the first frame of the fringe pattern sequence.

        >>> I = f.encode(frames=0)
        >>> I = f[0]
        >>> I = next(iter(f))

        Encode the last frame of the fringe pattern sequence.

        >>> I = f.encode(frames=-1)
        >>> I = f[-1]

        Encode the first two frames of the fringe pattern sequence.

        >>> I = f.encode(frames=(0, 1))
        >>> I = f[0, 1]
        >>> I = f[:2]

        Create a generator to receive the frames iteratively, i.e. in a lazy manner.

        >>> I = (frame for frame in f)
        """
        t0 = time.perf_counter()

        # frames
        if frames is None:
            # frames = np.arange(np.sum(self._N))
            frames = np.arange(self.H * np.sum(self._N))
        else:
            if not hasattr(frames, "__iter__"):
                frames = [frames]

            frames = np.array(frames, int).ravel() % self.T

            if self.FDM:
                frames = np.array([np.arange(i * self.D * self.K, (i + 1) * self.D * self.K) for i in frames]).ravel()

            if self.WDM:  # WDM before SDM
                N = 3
                frames = np.array([np.arange(i * N, (i + 1) * N) for i in frames]).ravel()

            if self.SDM or self.uwr == "FTM":  # WDM before SDM
                len_D0 = np.sum(self._N[0])
                frames = np.array([np.arange(i, i + len_D0 + 1, len_D0) for i in frames]).ravel()

        # modulate
        I = self._modulate(frames)

        # multiplex (reduce number of frames)
        if self.SDM or self.WDM or self.FDM or self.uwr == "FTM":
            I = self._multiplex(I)

        # apply inscribed circle
        if self.grid in ["polar", "log-polar"]:
            I *= grid.innercirc(self.Y, self.X)[None, :, :, None]

        # colorize (extended averaging)
        if self.H > 1 or np.any(self.h != 255):  # can be used for extended averaging
            I = self._colorize(I, frames)

        logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

        return I

    def decode(
        self,
        I: np.ndarray,
        Vmin: float = 0,  # todo: Emin
        verbose: bool | str = False,
        check_overexposure: bool = False,
        deinterlace: bool = False,
        despike: bool = False,
        denoise: bool = False,
    ) -> namedtuple:
        r"""Decode fringe patterns.

        Parameters
        ----------
        I : np.ndarray | list
            Fringe pattern sequence.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

            .. note:: It must have been encoded with the same parameters set to the Fringes instance as the encoded one.

        Vmin : float
            Minimum visibility for measurement to be valid
            (can accelerate decoding).

        verbose : bool or str, optional
            If this or the argument `verbose` of the Fringes instance is set to True,
            intermediate and verbose results are returned as well.

        check_overexposure: bool, optional
            If this flag is set to True, 'I' is checked for overexposure
            and if so a warning is logged.

        deinterlace : bool, optional
            If this flag is set to True, deinterlacing is activated.

        despike: bool, optional
            If this flag is set to True, single pixel outliers in the unwrapped phase map are replaced
            by their local neighborhood using a median filter.

        denoise: bool, optional
            If this flag is set to True, the unwrapped phase map is smoothened
            by a bilateral filter which is edge-preserving.

        Returns
        -------
        brightness : np.ndarray
            Local background signal.

        modulation : np.ndarray
            Local amplitude of the cosine signal.

        registration : np.ndarray
            Decoded coordinates.

            .. note:: The registration is a mapping in the same pixel grid as the camera sensor
              and contains the information where each camera pixel, i.e. each camera sightray,
              was looking at during the fringe pattern recording.

        phase : np.ndarray, optional
            Local phase.

        orders : np.ndarray, optional
            Fringe orders.

        residuals : np.ndarray, optional
            Residuals from the optimization-based unwrapping process.

        uncertainty : np.ndarray, optional
            uncertainty of positional decoding in pixel units

        visibility : np.ndarray, optional
            Local visibility (fringe contrast).

        exposure : np.ndarray, optional
            Local exposure (relative average intensity).

        Raises
        ------
        ValueError
            If the number of frames of `I` and the attribute `T` of the `Fringes` instance don't match.

        ValueError
            If 'WDM' is active but 'I' does not have 3 color channels.

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> I = f.encode()

        >>> a, b, x = f.decode(I)

        >>> a, b, x, p, k, r, u, d, g, V, E = f.decode(I, verbose=True)
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
        T, Y, X, C = vshape(I).shape  # extract Y, X, C from data as these parameters depend on the used camera
        I = I.reshape((T, Y, X, C))

        # check for overexposure
        if check_overexposure:
            if I.dtype.kind in "ui":
                lessbits = False  # todo: lessbits
                if lessbits and np.iinfo(I.dtype).bits > 8:  # data may contain fewer bits of information
                    Imax = int(np.ceil(np.log2(I.max())))  # same or next power of two
                    Imax += -Imax % 2  # same or next power of two divisible by two
                else:
                    Imax = np.iinfo(I.dtype).max

                if np.count_nonzero(I == Imax) > 0.1 * I.size:
                    logger.warning("'I' is probably overexposed and decoding might yield unreliable results.")

        # deinterlace
        if deinterlace:
            I = I._deinterlace(I)

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
        Vmin = float(min(max(0, Vmin), 1))  # todo: bmin from dtype, only warning?
        bri, mod, phi, reg, res = self._demodulate(I, Vmin, verbose)

        # verbose
        if verbose:
            unc, fid, dir, glo, vis, exp = self._verbose(I, bri, mod, reg)

        # # blacken where color value of hue was black
        # if self.H > 1 and C == 3:
        #     idx = np.sum(self.h, axis=0) == 0
        #     if np.any(idx):  # blacken where color value of hue was black
        #         bri[..., idx] = 0
        #         mod[..., idx] = 0
        #         reg[..., idx] = np.nan
        #         if verbose:
        #             res[..., idx] = np.nan
        #             unc[..., idx] = np.nan  # self.R / np.sqrt(12)  # todo: circular distribution
        #             phi[..., idx] = np.nan
        #             fid[..., idx] = np.nan
        #             dir[..., idx] = 0
        #             glo[..., idx] = 0
        #             vis[..., idx] = 0
        #             exp[..., idx] = 0

        # spatial unwrapping
        if self._ambiguous:
            logger.warning("Unwrapping is not spatially independent and only yields a relative phase map.")
            reg = self._unwrap(reg, bri)  # todo: res if verbose
        else:  # coordinate retransformation
            # todo: tests

            if self.D == 2:
                # todo: swapaxes
                if self.grid == "Cartesian":
                    if self.X >= self.Y:
                        reg[0] += self.X / 2 - 0.5
                        reg[0] %= self.X
                        reg[1] *= -1
                        reg[1] += self.Y / 2 - 0.5
                        reg[1] %= self.X
                    else:
                        reg[0] += self.X / 2 - 0.5
                        reg[0] %= self.Y
                        reg[1] *= -1
                        reg[1] += self.Y / 2 - 0.5
                        reg[1] %= self.Y

                # todo: polar, logpolar, spiral
                # if self.angle != 0:
                #     t = np.deg2rad(-self.angle)
                #
                #     if self.angle % 90 == 0:
                #         c = np.cos(t)
                #         s = np.sin(t)
                #         R = np.array([[c, -s], [s, c]])
                #         # R = np.matrix([[c, -s], [s, c]])
                #         ur = R[0, 0] * reg[0] + R[0, 1] * reg[1]
                #         vr = R[1, 0] * reg[0] + R[1, 1] * reg[2]
                #         # u = np.dot(uu, R)  # todo: matrix multiplication
                #         # v = np.dot(vv, R)
                #     else:
                #         tan = np.tan(t)
                #         ur = reg[0] - reg[1] * np.tan(t)
                #         vr = reg[0] + reg[1] / np.tan(t)
                #
                #     vv = (reg[1] - reg[0]) / (1 / tan - tan)
                #     uu = reg[0] + vv * tan
                #     reg = np.stack((uu, vv), axis=0)
                #     reg = np.stack((ur, vr), axis=0)

        if despike:
            reg = sp.ndimage.median_filter(reg, size=3, mode="nearest", axes=(1, 2))
            # todo: despike all channels

            # reg[:, -1, -1, ...] = 0
            # reg[:, -10, -10, ...] = 0
            # points = np.arange(Y), np.arange(X)
            # for d in range(self.D):
            #     for c in range(C):
            #         spikes = np.abs(reg[d, :, :, c] - median(reg[d, :, :, c])[0, :, :, 0]) > np.std(reg[d, :, :, c])  # bilateral
            #         values = reg[d, :, :, c]
            #         x = np.nonzero(spikes)
            #         values[x] = 0
            #         x = np.argwhere(spikes)
            #         a = sp.interpolate.interpn(points, values, x, method="cubic")
            #         reg[d] = sp.interpolate.interpn(points, values, x, method="cubic")

        if denoise:
            # # blurring due to uncertainty and PSF
            # u = self.u if self.indexing == "ij" else self.u[::-1]  # todo: D = 1, i.e. shape of sigma equal to axes?
            # sigma = np.sqrt(u ** 2 + self.PSF ** 2)
            # reg = sp.ndimage.gaussian_filter(reg, sigma, mode='nearest', axes=(1, 2))
            reg = bilateral(reg, k=3)
            # todo: denoise all channels

        # create named tuple to return
        if verbose:
            dec = namedtuple(
                "decoded",
                "brightness modulation registration residuals uncertainty phase order direct glob visibility exposure",  # naming it global is not allowed
            )(bri, mod, reg, res, unc, phi, fid, dir, glo, vis, exp)
        else:
            dec = namedtuple("decoded", "brightness modulation registration")(bri, mod, reg)

        logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

        return dec

    def _verbose(self, I: np.ndarray, a: np.ndarray, b: np.ndarray, x: np.ndarray, lessbits: bool = False):
        """Compute verbose output.

        Parameters
        ----------
        I : np.ndarray
            Fringe pattern sequence.

        a : np.ndarray
            Brightness.

        b : np.ndarray
            Modulation.

        x : np.ndarray
            Registration.

        lessbits : bool, optional
            The fringe pattern sequence 'I' may contain fewer bits of information than its corresponding dtype.
            This occurs if e.g. a 10 or 12 bit camera is used, for which the corresponding dtype would be 'uint16'.
            If 'lessbits' is True (the default), the number of bits is estimated based on the maximal value of 'I'.
            This affects the value of the exposure 'e'.

        Returns
        -------
        u : np.ndarray
            Uncertainty of positional decoding, in pixel units.

        k : np.ndarray
            Fringe orders.

        d : np.ndarray
            Direct illumination component.

        g : np.ndarray
            Global illumination component.

        V : np.ndarray
            Visibility (fringe contrast).

        E : np.ndarray
            Exposure (relative average intensity).
        """
        t0 = time.perf_counter()

        Y, X, C = a.shape[1:]

        upi = (
            np.sqrt(2)
            / np.sqrt(self.M)
            / np.sqrt(self._N[:, :, None, None, None])
            / b.reshape(self.D, self.K, Y, X, C)
            * self.ui
        )  # local phase uncertainties  # todo: M
        uxi = upi / (2 * np.pi) * self._l[:, :, None, None, None]  # local positional uncertainties
        u = np.sqrt(1 / np.sum(1 / uxi**2, axis=1))  # global positional uncertainty
        u = u.astype(np.float32, copy=False)

        # todo: from weighted average? this may differ from the k_i in decoder
        k = x[:, None, :, :, :] // self._l[:, :, None, None, None]
        k = k.astype(int, copy=False).reshape(self.D * self.K, Y, X, C)
        # p = (x[:, None, :, :, :] / self._l[:, :, None, None, None] - k) * 2 * np.pi - self.p0
        # p = p.astype(np.float32, copy=False).reshape(self.D * self.K, Y, X, C)

        d = 2 * b  # direct illumination component
        g = 2 * (a.reshape(self.D, 1, -1) - b.reshape(self.D, self.K, -1)).reshape(b.shape).clip(
            0
        )  # global illumination component

        V = np.minimum(
            1, b.reshape(self.D, self.K, Y, X, C) / np.maximum(a[:, None, :, :, :], np.finfo(np.float_).eps)
        )  # avoid division by zero
        V = V.astype(np.float32, copy=False).reshape(self.D * self.K, Y, X, C)

        if I.dtype.kind in "ui":
            if np.iinfo(I.dtype).bits > 8 and lessbits:  # data may contain fewer bits of information
                Imax = int(np.ceil(np.log2(I.max())))  # same or next power of two
                Imax += -Imax % 2  # same or next power of two which is divisible by two
            else:
                Imax = np.iinfo(I.dtype).max
        else:  # float
            # todo: attention: this should never happen, except when decolorized or averaged
            Imax = 1
        E = a / Imax
        E = E.astype(np.float32, copy=False)

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return u, k, d, g, V, E

    def _unwrap(
        self, phi: np.ndarray, b: np.ndarray, func: str = "ski", verbose: bool = False
    ) -> (np.ndarray, np.ndarray):  # todo: use b for quality guidance
        """Unwrap phase maps spatially.

        Parameters
        ----------
        phi : np.ndarray
            Phase maps to unwrap spatially, stacked along the first dimension.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
            The frames (first dimension) as well the color channels (last dimension)
            are unwrapped separately.

        b : np.ndarray, optional
            Modulation of the decoded phase.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        func : str, optional
            Unwrapping function to use. The default is 'ski'.

            - 'ski': `Scikit-image[1]_ <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_

            - else: `OpenCV[2]_ <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_

        verbose : bool, optional
            If this or the argument `verbose` of the Fringes instance is set to True,
            intermediate and verbose results are returned as well.

        Returns
        -------
        unwrapped : np.ndarray
            Unwrapped phase maps.

        References
        ----------
        .. [1] `Herráez et al.,
           "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path",
           Applied Optics,
           2002.
           <https://doi.org/10.1364/AO.41.007437>`_

        .. [2] `Lei et al.,
           "A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping",
           Optik - International Journal for Light and Electron Optics,
           2015.
           <https://doi.org/10.1016/j.ijleo.2015.04.070>`_
        """
        t0 = time.perf_counter()

        T, Y, X, C = vshape(phi).shape
        assert T % self.D == 0, "Number of frames of parameters and data don't match."

        func = "ski"  # todo: enable cv2 unwrapping

        if func in "cv2":  # OpenCV unwrapping
            # params = cv2.phase_unwrapping_HistogramPhaseUnwrapping_Params()
            params = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
            params.height = Y
            params.width = X
            # unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping_create(params)
            unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)

        reg = np.empty((self.D, Y, X, C), np.float32)
        if verbose:
            res = np.empty((self.D, Y, X, C), np.float32)

        for d in range(self.D):
            if self.K == 1:  # todo: self.K[d] == 1
                logger.info(f"Spatial phase unwrapping in 2D{' for each color indepently' if C > 1 else ''}.")
            else:
                logger.info(f"Spatial phase unwrapping in 3D{' for each color indepently' if C > 1 else ''}.")
                func = "ski"  # only ski can unwrap in 3D

            for c in range(C):
                if func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
                    # dtype must be np.float32  # todo: test this
                    if False:  # todo: isinstance(b, np.ndarray) and vshape(b).shape == phi.shape:
                        # todo: unwrap with mask
                        # todo: swapaxes ???
                        SNR = self.B[d, :, :, c] / self.ui
                        upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR  # local phase uncertainties
                        upin = upi / (2 * np.pi)  # normalized local phase uncertainties
                        uxi = upin * self._l[d]  # local positional uncertainties
                        ux = np.sqrt(
                            1 / np.sum(1 / uxi**2)
                        )  # global phase uncertainty (by inverse variance weighting of uxi)
                        mask = np.astype(ux < 0.5, copy=False)  # todo: which limit?

                        reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c], mask)  # todo: test this
                    else:
                        reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c])

                    if verbose:
                        res[d, :, :, c] = unwrapping_instance.getInverseReliabilityMap()  # todo: test this
                        # todo: res vs. rel
                else:  # Scikit-image algorithm is slower but delivers better results on edges
                    reg[d, :, :, c] = ski.restoration.unwrap_phase(phi[d, :, :, c])

                    if verbose:
                        res[d, :, :, c] = np.nan

            regmin = reg[d].min()
            if regmin < 0:
                reg[d] -= regmin

        reg *= self._l[:, 0, None, None, None] / (2 * np.pi)

        logger.debug(f"{1000 * (time.perf_counter() - t0)}ms")

        return reg  # todo: res if verbose

    # @staticmethod
    # def unwrap(phi: np.ndarray, mask: np.ndarray | None = None, func: str = "ski") -> np.array:
    #     """Unwrap phase maps spacially.
    #
    #     Parameters
    #     ----------
    #     phi : np.ndarray
    #         Phase maps to unwrap spatially, stacked along the first dimension.
    #         It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
    #         The frames (first dimension) as well the color channels (last dimension)
    #         are unwrapped separately.
    #
    #     mask : np.ndarray, optional
    #         Mask image with dtype 'np.uint8' used when some pixels do not hold any phase information.
    #
    #     func : str, optional
    #         Unwrapping function to use. The default is 'ski'.
    #
    #         - 'ski': `scikit-image <https://scikit-image.org/docs/stable/auto_examples/filters/plot_phase_unwrap.html>`_ [1]_
    #
    #         - else: `OpenCV <https://docs.opencv.org/4.7.0/df/d3a/group__phase__unwrapping.html>`_ [2]_
    #
    #     Returns
    #     -------
    #     unwrapped : np.ndarray
    #         Unwrapped phase maps.
    #
    #     References
    #     ----------
    #     .. [1] `Herráez et al.,
    #        "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path",
    #        Applied Optics,
    #        2002.
    #        <https://doi.org/10.1364/AO.41.007437>`_
    #
    #     .. [2] `Lei et al.,
    #        "A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping",
    #        Optik - International Journal for Light and Electron Optics,
    #        2015.
    #       <https://doi.org/10.1016/j.ijleo.2015.04.070>`_
    #     """
    #     # todo: counter + log
    #
    #     T, Y, X, C = vshape(phi).shape
    #     phi = phi.reshape((T, Y, X, C))
    #
    #     if func in "cv2":  # OpenCV unwrapping
    #         # params = cv2.phase_unwrapping_HistogramPhaseUnwrapping_Params()
    #         params = cv2.phase_unwrapping.HistogramPhaseUnwrapping.Params()
    #         params.height = Y
    #         params.width = X
    #         # unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping_create(params)
    #         unwrapping_instance = cv2.phase_unwrapping.HistogramPhaseUnwrapping.create(params)
    #
    #     uwr = np.empty_like(phi)
    #
    #     for t in range(T):
    #         for c in range(C):
    #             if func in "cv2":  # OpenCV algorithm is usually faster, but can be much slower in noisy images
    #                 # dtype must be np.float32  # todo: test this
    #                 if False:  # isinstance(mask, np.ndarray) and vshape(mask).shape == phi.shape:
    #                     # todo: unwrap with mask
    #                     mask = vshape(mask)
    #                     mask = np.astype(mask[t, :, :, c], copy=False)
    #                     uwr[t, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[t, :, :, c], mask)
    #                 else:
    #                     uwr[t, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[t, :, :, c])
    #
    #                 # # dtype of phase must be np.uint8, dtype of shadow_mask np.uint8  # todo: test this
    #                 # reg[d, :, :, c] = unwrapping_instance.unwrapPhaseMap(phi[d, :, :, c], shadowMask=shadow_mask)
    #             else:  # Scikit-image algorithm is slower but delivers better results on edges
    #                 uwr[t, :, :, c] = ski.restoration.unwrap_phase(phi[t, :, :, c])
    #
    #     return uwr

    def source(
        self,
        x: np.ndarray,
        b: np.ndarray | None = None,
        u: np.ndarray | float = 0,
        dx: float = 1,
        mode: str = "fast",
    ) -> np.ndarray:
        """Source activation heatmap.

        The decoded coordinates (having sub-pixel accuracy)
        are mapped from the camera grid
        to integer positions on the screen grid
        with weights from the modulation.

        This yields the source activation heatmap:
        a grid representing the screen (light source)
        with the pixel values being a relative measure
        of how much a screen (light source) pixel contributed
        to the exposure of the camera sensor.

        The dimensions of the screen are taken from the `Fringes` instance.

        Parameters
        ----------
        x : np.ndarray
            Registration, i.e. the decoded screen coordinates as seen by the camera.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.

        b : np.ndarray, optional
            Modulation. Used for weighting.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
            If `b` is an array, it must have the same height `Y`, width `X` and color channels `C` as `x`.
            If `b` is not given, equal weights are used.

        u : np.ndarray | float, optional
            Uncertainty.
            It is reshaped to video-shape (frames `T`, height `Y`, width `X`, color channels `C`) before processing.
            If `u` is an array, it must have the same height `Y`, width `X` and color channels `C` as `x`.
            Default is zero.

        dx : float, optional
            Size of one camera pixel, projected onto the screen, in units of screen pixels.
            Default is one.

        mode : str, optional
            By default, fast remapping is applied.
            Else, inverse distance weighted remapping is applied,
            which is more precise but also more time-consuming.

        Returns
        -------
        src : np.ndarray
            Source activation heatmap.

        Notes
        -----
        In fact, this is the inverse function of OpenCV’s remap() [3]_.

        References
        ----------
        .. [3] `OpenCV,
               "remap()",
               OpenCV,
               2024.
               <https://docs.opencv.org/4.9.0/da/d54/group__imgproc__transform.html#gab75ef31ce5cdfb5c44b6da5f3b908ea4>`_

        Examples
        --------
        >>> import fringes as frng
        >>> f = frng.Fringes()
        >>> I = f.encode()

        >>> I_rec = I  # todo: replace this line with recording data as in '../examples/record.py'
        >>> a, b, x = f.decode(I_rec)

        >>> src = f.source(x, b)
        """
        # b = = cv2.remap(src, x[0], x[1], cv2.INTER_LINEAR)  # this is the inverse function of what we want

        t0 = time.perf_counter()

        T, Y, X, C = vshape(x).shape

        # trim x
        x = x.reshape((-1, Y, X, C))
        if self.D == 1:
            if self.axis == 0:
                # x = np.vstack((x, np.zeros_like(x)))
                x = np.concatenate((x, np.zeros_like(x)), axis=0)
            else:
                # x = np.vstack((np.zeros_like(x), x))
                x = np.concatenate((np.zeros_like(x), x), axis=0)
        elif self.indexing == "ij":
            x = x[::-1]  # returns a view
            # b does not need to be changed because there is only one fused value for each (x, y)-coordinate
            # u does not need to be changed because there is only one value for each (x, y)-coordinate
        valid = (x[0] <= self.X) * (x[1] <= self.Y)

        # trim b
        if isinstance(b, np.ndarray):
            assert x.shape[1:] == b.shape[1:], "'x' and 'b' have different width, height or number of color channels"
            b = b.reshape((-1, Y, X, C))
            # b = np.max(b, axis=0)
            b = np.mean(b, axis=0)
            # todo: remap for each b?

        if mode == "fast":
            if not isinstance(b, np.ndarray):
                b = np.ones((Y, X, C), np.uint8)

            # trim u
            if isinstance(u, np.ndarray):
                assert (
                    x.shape[1:] == u.shape[1:]
                ), "'x' and 'u' have different width, height or number of color channels"
                u = u.reshape((-1, Y, X, C))
                u = np.maximum(u, self.u)

            src = np.zeros((self.Y, self.X, C), np.float32)

            # x = x[::-1]  # returns a view
            # idx = np.rint(x).astype(int, copy=False)
            # for c in range(C):  # looping through color channels reduces memory consumption
            #     src[idx[1].ravel(), idx[0].ravel(), c] += b[..., c].ravel()  # ravel() returns a view
            # todo: advanced indexing with nan?
            b[~valid] = 0
            src = _remap(src, x, b)  # todo: if u is array -> also increment region around rint pixel

            # blurring due to uncertainty and PSF
            u = self.u if self.indexing == "ij" else self.u[::-1]  # todo: D = 1, i.e. shape of sigma equal to axes?
            sigma = np.sqrt(u**2 + self.PSF**2)
            src = sp.ndimage.gaussian_filter(src, sigma, mode="nearest", axes=(0, 1))

            # blurring due to pixel size
            dx = 3
            dx_ = int(dx + 0.5)
            if dx > 1:
                src = sp.ndimage.uniform_filter(src, size=dx_, mode="reflect", axes=(0, 1))
        else:
            x = x[:, valid].reshape(2, -1, C).swapaxes(0, 1)  # n data points of dimension m
            if b is not None:
                b = b[valid].reshape(-1, C)  # n data points of dimension m

            # todo: use U if given as an array (but how?)

            u = np.prod(self.u, axis=0) ** (
                1 / self.D
            )  # geometric mean averages the semi-axes of the uncertainty ellipses
            sigma = np.sqrt(u**2 + self.PSF**2)
            dr = dx / np.sqrt(np.pi)  # mapping radius of square (=dx/2) to radius of circle (dr) with same area
            a = 3  # number of standard deviations
            R = dr + a * sigma

            src = np.empty((self.Y, self.X, C), np.float32)
            for c in range(C):
                kdtree = sp.spatial.KDTree(x[:, :, c])
                for xs in range(self.X):
                    for ys in range(self.Y):
                        # todo: choose r pixel individually
                        i = kdtree.query_ball_point(x=(xs, ys), r=R, p=2)  # , workers=-1)  # list of indices

                        if not i:  # empty list, i.e. no points found within distance R
                            v = 0
                        elif b is None:
                            # v = np.sum(np.ones(len(idx)) * w)
                            # v = np.sum(np.ones(len(idx)))
                            v = len(i)
                        else:
                            xr = x[i, 0, c]  # returns a view
                            yr = x[i, 1, c]  # returns a view
                            d = np.sqrt((xs - xr) ** 2 + (ys - yr) ** 2)  # distance

                            # inverse distance weighting using modified Shepard's method:
                            # https://en.wikipedia.org/wiki/Inverse_distance_weighting
                            if 0 in d:
                                # z = d == 0
                                # w[z] = 1
                                # w[~z] = 0
                                w = (d == 0).astype(float, copy=False)
                            else:
                                w = 1 / (d**2)

                            # v = np.sum(b[i, c] * w)
                            v = np.dot(b[i, c], w)

                        src[ys, xs, c] = v

        mx = src.max()
        if mx > 0 and mx != 1:
            src /= mx

        logger.info(f"{1000 * (time.perf_counter() - t0)}ms")

        return src

    def brightfield_inverse(self, src: np.ndarray, t: float = 0.1, k: int = 3) -> np.ndarray:
        """Inverse bright-field

        with radiomatric compensation
        assuming a linear response function for display/projector and camera.

        Parameters
        ----------
        src : np.ndarray
            Source activation heatmap.

        t : float
            Threshold within [0, 1].

        k : int
            Edge ength of morphological operator.

        Returns
        -------
        bfi : Inverse bright-field.
        """
        t = np.clip(t, 0, 1)
        t = 0.2
        src[src < t] = np.nan  # 0

        # radiometric compensation
        bfi = 1 / src  #  - 1
        bfi /= np.nanmax(bfi)
        bfi *= self.Imax
        bfi[bfi == np.nan] = 0
        bfi = bfi.astype(self.dtype, copy=False)

        # blurring due to uncertainty and PSF
        # u = self.u if self.indexing == "ij" else self.u[::-1]  # todo: D = 1, i.e. shape of sigma equal to axes?
        # sigma = np.sqrt(u ** 2 + self.PSF ** 2)
        # a = 3
        # k = 1 + dx + 2 * a * sigma  # 2: both sides
        # k = np.ceil(k).astype(int)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        T, Y, X, C = vshape(bfi).shape
        for t in range(T):
            for c in range(C):
                bfi[t, :, :, c] = cv2.dilate(bfi[t, :, :, c], kernel)

        return bfi  # todo: brightfield_inverse

    def brightfield(self, src: np.ndarray, t: float = 0.1, k: int = 3) -> np.ndarray:
        """Bright-field.

        Parameters
        ----------
        src : np.ndarray
            Source activation heatmap.

        t : float
            Threshold within [0, 1].

        Returns
        -------
        bf : Bright-field.
        """
        t = np.clip(t, 0, 1)
        bf = (src > t).astype(self.dtype, copy=False) * self.Imax

        if k:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            T, Y, X, C = vshape(bf).shape
            for t in range(T):
                for c in range(C):
                    bf[t, :, :, c] = cv2.dilate(bf[t, :, :, c], kernel)

        return bf

    def darkfield(self, src: np.ndarray, t: float = 0.1, k: int = 3) -> np.ndarray:
        """Dark-field.

        Parameters
        ----------
        src : np.ndarray
            Source activation heatmap.

        t : float
            Threshold within [0, 1].

        Returns
        -------
        df : Dark-field.
        """
        t = np.clip(t, 0, 1)
        df = (src <= t).astype(self.dtype, copy=False) * self.Imax

        if k:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            T, Y, X, C = vshape(df).shape
            for t in range(T):
                for c in range(C):
                    df[t, :, :, c] = cv2.dilate(df[t, :, :, c], kernel)

        return df

    def _trim(self, a: np.ndarray) -> np.ndarray:
        """Change ndim of `a` to 2 and limit its shape."""
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

        Returns
        -------
        None

        Raises
        ______
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

        b.shape = T, Y, X, C = vshape(b).shape

        assert T == self.D * self.K  # todo: description of AssertionError

        b = b.reshape(self.D, self.K, self.Y, self.X, self.C)

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

    def mtf(self, v: int | float | tuple[int | float] | list[int | float] | np.ndarray | None = None) -> np.ndarray:
        """Modulation Transfer Function,

        normalized to values within [0, 1].

        Parameters
        ----------
        v: array_like
            Spatial frequencies at which to determine the normalized modulation.

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
        elif self.PSF > 0:  # determine MTF from PSF
            b = self.B * np.exp(-2 * (np.pi * self.PSF * v_) ** 2)  # todo: fix
            # todo: what is smaller: dl or lv?
        else:
            b = 1 - v_ / self.vmax  # approximation of [Bothe2008]
        # else:
        #     b = np.ones(v_.shape)

        return b

    @property
    def T(self) -> int:
        """Number of frames."""

        T = self.H * np.sum(self._N)

        if self.FDM:  # todo: fractional periods
            T /= self.D * self.K

        if self.SDM:
            T /= self.D

        if self.WDM:
            if np.all(self.N == 3):  # WDM along shifts
                T /= 3
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

        if self.uwr == "FTM":
            T = 1

        return int(T)  # use int() to ensure type is "int" instead of "numpy.core.multiarray.scalar"  # todo: necessary?

    @T.setter
    def T(self, T: int):
        # attention: _params may change even if Tnew == Told

        _T = int(min(max(1, T), self._Tmax))

        if _T == 1:  # WDM + SDM
            # todo: FTM?
            if self.grid not in self._grids[:2]:
                logger.error(f"Couldn't set 'T = 1': grid not in {self._grids[:2]}'.")
                return

            self.H = 1
            self.K = 1

            self.FDM = False  # reset FDM before setting N
            self.N = 3  # set N before WDM
            self.WDM = True

            if self.D == 2:
                self.SDM = True

            self.v = 1
        elif _T == 2:  # WDM
            if self.grid not in self._grids[:2]:
                logger.error(f"Couldn't set 'T = 1': grid not in {self._grids[:2]}'.")
                return

            self.H = 1
            self.D = 2
            self.K = 1

            self.FDM = False  # reset FDM before setting N
            self.SDM = False
            self.N = 3  # set N before WDM
            self.WDM = True

            self.v = 1
        else:
            # set boundaries
            self.FDM = False
            self.SDM = False
            self.WDM = False

            # todo: T == 4 -> no mod
            #  T == 5 -> FDM if _T >= self._Nmin?

            # try D == 2  # todo: mux
            if _T < 2 * self._Nmin:
                self.D = 1
            else:
                self.D = 2

            # try to keep 'H'
            Hmax = _T // (self.D * self._Nmin)
            if self.H > Hmax:
                while _T % Hmax != 0:
                    Hmax -= 1
                self.H = Hmax
                _T //= self.H

            # try to keep 'K'
            Kmin = _T // (self.D * self._Nmin)
            if self.K > Kmin:
                self.K = Kmin

            # ensure UMR >= R
            if self._ambiguous:
                imin = np.argmin(self._v, axis=0)
                self._v[imin] = 1

            # set 'N'
            N = np.full((self.D, self.K), _T // (self.D * self.K))
            dT = _T - np.sum(N)
            if dT > 0:
                k = int(dT // self.D)
                N[:, :k] += 1
                if dT % self.D != 0:
                    d = dT % self.D
                    N[:d, k] += 1
            self.N = N

    @property
    def Y(self) -> int:
        """Height of fringe patterns.
        [Y] = px."""
        return self._Y

    @Y.setter
    def Y(self, Y: int):
        _Y = int(min(max(1, Y), self._Ymax, self._Pmax / self.X))

        if self._Y != _Y:
            self._Y = _Y
            logger.debug(f"{self._Y=}")
            self._UMR = None  # empty cache
            self.UMR  # compute
            # self._MM_ = None  # empty cache
            # self.MM_  # compute

            if self._X == self._Y == 1:
                self.D = 1
                self.axis = 0
            elif self._X == 1:
                self.D = 1
                self.axis = 1
            elif self._Y == 1:
                self.D = 1
                self.axis = 0

    @property
    def X(self: int) -> int:
        """Width of fringe patterns.
        [X] = px."""
        return self._X

    @X.setter
    def X(self, X: int):
        _X = int(min(max(1, X), self._Xmax, self._Pmax / self.Y))

        if self._X != _X:
            self._X = _X
            logger.debug(f"{self._X=}")
            self._UMR = None  # empty cache
            self.UMR  # compute
            # self._MM_ = None  # empty cache
            # self.MM_  # compute

            if self._X == self._Y == 1:
                self.D = 1
                self.axis = 0
            elif self._X == 1:
                self.D = 1
                self.axis = 1
            elif self._Y == 1:
                self.D = 1
                self.axis = 0

    @property
    def C(self) -> int:
        """Number of color channels."""
        return 3 if self.WDM or not self._monochrome else 1

    @property
    def R(self) -> np.ndarray:
        """Lengths of fringe patterns for each direction.
        [R] = px."""

        R = np.array([self.X, self.Y])

        if self.indexing == "ij":
            R = R[::-1]

        if self.D == 1:
            R = np.atleast_1d(R[self.axis])

        return R

    @property
    def alpha(self) -> float:
        """Factor for extending the coding range `L`."""
        # alpha = float(1 + 2 * x0 / np.max(self.R))
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        _alpha = float(min(max(1, alpha), self._alphamax))

        if self._alpha != _alpha:
            self._alpha = _alpha
            logger.debug(f"{self._alpha=}")
            logger.debug(f"{self.x0=}")
            self._UMR = None  # N -> fmax -> f -> v  # empty cache
            self.UMR  # compute
            self._MM_ = None  # empty cache
            self.MM_  # compute

    @property
    def x0(self) -> np.ndarray:
        """Coordinate offset."""
        return self.R * (self.alpha - 1) / 2

    @property
    def L(self) -> float:
        """Coding range.
        [L] = px."""
        return float(np.max(self.R) * self.alpha)

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

        if (self.SDM or self.uwr == "FTM") and self.grid not in self._grids[:2]:
            logger.error(f"Couldn't set 'grid': grid not in {self._grids[:2]}'.")
            return

        if self._grid != _grid and _grid in self._grids:
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

            self.SDM = False
            if self._K == 1:
                self.FDM = False
        elif self._D < _D:  # this means that D_ == 2
            self._D = _D
            logger.debug(f"{self._D=}")

            self.N = np.append(self._N, np.tile(self._N[-1, :], (_D - self._N.shape[0], 1)), axis=0)
            self.v = np.append(self._v, np.tile(self._v[-1, :], (_D - self._v.shape[0], 1)), axis=0)
            self.f = np.append(self._f, np.tile(self._f[-1, :], (_D - self._f.shape[0], 1)), axis=0)

            self.B = self.B

    @property
    def axis(self) -> int:
        """Axis along which to shift if number of directions equals one.

        Either `0` or `1`."""
        return self._axis

    @axis.setter
    def axis(self, axis: int):
        _axis = int(min(max(0, axis), 1))

        if self._axis != _axis:
            self._axis = _axis
            logger.debug(f"{self._axis=}")
            self._UMR = None  # empty cache
            self.UMR  # compute

    @property
    def M(self) -> np.ndarray:
        """Number of averaged intensity samples."""
        # M = np.sum(self.h, axis=0) / 255
        M = np.count_nonzero(self.h, axis=0)  # todo

        # M = np.atleast_1d(M[0]) if self._monochrome else M
        M = M[0] if self._monochrome else M
        M = M.min()  # todo: array of M
        return int(
            M
        )  # use int() or float() to ensure type is "int" instead of "numpy.core.multiarray.scalar"  # todo: necessary?

    @M.setter
    def M(self, M: int):
        # try:
        #     iter(M)
        # except TypeError:
        #     ...

        # if hasattr(M, "__iter__"):
        #     ...
        # else:
        #     ...

        _M = int(min(M, self._Mmax))
        self.h = "w" * _M
        # todo: float

    @property
    def H(self) -> int:
        """Number of hues."""
        return self.h.shape[0]

    @H.setter
    def H(self, H: int):
        _H = int(min(max(1, H), self._Hmax))

        if self.H != _H:
            if self.WDM:
                logger.error("Couldn't set 'H': WDM is active.")
                return
                self.h = "w" * _H
            elif _H == 1:
                self.h = "w"
            elif _H == 2:
                self.h = "rb"
            else:
                rgb = "rgb" * int(np.ceil(_H / 3))  # repeat "rgb" this many times
                self.h = rgb[:_H]

    @property
    def h(self) -> np.ndarray:
        """Hues i.e. colors of fringe patterns.

        Possible values are any sequence of RGB color triples within the interval [0, 255].
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
    def h(self, h: int | tuple[int] | list[int] | np.ndarray | str):
        if isinstance(h, str):
            LUT = {
                "r": [255, 0, 0],
                "g": [0, 255, 0],
                "b": [0, 0, 255],
                "c": [0, 255, 255],
                "m": [255, 0, 255],
                "y": [255, 255, 0],
                "w": [255, 255, 255],
            }
            if all(c.lower() in LUT.keys() for c in h):
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
            _h = _h[: self._Hmax, :3]
        else:
            _h = _h[: self._Hmax, :3, ..., -1]

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

    # @property
    # def TDM(self) -> bool:
    #     """Temporal division multiplexing."""
    #     return self.T > 1

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

            if self.grid not in self._grids[:2]:
                _SDM = False
                logger.error(f"Couldn't set 'SDM': grid not in {self._grids[:2]}'.")

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
    def K(self) -> int:
        """Number of sets (number of fringe patterns with different spatial frequencies)."""
        return self._K

    @K.setter
    def K(self, K: int):
        # todo: different K for each D: use array of arrays
        # a = np.ones(2)
        # b = np.ones(5)
        # c = np.array([a, b])

        Kmax = (self._Nmax - 1) / 2 / self.D if self.FDM else self._Kmax  # todo: check if necessary
        _K = int(min(max(1, K), Kmax))

        if self._K > _K:  # remove elements
            self._K = _K
            logger.debug(f"{self._K=}")

            self.N = self._N[:, : self.K]
            self.v = self._v[:, : self.K]
            self.f = self._f[:, : self.K]

            if self._D == self._K == 1:
                self.FDM = False
        elif self._K < _K:  # add elements
            self._K = _K
            logger.debug(f"{self._K=}")

            self.N = np.append(
                self._N, np.tile(self._N[0, 0], (self.D, _K - self._N.shape[1])), axis=1
            )  # don't append N from _defaults, this might be in conflict with WDM!
            v = self.L ** (1 / np.arange(self._v.shape[1] + 1, _K + 1))
            self.v = np.append(self._v, np.tile(v, (self.D, 1)), axis=1)
            self.f = np.append(
                self._f,
                np.tile(self._defaults["f"][0, 0], (self.D, _K - self._f.shape[1])),
                axis=1,
            )

            self.B = self.B

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
        """Number of phase shifts."""
        if self.D == 1 or len(np.unique(self._N, axis=0)) == 1:  # sets in directions are identical
            N = self._N[0]  # 1D
        else:
            N = self._N  # 2D
        return N

    @N.setter
    def N(self, N: int | tuple[int] | list[int] | np.ndarray):
        if self.K == 1 and np.all(N == 1):  # FTM
            _N = np.array([[1], [1]], int)
            if not np.all(self.N == 1):
                self.SDM = True
        else:
            _N = np.array(N, int).clip(self._Nmin, self._Nmax)  # make array, cast to dtype, clip
            if np.all(self.N == 1):
                self.SDM = False

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

        if self.WDM and not np.all(_N == 3):
            logger.error("Couldn't set 'N': At least one Shift != 3.")
            return

        if self.FDM and not np.all(_N == _N[0, 0]):
            _N = np.tile(_N[0, 0], _N.shape)

        if not np.array_equal(self._N, _N):
            self._N = _N
            logger.debug(f"self._N = {str(self._N).replace(chr(10), ',')}")
            self.D, self.K = self._N.shape
            logger.debug(f"{self.T=}")
            self._UMR = None  # N -> fmax -> f -> v  # empty cache
            self.UMR  # compute

    @property
    def lmin(self) -> float:
        """Minimum resolvable wavelength.
        [lmin] = px."""

        # don't use self._fmax, else circular loop
        if self.FDM and self.static:
            fmax = min((self._Nmin - 1) / 2, self.L / self._lmin)
            return min(self._lmin, self.L / fmax)
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

        When L changes, v is kept constant and only l is changed."""
        return self.L / self.v  # todo: round to number of digits?

    @l.setter
    def l(self, l: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(l, str):
            if "," in l:
                l = np.fromstring(l, sep=",")
            elif l == "optimal":
                lmin = int(np.ceil(self.lmin))
                lmax = int(
                    np.ceil(
                        max(
                            self.L / lmin,  # todo: only if b differ slightly
                            self.lmin,
                            min(self.L / self.vopt, self.L),
                            np.sqrt(self.L),
                        )
                    )
                )

                if lmin == lmax and lmax < self.L:
                    lmax += 1

                if lmax < self.L and not sympy.isprime(lmax):
                    lmax = sympy.ntheory.generate.nextprime(lmax, 1)  # ensures lcm(a, lmax) >= L for all a >= lmin

                n = lmax - lmin + 1

                l_ = np.array([lmin])
                l_max = lmin + 1
                lcm = l_
                while lcm < self.L:
                    lcm_new = np.lcm(lcm, l_max)
                    if lcm_new > lcm:
                        l_ = np.append(l_, l_max)
                        lcm = lcm_new
                    l_max += 1
                K = min(len(l_), self.K)

                C = sp.special.comb(n, K, exact=True, repetition=True)  # number of unique combinations
                combos = it.combinations_with_replacement(range(lmin, lmax + 1), K)

                # B = int(np.ceil(np.log2(lmax - 1) / 8)) * 8  # number of bits required to store integers up to lmax
                # combos = np.fromiter(combos, np.dtype((f"uint{B}", K)), C)

                kroot = self.L ** (1 / K)
                if self.lmin <= kroot:
                    lcombos = np.array(
                        [l for l in combos if np.any(np.array([l]) > kroot) and np.lcm.reduce(l) >= self.L]
                    )
                else:
                    lcombos = np.array([l for l in combos if np.lcm.reduce(l) >= self.L])

                # lcombos = filter(lcombos, K, self.L, lmin)

                # idx = np.argmax(np.sum(1 / lcombos**2, axis=1))
                # l = lcombos[idx]

                v = self.L / lcombos
                b = self.mtf(v)
                var = 1 / self.M / self.N * lcombos**2 / b**2  # todo: D, M
                idx = np.argmax(np.sum(1 / var, axis=1))

                l = lcombos[idx]

                if K < self.K:
                    l = np.concatenate((np.full(self.K - K, lmin), l))

                # while lmax < self.L and np.gcd(lmin, lmax) != 1:
                #     lmax += 1  # maximum number of iterations? = min(next prime after lmax - lmax, max(0, L - lmax, ))
                # l = np.array([lmin] * (self.K - 1) + [lmax])

                # vmax = int(max(1 if self.K == 1 else 2, self.vmax))  # todo: ripples from int()
                # v = np.array([vmax] * (self.K - 1) + [vmax - 1])  # two consecutive numbers are always coprime
                # lv = self.L / v
                # lv = np.maximum(self._lmin, np.minimum(lv, self.L))
                #
                # idx = np.argmax((np.sum(1 / (l ** 2), axis=0), np.sum(1 / (lv ** 2), axis=0)))
                # l = l if idx == 0 else lv
                # print("l" if idx == 0 else "v")
            elif l == "close":
                lmin = int(max(np.ceil(self.lmin), self.L ** (1 / self.K) - self.K))
                l = lmin + np.arange(self.K)
                while np.lcm.reduce(l) < self.L:
                    l += 1
            elif l == "small":
                lmin = int(np.ceil(self.lmin))
                lmax = int(np.ceil(self.L ** (1 / self.K)))  # wavelengths are around kth root of self.L

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
                        lmin, min(lmax, int(np.ceil(self.L)))
                    )  # max in outer condition ensures lmax >= lmin even if L < lmin
                    if lmin == lmax and lmax < self.L:
                        lmax += 1  # ensures lmin and lmax differ so that lcm(l) >= L

                    n = lmax - lmin + 1
                    K = min(self.K, n)  # ensures K <= n
                    C = sp.special.comb(n, K, exact=True, repetition=False)  # number of unique combinations
                    combos = np.array(
                        [
                            c
                            for c in it.combinations(range(lmin, lmax + 1), K)
                            if np.any(np.array([c]) > self.L ** (1 / self.K)) and np.lcm.reduce(c) >= self.L
                        ]
                    )

                    idx = np.argmax(np.sum(1 / combos**2, axis=1))
                    l = combos[idx]

                if K < self.K:
                    l = np.concatenate((l, np.arange(l.max() + 1, l.max() + 1 + self.K - K)))
            elif l == "exponential":
                l = np.concatenate(([np.inf], np.geomspace(self.L, self.lmin, self.K - 1)))
            elif l == "linear":
                l = np.concatenate(([np.inf], np.linspace(self.L, self.lmin, self.K - 1)))
            else:
                return

        self.v = self.L / np.array(l, float)

    @property
    def _l(self) -> np.ndarray:  # kept for backwards compatibility with fringes-GUI
        """Wavelengths of fringe periods.
        [l] = px.

        When L changes, v is kept constant and only l is changed."""
        return self.L / self._v

    @property
    def vmax(self) -> float:
        """Maximum resolvable spatial frequency."""
        return self.L / self.lmin

    @property
    def vopt(self) -> float:
        """Optimal spatial frequency for minimal decoding uncertainty."""
        # todo: test
        if self._mtf is not None:  # interpolate from measurement
            v = np.arange(1, self.vmax + 1)
            b = self.mtf(v)
            N = self._N.ravel()[np.argpartition(self._N.ravel(), int(self._N.size // 2))[int(self._N.size // 2)]]
            var = 1 / self.M / N / (v**2) / b**2  # todo: D, M
            idx = np.argmax(np.sum(1 / var, axis=1))
            vopt = v[idx]
        elif self.PSF > 0:  # determine from PSF
            vopt_ = 1 / (2 * np.pi * self.PSF)  # todo
            lopt = 1 / vopt_
            vopt = self.L / lopt
            vopt = self.vmax / 2  # approximation [Bothe2008]
        else:
            vopt = int(self.vmax)

        return vopt

    @property
    def v(self) -> np.ndarray:
        """Spatial frequencies (number of periods/fringes across maximum coding length 'L')."""
        if self.D == 1 or len(np.unique(self._v, axis=0)) == 1:  # sets in directions are identical
            v = self._v[0]  # 1D
        else:
            v = self._v  # 2D
        return v  # todo: round to number of digits?

    @v.setter
    def v(self, v: int | float | tuple[int | float] | list[int | float] | np.ndarray | str):
        if isinstance(v, str):
            if "," in v:
                v = np.fromstring(v, sep=",")
            if v == "optimal":
                # |{v}| = 2
                vmax = int(max(1 if self.K == 1 else 2, self.vopt))
                v = np.array([vmax] * (self.K - 1) + [vmax - 1])  # two consecutive numbers are always coprime

                # # # |{v}| = K
                # vmax = int(max(self.K, self.vopt))
                # v = vmax - np.arange(self.K)
            elif v == "exponential":
                # K = int(np.ceil(np.log2(self.vmax))) + 1  # + 1: 2 ** 0 = 1
                v = np.concatenate(([0], np.geomspace(1, self.vmax, self.K - 1)))
            elif v == "linear":
                v = np.concatenate(([0], np.linspace(1, self.vmax, self.K - 1)))
            else:
                return

        _v = np.array(v, float).clip(0, self.vmax)  # make array, cast to dtype, clip

        if not _v.size:  # empty array
            return

        _v = self._trim(_v)

        if self.FDM:
            if self.static:
                if (
                    _v.size != self.D * self.K
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
            self.UMR  # compute
            self._MM_ = None  # empty cache
            self.MM_  # compute

    @property
    def _fmax(self):
        """Maximum temporal frequency (maximum number of periods to shift over)."""
        return (
            min((self._Nmin - 1) / 2, self.vmax) if self.FDM and self.static else (self._Nmax - 1) / 2
        )  # todo: Nmin vs. Nmax

    @property
    def f(self) -> np.ndarray:
        """Temporal frequency (number of periods to shift over)."""
        if self.D == 1 or len(np.unique(self._f, axis=0)) == 1:  # sets in directions are identical
            f = self._f[0]  # 1D
        else:
            f = self._f  # 2D
        return f

    @f.setter
    def f(self, f: int | float | tuple[int | float] | list[int | float] | np.ndarray):
        _f = np.array(f, float).clip(-self._fmax, self._fmax)  # make array, cast to dtype, clip

        if not _f.size:  # empty array
            return

        _f = self._trim(_f)

        # 'f' must not be divisible by 'N'  # todo: test this
        D = min(_f.shape[0], self._N.shape[0])
        K = min(_f.shape[1], self._N.shape[1])
        if np.any(_f[:D, :K] % self._N[:D, :K] == 0):
            # _f = np.ones(_f.shape)
            _f[:D, :K][_f[:D, :K] % self._N[:D, :K] == 0] = 1  # todo: N = np.minimum(N, np.ceil(2 * f + 1)) but <= Nmax

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

        if 0 not in _f and not np.array_equal(self._f, _f):  # todo: if o in f: log error / warning / rise ValueError
            self._f = _f
            logger.debug(f"self._f = {str((-self._f if self.reverse else self._f).round(3)).replace(chr(10), ',')}")
            self.D, self.K = self._f.shape
            self.N = self._N  # todo: remove if fractional periods is implemented, log warning

    @property
    def p0(self) -> float:
        """Phase offset within interval (-2pi, +2pi).

        It can be used to e.g. let the fringe patterns start (at the origin) with a gray value of zero.
        """
        return self._p0

    @p0.setter
    def p0(self, p0: float):
        _p0 = float(np.sign(p0) * np.abs(p0) % (2 * np.pi))

        if self._p0 != _p0:
            self._p0 = _p0
            logger.debug(f"self._p0 = {self._p0 / np.pi} \u03C0")  # π

    @property
    def _monochrome(self) -> bool:
        """True if all hues are monochromatic, i.e. the RGB values are identical for each hue."""
        return all(len(set(h)) == 1 for h in self.h)

    @property
    def _ambiguous(self) -> bool:
        """True if unambiguous measurement range is larger than the screen length."""
        return bool(np.any(self.UMR < self.R * self.alpha))

    @property
    def indexing(self) -> str:
        """Indexing convention.

        Cartesian indexing `xy` (the default) will index the row first,
        while matrix indexing `ij` will index the colum first.
        """
        # todo: legacy: return "xi" i.e. matrix indexing but row first
        return "ij" if self.mode.lower() == "mrd" else self._indexing

    @indexing.setter
    def indexing(self, indexing):
        _indexing = str(indexing).lower()

        if self._indexing != _indexing and _indexing in self._indexings:
            self._indexing = _indexing
            logger.debug(f"{self._indexing=}")
            self._UMR = None  # empty cache
            self.UMR  # compute
            self._MM_ = None  # empty cache
            self.MM_  # compute

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
        """Mode for remapping.

        The following values can be set:\n
        - 'fast'\n
        - 'precise'
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str):
        _mode = str(mode)

        if self._mode != _mode and _mode in self._modes:
            self._mode = _mode
            logger.debug(f"{self._mode=}")

    @property
    def uwr(self) -> str:
        """Phase unwrapping method."""

        if self.K == 1 and np.all(self._N == 1) and self.grid in self._grids[:2]:
            # todo: v >> 1, i.e. l ~ 8
            uwr = "FTM"  # Fourier-transform method
        elif np.all(self.v <= 1):
            uwr = "none"  # only averaging
        elif self._ambiguous:
            uwr = "spatial"
        else:
            uwr = "temporal"

        return uwr

    @property
    def gamma(self) -> float:
        """Gamma correction factor used to compensate nonlinearities of the display response curve."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float):
        _gamma = float(min(max(0, gamma), self._gammamax))

        if self._gamma != _gamma and _gamma != 0:
            self._gamma = _gamma
            logger.debug(f"{self._gamma=}")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Shape of fringe pattern sequence in video shape (frames, height, with, color channels)."""
        return self.T, self.Y, self.X, self.C

    @property
    def dtype(self) -> np.dtype:
        """Data type.

        The following values can be set:\n
        - 'uint8'\n
        - 'uint16'\n
        - 'float32'\n
        - 'float64'\n
        """
        return np.dtype(self._dtype)  # this is a hotfix for setting _dtype directly as a str in init

    @dtype.setter
    def dtype(self, dtype: np.dtype | str):
        _dtype = np.dtype(dtype)

        if self._dtype != _dtype and str(_dtype) in self._dtypes:
            self._dtype = _dtype
            logger.debug(f"{self._dtype=}")
            logger.debug(f"self.A = {self.A}")
            logger.debug(f"self.B = {self.B}")

    @property
    def Imax(self) -> int:
        """Maximum gray value."""
        return np.iinfo(self.dtype).max if self.dtype.kind in "ui" else 1

    @property
    def _Amin(self):
        """Minimum bias."""
        return self.B / self._Vmax

    @property
    def _Amax(self):
        """Maximum bias."""
        return self.Imax - self._Amin

    @property
    def A(self) -> float:
        """Bias."""
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
        """Maximum relative bias (exposure)."""
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
        """Fringe visibility (fringe contrast) ∈ [0, 1]."""
        return self._V

    @V.setter
    def V(self, V: float):
        _V = float(min(max(0, V), self._Vmax))

        if self._V != _V:
            self._V = _V
            logger.debug(f"{self.V=}")
            logger.debug(f"{self.B=}")

    @property
    def PSF(self) -> float:  # todo: magnification?
        """Standard deviation of Point Spread Function for defocus.
        [PSF] = px."""
        return self._PSF

    @PSF.setter
    def PSF(self, PSF):
        _PSF = float(max(0, PSF))

        if self._PSF != _PSF:
            self._PSF = _PSF
            logger.debug(f"{self._PSF=}")

    @property
    def UMR(self) -> np.ndarray:
        """Unambiguous measurement range.
        [UMR] = px

        The coding is only unique within the interval [0, UMR); after that it repeats itself.

        The UMR is derived from 'l' and 'v':\n
        - If 'l' ∈ ℕ, UMR = lcm('l'), with lcm() being the least common multiple.\n
        - Else, if 'v' ∈ ℕ, UMR = 'L' / gcd('v'), with gcd() being the greatest common divisor.\n
        - Else, if 'l' ∨ 'v' ∈ ℚ, lcm() resp. gcd() are extended to rational numbers.\n
        - Else, if 'l' ∧ 'v' ∈ ℝ ∖ ℚ, UMR = prod('l'), with prod() being the product operator.
        """
        # cache
        if self._UMR is not None:
            return self._UMR
        elif not self._N.shape == self._v.shape == self._f.shape:
            # logger.warning(
            #     "Could not determine 'UMR' because of unequal shapes of 'N', 'v', 'f'. "
            #     "Returning old, cached value instead."
            # )  # attention: logging shouldn't be necessary because instance will never stay in this state
            return self._UMR

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
                UMR[d] = self.L / np.gcd.reduce(v.astype(int, copy=False))
            elif np.allclose(l, np.rint(l), rtol=0, atol=atol):  # all l are integers within tolerance
                UMR[d] = np.lcm.reduce(np.rint(l).astype(int, copy=False))
            elif np.allclose(v, np.rint(v), rtol=0, atol=atol):  # all v are integers within tolerance
                UMR[d] = self.L / np.gcd.reduce(np.rint(v).astype(int, copy=False))
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
                        UMR[d] = self.L / (np.gcd.reduce(vs.astype(int, copy=False)) / 10**Dv)
                        logger.debug("Extended 'gcd' to rational numbers.")
                else:  # irrational numbers or rational numbers with more digits than "precision"
                    UMR[d] = np.prod(l)

        # ...

        self._UMR = UMR
        logger.debug(f"self.UMR = {str(self._UMR)}")

        if self._ambiguous:
            logger.warning("UMR < R. Unwrapping will not be spatially independent and only yield a relative phase map.")

        return self._UMR

    @property
    def _m(self) -> np.ndarray:
        """Moduli for Chinese remainder theorem."""

        # todo: cache this, cf. UMR, MM_

        precision = np.finfo(float).precision - 2  # todo: 13
        atol = 10**-precision

        m = np.empty((self.D, self.K), int)  # mduli, i.e. integer ratios
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
        return m

    @property
    def t(self) -> np.ndarray:
        """Greatest common divisor of moduli."""
        # think of it as gear tooth size
        t = self._l / self._m
        return np.mean(t, axis=1)  # each ' t# should be identical; we average to get more precision

    @property
    def MM_(self) -> np.ndarray:
        """Coefficients for Chinese remainder theorem."""

        if self._MM_ is not None:  # cache
            return self._MM_  # todo: resetting cache doesn't work for some reasons...fixed? -> test!

        MM_ = np.empty((self.D, self.K), int)  # Solution of multiple congruences.
        for d in range(self.D):
            lcm = np.lcm.reduce(self._m[d])

            if lcm > 0:
                for i in range(self.K):
                    # try:
                    #     M = int(lcm / self._m[d, i])
                    # except ZeroDivisionError:
                    #     MM_[d, i] = 0  # no solution
                    #     continue
                    M = int(lcm / self._m[d, i])

                    # if np.any(MM_[d, :i] % self._m[d, i] == 1):  # one element already equals 1 modulo m[i]
                    #     MM_[d, i] = -1  # 0  # todo: WAVG them
                    #     continue

                    try:
                        # M_ = pow(M, -1, self._m[d, i])  # only for Python 3.8+
                        M_ = sympy.core.numbers.mod_inverse(M, self._m[d, i])
                        MM_[d, i] = M * M_
                    except ValueError:
                        MM_[d, i] = 0  # no solution found

                    if np.any(MM_[d, :i] % self._m[d, i] == 1):  # one element already equals 1 modulo m[i]
                        MM_[d, i] *= -1  # todo: WAVG them
                        continue
            else:
                MM_[d, :] = 0

        self._MM_ = MM_
        logger.debug(f"self._MM_ = {str(self._MM_)}")

        return self._MM_

    @property
    def eta(self) -> float:
        """Coding efficiency."""
        eta = self.R / self.UMR
        eta[self.UMR < self.R] = 0
        return eta

    @property
    def ui(self) -> float:
        """Intensity noise."""
        return self._ui

    @ui.setter
    def ui(self, ui):
        _ui = float(max(0, ui))

        if self._ui != _ui:
            self._ui = _ui
            logger.debug(f"{self._ui=}")

    @property
    def upi(self) -> float:
        """Phase uncertainties.
        [upi] = rad"""
        b = self.B * self.mtf(self._v)
        SNR = b / self.ui
        upi = np.sqrt(2) / np.sqrt(self.M) / np.sqrt(self._N) / SNR
        return upi

    @property
    def uxi(self) -> np.ndarray:
        """Positional uncertainties."""
        uxi = self.upi / (2 * np.pi) * self._l  # local positional uncertainties
        return uxi

    @property
    def u(self) -> np.ndarray:
        """Measurement uncertainty.
        [u] = px.

        It is based on the phase noise model
        and propagated through the unwrapping process and the phase fusion."""

        # todo: rename to ux?

        ux = np.sqrt(1 / np.sum(1 / self.uxi**2, axis=1))  # global positional uncertainty
        # todo: factor out L ?!
        return ux

    @property
    def SNR(self) -> np.ndarray:
        """Signal-to-noise ratio of the phase shift coding.

        It is a measure of how many points can be distinguished within the screen length [0, R).
        """
        return self.R / self.u

    @property
    def SNRdB(self) -> np.ndarray:
        """Signal-to-noise ratio.
        [SNRdB] = dB."""
        return 20 * np.log10(self.SNR)

    @property
    def DR(self) -> np.ndarray:
        """Dynamic range of the phase shift coding.

        It is a measure of how many points can be distinguished within the unambiguous measurement range [0, UMR).
        """
        return self.UMR / self.u

    @property
    def DRdB(self) -> np.ndarray:
        """Dynamic range. [DRdB] = dB."""
        return 20 * np.log10(self.DR)

    @property  # todo: find name and write docs
    def _efficiency(self) -> np.ndarray:
        """Coding efficiency."""
        return self.SNR / self.T

    @property
    def _params(self) -> dict:
        """Base parameters required for en- & decoding fringe patterns.

        This contains all property objects of the class which have a setter method,
        i.e. are (usually) not derived from others.
        """
        params = {"__version__": _version()}
        for p in sorted(dir(self)):  # sorted() ensures setting params in the right order in __init__()
            if not p.startswith("_"):  # p != "_params"
                if isinstance(getattr(type(self), p, None), property) and getattr(type(self), p, None).fset is not None:
                    param = getattr(self, p, None)
                    if isinstance(param, np.ndarray):
                        params[p] = param.tolist()
                    elif isinstance(param, np.dtype):
                        params[p] = str(param)
                    else:
                        params[p] = param

        return params

    @_params.setter
    def _params(self, params: dict = {}):
        # todo: if no version string or version <= 1.1.1:
        # iterating self._params ensures that only properties with a setter method are set
        params_old = self._params.copy()

        for k, v in self._params.items():
            if k in params and k not in ["T", "__version__"]:
                setattr(self, k, params[k])

        for k, v in self._params.items():
            if k in params:
                if k in "Nlvf" and np.array(v).ndim != np.array(params[k]).ndim:
                    if not np.array_equal(v, params[k][0]):
                        break
                else:
                    if not np.array_equal(v, params[k]):
                        break
        else:  # else clause executes only after the loop completes normally, i.e. did not encounter a break
            return

        logger.warning(f"'{k}' got overwritten by interdependencies. Choose a consistent parameter set.")
        self._params = params_old

    _glossary: dict = {
        k: v.__doc__.splitlines()[0]
        for k, v in sorted(vars().items())
        if not k.startswith("_") and isinstance(v, property)
    }
    _defaults: dict = dict(sorted(__init__.__kwdefaults__.items()))

    # restrict instance attributes to the ones listed here
    # (commend the next line out to prevent this)
    __slots__ = tuple("_" + k for k in _defaults.keys() if k not in "HMTlAB") + (
        "_mtf",
        "_UMR",
        "_MM_",
        "_t",
        "_UMR_params",
    )

    # continuing the class docstring following the NumPy style guide:
    # https://numpydoc.readthedocs.io/en/latest/format.html#class-docstring
    __doc__ += "\n\nParameters\n----------"
    __doc__ += "\n*args : tuple\n    Non-keyword arguments are explicitly discarded."
    __doc__ += "\n    Only keyword arguments are considered."

    for __k, __v in _defaults.items():
        # do this in for loop instead of list comprehension,
        # because for the latter, names in class scope are not accessible
        if __k in ["N", "l", "v", "l", "h"]:
            __doc__ += f"\n{__k} : array_like, default = {__v}\n    {_glossary[__k]}"
        else:
            __doc__ += f"\n{__k} : {__init__.__annotations__[__k]}, default = {__v}\n    {_glossary[__k]}"

    # __doc__ += "\n\nAttributes (derived from Parameters)\n------------------------------------"
    # for __k, __v in sorted(vars().items()):
    #     # do this in for loop instead of list comprehension,
    #     # because for the latter, names in class scope are not accessible
    #     if __k not in _defaults and isinstance(__v, property) and not __k.startswith("_"):
    #         __doc__ += f"\n{__k} : {type(__v)}\n    {_glossary[__k]}"  # skip default values as they are evaluated only at runtime
    #
    # __doc__ += "\n\nMethods\n-------"
    # __doc__ += f"\nencode\n    {encode.__doc__.splitlines()[0]}"
    # __doc__ += f"\ndecode\n    {decode.__doc__.splitlines()[0]}"
    # todo: add all public methods (which should be public?)
    # for __k, __v in sorted(vars().items()):
    #     # do this in for loop instead of list comprehension,
    #     # because for the latter, names in class scope are not accessible
    #     if __k not __k.startswith("_") and callable(__v):
    #         __doc__ += f"\n{__k}\n : {__v.__doc__.splitlines()[0]}"

    del __k, __v
