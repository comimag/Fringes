Optimal Coding Strategy
=======================

As makes sense intuitively, more sets `K` as well as more shifts `N` per set reduce the uncertainty `u` after decoding.
A minimum of 3 shifts is needed to solve for the 3 unknowns brightness `A`, modulation `B` and coordinates `ξ`.
Any additional 2 shifts compensate for one harmonic of the recorded fringe pattern.
Therefore, higher accuracy can be achieved using more shifts `N`, but the time required to capture them
sets a practical upper limit to the feasible number of shifts.

Generally, shorter wavelengths `l` (or equivalently more periods `v`) reduce the uncertainty `u`,
but the resolution of the camera and the display must resolve the fringe pattern spatially.
Hence, the used hardware imposes a lower bound for the wavelength (or upper bound for the number of periods).

Also, small wavelengths might result in a smaller unambiguous measurement range `UMR`.
If two or more sets `K` are used and their wavelengths `l` resp. number of periods `v` are relative primes,
the unmbiguous measurement range can be increased many times.
As a consequence, one can use much smaller wavelenghts `l` (larger number of periods `v`).


However, it must be assured that the unambiguous measurment range is always equal or larger than both,
the width `X` and the height `Y`.
Else, [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-) would yield wrong results and thus instead
[spatial phase unwrapping](#spatial-phase-unwrapping--spu-) is used.
Be aware that in the latter case only a relative phase map is obtained,
which lacks the information of where exactly the camera sight rays were looking at during acquisition.

To simplify finding and setting the optimal parameters, the following methods can be used:
- `setMTF()`: The optimal `vmax` is determined automativally [[18]](#18-bothe-2008)
by measuring the **modulation transfer function** `MTF`.\
  Therefore, a sequence of exponentially increasing `v` is acquired:
    1. Set `v` to `'exponential'`.
    2. Encode, acquire and decode the fringe pattern sequence.
    3. Call the function `setMTF(B)` with the argument `B` from decoding.
- `v` can be set to `'auto'`. This automatically determines the optimal integer set of `v`
  based on the maximal resolvable spatial frequency `vmax`.
-  Equivalently, `l` can also be set to `'auto'`. This will automatically determine the optimal integer set of `l`
  based on the minimal resolvable wavelength `lmin = L / vmax`.
- `T` can be set directly, based on the desired acquisition time.
  The optimal `K`, `N` and the [multiplexing](#multiplexing) methods will be determined automatically.

Alternatively, simply use the function `auto()`
to automatically set the optimal `v`, `T` and [multiplexing](#multiplexing) methods.

## __Optimal Coding Strategy__
As makes sense intuitively, more sets `K` as well as more shifts `N` per set reduce the uncertainty `u` after decoding.
A minimum of 3 shifts is needed to solve for the 3 unknowns brightness `A`, modulation `B` and coordinate `ξ`.
Any additional 2 shifts compensate for one harmonic of the recorded fringe pattern.
Therefore, higher accuracy can be achieved using more shifts `N`, but the time required to capture them
sets a practical upper limit to the feasible number of shifts.

Generally, shorter wavelengths `l` (or equivalently more periods `v`) reduce the uncertainty `u`,
but the resolution of the camera and the display must resolve the fringe pattern spatially.
Hence, the used hardware imposes a lower bound for the wavelength (or upper bound for the number of periods).

Also, small wavelengths might result in a smaller unambiguous measurement range `UMR`.
If two or more sets `K` are used and their wavelengths `l` resp. number of periods `v` are relative primes,
the unambiguous measurement range can be increased many times.
As a consequence, one can use much smaller wavelenghts `l` (larger number of periods `v`).

However, it must be assured that the unambiguous measurment range is always equal or larger than both,
the width `X` and the height `Y`.
Else, [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-) would yield wrong results and thus instead
[spatial phase unwrapping](#spatial-phase-unwrapping--spu-) is used.
Be aware that in the latter case only a relative phase map is obtained,
which lacks the information of where exactly the camera pixels were imaged to during acquisition.

To simplify finding and setting the optimal parameters, one can choose from the followng options:
- `v` can be set to `'optimal'`.
  This automatically determines the optimal integer set of `v`,
  based on the maximal resolvable spatial frequency `vmax`.\
- Equivalently, `l` can also be set to `'optimal'`.
  This will automatically determine the optimal integer set of `l`,
  based on the minimal resolvable wavelength `lmin = L / vmax`.
- `T` can be set directly, based on the desired acquisition time.
  The optimal `K`, `N` and  - if necessary - the multiplexing methods will be determined automatically.
- Instead of the options above, one can simply use the function `optimize()`
  to automatically set the optimal `v`, `l`, `T` and multiplexing methods.

However, those methods only perform optimally
if the recorded modulation `B` is known (or can be estimated) for each certain spatial frequencies `v`.
1. Option A: Measure the **modulation transfer function (MTF)** at a given number of sample points:
   1. Set `K` to the required number of sample ponts (usually 10 is a good value).
   2. Set `v` to `'exponential'`.
      This will create spatial frequencies `v` spaced evenly on a log scale (a geometric progression),
      starting from `0` up to `vmax`.
   3. Encode, acquire and decode the fringe pattern sequence.
   4. Mask the values of `B` with nan where the camera wasn't looking at the screen.
   5. Call `Bv(B)` with the estimated modulation from the measurement as the argument.
   6. Finlly, to get the modulation `B` at certain spatial frequencies `v`, simply call `MTF(v)`.
      This method interpolates the modulation from the measurements `Bv` at the points `v`.
2. Option B: Estimate the **magnification** and the **Point Spread Function (PSF)** of the imaging system:
   1. Set the attributes `magnification` and `PSF` of the `Fringes` instance.
      `PSF` is given as the standard deviation of the Point Spread Function.
   2. Finlly, to get the modulation `B` at certain spatial frequencies `v`, simply call `MTF(v)`.
      This method computes the modulation from the specified attributes `magnifiction` and `PSF` directly.