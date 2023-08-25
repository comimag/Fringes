# Fringes
![PyPI](https://img.shields.io/pypi/v/fringes)
![GitHub top language](https://img.shields.io/github/languages/top/comimag/fringes)
![Read the Docs](https://img.shields.io/readthedocs/fringes)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI - License](https://img.shields.io/pypi/l/fringes)
![PyPI - Downloads](https://img.shields.io/pypi/dm/fringes)

<!---
![GitHub](https://img.shields.io/github/license/comimag/fringes)
--->

Easy to use tool for generating and analyzing fringe patterns with phase shifting algorithms.

![Coding Scheme](https://raw.githubusercontent.com/comimag/fringes/develop/docs/getting-started/coding-scheme.gif)\
Figure 1: Phase Shift Coding Scheme.

<!---
link to  paper, please cite
--->

<!---
## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Graphical User Interface](#graphical-user-interface)
- [Optimal Coding Strategy](#optimal-coding-strategy)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Project Status](#project-status)
- [References](#references)
--->

## Installation
You can install `fringes` directly from [PyPi](https://pypi.org/) via `pip`:

```
pip install fringes
```

## Usage
You instantiate, parameterize and deploy the `Fringes` class:

```python
import fringes as frng  # import module

f = frng.Fringes()      # instantiate class

f.glossary              # get glossary
f.X = 1920              # set width of the fringe patterns
f.Y = 1080              # set height of the fringe patterns
f.K = 2                 # set number of sets
f.N = 4                 # set number of shifts
f.v = [9, 10]           # set spatial frequencies
f.T                     # get number of frames
                            
I = f.encode()          # encode fringe patterns
A, B, x = f.decode(I)   # decode fringe patterns
```

All parameters are accesible by the respective attributes of the `Fringes` instance
(a glossary of them is obtained by the attribute `glossary`).
They are implemented as class properties (managed attributes).
Note that some attributes have subdependencies, hence dependent attributes might change as well.
Circular dependencies are resolved automatically.

For generating the fringe pattern sequence `I`, use the method `encode()`.\
It returns a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) 
in videoshape (frames `T`, width `X`, height `Y`, color channels `C`).

For analyzing (recorded) fringe patterns, use the method `decode()`.\
It returns the Numpy arrays brightness `A`, modulation `B` and coordinate `x`.

## Graphical User Interface
Do you need a GUI? `Fringes` has a sister project which is called `Fringes-GUI`: https://pypi.org/project/fringes-gui/

<!---
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
--->

## Troubleshooting
<!---
- __`poetry install` does not work__  
  First, ensure that poetry is installed correctly as descibed on the [Poetry Website](https://python-poetry.org/docs/).\
  Secondly, ensure the correct python version is installed on your system, as specified in the file `pyproject.toml`!\
  Thirdly, this can be caused by a proxy which `pip` does not handle correctly.
  Manually setting the proxy in the Windows settings or even adding a system variable 
  `https_proxy = http://YOUR_PROXY:PORT` can resolve this.
--->

- __Decoding takes a long time__  
  This is most likely due to the just-in-time compiler [Numba](https://numba.pydata.org/), 
  which is used for this computationally expensive :
  During the first execution, an initial compilation is executed. 
  This can take several tens of seconds up to single digit minutes, depending on your CPU.
  However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster, 
  approaching the speeds of code written in C.

<!---
- __My decoded coordinates show lots of noise__
  - Make sure the exposure of your camera is adjusted so that the fringe patterns show up with maximum contrast.
    Try to avoid under- and overexposure during acquisition.
  - Try using more, sets `K` and/or shifts `N`.
  - Adjust the used wavelengths `l` resp. number of periods `v` to ensure the unamboguous measurement range
    is larger than the pattern length, i.e. <code>UMR &ge; L</code>.
  - If the decoded modulation is much lower than the decoded brightness,
    try to use larger wavelengths `l` resp. smaller number of periods `v`.\
    If the decoded modulation remains low even with very large wavelengths (less than five periods per screen length),
    and you are conducting a deflectometric mesurement, the surface under test is probably too rough.
    Since deflectometry is for specular and glossy surfaces only, it isn't suited for scattering ones.
    You should consider a different measurement technique, e.g. fringe projection.

- __My decoded coordinates show systematic offsets__
  - First, ensure that the correct frames were captured while acquiring the fringe pattern sequence.
    If the timings are not set correctly, the sequence may be a frame off.
  - Secondly, this might occur if either the camera or the display used have a gamma value very different from 1.
    - Typically, screens have a gamma value of 2.2; therefore compensate by setting the inverse value
      <code>gamma<sup>-1</sup> = 1 / 2.2 &approx; 0.45</code> to the `gamma` attribute of the `Fringes` instance.\
      Alternatively, change the gamma value of the light source or camera directly.
    - You can use the static method `gamma_auto_correct` to
      automatically estimate and apply the gamma correction factor to linearize the display/camera response curve.
    - You might also use more shifts `N` to compensate for the dominant harmonics of the gamma-nonlinearities.
--->

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License

## Project Status
This package is under active development, so features and functionally will be added in the future.
Feature requests are warmly welcomed!
