# Fringes
Phase shifting algorithms for encoding and decoding sinusoidal fringe patterns.

## Description

### Background
Many applications, such as fringe projection [[11]](#11-burke-2002) or deflectometry [[1]](#1-burke-2022),
require the ability to encode positional data.
To do this, fringe patterns are used to encode the position on the screen / projector (in pixel coordinates)
at which the camera pixels were looking at during acquisition.

--- FIGURE coding ---

```
I = A + B * cos(2πvξ/L - 2πft - φ₀)
  = A + B * cos(kx - wt - φ₀)
  = A + B * cos(Φ)

```

- **Encoding**
  - **Spatial Modulation**\
The x- resp. y-coordinate `ξ` of the screen/projector is normalized into the range `[0, 1)`
by dividing through the maximum coordinate `L`
and used to modulate the luminance in a sinusoidal fringe pattern `I`
with offset `A`, amplitude `B` and spatial frequency `v`.\
  - **Temporal Modulation**\
The pattern is then shifted `N` times with an equidistant phase shift of `2πf/N` radian each.
An additional phase offset `φ₀` may be set, e.g. to let the fringe patterns start with a gray value of zero.
- **Decoding**
  - **Temporal Demodulation**\
From these shifts, the phase map `φ` is determined [[13]](#13-burke-2012). Due to the trigonometric functions used,
the global phase `Φ` is wrapped into the interval <code>[0, 2 &pi;]</code> with `v` periods:
<code>φ &equiv; Φ mod 2&pi;</code>.
  - **Spatial Demodulation / Phase Unwrapping**\
If only one set with spatial frequency <code>v &le; 1</code> is used,
no unwrapping is required because one period covers the complete coding range.
Hence, the coordinates `ξ` are computed directly by scaling: <code>ξ = φ / (2&pi;) * L / v</code>.
This constitutes the registration, which is a mapping in the same pixel grid as the camera sensor
and contains the information where each camera pixel, i.e. each camera sightray, was looking at
during the fringe pattern acquisition.
Note that in contrast to binary coding schemes, e.g. Gray code,
the coordinates are obtained with sub-pixel precision.
    - **Temporal Phase Unwrapping (TPU)**\
If multiple sets with different spatial frequencies `v` are used
and the [unmbiguous measurement range](#quality-metrics) is larger than the coding range <code>UMR &ge; L</code>,
the ambiguity of the phase map is resolved by
generalized multi-frequency temporal phase unwrapping [[14]](#14-kludt-2024).
    - **Spatial Phase Unwrapping (SPU)**\
However, if only one set with `v > 1` is used, or multiple sets but  `UMR < L`, the ambiguous phase `φ`
is unwrapped analyzing the neighbouring phase values [[15]](#15-herráez-2002) [[16]](#16-lei-2015).
This only yields a relative phase map, therefore absolute positions are unknown.

### Features
<!---
- Generalized Temporal Phase Unwrappting (GTPU)[[14](#14-kludt-2024)]
--->
- Generalized Temporal Phase Unwrappting (GTPU)
- Uncertainty Propagation
- Computation of Residuals
- Deinterlacing
- Multiplexing
- Filtering Phase Maps
- Remapping

## Contents
- [Installation](#installation)
- [Usage](#usage)
- [Graphical User Interface](#graphical-user-interface)
- [Attributes](#attributes)
- [Methods](#methods)
- [Optimal Coding Strategy](#optimal-coding-strategy)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Project Status](#project-status)

## Installation
You can install `fringes` directly from [PyPi](https://pypi.org/) via `pip`:

```
pip install fringes
```

## Usage
You instantiate and deploy the `Fringes` class:

```python
import fringes as frng

f = frng.Fringes()      # instanciate class
```

For creating the fringe pattern sequence `I`, use the method `encode()`.
It will return a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) 
in [videoshape](#video-shape) (frames, width, height, color channels).

```python
I = f.encode()          # encode fringe patterns
```

For analyzing (recorded) fringe patterns, use the method `decode()`.
It will return a [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple), 
containing the Numpy arrays brightness `A`, modulation `B` and the coordinates `ξ`,
all in [videoshape](#video-shape).

```python
A, B, xi = f.decode(I)  # decode fringe patterns
```

All parameters are accesible by the respective attributes of the `Fringes` class.

```python
f.X = 1920              # set width of the fringe patterns
f.Y = 1080              # set height of the fringe patterns
f.K = 2                 # set number of sets
f.N = 4                 # set number of shifts
f.v = [9, 10]           # set spatial frequencies
f.T                     # get the number of frames
```

A glossary of them is obtained by the class attribute `doc`.

```python
frng.Fringes.doc        # get glossary
```

You can change the [logging level](https://docs.python.org/3/library/logging.html#levels) of a `Fringes` instance.
Changing it to `'DEBUG'` gives you verbose feedback on which parameters are changes
and how long functions take to execute.

```python
f.logger.setLevel("DEBUG")
```
<!---
## Graphical User Interface
Do you need a GUI? `Fringes` has a sister project that is called `Fringes GUI`:
https://pypi.org/project/fringes-gui/
--->
## Attributes
All parameters are parsed when setting, so usually several input formats are accepted, e.g.
`bool`, `int`, `float`, `str` for scalars and additionally `list`, `tuple`, `ndarray` for arrays.

Note that parameters might have circular dependencies which are resolved automatically,
hence dependent parameters might change as well. 
The attributes in rectangular boxes are readonly, i.e. they are inferred from the others.
Only the ones in white boxes will never influence others.

![Parameter Interdependencies](docs/interdependencies.svg)\
Parameter and their Interdependencies.

### __Coordinate System__
The following coordinate systems can be used by setting `grid` to:
- `'image'`: The top left corner pixel of the grid is the origin (0, 0)
and positive directions are right- resp. downwards.
- `'Cartesian'`: The center of grid is the origin (0, 0) and positive directions are right- resp. upwards.
- `'polar'`: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.
- `'log-polar'`: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.

`D` denotes the number of directions to be encoded.
If <code>D &equiv; 1</code>, the parameter `axis` is used to define along which axis of the coordinate system
(index 0 or 1) the fringes are shifted.

`angle` can be used to tilt the coordinate system. The origin stays the same.

### __Video Shape__
Standardized `shape` (`T`, `Y`, `X`, `C`) of fringe pattern sequence, with
- `T`: number of frames
- `Y`: height
- `X`: width
- `C`: number of color channels

`T` depends on the paremeters `H`, `D`, `K`, `N` and the [multiplexing](#multiplexing) methods:\
If all `N` are identical, then `T = H * D * K * N` with `N` as a scalar,
else <code>T = H * &sum; N<sub>i</sub></code> with `N` as an array.\
If a [multiplexing](#multiplexing) methods is activated, `T` reduces further.

The length `L` is the maximum of `X` and `Y`.

`C` depends on the [coloring](#coloring-and-averaging) and [multiplexing](#multiplexing) methods.

`size` is the product of `shape`.

### __Set__
Each set consists of the following parameters:
- `N`: number of shifts
- `l`: wavelength [px]
- `v`: spatial frequency, i.e. number of periods
- `f`: temporal frequency, i.e. number of periods to shift over

Each is an array with shape (direction `D`, number of sets`K`).\
For example, if <code>N.shape &equiv; (2, 3)</code>, it means that we encode `D = 2` directions with `K = 3` sets each.

Changing `D` or `K` directly, changes the shape of all set parameters.
When setting a set parameter with a new shape (`D'`, `K'`),
`D` and `K` are updated as well as the shape of the other set parameters.

Per direction at least one set with `N >= 3` is necessary
to solve for the three unknowns brightness `A`, modulation `B` and coordinates `ξ`.

`l` and `v` are related by `l = L / v`
When `L` changes, `v` is kept constant and only `l` is changed.

Usually `f = 1`
and `f` is essentially only changed if [frequency division multiplexing](#multiplexing) `FDM` is activated.

`reverse` is a boolean which reverses the direction of the shifts (by multiplying `f` with `-1`).

`o` denotes the phase offset `φ₀` which can be used to e.g. let the fringe patterns start (in origin) with a gray value of zero

`UMR` denotes the unambiguous measurement range.
The coding is only unique in the interval `[0, UMR]`, after that it repeats itself.
The `UMR` is determined from `l` and `v`:\
- If <code>l &isin; &#8469;</code>, <code>UMR = lcm(l<sub>i</sub>)</code> with `lcm` being the least common multiple.\
- Else, if <code>v &isin; &#8469;</code>,
  <code>UMR = `L`/ gcd(v<sub>i</sub>)</code> with `gcd` being the greatest common divisor.\
- Else, if <code>l &and; v &isin; &#8474;</code>, `lcm` resp. `gdc` are extended to rational numbers.
- Else, if <code>l &and; v &isin; &#8477; \ &#8474;</code>, `l` and `v` are approximated by rational numbers
  with a fixed length of decimal digits.

### __Coloring and Averaging__
The fringe pattern sequence `I` can be colorized by setting the hue `h` to any RGB color tuple
in the interval `[0, 255]`. However, black `(0, 0, 0)` is not allowed. `h` must be in shape `(H, 3)`:\
`H` is the number of hues and can be set directly; 3 is the length of the RGB color tuple.\
The hues `h` can also be set by assigning any combination of the following characters as a string:
- `'r'`: red
- `'g'`: green
- `'b'`: blue
- `'c'`: cyan
- `'m'`: magenta
- `'y'`: yellow
- `'w'`: white

`C` is the number of color channels required for either the set of hues `h`
or [wavelength division multiplexing](#multiplexing).
For example, if all hues are monochromatic, i.e. the RGB values are identical for each hue, `C` equals 1, else 3.

Repeating hues will be fused by averaging them before decoding.\
`M`is the number of averaged intensity samples and can be set directly.

### __Multiplexing__
The following multiplexing methods can be activated by setting them to `True`:
- `SDM`: Spatial Division Multiplexing [[2]](#2)\
  This results in crossed fringe patterns. The amplitude `B` is halved.\
  It can only be activated if we have two directions <code>D &equiv; 2</code>.\
  The number of frames `T` is reduced by a factor of 2.
- `WDM`: Wavelength Divison Multiplexing [[3]](#3)\
  All shifts `N`must equal 3. Then, the shifts are multiplexed in the color channel,
  resulting in an RGB fringe pattern.\
  The number of frames `T` is reduced by a factor of 3.
- `FDM`: Frequency Division Multiplexing [[4]](#4), [[5]](#5)\
  Here, the directions `D` and the sets `K`are multiplexed.
  Hence, the amplitude `B` is reduced by a factor of `D` * `K`.\
  It can only be activated if <code>D &or; K > 1</code> i.e. `D * K > 1`.\
  This results in crossed fringe patterns if <code>D &equiv; 2</code>.\
  Each set per direction receives an individual temporal frequency `f`,
  which is used in [temporal demodulation](#temporal-demodulation) to distinguish the individual sets.\
  A minimal number of shifts <code>N<sub>min</sub> &ge; &LeftCeiling;</sub> 2 * f<sub>max</sub> + 1 &RightCeiling;</code>
  is required to satisfy the sampling theorem and `N` is updated automatically if necessary.\
  If one wants a static pattern, i.e. one that remains congruent when shifted, set `static` to `True`.

`SDM`and `WDM`can be used together [[6]](#6) (reducing `T` by a factor of `2 * 3 = 6`), `FDM` with neighter.

By default, the aforementioned multiplexing methods are deactivated,
so we then only have `TDM`: Time Divison Multiplexing.

### __Data Type__
`dtype` denotes the Amplitudes of the fringe pattern sequence `I`.\
Possible values are:
- `'bool'`
- `'uint8'` (the default)
- `'uint16'`
- `'float32'`
- `'float64'`

The total number of bytes `nbytes` consumed by the fringe pattern sequence
as well as its maximum gray value `Imax` are derived directly from it:\
`Imax = 1` for `float` and `bool`,
and <code>Imax = 2<sup>Q</sup> - 1</code> for `unsigned integers` with `Q` bits.

`Imax` in turn limits the offset `A` and the amplitude `B`.
The fringe visibility (also called fringe contrast) is `V = A / B` with <code>V &isin; [0, 1]</code>.

The quantization step size `q` is also derived from `dtype`:
`q = 1` for `bool` and `Q`-bit `unsigned integers`, 
and for `float` its corresponding [resolution](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html).

The standard deviation of the quantization noise  is <code>QN = q / &radic; 12</code>.

### Unwrapping
- `PU` denotes the phase unwrapping method and is eihter `'none'`, `'temporal'`, `'spatial'` or `'SSB'`.
See [spatial demodulation](#spatial-demodulation--phase-unwrapping--pu-) for more details.
- `mode` denotes the mode used for [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-).
  Choose either `'fast'` (the default) or `'precise'`.
- `Vmin` denotes the minimal fringe visibility (fringe contrast) required for the measurement to be valid
and is in the interval `[0, 1]`. During decoding, pixels with less are discarded, which can spead up the computation.
- `verbose`can be set to `True` to also receive the wrapped phase map `φ`,
the fringe orders `k` and the residuals `r` from decoding.
- `SSB` denotes **Single Sideband Demodulation** [[17]](#17-takeda) and is deployed
if <code>K &equiv; H &equiv; N &equiv; 1</code>, i.e. <code>T &equiv; 1</code>
and the [coordinate system](#coordinate-system) is eighter `'image'` or `'Cartesian'`.

### __Quality Metrics__
`eta` denotes the coding efficiency `L / UMR`. It makes no sense to choose `UMR` much larger than `L`,
because then a significant part of the coding range is not used.

`u` denotes the minimum possible uncertainty of the measurement in pixels.
It is based on the phase noise model from [[7]](#7),
propagated through the [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-) and converted from phase to pixel units.
It is influenced by the fringe parameters
- `M`: number of [averaged](#coloring-and-averaging) intensity samples
- `N`: number of phase shifts
- `l`: wavelength of the fringes
- `B`: measured amplitude

and the measurement hardware-specific noise sources [[8]](#8), [[9]](#9)
- `PN`: photon noise of light itself
- `DN`: dark noise of the used camera
- `QN`: quantization noise of the light source or camera

The maximum possible dynamic range of the measurement is `DR = UMR / u`.
It describes how many points can be discriminated on the interval `[0, UMR]`.
It remains constant if `L` resp. `l` are scaled (the scaling factor cancels out).

## Methods
- `load(fname)`\
  Load a parameter set from the file `fname` to a `Fringes` instance.
- `save(fname)`\
  Save the parameter set of the current `Fringes` instance to the file `fname`.
  If `fname` is not provided, the default is `params.yaml` within the package's directory 'fringes'.
- `reset()`\
  Reset the parameter set of the current `Fringes` instance to the default values.
- `auto(T)`\
  Automatically set the [optimal parameters](#optimal-coding-strategy) based on the argument `T` (number of frames).
  This takes also into account the minimum resolvable wavelength `lmin` and the length of the fringe patterns `L`.
- `setMTF(B)`\
  Compute the normalized modulation transfer function at spatial frequencies v
  and use the result to set the optimal `lmin`.
  `B` is the modulation from decoding. For more details, see [Optimal Coding Strategy](#optimal-coding-strategy).
- `coordinates()`\
  Generate the coordinate matrices of the defined [coordinate system](#coordinate-system).
- `encode(frames)`\
  [Encode](#encoding) the fringe pattern sequence `I`.\
  The frames <code>I<sub>t</sub></code> can be encoded indiviually
  by passing the frame indices `frames`, either as an `integer` or a `tuple`.
  The default is `None` in which case all frames are encoded.\
  To receive the frames iteratively (i.e. in a lazy manner), simply iterate over the instance.\
- `decode(I, verbose)`\
  [Decode](#decoding) the fringe pattern sequence `I`.\
  If either the argument `verbose` or the attribute with the same name is `True`,
  additional infomation is computed and retuned: phase maps `φ`, residuals `r` and fringe orders `k`.\
  If the argument `denoise` is `True`, the unwrapped phase map will be smoothened using a bilateral filter
  which is edge-preserving.\
  If the argument `denspike` is `True`, single pixel outliers in the unwrapped phase map
  will be replaced by their local neighbourhood.
- `remap(reg, mod)`\
  Mapping decoded coordinates `reg` i.e. `ξ` (having sub-pixel accuracy)
  from camera grid to (integer) positions on pattern/screen grid
  with weights from modulation `mod` i.e. `B`.
  Default for `mod` is `None`, in which case all weights are assumed to equal one.
  This yields a grid representing the screen (light source)
  with the pixel values being a relative measure
  of how much a screen (light source) pixel contributed
  to the exposure of the camera sensor.
- `deinterlace(I)`\
  Deinterlace a fringe pattern sequence `I` acquired with a line scan camera
  while each frame has been displayed and captured
  while the object has been moved by one pixel.

The next methods are class-methods:
- `unwrap(phi)`\
  [Unwrap](#spatial-phase-unwrapping--spu-) the phase map `phi` i.e. `φ` spacially.

The next methods are package-methods:
- `vshape(I)`\
  Transforms video data of arbitrary shape and dimensionality into the standardized shape `(T, Y, X, C)`, where
  `T` is number of frames, `Y` is height, `X` is width, and `C` is number of color channels.
  Ensures that the array becomes 4-dimensional and that the size of the last dimension,
  i.e. the number of color channels <code>C &isin; {1; 3; 4}</code>. To do this, leading dimensions may be flattened.
- `curvature(registration)`\
  Returns a curvature map. 
- `relief(curvature)`\
  Local height map by local integration via an inverse laplace filter.

## __Optimal Coding Strategy__
As makes sense intuitively, more sets `K` as well as more shifts `N` per set reduce the uncertainty `u` after decoding.
A minimum of 3 shifts is needed to solve for the 3 unknowns brightness `A`, modulation `B` and coordinates `ξ`.
Any additional 2 shifts compensate for one harmonic of the recorded fringe pattern.
Therefore, higher accuracy can be achieved using more shifts `N`, but the time required to capture them 
sets a practical upper limit to the feasible number of shifts.

Generally, shorter wavelengths `l` (or equivalently more periods `v`) reduce the uncertainty `u`,
but the resolution of the camera and the display must resolve the fringe pattern spatially.
Hence, the used hardware imposes a lower bound for the wavelength (or upper bound for the number of periods).
   
Also, small wavelengths might result in a smaller unambiguous measurement range `UMR`.
If two or more sets `K` are used and their wavelengths `l` resp. number of periods `V` are relative primes,
the unmbiguous measurement range can be increased many times.
As a consequence, one can use much smaller wavelenghts `l` (larger number of periods `v`).
However, it must be assured that the unambiguous measurment range is always equal or larger than both,
the width `X` and the height `Y`.
Else, [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-) will yield wrong results and instead
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
-  Equivalently, `l` can also be set to `'auto'`. This will automatically determine the optimal integer set `l`
  based on the minimal resolvable wavelength `lmin = L / vmax`.
- `T` can be set directly, based on the desired acquisition time.
  The optimal `K`, `N` and the [multiplexing](#multiplexing) methods will be determined automatically.

Alternatively, simply use the function `auto()`
to automatically set the optimal `v`, `T` and [multiplexing](#multiplexing) methods.

## Troubleshooting
<!---
- __`poetry install` does not work__
  
  First, ensure that poetry is installed correctly as descibed onthe [Poetry Website](https://python-poetry.org/docs/).\
  Secondly, ensure the correct python version is installed on your system, as specified in the file `pyproject.toml`!\
  Thirdly, this can be caused by a proxy which `pip` does not handle correctly.
  Manually setting the proxy in the Windows settings or even adding a system variable 
`https_proxy = http://YOUR_PROXY:PORT` can resolve this.
--->

- __Encoding/Decoding takes a long time__
  
  This is probably related to the just-in-time compiler [Numba](https://numba.pydata.org/) 
  used for computationally expensive functions:
  During the first execution of any function decorated with it, an initial compilation is executed. 
  This can take several tens of seconds up to single digit minutes, depending on your CPU.
  However, for any subsequent execution, the compiled code is buffered and the code of the function runs much faster, 
  approaching the speeds of code written in C.


- __My decoded coordinates show lots of noise__

  Try using more, sets `K` and/or shifts `N` and adjust the used wavelengths `l` resp. number of periods `v`.\
  Also, make sure the exposure of your camera is adjusted so that the fringe patterns show up with maximum contrast.\
  Try to avoid under- and overexposure during acquisition.


- #### My decoded coordinates show systematic offsets
  First, ensure that the correct frame was captured while acquiring the fringe pattern sequence.
  If the timings are not set correctly, the sequence may be a frame off.\
  Secondly, this might occur if either the camera or the display used have a gamma value very different from 1.
  Typical screens have a gamma value of 2.2;   therefore compensate by setting the inverse value
  <code>gamma<sup>-1</sup> = 1 / 2.2 &approx; 0.45</code> to the `gamma`attribute of the `Fringes` instance.
  Alternatively, change the gamma value of the light source or camera directly.
  You might also use more shifts `N` to compensate for the dominant harmonics of the gamma-nonlinearities.

## References

#### [11] [Burke 2002]
[J. Burke, T. Bothe, W. Osten, and C. Hess,
“Reverse engineering by fringe projection,”
in Interferometry XI: Applications (W. Osten, ed.), vol. 4778, pp. 312–324, SPIE,
2002.](https://doi.org/10.1117/12.473547)

#### [1] [Burke 2022] 
[J. Burke, A. Pak, S. Höfer, M. Ziebarth, M. Roschani, J. Beyerer,
"Deflectometry for specular surfaces: an overview",
arXiv:2204.11592v1 [physics.optics],
2022.](https://arxiv.org/abs/2204.11592)

#### [2] [Park2008]?

#### [3] [Huang 1999]

#### [4] [Liu 2014] [Liu2010] [Park2008]?

#### [5] [Kludt 2018]

#### [6] [Trumper 2016]

#### [7] [Surrel 1997]

#### [8] [EMVA1288]

#### [9] [Bothe2008]

#### [10] [Fischer]

#### [13] [Burke 2012]
[J. Burke,
"Phase Decoding and Reconstruction",
vol. Optical Methods for Solid Mechanics: A Full-Field Approach, ch. 3, pp. 83–141. Wiley, Weinheim,
2012.](https://www.wiley.com/en-us/Optical+Methods+for+Solid+Mechanics%3A+A+Full+Field+Approach-p-9783527411115)

#### [14] [Kludt 2024]

#### [15] [Herráez 2002]
[Miguel Arevallilo Herráaez, David R. Burton, Michael J. Lalor, and Munther A. Gdeisat.
Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path.
Appl. Opt., 41(35):7437-7444,
Dec 2002.](https://opg.optica.org/ao/abstract.cfm?uri=ao-41-35-7437)

#### [16] [Lei 2015]
[Hai Lei, Xin yu Chang, Fei Wang, Xiao-Tang Hu, and Xiao-Dong Hu.
A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping.
Optik - International Journal for Light and Electron Optics, 126(18):1640 - 1644,
2015.](https://www.sciencedirect.com/science/article/abs/pii/S0030402615003228?via%3Dihub)

#### [17] [Takeda]

#### [18] [Bothe 2008]

#### [19]
Inverse Laplace Filter

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
