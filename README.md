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

<!---
link to  paper, please cite
--->

Easy to use tool for generating and analyzing fringe patterns with phase shifting algorithms.

![Coding Scheme](https://raw.githubusercontent.com/comimag/fringes/main/docs/getting_started/coding-scheme.gif)\
Phase Shift Coding Scheme.

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

## Documentation
The documentation can be found here: https://fringes.readthedocs.io

## Troubleshooting
<!---
- __`poetry install` does not work__  
  First, ensure that poetry is installed correctly as descibed on the [Poetry Website](https://python-poetry.org/docs/).\
  Secondly, ensure the correct python version is installed on your system, as specified in the file `pyproject.toml`!\
  Thirdly, this can be caused by a proxy which `pip` does not handle correctly.
  Manually setting the proxy in the Windows settings
  or even adding a system variable `https_proxy = http://YOUR_PROXY:PORT` can resolve this.
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
