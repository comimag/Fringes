# Fringes
[![PyPI](https://img.shields.io/pypi/v/fringes)](https://pypi.org/project/fringes/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fringes)
[![Read the Docs](https://img.shields.io/readthedocs/fringes)](https://fringes.readthedocs.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI - License](https://img.shields.io/pypi/l/fringes)](https://github.com/comimag/fringes/blob/main/LICENSE.txt)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fringes)](https://pypistats.org/packages/fringes)

<!---
![GitHub top language](https://img.shields.io/github/languages/top/comimag/fringes)
![GitHub issues](https://img.shields.io/github/issues/comimag/fringes)
![GitHub](https://img.shields.io/github/license/comimag/fringes)
--->

<!---
link to  paper, please cite
--->

Easy-to-use tool for parameterizing, generating and analyzing fringe patterns with phase shifting algorithms.

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

All [parameters](https://fringes.readthedocs.io/en/latest/user_guide/params.html)
are accesible by the respective attributes of the `Fringes` instance
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
Do you need a GUI? `Fringes` has a sister project which is called `Fringes-GUI`:
https://pypi.org/project/fringes-gui/

## Documentation
The documentation can be found here:
https://fringes.readthedocs.io

## Troubleshooting
- __Decoding takes a long time__  
  This is most likely due to the just-in-time compiler [Numba](https://numba.pydata.org/), 
  which is used for this computationally expensive function:
  During the first execution, an initial compilation is executed. 
  This can take several tens of seconds up to single digit minutes, depending on your CPU.
  However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster, 
  approaching the speeds of code written in C.

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
