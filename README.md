# Fringes
[![PyPI](https://img.shields.io/pypi/v/fringes)](https://pypi.org/project/fringes/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fringes)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Read the Docs](https://img.shields.io/readthedocs/fringes)](https://fringes.readthedocs.io)
[![PyPI - License](https://img.shields.io/pypi/l/fringes)](https://github.com/comimag/fringes/blob/main/LICENSE.txt)
[![Static Badge](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10936353-blue)](https://zenodo.org/doi/10.5281/zenodo.10936353)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fringes)](https://pypistats.org/packages/fringes)

<!---
[![PyPI - Downloads](https://img.shields.io/pypi/dm/fringes)](https://pypistats.org/packages/fringes)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/comimag/fringes/python-package.yml)
![GitHub top language](https://img.shields.io/github/languages/top/comimag/fringes)
![https://img.shields.io/badge/python-%3E=3.9-blue](https://img.shields.io/badge/python-%3E=3.9-blue)
![GitHub](https://img.shields.io/github/license/comimag/fringes)
[![Downloads](https://static.pepy.tech/badge/fringes)](https://pepy.tech/project/fringes)
--->

<!---
link to  paper, please cite
--->

Easily create customized fringe patterns
and analyse them using phase shifting algorithms.

![coding-cheme](https://raw.githubusercontent.com/comimag/fringes/main/docs/source/01_start/coding-scheme.gif)\
Figure 1: Phase Shifting Coding Scheme.

## Features
- [Create](https://fringes.readthedocs.io/en/main/02_tutorial/fundamentals.html#encoding) and
  [analyze](https://fringes.readthedocs.io/en/main/02_tutorial/fundamentals.html#decoding)
  customized fringe patterns
- [Parameterize](https://fringes.readthedocs.io/en/main/02_tutorial/params.html) the phase shifting algorithm
- Generalized Temporal Phase Unwrapping (GTPU)
- [Multiplexing](https://fringes.readthedocs.io/en/main/02_tutorial/mux.html) fringe patterns
- [Filtering](https://fringes.readthedocs.io/en/main/02_tutorial/filter.html) methods

<!---
todo: add reference to GTPU-paper
- Uncertainty Propagation
- [Optimal Coding Strategy](https://fringes.readthedocs.io/en/main/user_guide/optimal.html)
--->

## Installation
You can install `fringes` directly from [PyPi](https://pypi.org/) via `pip`:

```
pip install fringes
```

## Usage
You instantiate, parameterize and deploy the `Fringes` class:

```python

from fringes import Fringes

f = Fringes()              # instantiate class
```

All [parameters](https://fringes.readthedocs.io/en/main/02_tutorial/params.html)
are accessible as class properties (managed attributes) of the `Fringes` instance.

```python
f.X = 1920                 # set width of the fringe patterns
f.Y = 1080                 # set height of the fringe patterns
f.K = 2                    # set number of sets
f.N = 4                    # set number of shifts
f.v = [9, 10]              # set spatial frequencies
T = f.T                    # get number of frames
```

For generating the fringe pattern sequence `I`, use the method `encode()`.
It returns a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) in video-shape (frames, height, width, color channels).

```python
I = f.encode()             # encode fringe patterns
```

Now display each frame of the fringe pattern sequence on a screen and capture the scene with a camera
according to the following pseudocode
(a minimal working example is depicted
[here](https://fringes.readthedocs.io/en/main/01_start/usage.html#minimal-working-example)):

```python
# allocate image stack
I_rec = []

for t in range(f.T):
    # display frame on screen
    frame = I[t]
    ...

    # capture scene with camera
    image = ...
    
    # append to image stack
    I_rec.append(image)
```

For analyzing (recorded) fringe patterns, use the method `decode()`.
It returns the Numpy arrays brightness `a`, modulation `b` and coordinate `x`.

```python
a, b, x = f.decode(I_rec)  # decode fringe patterns
```

> Note:\
For the computationally expensive ``decode()``-function
we make use of the just-in-time compiler [Numba](https://numba.pydata.org/).
During the first execution, an initial compilation is executed.
This can take several tens of seconds up to single digit minutes, depending on your CPU and energy settings.
However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster,
approaching the speeds of code written in C.

A minimal working example - including image recording - can be found in the
[documentation](https://fringes.readthedocs.io/en/main/01_start/usage.html#minimal-working-example).

## Graphical User Interface
<!---
Do you need a GUI? `Fringes` has a sister project which is called `Fringes-GUI`:
--->
https://pypi.org/project/fringes-gui/

## Documentation
https://fringes.readthedocs.io

## License
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](
https://github.com/comimag/Fringes/blob/main/LICENSE.txt)

## Citation
If you use this software, please cite it using this DOI:
[10.5281/zenodo.10936353](https://zenodo.org/doi/10.5281/zenodo.10936353)\
This DOI represents all versions, i.e. the concept of this software package,
and will always resolve to the latest one.

If you want to cite a specific version,
please choose the respective DOI from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10936353).

## Support
I was looking for a user-friendly tool to configure,
encode and decode customized fringe patterns with phase shifting algorithms.
Since I couldn't find any, I started developing one myself.
It is intended for non-commercial, academic and educational use.

However, I do this entirely in my free time.
If you like this package and can make use of it, I would be happy about a donation.
It will help me keep it up-to-date and add more features in the future.

<!---
[![Liberapay](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/comimag/donate/)
[![](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=EHBGZ229DKUC4)
--->

[![paypal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=PayPal&logoColor=white)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=EHBGZ229DKUC4)

Thank you!
