# Fringes

[//]: # (<img src="https://raw.githubusercontent.com/comimag/fringes/main/docs/source/01_start/coding-scheme.gif" alt="drawing" width="500"/>)

![coding-cheme](https://raw.githubusercontent.com/comimag/fringes/main/docs/source/01_start/coding-scheme.gif)

[![PyPI](https://img.shields.io/pypi/v/fringes)](https://pypi.org/project/fringes/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Read the Docs](https://img.shields.io/readthedocs/fringes)](https://fringes.readthedocs.io)
[![PyPI - License](https://img.shields.io/pypi/l/fringes)](https://github.com/comimag/fringes/blob/main/LICENSE.txt)
[![Static Badge](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.10936353-blue)](https://zenodo.org/doi/10.5281/zenodo.10936353)

[//]: # (![PyPI - Python Version]&#40;https://img.shields.io/pypi/pyversions/fringes&#41;)
[//]: # (![GitHub Actions Workflow Status]&#40;https://img.shields.io/github/actions/workflow/status/comimag/fringes/python-package.yml&#41;)
[//]: # (![GitHub]&#40;https://img.shields.io/github/license/comimag/fringes&#41;)
[//]: # ([![PyPI - Downloads]&#40;https://img.shields.io/pypi/dm/fringes&#41;]&#40;https://pypistats.org/packages/fringes&#41;)
[//]: # ([![Downloads]&#40;https://static.pepy.tech/badge/fringes&#41;]&#40;https://pepy.tech/project/fringes&#41;)

Easily configure, encode and decode sinusoidal fringe patterns
using phase shifting algorithms.

## Features
- [Configure](https://fringes.readthedocs.io/en/main/02_tutorial/params.html) the phase shifting algorithm
- [Encode](https://fringes.readthedocs.io/en/main/02_tutorial/fundamentals.html#encoding) and
  [decode](https://fringes.readthedocs.io/en/main/02_tutorial/fundamentals.html#decoding)
  fringe patterns
- [Spatial Phase Unwrapping](https://fringes.readthedocs.io/en/main/02_tutorial/fundamentals.html#uwr)
- [Generalized Temporal Phase Unwrapping](https://fringes.readthedocs.io/en/main/02_tutorial/fundamentals.html#uwr) 
- [Multiplexing](https://fringes.readthedocs.io/en/main/02_tutorial/mux.html) fringe patterns
- [Filtering](https://fringes.readthedocs.io/en/main/02_tutorial/filter.html) methods
<!--
todo: add reference to GTPU-paper
add reference to SPU
- Uncertainty Propagation
- [Optimal Coding Strategy](https://fringes.readthedocs.io/en/main/user_guide/optimal.html)
-->

## Installation
You can install `fringes` directly from [PyPi](https://pypi.org/) via `pip`:

```
pip install fringes
```

## Usage
You instantiate, parameterize and deploy the `Fringes` class:

```python

from fringes import Fringes  # import the Fringes class

f = Fringes()                # instantiate Fringes object
```

All [parameters](https://fringes.readthedocs.io/en/main/02_tutorial/params.html)
are accessible as class properties (managed attributes) of the `Fringes` instance.

```python
f.X = 1920                   # set width of the fringe patterns
f.Y = 1080                   # set height of the fringe patterns
f.N = 4                      # set number of shifts
f.v = [9, 10]                # set spatial frequencies
T = f.T                      # get number of frames
```

To encode the fringe pattern sequence `I`, use the method `encode()`.
It returns a [NumPy array](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) in video-shape (frames, height, width, color channels).

```python
I = f.encode()               # encode fringe patterns
```

Now display each frame of the fringe pattern sequence on a screen and capture the scene with a camera
according to the following pseudocode
(a minimal working example is depicted
[here](https://fringes.readthedocs.io/en/main/01_start/usage.html#minimal-working-example)):

```python
# allocate image stack
Irec = []

for t in range(f.T):
    # display frame on screen
    frame = I[t]
    ...

    # capture scene with camera
    image = ...
    
    # append to image stack
    Irec.append(image)
```

To decode (recorded) fringe patterns, use the method `decode()`.
It returns the Numpy arrays
[brightness](https://fringes.readthedocs.io/en/main/02_tutorial/filter.html#brightness) `a`,
[modulation](https://fringes.readthedocs.io/en/main/02_tutorial/filter.html#modulation) `b` and
[(screen) coordinate](https://fringes.readthedocs.io/en/main/02_tutorial/filter.html#coordinate) `x`.

```python
a, b, x = f.decode(Irec)     # decode fringe patterns
```

> Note:\
For the computationally expensive ``decode()``-function
we make use of the just-in-time compiler [Numba](https://numba.pydata.org/).
During the first execution, an initial compilation is executed.
This can take several tens of seconds up to single digit minutes, depending on your CPU and energy settings.
However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster,
approaching the speeds of code written in C.

More complete code examples can be found in the
[examples](https://github.com/comimag/Fringes/tree/main/examples) directory.

<!---
## Citation
If you use this software, please cite it using this DOI:
[10.5281/zenodo.10936353](https://zenodo.org/doi/10.5281/zenodo.10936353)\
This DOI represents all versions, i.e. the concept of this software package,
and will always resolve to the latest one.

If you want to cite a specific version,
please choose the respective DOI from [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10936353).

link to  paper, please cite
--->

## Support
I was looking for a user-friendly tool to configure,
encode and decode customized fringe patterns with phase shifting algorithms.
Since I couldn't find any, I started developing one myself.
It is intended for non-commercial, academic and educational use.

However, I do this entirely in my free time.
If you like this package and can make use of it, I would be happy about a donation.
It will help me keep it up-to-date and add more features in the future.

[//]: # ([![Liberapay]&#40;https://liberapay.com/assets/widgets/donate.svg&#41;]&#40;https://liberapay.com/comimag/donate/&#41;)

[![paypal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=PayPal&logoColor=white)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=EHBGZ229DKUC4)

Thank you!
