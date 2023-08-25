Usage
=====

This package provides the handy Fringes` class,
which handles all the required parameters
for configuring fringe pattern sequences
and provides methods for fringe analysis.

Command-line use
---------------

You instantiate, parameterize and deploy the `Fringes` class:

.. code-block:: python

    import fringes as frng

    f = frng.Fringes()

    f.X = 1920                  # set width of the fringe patterns
    f.Y = 1080                  # set height of the fringe patterns
    f.K = 2                     # set number of sets
    f.N = 4                     # set number of shifts
    f.v = [9, 10]               # set spatial frequencies
    f.T                         # get number of frames

You can change the `logging level <https://docs.python.org/3/library/logging.html#levels>`_ of a `Fringes` instance.
For example, changing it to `'DEBUG'` gives you verbose feedback on which parameters are changed
and how long functions take to execute.

.. code-block:: python

    f.logger.setLevel("DEBUG")

All parameters are accesible by the respective attributes of the `Fringes` instance
(a glossary of them is obtained by the class attribute `glossary`).
They are implemented as class properties (managed attributes),
which are parsed when setting,
so usually several input types are accepted
(e.g. `bool`, `int`, `float` for scalars
and additionally `list`, `tuple`, `ndarray` for `arrays <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_).
Note that some attributes have subdependencies (cf. next Figure),
hence dependent attributes might change as well.
Circular dependencies are resolved automatically.

.. code-block:: python

    f.X = 1920                  # set width of the fringe patterns
    f.Y = 1080                  # set height of the fringe patterns
    f.K = 2                     # set number of sets
    f.N = 4                     # set number of shifts
    f.v = [9, 10]               # set spatial frequencies

    f.T                         # get number of frames

For generating the fringe pattern sequence `I`, use the method `encode()`.
It returns a `Numpy array <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_
in videoshape (frames `T`, width `X`, height `Y`, color channels `C`).

.. code-block:: python

    I = f.encode()              # encode fringe patterns

For analyzing (recorded) fringe patterns, use the method `decode()`.
It returns the Numpy arrays brightness `A`, modulation `B` and coordinate `x`.

.. code-block:: python

    A, B, x = f.decode(I)       # decode fringe patterns

Graphical User Interface
------------------------

Do you prefer to interact with a GUI?
Fringes has a sister project which is called Fringes-GUI: https://pypi.org/project/fringes-gui/
