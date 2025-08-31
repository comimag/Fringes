.. default-role:: math

Parameters
==========
All parameters are implemented as
`properties <https://docs.python.org/3/library/functions.html#property>`_ (managed attributes).
They are parsed when set, so usually several input types are accepted,
e.g. ``bool``, ``int``, ``float`` for `numbers <https://docs.python.org/3.13/library/numbers.html#module-numbers>`_
and additionally ``list``, ``tuple``, ``numpy.ndarray`` for `sequences <https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence>`_.

Note that some attributes have sub-dependencies,
hence dependent attributes might change as well.
Circular dependencies are resolved automatically.

.. (cf. :numref:`interdependencies`),
   .. _interdependencies:
   .. figure:: params/interdependencies.svg
       :align: center
       :alt: interdependencies

    Parameters and their interdependencies.

Video Shape
-----------
:attr:`~fringes.fringes.Fringes.shape` is the standardized shape (
:attr:`~fringes.fringes.Fringes.T`, :attr:`~fringes.fringes.Fringes.Y`,
:attr:`~fringes.fringes.Fringes.X`, :attr:`~fringes.fringes.Fringes.C`) of the fringe pattern sequence, with

- :attr:`~fringes.fringes.Fringes.T`: number of frames
- :attr:`~fringes.fringes.Fringes.Y`: height (in pixel units)
- :attr:`~fringes.fringes.Fringes.X`: width (in pixel units)
- :attr:`~fringes.fringes.Fringes.C`: number of color channels
  (depends on the :ref:`coloring <coloring and averaging>` and `multiplexing`_ schemes activated).

.. :attr:`~fringes.fringes.Fringes.T` = :attr:`~fringes.fringes.Fringes.H` `\cdot \sum` :attr:`~fringes.fringes.Fringes.N`.
   If a `multiplexing`_ scheme is activated, :attr:`~fringes.fringes.Fringes.T` reduces further.

:attr:`~fringes.fringes.Fringes.L` is the maximum of :attr:`~fringes.fringes.Fringes.X` and :attr:`~fringes.fringes.Fringes.Y` and denotes the length (in pixel units) to be encoded.
It can be extended by the factor :attr:`~fringes.fringes.Fringes.a` to :attr:`~fringes.fringes.Fringes.Lext`.

.. :attr:`~fringes.fringes.Fringes.size` is the product of :attr:`~fringes.fringes.Fringes.shape`.

Coordinate System
------------------
The following coordinate systems can be used by setting :attr:`~fringes.fringes.Fringes.grid` to:

- ``'image'``: The top left corner pixel of the grid is the origin (0, 0) and positive directions are right- resp. downwards.

.. - ``'Cartesian'``: The center of grid is the origin (0, 0) and positive directions are right- resp. upwards.
   - ``'polar'``: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.
   - ``'log-polar'``: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.
   - ``'spiral'``: The origin (0, 0) resides in the center of the generated grid and positive directions are counterclockwise, cf. [Klu18]_.

:attr:`~fringes.fringes.Fringes.indexing` denotes the indexing convention.
Possible values are:

- ``'xy'``: *Cartesian indexing* will index the row first;
- ``'ij'``: *matrix indexing* will index the colum first.

:attr:`~fringes.fringes.Fringes.D` denotes the number of directions to be encoded.

:attr:`~fringes.fringes.Fringes.axis` is used to define along which axis of the coordinate system (index 0 or 1)
the fringe pattern is modulated and shifted if :attr:`~fringes.fringes.Fringes.D` = 1.

..
    :attr:`~fringes.fringes.Fringes.angle` can be used to tilt the coordinate system. The origin remains the same.

Set
---
Each set consists of the following attributes:

.. (cf. black box in :numref:`interdependencies`):

- :attr:`~fringes.fringes.Fringes.N`: number of shifts
- :attr:`~fringes.fringes.Fringes.l`: wavelength (in pixel units)
- :attr:`~fringes.fringes.Fringes.v`: spatial frequency, i.e. number of periods (per screen length :attr:`~fringes.fringes.Fringes.L`)
- :attr:`~fringes.fringes.Fringes.f`: temporal frequency, i.e. number of periods to shift over

Each is an array with shape (number of directions :attr:`~fringes.fringes.Fringes.D`, number of sets :attr:`~fringes.fringes.Fringes.K`).
For example, if :attr:`~fringes.fringes.Fringes.N`.shape = (2, 3), it means that we encode :attr:`~fringes.fringes.Fringes.D` = 2 directions with :attr:`~fringes.fringes.Fringes.K` = 3 sets each.
Changing :attr:`~fringes.fringes.Fringes.D` or :attr:`~fringes.fringes.Fringes.K` directly, changes the shape of all set attributes.
When setting a set attribute with a new shape (:attr:`~fringes.fringes.Fringes.D`', :attr:`~fringes.fringes.Fringes.K`'),
:attr:`~fringes.fringes.Fringes.D` and :attr:`~fringes.fringes.Fringes.K` are updated as well as the shape of the other set attributes.

If a set attribute is 1D, then it is stacked to match the number of directions :attr:`~fringes.fringes.Fringes.D`.

If a set attribute is 0D, i.e. a scalar, then all values are simply replaced by the new one.

:attr:`~fringes.fringes.Fringes.l` and :attr:`~fringes.fringes.Fringes.v` are related by :attr:`~fringes.fringes.Fringes.l` = :attr:`~fringes.fringes.Fringes.L` / :attr:`~fringes.fringes.Fringes.v`.
When :attr:`~fringes.fringes.Fringes.L` changes, :attr:`~fringes.fringes.Fringes.v` is kept constant and only :attr:`~fringes.fringes.Fringes.l` is changed.

Usually :attr:`~fringes.fringes.Fringes.f` = 1 and is essentially only changed if :ref:`frequency division multiplexing <multiplexing>` :attr:`~fringes.fringes.Fringes.FDM` is activated.

:attr:`~fringes.fringes.Fringes.reverse` is a boolean which reverses the direction of the shifts (by multiplying :attr:`~fringes.fringes.Fringes.f` with -1).

:attr:`~fringes.fringes.Fringes.p0` denotes the phase offset, which can be used to
e.g. let the fringe patterns start (at the origin) with a gray value of zero.

Intensity Values
----------------
:attr:`~fringes.fringes.Fringes.bits` is the number of bits.

:attr:`~fringes.fringes.Fringes.dtype` denotes the data type which can hold `2` ** :attr:`~fringes.fringes.Fringes.bits` of information.
Possible values are:

- ``'uint8'`` (default)
- ``'uint16'``
- ``'float32'``
- ``'float64'``

.. :attr:`~fringes.fringes.Fringes.nbytes` is the total bytes consumed by fringe pattern sequence.

.. :attr:`~fringes.fringes.Fringes.q` is the quantization step size and equals 1 for ``bool``, `2^r` for r-bit ``unsigned integers``,
   and for ``float`` its corresponding `resolution <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>`_.

.. :attr:`~fringes.fringes.Fringes.q` is the quantization step size and equals `2^r` for r-bit ``unsigned integers``
   and for ``float`` its corresponding `resolution <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>`_.

.. :attr:`~fringes.fringes.Fringes.Imax` is the maximum gray value and equals 1 for ``float`` and ``bool``,
   and `2^r - 1` for ``unsigned integers`` with r bits.

:attr:`~fringes.fringes.Fringes.Imax` is the maximum gray value and equals 1 for ``float``
and `2` ** :attr:`~fringes.fringes.Fringes.bits` for ``unsigned integers``.

:attr:`~fringes.fringes.Fringes.A` is the offset, also called brightness (of the background).
It is limited by :attr:`~fringes.fringes.Fringes.Imax`.

:attr:`~fringes.fringes.Fringes.B` is the amplitude of the cosinusoidal fringes.
It is limited by :attr:`~fringes.fringes.Fringes.Imax`.

:attr:`~fringes.fringes.Fringes.V` is the fringe :ref:`visibility <visibility and Exposure>` (also called fringe contrast).
:attr:`~fringes.fringes.Fringes.V` = :attr:`~fringes.fringes.Fringes.A` / :attr:`~fringes.fringes.Fringes.B`, with :attr:`~fringes.fringes.Fringes.V` `\in [0, 1]`.

:attr:`~fringes.fringes.Fringes.E` is the :ref:`exposure <visibility and Exposure>` (relative brightness) and is within the range `[0, 1]`.

:attr:`~fringes.fringes.Fringes.g` denotes the gamma correction factor and can be used to compensate non-linearities of the display response curve.

Coloring and Averaging
----------------------
The fringe patterns can be colorized by setting the hue :attr:`~fringes.fringes.Fringes.h` to any RGB color triplet within the interval [0, 255].
However, black (0, 0, 0) is not allowed.
:attr:`~fringes.fringes.Fringes.h` must be in shape (:attr:`~fringes.fringes.Fringes.H`, 3),
where :attr:`~fringes.fringes.Fringes.H` is the number of hues and 3 is the length of the RGB color triplet.

The hues :attr:`~fringes.fringes.Fringes.h` can also be set by assigning any combination of the following characters as a string:

- ``'r'``: red
- ``'g'``: green
- ``'b'``: blue
- ``'c'``: cyan
- ``'m'``: magenta
- ``'y'``: yellow
- ``'w'``: white

:attr:`~fringes.fringes.Fringes.C` is the number of color channels required for either the set of hues :attr:`~fringes.fringes.Fringes.h`
or :ref:`wavelength division multiplexing <multiplexing>`.
For example, if all hues are monochromatic, i.e. the RGB values are identical for each hue, :attr:`~fringes.fringes.Fringes.C` equals 1, else 3.

Repeating hues will be fused by averaging them before decoding.

:attr:`~fringes.fringes.Fringes.M` is the number of averaged intensity samples.

Multiplexing
------------
The following multiplexing methods can be activated by setting them to ``True``:

- :attr:`~fringes.fringes.Fringes.SDM`: Spatial Division Multiplexing

  This results in crossed fringe patterns.
  It can only be activated if we have two directions :attr:`~fringes.fringes.Fringes.D` = 2.
  The number of frames :attr:`~fringes.fringes.Fringes.T` is reduced by a factor of 2.

- :attr:`~fringes.fringes.Fringes.WDM`: Wavelength Division Multiplexing

  The shifts are multiplexed into the color channel, resulting in an RGB fringe pattern.
  All shifts :attr:`~fringes.fringes.Fringes.N` must equal 3.
  The number of frames :attr:`~fringes.fringes.Fringes.T` is reduced by a factor of 3.

- :attr:`~fringes.fringes.Fringes.FDM`: Frequency Division Multiplexing

  Here, the directions :attr:`~fringes.fringes.Fringes.D` and the sets :attr:`~fringes.fringes.Fringes.K` are multiplexed.
  This results in crossed fringe patterns if :attr:`~fringes.fringes.Fringes.D` = 2.
  It can only be activated if :attr:`~fringes.fringes.Fringes.D` > 1 or :attr:`~fringes.fringes.Fringes.K` > 1.
  If one wants a static pattern, i.e. one that remains congruent when shifted, set :attr:`~fringes.fringes.Fringes.static` to ``True``.

:attr:`~fringes.fringes.Fringes.SDM` and :attr:`~fringes.fringes.Fringes.WDM` can be used together
(reducing :attr:`~fringes.fringes.Fringes.T` by a factor of 2 * 3 = 6),
:attr:`~fringes.fringes.Fringes.FDM` with neither.

For more details, please refer to :doc:`Multiplex <mux>`.

.. Unwrapping
   ----------

   :attr:`~fringes.fringes.Fringes.uwr` denotes the phase unwrapping method and is eihter ``'none'``, ``'temporal'``, ``'spatial'`` or ``'FTM'``.
   See :ref:`unwrapping <uwr>` for more details.

.. :attr:`~fringes.fringes.Fringes.mode` denotes the mode used for [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-).
   Choose either ``'fast'`` (the default) or ``'precise'``.

Quality Metrics
---------------

:attr:`~fringes.fringes.Fringes.UMR` denotes the unambiguous measurement range.
The coding is only unique within the interval [0, :attr:`~fringes.fringes.Fringes.UMR`); after that it repeats itself.

The :attr:`~fringes.fringes.Fringes.UMR` is derived from :attr:`~fringes.fringes.Fringes.l` and :attr:`~fringes.fringes.Fringes.v`:

- If :attr:`~fringes.fringes.Fringes.l` `\in \mathbb{N}`, :attr:`~fringes.fringes.Fringes.UMR` = `lcm(` :attr:`~fringes.fringes.Fringes.l` `)`, with `lcm` being the least common multiple.
- Else, if :attr:`~fringes.fringes.Fringes.v` `\in \mathbb{N}`, :attr:`~fringes.fringes.Fringes.UMR` = :attr:`~fringes.fringes.Fringes.L` / `gcd(` :attr:`~fringes.fringes.Fringes.v` `)`, with `gcd` being the greatest common divisor.
- Else, if :attr:`~fringes.fringes.Fringes.v` `\lor` :attr:`~fringes.fringes.Fringes.l` `\in \mathbb{Q}` , `lcm` resp. `gcd` are extended to rational numbers.
- Else, if :attr:`~fringes.fringes.Fringes.v` `\land` :attr:`~fringes.fringes.Fringes.l` `\in \mathbb{R} \setminus \mathbb{Q}` , :attr:`~fringes.fringes.Fringes.UMR` = `prod(` :attr:`~fringes.fringes.Fringes.l` `)`, with `prod` being the product operator.

.. :attr:`~fringes.fringes.Fringes.u` denotes the minimum possible uncertainty of the measurement in pixels.
   It is based on the phase noise model from [Sur97]_
   and propagated through the unwrapping process and the phase fusion.
   It is influenced by the parameters

   - :attr:`~fringes.fringes.Fringes.M`: number of averaged intensity samples,
   - :attr:`~fringes.fringes.Fringes.N`: number of phase shifts,
   - :attr:`~fringes.fringes.Fringes.l`: wavelengths,
   - `\hat{B}`: measured modulation and
   - `\hat{u_I}`: intensity noise (caused by the measurement hardware [EMV]_, [Bot08]_).

   .. - :attr:`~fringes.fringes.Fringes.quant`: quantization noise of the light source or camera,
      - :attr:`~fringes.fringes.Fringes.dark`: dark noise of the used camera,
      - :attr:`~fringes.fringes.Fringes.shot`: photon noise of light itself,
      - :attr:`~fringes.fringes.Fringes.gain`: system gain of the used camera.

   :attr:`~fringes.fringes.Fringes.SNR` = :attr:`~fringes.fringes.Fringes.L` / :attr:`~fringes.fringes.Fringes.u`
   is the signal-to-noise ratio of the phase shift coding
   and is a measure of how many points can be distinguished within the screen length [0, :attr:`~fringes.fringes.Fringes.L`).
   It remains constant if :attr:`~fringes.fringes.Fringes.L` and hence :attr:`~fringes.fringes.Fringes.l` is scaled (the scaling factor cancels out).

   :attr:`~fringes.fringes.Fringes.DR` = :attr:`~fringes.fringes.Fringes.UMR` / :attr:`~fringes.fringes.Fringes.u`
   is the dynamic range of the phase shift coding
   and is a measure of how many points can be distinguished within the unambiguous measurement range `[0,` :attr:`~fringes.fringes.Fringes.UMR` `)`.
   Again, it remains constant if :attr:`~fringes.fringes.Fringes.L` and hence :attr:`~fringes.fringes.Fringes.l` is scaled (the scaling factor cancels out).

:attr:`~fringes.fringes.Fringes.eta` = :attr:`~fringes.fringes.Fringes.L` / :attr:`~fringes.fringes.Fringes.UMR`
is the spatial coding efficiency
and is a measure of how well the coding range :attr:`~fringes.fringes.Fringes.UMR` fits the screen length :attr:`~fringes.fringes.Fringes.L`.
It makes no sense to choose :attr:`~fringes.fringes.Fringes.UMR` much larger than :attr:`~fringes.fringes.Fringes.L`,
because then a significant part of the coding range remains unused.

.. :attr:`~fringes.fringes.Fringes.eta_temp` = :attr:`~fringes.fringes.Fringes.SNR` / :attr:`~fringes.fringes.Fringes.T`
   is the temporal coding efficiency
   and is a measure of how many code words, i.e. frames :attr:`~fringes.fringes.Fringes.T`,
   can distinguish how many screen points :attr:`~fringes.fringes.Fringes.SNR`.

.. [Bot08]
   `Bothe,
   "Grundlegende Untersuchungen zur Formerfassung mit einem neuartigen Prinzip der Streifenprojektion und Realisierung in einer kompakten 3D-Kamera",
   Dissertation,
   ISBN 978-3-933762-24-5,
   BIAS Bremen,
   2008.
   <https://www.amazon.de/Grundlegende-Untersuchungen-Formerfassung-Streifenprojektion-Strahltechnik/dp/3933762243/ref=sr_1_2?qid=1691575452&refinements=p_27%3AThorsten+B%C3%B6th&s=books&sr=1-2>`_

.. [EMV]
   `EMVA,
   "Standard for Characterization of Image Sensors and Cameras Release 4.0 Linear",
   European Machine Vision Association,
   2021.
   <https://www.emva.org/standards-technology/emva-1288/emva-standard-1288-downloads-2/>`_

.. [Klu18]
   `Kludt and Burke,
   "Coding strategies for static patterns suitable for UV deflectometry",
   Forum Bildverarbeitung 2018,
   2018.
   <https://publikationen.bibliothek.kit.edu/1000088264>`_

.. [Sur97]
   `Surrel,
   "Additive noise effect in digital phase detection",
   Applied Optics,
   1997.
   <https://doi.org/10.1364/AO.36.000271>`_
