.. default-role:: math
.. _coloring: `coloring and averaging`_
.. _frequency division multiplexing: `multiplexing`_
.. _wavelength division multiplexing: `multiplexing`_

Parameters
==========

All parameters are parsed when setting, so usually several input formats are accepted, e.g.
``bool``, ``int``, ``float``, ``str`` for scalars and additionally ``list``, ``tuple``, ``ndarray`` for arrays.

Note that parameters might have circular dependencies which are resolved automatically,
hence dependent parameters might change as well.

.. figure:: interdependencies.svg
    :align: center
    :alt: interdependencies

    Parameters and their Interdependencies.

Video Shape
-----------

Standardized ``shape`` (``T``, ``Y``, ``X``, ``C``) of fringe pattern sequence, with

- ``T``: number of frames
- ``Y``: height (in pixel units)
- ``X``: width (in pixel units)
- ``C``: number of color channels

``T`` = ``H`` `\cdot \sum` ``N``.
If a `multiplexing`_ scheme is activated, ``T`` reduces further.

The length ``L`` is the maximum of ``X`` and ``Y`` and denotes the length (in pixel units) to be ancoded.
It can be extended by the factor ``alpha``.

``C`` depends on the `coloring`_ and `multiplexing`_ schemes activated`.

``size`` is the product of ``shape``.

Coordinate System
------------------

The following coordinate systems can be used by setting ``grid`` to:

- ``image``: The top left corner pixel of the grid is the origin (0, 0) and positive directions are right- resp. downwards.
- ``Cartesian``: The center of grid is the origin (0, 0) and positive directions are right- resp. upwards.
- ``polar``: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.
- ``log-polar``: The center of grid is the origin (0, 0) and positive directions are clockwise resp. outwards.

``D`` denotes the number of directions to be encoded.

``axis`` is used to define along which axis of the coordinate system (index 0 or 1)
the fringe pattern is shifted if ``D`` = 1.

``angle`` can be used to tilt the coordinate system. The origin stays the same.

Set
---

Each set consists of the following attributes (cf. black box in above Figure):

- ``N``: number of shifts
- ``l``: wavelength (in pixel units)
- ``v``: spatial frequency, i.e. number of periods (per screen length ``L``)
- ``f``: temporal frequency, i.e. number of periods to shift over

Each is an array with shape (number of directions ``D``, number of sets ``K``).
For example, if ``N``.shape = (2, 3), it means that we encode ``D`` = 2 directions with ``K`` = 3 sets each.
Changing ``D`` or ``K`` directly, changes the shape of all set attributes.
When setting a set attribute with a new shape (``D'``, ``K'``),
``D`` and ``K`` are updated as well as the shape of the other set attributes.

If a set attribute is 1D, then it is stacked to match the number of directions ``D``.

If a set attribute is 0D i.e. a scalar, then all values are simply replaced by the new one.

``l`` and ``v`` are related by ``l`` = ``L`` / ``v``.
When ``L`` changes, ``v`` is kept constant and only ``l`` is changed.

Usually ``f`` = 1 and is essentially only changed if `frequency division multiplexing`_ ``FDM`` is activated.

``reverse`` is a boolean which reverses the direction of the shifts (by multiplying ``f`` with -1).

``o`` denotes the phase offset, which can be used to
e.g. let the fringe patterns start (at the origin) with a gray value of zero.

Coloring and Averaging
----------------------

The fringe patterns can be colorized by setting the hue ``h`` to any RGB color triple within the interval [0, 255].
However, black (0, 0, 0) is not allowed.
``h`` must be in shape (``H``, 3)`:

``H`` is the number of hues and can be set directly; 3 is the length of the RGB color tuple.

The hues ``h`` can also be set by assigning any combination of the following characters as a string:

- ``'r'``: red
- ``'g'``: green
- ``'b'``: blue
- ``'c'``: cyan
- ``'m'``: magenta
- ``'y'``: yellow
- ``'w'``: white

``C`` is the number of color channels required for either the set of hues ``h``
or `wavelength division multiplexing`_.
For example, if all hues are monochromatic, i.e. the RGB values are identical for each hue, ``C`` equals 1, else 3.

Repeating hues will be fused by averaging them before decoding.

``M`` is the number of averaged intensity samples and can be set directly.

Multiplexing
------------

The following multiplexing methods can be activated by setting them to ``True``:

- ``SDM``: Spatial Division Multiplexing [1]_

  This results in crossed fringe patterns. The amplitude ``B`` is halved.
  It can only be activated if we have two directions ``D`` = 2.
  The number of frames ``T`` is reduced by a factor of 2.

- ``WDM``: Wavelength Divison Multiplexing [2]_

  All shifts ``N`` must equal 3. Then, the shifts are multiplexed into the color channel,
  resulting in an RGB fringe pattern.
  The number of frames ``T`` is reduced by a factor of 3.

- ``FDM``: Frequency Division Multiplexing [3]_, [4]_, [5]_

  Here, the directions ``D`` and the sets ``K`` are multiplexed.
  Hence, the amplitude ``B`` is reduced by a factor of ``D`` * ``K``.
  It can only be activated if ``D`` > 1 or ``K`` > 1.
  This results in crossed fringe patterns if ``D`` = 2.
  Each set per direction receives an individual temporal frequency ``f```,
  which is used in [temporal demodulation](#temporal-demodulation) to distinguish the individual sets.
  A minimal number of shifts ``Nmin`` `\ge \lceil` 2 * ``fmax`` `\rceil` + 1
  is required to satisfy the sampling theorem and ``N`` is updated automatically if necessary.
  If one wants a static pattern, i.e. one that remains congruent when shifted, set ``static`` to ``True``.

``SDM`` and ``WDM`` can be used together [6]_ (reducing ``T`` by a factor of 2 * 3 = 6), ``FDM`` with neighter.

``TDM``: By default, the aforementioned multiplexing methods are deactivated,
so we then only have Time Divison Multiplexing.

Data Type
---------

``dtype`` denotes the data type of the fringe pattern sequence.
Possible values are:

- ``bool``
- ``uint8`` (default)
- ``uint16``
- ``float32``
- ``float64``

``nbytes`` is the total bytes consumed by fringe pattern sequence.

``q`` is the quantization step size.
``q`` = 1 for ``bool`` and `2^r` for r-bit ``unsigned integers``,
and for ``float`` its corresponding `resolution <https://numpy.org/doc/stable/reference/generated/numpy.finfo.html>`_.

``Imax`` is the maximum gray value.
``Imax`` = 1 for ``float`` and ``bool``, and ``Imax`` = `2^r - 1` for ``unsigned integers`` with r bits.

``A`` is the offset, also called brightness (of the background).
It is limited by ``Imax``.

``B`` is the amplitude of the cosinusoidal fringes.
It is limited by ``Imax``.

``V`` is the fringe visibility (also called fringe contrast).
``V`` = ``A`` / ``B``, where ``V`` and is the range [0, 1].

``beta`` is the relative brightness (exposure) and is within the range [0, 1].

``gamma`` denotes the gamma correction factor and can be used to compensate nonlinearities of the display response curve.

Unwrapping
----------

``PU`` denotes the phase unwrapping method and is eihter ``'none'``, ``'temporal'``, ``'spatial'`` or ``'FTM'``.
See [spatial demodulation](#spatial-demodulation--phase-unwrapping--pu-) for more details.

``mode`` denotes the mode used for [temporal phase unwrapping](#temporal-phase-unwrapping--tpu-).
Choose either ``'fast'`` (the default) or ``'precise'``.

..
    ``umin`` denotes the minimal unvertainty required for the measurement to be valid
    and is in the interval `[0, 1]`. During decoding, pixels with less are discarded, which can speed up the computation.

``verbose`` can be set to ``True`` to also receive
the wrapped phase maps `\varphi_i`, the fringe orders `k` and the residuals `R` from decoding.

``FTM`` denotes Fourier Transform Method and is deployed if ``T`` = 1
and the `coordinate system`_ is eighter ``'image'`` or ``'Cartesian'``.

Quality Metrics
---------------

``UMR`` denotes the unambiguous measurement range.
The coding is only unique within the interval [0, ``UMR``); after that it repeats itself.

The ``UMR`` is derived from ``l`` and ``v``:

- If ``l`` `\in \mathbb{N}`, ``UMR`` = `lcm(` ``l`` `)` with `lcm` being the least common multiple.
- Else, if ``v`` `\in \mathbb{N}`, ``UMR`` = ``L`` / `gcd(` ``v`` `)` with `gcd` being the greatest common divisor.
- Else, if ``v`` `\lor` ``l`` `\in \mathbb{Q}` , `lcm` resp. `gcd` are extended to rational numbers.
- Else, if ``v`` `\land` ``l`` `\in \mathbb{R} \setminus \mathbb{Q}` , ``l`` and ``v`` are approximated by rational numbers
  with a fixed length of decimal digits.

``eta`` denotes the coding efficiency ``L`` / ``UMR``.
It makes no sense to choose ``UMR`` much larger than ``L``,
because then a significant part of the coding range is not used.

``u`` denotes the minimum possible uncertainty of the measurement in pixels.
It is based on the phase noise model from [7]_
and propagated through the unwrapping process and the phase fusion.
It is influenced by the parameters

- ``M``: number of averaged intensity samples,
- ``N``: number of phase shifts,
- ``l``: wavelengths of the fringes,
- ``B``: measured amplitude

and the measurement hardware-specific noise sources [8]_, [9]_

- ``quant``: quantization noise of the light source or camera,
- ``dark``: dark noise of the used camera,
- ``shot``: photon noise of light itself,
- ``gain``: system gain of the used camera.

``DR`` = ``UMR`` / ``u`` is the dynamic range of the phase shift coding
and is a measure of how many points can be distinguished within the unambiguous measurement range [0, ``UMR``).
It remains constant if ``L`` and hence ``l`` is scaled (the scaling factor cancels out).

``SNR`` = ``L`` / ``u`` is the signal-to-noise ratio of the phase shift coding
and is a masure of how many points can be distinguished within the screen length [0, ``L``).
Again, it remains constant if ``L`` and hence ``l`` is scaled (the scaling factor cancels out).

.. [1] `Park,
        "A twodimensional phase-shifting method for deflectometry",
        International Symposium on Optomechatronic Technologies,
        2008.
        <https://doi.org/10.1117/12.816472>`_

.. [2] `Huang,
        "Color-encoded digital fringe projection technique for high-speed three-dimensional surface contouring",
        Optical Engineering,
        1999.
        <https://doi.org/10.1117/1.602151>`_

.. [3] `Liu et al.,
        "Dual-frequency pattern scheme for high-speed 3-D shape measurement",
        Optics Express,
        2010.
        <https://doi.org/10.1364/OE.18.005229>`_

.. [4] `Liu et al.,
        "Fast and accurate deflectometry with crossed fringes",
        Advanced Optical Technologies,
        2014.
        <https://doi.org/10.1515/aot-2014-0032>`_

.. [5] `Kludt and Burke,
        "Coding strategies for static patterns suitable for UV deflectometry",
        Forum Bildverarbeitung 2018,
        2018.
        <https://publikationen.bibliothek.kit.edu/1000088264>`_

.. [6] `Trumper et al.,
        "Instantaneous phase shifting deflectometry",
        Optics Express,
        2016.
        <https://doi.org/10.1364/OE.24.027993>`_

.. [7] `Surrel,
        "Additive noise effect in digital phase detection",
        Applied Optics,
        1997.
        <https://doi.org/10.1364/AO.36.000271>`_

.. [8] `EMVA,
        "Standard for Characterization of Image Sensors and Cameras Release 4.0 Linear",
        European Machine Vision Association,
        2021.
        <https://www.emva.org/standards-technology/emva-1288/emva-standard-1288-downloads-2/>`_

.. [9] `Bothe,
        "Grundlegende Untersuchungen zur Formerfassung mit einem neuartigen Prinzip der Streifenprojektion und Realisierung in einer kompakten 3D-Kamera",
        Dissertation,
        ISBN 978-3-933762-24-5,
        BIAS Bremen,
        <https://www.amazon.de/Grundlegende-Untersuchungen-Formerfassung-Streifenprojektion-Strahltechnik/dp/3933762243/ref=sr_1_2?qid=1691575452&refinements=p_27%3AThorsten+B%C3%B6th&s=books&sr=1-2>`_
