.. default-role:: math

Multiplex
=========
Multiplexing can be used to reduce the number of frames :attr:`~fringes.fringes.Fringes.T`.
This accelerates the recording time
at the cost of increased measurement uncertainty.

Spatial Division Multiplexing
-----------------------------
In spatial division multiplexing (:attr:`~fringes.fringes.Fringes.SDM`) [Par08]_, the fringes for each direction are additively superimposed,
which results in crossed fringe patterns, cf. :numref:`SDM`.
The amplitude :attr:`~fringes.fringes.Fringes.B` is halved,
i.e. for each direction only have the signal strength is available.
The number of frames :attr:`~fringes.fringes.Fringes.T` is halved.

.. literalinclude:: /../../examples/spatial_division_multiplexing.py
   :language: python
   :emphasize-lines: 11, 12
   :linenos:

.. _SDM:
.. figure:: mux/SDM.png
    :scale: 20%
    :align: center

    Spatial division multiplexing (SDM).

In the decoding stage, the recorded fringe pattern sequence `I^*` is Fourier-transformed
and the directions are separated in frequency space.
Because this is done within the camera frame of reference,
the demultiplexed directions only correspond to the encoded ones when the camera and scree are well aligned,
i.e. they must face each other directly.
Otherwise, the decoded coordinate directions can not be assigned to the screen axes correctly.

Wavelength Division Multiplexing
--------------------------------
In wavelength division multiplexing (:attr:`~fringes.fringes.Fringes.WDM`) [Hua99]_,
the shifts are multiplexed into the color channel,
resulting in an RGB fringe pattern, cf. :numref:`WDM`.
Therefore it is required that all shifts :attr:`~fringes.fringes.Fringes.N` `= 3`.
The number of frames :attr:`~fringes.fringes.Fringes.T` is cut into thirds.

.. literalinclude:: /../../examples/wavelength_division_multiplexing.py
   :language: python
   :emphasize-lines: 11, 12
   :linenos:

.. _WDM:
.. figure:: mux/WDM.png
    :scale: 20%
    :align: center

    Wavelength division multiplexing (WDM).

This works best when an RGB-prism-based camera is used,
because its spectral bands don't overlap and hence the RGB-channels can be separated sharply.
Additionally, a white balance has to be executed to ensure equal irradiance readings in all color channels.

Also, the effect of color absorption by the surface material cannot be neglected.
This means that the test object itself must not have any color.

Overall, less light is available per pixel because it is divided into the three color channels.
Therefore, it requires about 3 times the exposure time compared to grayscale patterns.

Spatial and wavelength division multiplexing can be used together [Tru16]_.
If only one set :attr:`~fringes.fringes.Fringes.K` `= 1` per direction is used,
only one frame :attr:`~fringes.fringes.Fringes.T` `= 1` is necessary, cf. :numref:`SDM+WDM`.
This allows single shot applications to be implemented.

.. literalinclude:: /../../examples/spatial_wavelength_division_multiplexing.py
   :language: python
   :emphasize-lines: 11, 12, 13, 14, 15
   :linenos:

.. _SDM+WDM:
.. figure:: mux/SDM+WDM.png
    :scale: 20%
    :align: center

    Spatial and wavelength division multiplexing combined.

Frequency Division Multiplexing
-------------------------------
In frequency division multiplexing (:attr:`~fringes.fringes.Fringes.FDM`) [Liu14b]_, [Liu10]_,
the directions :attr:`~fringes.fringes.Fringes.D`
and the sets :attr:`~fringes.fringes.Fringes.K` are additively superimposed.
Hence, the amplitude :attr:`~fringes.fringes.Fringes.B` is reduced
by a factor of :attr:`~fringes.fringes.Fringes.D` `*` :attr:`~fringes.fringes.Fringes.K`.
This results in crossed fringe patterns if we have :attr:`~fringes.fringes.Fringes.D` `= 2` directions,
cf. :numref:`FDM-D2K1` and :numref:`FDM-D2K2`.

Each set `k` per direction `d` receives an individual temporal frequency `f_{d,k}`,
which is used in :ref:`temporal demodulation <encoding>`
to distinguish the individual sets.
A minimal number of shifts
`N_{min} \ge \lceil 2 * f_{max} \rceil + 1`
is required to satisfy the sampling theorem.

.. literalinclude:: /../../examples/frequency_division_multiplexing_2.py
   :language: python
   :emphasize-lines: 11, 12, 13, 14, 15
   :linenos:

.. _FDM-D2K1:
.. figure:: mux/FDM_D2K1.png
    :scale: 20%
    :align: center
    :alt: FDM-D2K1

    Frequency division multiplexing (FDM).
    Two directions are superimposed.

----

.. literalinclude:: /../../examples/frequency_division_multiplexing_1.py
   :language: python
   :emphasize-lines: 11, 12, 13, 14, 15
   :linenos:

.. _FDM-D1K2:
.. figure:: mux/FDM_D1K2.png
    :scale: 20%
    :align: center
    :alt: FDM-D1K2

    Frequency division multiplexing (FDM).
    Two sets are superimposed.

----

.. literalinclude:: /../../examples/frequency_division_multiplexing_3.py
   :language: python
   :emphasize-lines: 11, 12, 13, 14, 15
   :linenos:

.. _FDM-D2K2:
.. figure:: mux/FDM_D2K2.png
    :scale: 20%
    :align: center
    :alt: FDM-D2K2

    Frequency division multiplexing (FDM).
    Two directions and two sets are superimposed.

----

If one wants a static pattern, i.e. one that remains congruent when phase shifted,
the spatial frequencies must be integers:
:attr:`~fringes.fringes.Fringes.v` `\in \mathbb{N}`,
must not share any common divisor except one:
`gcd(` :attr:`~fringes.fringes.Fringes.v` `)= 1`,
and the temporal frequencies must equal the spatial ones:
:attr:`~fringes.fringes.Fringes.v` `=` :attr:`~fringes.fringes.Fringes.f`.
With static/congruent patterns, one can realize phase shifting by moving printed patterns [Klu18]_.

.. literalinclude:: /../../examples/frequency_division_multiplexing_4.py
   :language: python
   :emphasize-lines: 10, 11, 12, 13, 14, 15, 16
   :linenos:

.. _FDM-D2K2-static:
.. figure:: mux/FDM_D2K2_static.png
    :scale: 20%
    :align: center
    :alt: FDM-D2K2-static

    Frequency division multiplexing (FDM).
    Two directions and two sets are superimposed.
    This is a static pattern which remains congruent when phase shifted.

.. Fourier Transform Method
   ------------------------
   If only a single frame, i.e. :attr:`~fringes.fringes.Fringes.T` `= 1`, is recorded using a crossed fringe pattern,
   the phase signal introduced by the object's distortion of the fringe pattern
   can be extracted with a purely spatial analysis by virtue of the Fourier-transform method (FTM) [Tak82]_:

   .. literalinclude:: /../../examples/fourier_transform_method.py
      :language: python
      :emphasize-lines: 11, 12, 13, 14
      :linenos:

   .. _FTM:
   .. figure:: mux/FTM.png
       :scale: 20%
       :align: center
       :alt: FTM

       Fourier Transform Method (FTM).
       Two directions are superimposed.
       Only this one frame is required.

   The recorded phase consists of a carrier with the spatial frequency `\nu_r`
   (note that `\nu_r` denotes the spatial frequency in the recorded camera frame,
   therefore `\nu` and `\nu_r` are related by the imaging of the optical system but not identical):
   `\varPhi_r = \varPhi_c + \varPhi_s = 2 \pi \nu_r + \varPhi_s`.
   If the offset `A`, the amplitude `B` and the signal phase `\varPhi_s` vary slowly
   compared with the variation introduced by the spatial-carrier frequency `\nu_r`,
   i.e. the surface is rather smooth and has no sharp edges,
   and the spatial carrier frequency is high enough, i.e. `\nu_r >> 1`,
   their spectra can be separated and therefore filtered in frequency space.

   For this purpose, the recorded fringe pattern is Fourier transformed
   by the use of the two-dimensional fast-Fourier-transform (2DFFT) algorithm - hence the name -
   and processed in its spatial frequency domain.
   Here, the Fourier spectra are separated by the carrier frequency `\nu_r`, as can be seen in :numref:`spectra`.
   We filter out the background variation `A`, select either of the two spectra on the carrier,
   and translate it by `\nu_r` on the frequency axis towards the origin.

   .. _spectra:
   .. figure:: mux/spectra.png
       :scale: 25%
       :align: center

       In this image, the spatial frequency `\nu_r` is denoted as `f_0`.
       (A) Separated Fourier spectra; (B) single spectrum selected and translated to the origin.
       Source: [Mas01]_.

   Again using the 2DFFT algorithm, we compute the inverse Fourier-transform.
   Now we have the signal phase `\varPhi_s` in the imaginary part
   completely separated from the unwanted amplitude variation `B` in the real part.
   Subsequently, a spatial phase-unwrapping algorithm may be applied to remove any remaining phase jumps.

   .. _liu2014:
   .. figure:: mux/Liu2014.png
       :scale: 25%
       :align: center

       Recorded reflection of a crossed fringe pattern with high spatial frequency :attr:`~fringes.fringes.Fringes.v`.
       Source: [Liu14a]_.

   .. todo: show example images of decoded

   .. todo: attention: no a! resp. a is without content i.e. empty

   This phase unwrapping method is not critical if the signal-to-noise ratio is higher than 10
   and the gradients of the signal phase `\varPhi_s` are less than `\pi` per pixel.

   .. warning::
      This only yields a relative phase map, therefore absolute positions remain unknown.

.. [Hua99]
   `Huang,
   "Color-encoded digital fringe projection technique for high-speed three-dimensional surface contouring",
   Optical Engineering,
   1999.
   <https://doi.org/10.1117/1.602151>`_

.. [Klu18]
   `Kludt and Burke,
   "Coding strategies for static patterns suitable for UV deflectometry",
   Forum Bildverarbeitung 2018,
   2018.
   <https://publikationen.bibliothek.kit.edu/1000088264>`_

.. [Liu10]
   `Liu et al.,
   "Dual-frequency pattern scheme for high-speed 3-D shape measurement",
   Optics Express,
   2010.
   <https://doi.org/10.1364/OE.18.005229>`_

.. .. [Liu14a]
   `Liu et al.,
   "Fast and accurate Deflectometry with Crossed Fringes",
   arXiv,
   2014.
   <https://doi.org/10.48550/arXiv.1404.5727>`_

.. [Liu14b]
   `Liu et al.,
   "Fast and accurate deflectometry with crossed fringes",
   Advanced Optical Technologies,
   2014.
   <https://doi.org/10.1515/aot-2014-0032>`_

.. .. [Mas01]
   `Massig and Heppner,
   "Fringe-pattern analysis with high accuracy by use of the Fourier-transform method: theory and experimental tests",
   Applied Optics,
   2001.
   <https://doi.org/10.1364/AO.40.002081>`_

.. [Par08]
   `Park,
   "A twodimensional phase-shifting method for deflectometry",
   International Symposium on Optomechatronic Technologies,
   2008.
   <https://doi.org/10.1117/12.816472>`_

.. .. [Tak82]
   `Takeda et al.,
   "Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry",
   Journal of the Optical Society of America,
   1982.
   <https://doi.org/10.1364/JOSA.72.000156>`_

.. [Tru16]
   `Trumper et al.,
   "Instantaneous phase shifting deflectometry",
   Optics Express,
   2016.
   <https://doi.org/10.1364/OE.24.027993>`_
