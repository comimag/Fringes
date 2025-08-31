Optimal Coding Strategy
=======================

Increase Shifts and Sets
------------------------
As makes sense intuitively, more sets :attr:`~fringes.fringes.Fringes.K` as well as more shifts :attr:`~fringes.fringes.Fringes.N` per set reduce the uncertainty :attr:`~fringes.fringes.Fringes.u` after decoding.
A minimum of three shifts is needed to solve for the three unknowns brightness ``a``, modulation ``b`` and coordinates ``x``.
Any additional two shifts compensate for one harmonic of the recorded fringe pattern.
Therefore, higher accuracy can be achieved using more shifts :attr:`~fringes.fringes.Fringes.N`, but the time required to capture them
sets a practical upper limit to the feasible number of shifts.

Increase Spatial Frequency
--------------------------
Generally, shorter wavelengths :attr:`~fringes.fringes.Fringes.l` (or equivalently more periods :attr:`~fringes.fringes.Fringes.v`) reduce the uncertainty :attr:`~fringes.fringes.Fringes.u`,
but the resolution of the camera and the display must resolve the fringe pattern spatially.
Hence, the used hardware imposes a lower bound for the wavelength :attr:`~fringes.fringes.Fringes.lmin`
(or upper bound for the number of periods :attr:`~fringes.fringes.Fringes.vmax`).

Also, small wavelengths might result in a smaller unambiguous measurement range :attr:`~fringes.fringes.Fringes.UMR`.
If two or more sets :attr:`~fringes.fringes.Fringes.K` are used and their wavelengths :attr:`~fringes.fringes.Fringes.l` resp. number of periods :attr:`~fringes.fringes.Fringes.v` are relative primes,
the unambiguous measurement range can be increased many times.
As a consequence, one can use much smaller wavelengths :attr:`~fringes.fringes.Fringes.l` (larger number of periods :attr:`~fringes.fringes.Fringes.v`).

However, it must be assured that the unambiguous measurement range is always equal or larger than both,
the width :attr:`~fringes.fringes.Fringes.X` and the height :attr:`~fringes.fringes.Fringes.Y`.
Else, :ref:`temporal phase unwrapping <tpu>` would yield wrong results and thus instead
:ref:`spatial phase unwrapping <spu>` is used.
Be aware that in the latter case only a relative phase map is obtained,
which lacks the information of where exactly the camera sight rays were looking at during recording.

Automatic Optimization
----------------------
Instead of the options above, one can simply use the function :func:`~fringes.fringes.Fringes.optimize`:

a) If called with the argument ``T``,
   the parameters of the :class:`~fringes.fringes.Fringes` instance are optimized
   to yield the minimal uncertainty possible using the given number of frames ``T``.
b) Else, if called with the argument ``umax``, and :attr:`~fringes.fringes.Fringes.ui` is specified,
   the optimal parameters are determined that allow a maximal uncertainty ``umax``
   with the number of frames :attr:`~fringes.fringes.Fringes.T`.

c) Else, if called with no argument,
   the parameters of the :class:`~fringes.fringes.Fringes` instance are optimized
   to yield the minimal uncertainty possible using the number of frames :attr:`~fringes.fringes.Fringes.T`.

.. To simplify finding and setting the optimal parameters, one can choose from the following options:

   - :attr:`~fringes.fringes.Fringes.v` can be set to ``'optimal'``.
     This automatically determines the optimal integer set of :attr:`~fringes.fringes.Fringes.v`,
     based on the maximal resolvable spatial frequency :attr:`~fringes.fringes.Fringes.vmax`.
   - Equivalently, :attr:`~fringes.fringes.Fringes.l` can also be set to ``'optimal'``.
     This will automatically determine the optimal integer set of :attr:`~fringes.fringes.Fringes.l`,
     based on the minimal resolvable wavelength :attr:`~fringes.fringes.Fringes.lmin` = :attr:`~fringes.fringes.Fringes.L` / :attr:`~fringes.fringes.Fringes.vmax`.
   - :attr:`~fringes.fringes.Fringes.T` can be set directly, based on the desired recording time.
     The optimal :attr:`~fringes.fringes.Fringes.K`, :attr:`~fringes.fringes.Fringes.N` and  - if necessary - the :ref:`multiplexing <multiplex>` will be determined automatically.
   - Instead of the options above, one can simply use the function :func:`~fringes.fringes.Fringes.optimize`:
     If :attr:``umax`` is specified, the optimal parameters are determined
     that allow a maximal uncertainty of :attr:``umax`` with a minimum number of frames.
     Else, the parameters of the :class:`~fringes.fringes.Fringes` instance are optimized to yield the minimal uncertainty possible
     using the given number of frames :attr:`~fringes.fringes.Fringes.T`.

In order for this method to perform optimally,
the modulation ``b`` at the spatial frequencies :attr:`~fringes.fringes.Fringes.v`
must be known or at least be able to be estimated.

a) Measure the **modulation transfer function (MTF)** at a given number of sample points:

   1. Set :attr:`~fringes.fringes.Fringes.K` to the required number of sample points (usually > 10 is a good value).
   2. Set :attr:`~fringes.fringes.Fringes.v` to ``'exponential'``.
      This will create spatial frequencies :attr:`~fringes.fringes.Fringes.v` spaced evenly on a log scale (a geometric progression),
      starting from 0 up to :attr:`~fringes.fringes.Fringes.vmax`.
   3. Encode, record and decode the fringe pattern sequence.
      A minimal working example is given :ref:`here <minimal working example>`.
   4. Optionally for better precision:
      Mask the values of ``b`` with *nan* where the camera wasn't looking at the screen.
      The decoded modulation ``b`` can be used as an indicator.
   5. Call the method :func:`~fringes.fringes.Fringes.set_mtf`
      with the estimated modulation ``b`` from the measurement as the argument.
   6. Finally, the modulation ``b'`` at certain spatial frequencies :attr:`~fringes.fringes.Fringes.v`'
      will be interpolated by the method :meth:`~fringes.fringes.Fringes.mtf`.

      .. literalinclude:: /../../examples/modulation_transfer_function_1.py
      :language: python
      :emphasize-lines: 17, 18
      :linenos:

b) As an approximation, a linear MTF is assumed [Bot08]_:
   It starts at :attr:`~fringes.fringes.Fringes.v` = 0 with b = 1 and ends at :attr:`~fringes.fringes.Fringes.v` = :attr:`~fringes.fringes.Fringes.vmax` with b = 0 (cf. :numref:`mtf`).
   Therefore, the optimal wavelength is :attr:`~fringes.fringes.Fringes.vopt` = :attr:`~fringes.fringes.Fringes.vmax` / 2.

   1. Set :attr:`~fringes.fringes.Fringes.lmin` (:attr:`~fringes.fringes.Fringes.vmax`)
      to the minimum (maximum) resolvable wavelength (spatial frequency),
      that the recording camera can just about resolve

   .. literalinclude:: /../../examples/modulation_transfer_function_2.py
      :language: python
      :emphasize-lines: 8
      :linenos:

..
   Estimate the **magnification** and the **Point Spread Function (PSF)** of the imaging system:

   1. Set the attributes :attr:`~fringes.fringes.Fringes.magnification` and :attr:`~fringes.fringes.Fringes.PSF`.
      In the further, the modulation ``b'`` for certain spatial frequencies :attr:`~fringes.fringes.Fringes.v`'
      will be computed from the specified attributes :attr:`~fringes.fringes.Fringes.magnification`
      and :attr:`~fringes.fringes.Fringes.PSF` directly.

   .. todo: formula from paper
      at which the fringes vanish completely.
      start to faint.

   .. code-block:: python

      from fringes import Fringes     # import the Fringes class

      f = Fringes()         # instantiate Fringes object
      # f.magnification = ...
      f.PSF = 5                  # set standard deviation of the point spread function

      # v_new = 13, 7, 89          # define arbitrary new frequencies
      # mtf_est = f.mtf(v_new)       # get estimated modulation transfer values

      f.optimize()

.. c) As a last resort, a constant modulation transfer function is assumed: MTF(:attr:`~fringes.fringes.Fringes.v`') = 1.
   In this case the optimal frequency will be :attr:`~fringes.fringes.Fringes.vmax`.

.. [Bot08]
   `Bothe,
   "Grundlegende Untersuchungen zur Formerfassung mit einem neuartigen Prinzip der Streifenprojektion und Realisierung in einer kompakten 3D-Kamera",
   Dissertation,
   ISBN 978-3-933762-24-5,
   BIAS Bremen,
   2008.
   <https://www.amazon.de/Grundlegende-Untersuchungen-Formerfassung-Streifenprojektion-Strahltechnik/dp/3933762243/ref=sr_1_2?qid=1691575452&refinements=p_27%3AThorsten+B%C3%B6th&s=books&sr=1-2>`_
