Troubleshooting
===============

.. dropdown:: ``poetry install`` does not work

  - Ensure that poetry is installed correctly as descibed on the `Poetry Website <https://python-poetry.org/docs/>`_.
  - Ensure the correct python version is installed on your system, as specified in the file `pyproject.toml`.
  - This can be caused by a proxy which `pip` does not handle correctly.
    Manually setting the proxy in the Windows settings
    or even adding a system variable `https_proxy = http://YOUR_PROXY:PORT` can resolve this.

.. dropdown:: Decoding takes a long time

  This is most likely due to the just-in-time compiler `Numba <https://numba.pydata.org/>`_,
  which is used for this computationally expensive function:
  During the first execution, an initial compilation is executed. 
  This can take several tens of seconds up to single digit minutes, depending on your CPU and energy settings.
  However, for any subsequent execution, the compiled code is cached and the code of the function runs much faster, 
  approaching the speeds of code written in C.

.. dropdown:: My decoded coordinates show systematic offsets

  - Make sure the exposure of your camera is adjusted so that the fringe patterns show up with maximum contrast.
    Specifically avoid overexposure during recording.
  - Ensure that the correct frames were captured while recording the fringe pattern sequence.
    If the timings are not set correctly, the sequence may be a frame off.
  - This might occur if either the camera or the display used have a gamma value very different from 1.

    a) Typically, screens have a gamma value of 2.2; therefore compensate by setting the inverse value
       :math:`\gamma^{-1} = 1 / 2.2 \approx 0.45` to the :attr:`~fringes.fringes.Fringes.gamma` attribute
       of the :class:`~fringes.fringes.Fringes` instance.
       Alternatively, change the gamma value of the light source or camera directly.

    b) You might also use more shifts :attr:`~fringes.fringes.Fringes.N`
       to compensate for the dominant harmonics of the gamma-non-linearities.

    c) You can use the function :func:`~fringes.util.gamma_auto_correct`
       to automatically estimate and apply the gamma correction factor to linearize the display/camera response curve.\

  - Ensure that the unambiguous measurement range is larger than the pattern length,
    i.e. :attr:`~fringes.fringes.Fringes.UMR` :math:`\ge` :attr:`~fringes.fringes.Fringes.L`.
    If not, adjust the used wavelengths :attr:`~fringes.fringes.Fringes.l`
    resp. number of periods :attr:`~fringes.fringes.Fringes.v` accordingly,
    or reset them by setting either of them to 'default'.

.. dropdown:: My decoded coordinates show lots of noise

  - Try using more, sets :attr:`~fringes.fringes.Fringes.K` and/or shifts :attr:`~fringes.fringes.Fringes.N`.
  - If the decoded modulation is much lower than the decoded brightness,
    try to use larger wavelengths :attr:`~fringes.fringes.Fringes.l`
    resp. smaller number of periods :attr:`~fringes.fringes.Fringes.v`.
  - If the decoded modulation remains low even with very large wavelengths :attr:`~fringes.fringes.Fringes.l`
    (less than five periods :attr:`~fringes.fringes.Fringes.v` per screen length :attr:`~fringes.fringes.Fringes.L`),
    and you are conducting a deflectometric measurement, the surface under test is probably too rough.
    Since deflectometry is for specular and glossy surfaces only, it isn't suited for scattering ones.
    You should consider a different measurement technique, such as e.g. fringe projection.
