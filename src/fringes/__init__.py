"""Phase shifting algorithms for encoding and decoding sinusoidal fringe patterns."""

import logging

from .fringes import Fringes
from .util import _version, vshape, curvature  # todo: height, ...

logger = logging.getLogger(__name__)

__version__ = _version()
