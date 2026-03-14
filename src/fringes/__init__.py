import logging
from importlib.metadata import version

from fringes.fringes import Fringes
from fringes.util import vshape

logger = logging.getLogger(__name__)

__version__ = version(__package__)  # installed version

__all__ = ["Fringes", "vshape"]
