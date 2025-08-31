from importlib.metadata import version
import logging

from fringes.decoder import spu
from fringes.fringes import Fringes
from fringes.util import vshape

logger = logging.getLogger(__name__)

__version__ = version(__package__)  # installed version
