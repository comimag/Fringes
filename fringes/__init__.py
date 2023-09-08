import importlib
import os
import toml
import argparse

from .fringes import Fringes
from .util import vshape, curvature, height, circular_distance

# use verion string in pyproject.toml as the single source of truth
try:  # PackageNotFoundError
    fname = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    version = toml.load(fname)["tool"]["poetry"]["version"]
except FileNotFoundError or KeyError:
    version = importlib.metadata.version("fringes")

__version__ = version


def main():
    parser = argparse.ArgumentParser  # todo: argparser
    for p in Fringes.params:
        -p,
        parser.add_argument(
            "-X",
            "--width",
        )

    args = parser.parse_args()

    q = 1


if __name__ == "__main__":
    main()
