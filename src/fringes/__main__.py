import argparse
import importlib
import logging
import os
import sys

import numpy as np
import toml

from fringes import Fringes

logger = logging.getLogger(__name__)
# logger = logging.getLogger("fringes")


def main():
    #  use description string in pyproject.toml as the single source of truth
    try:
        # in order not to confuse an installed version of a package with a local one,
        # first try the local one (not being installed)
        data = toml.load(os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"))
        desc = data["project"]["description"]  # Python Packaging User Guide expects description here
    except KeyError:
        desc = data["tool"]["poetry"]["description"]  # Poetry expects description here
    except FileNotFoundError:
        name = "fringes"  # installed name
        desc = importlib.metadata.summary(name)  # installed description

    # parser = argparse.ArgumentParser(description=desc, argument_default=argparse.SUPPRESS)
    parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument("--config", "-c", type=str, help=Fringes.load.__doc__)
    parser.add_argument("--directions", "-D", type=int, choices=[1, 2], help=Fringes.D.__doc__)
    parser.add_argument("--sets", "-K", type=int, help=Fringes.K.__doc__)
    parser.add_argument("--shifts", "-N", type=int, help=Fringes.N.__doc__)
    parser.add_argument("--periods", "-p", nargs="+", type=float, help=Fringes.v.__doc__)
    parser.add_argument("--lambdas", "-l", nargs="+", type=float, help=Fringes.l.__doc__)
    parser.add_argument("--alpha", "-a", type=float, help=Fringes.alpha.__doc__)
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Verbosity of logging level.")
    parser.add_argument("--encode", "-e", action="store_true", help=Fringes.encode.__doc__)
    parser.add_argument("--decode", "-d", action="store_true", help=Fringes.decode.__doc__)
    parser.add_argument("--input", "-i", type=str, help="Filename of the recorded image sequence to be decoded.")
    parser.add_argument("--output", "-o", type=str, help="Filename where to save the decoded data to.")
    args = parser.parse_args()

    if args.verbose == 0:
        level = "WARNING"
    elif args.verbose == 1:
        level = "INFO"
    else:
        level = "DEBUG"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s.%(funcName)s(): %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)  # todo: log to stdout
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    f = Fringes()

    # if hasattr(args, "config") and args.config is not None:
    #     f.load(args.config)

    if hasattr(args, "alpha") and args.alpha is not None:
        f.alpha = args.alpha

    if hasattr(args, "lambdas") and args.lambdas is not None:
        f.l = args.lambdas

    if hasattr(args, "periods") and args.periods is not None:
        f.v = args.periods

    if hasattr(args, "shifts") and args.shifts is not None:
        f.N = args.shifts

    if hasattr(args, "sets") and args.sets is not None:
        f.K = args.sets

    if hasattr(args, "directions") and args.directions is not None:
        f.D = args.directions

    if hasattr(args, "encode") and args.encode is not False:
        I = f.encode()
    elif hasattr(args, "input") and args.input is not None:
        I = np.load(args.input)

    if hasattr(args, "decode") and args.decode is not False and "I" in locals():
        dec = f.decode(I)

    if hasattr(args, "output") and args.output is not None:
        if "I" in locals():
            np.save(args.output, I)

        if "dec" in locals():
            dict = dec._asdict()
            np.savez(args.output, **dict)


if __name__ == "__main__":
    main()
