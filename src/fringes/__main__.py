import argparse
import importlib
import logging
import os
import sys

import numpy as np

# from rich.logging import RichHandler
# from rich_argparse import RichHelpFormatter

from fringes import Fringes

# from fringes import gui

logger = logging.getLogger(__name__)


LONG_FLAGS = {
    "Y": "height",
    "X": "width",
    "D": "directions",
    "K": "sets",
    "N": "shifts",
    "l": "lambdas",  # wavelengths
    "v": "periods",
    "f": "frequencies",
    "h": "colors",
    "p0": "phase-offset",
    "A": "offset",
    "B": "modulation",
    "E": "exposure",
    "V": "visibility",
    "g": "gamma",
    "a": "alpha",
}  # info: "[argparse] allows long options to be abbreviated to a prefix, if the abbreviation is unambiguous[...]"
SHORT_FLAGS = {
    "h": "c",
    "v": "p",
}


class MyFormatter(argparse.MetavarTypeHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):  # RichHelpFormatter):
    pass  # https://python-forum.io/thread-28707.html


def main():
    parser = argparse.ArgumentParser(
        description=importlib.metadata.metadata(__package__)["summary"],
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.MetavarTypeHelpFormatter,  # MyFormatter
    )
    parser.add_argument("--version", action="version", version=importlib.metadata.version(__package__))
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity of logging level.")
    parser.add_argument(
        "-i",
        "--infile",
        action="store",
        type=str,
        help="Input file. If unspecified, encodes the fringe pattern sequence. Else, it decodes the given input.",
    )
    parser.add_argument("-o", "--outfile", action="store", type=str, default="outfile.npy", help="Output file.")
    # parser.add_argument("-g", "--gui", action="store_true", help="Start GUI.")
    # todo: check that adding arguments from setters has no conflict with add_argument()-lines in this document
    for k in sorted(Fringes._setters, key=lambda k: LONG_FLAGS.get(k, k.lower())):  # key=str.lower
        if k == "grid":
            continue  # todo: enable grid

        if len(k) == 1:
            flags = f"-{SHORT_FLAGS.get(k, k)}", f"--{LONG_FLAGS.get(k, k.lower())}"
        else:
            flags = (f"--{LONG_FLAGS.get(k, k.lower())}",)
        parser.add_argument(
            *flags,
            dest=SHORT_FLAGS.get(k, k),
            action=argparse.BooleanOptionalAction if isinstance(Fringes._types.get(k, None), bool) else "store",
            nargs="+" if Fringes._types.get(k, None) is np.ndarray or k == "l" else "?",
            type=(
                type(np.array(Fringes._defaults[k]).item(0))
                if Fringes._types.get(k, None) is tuple
                else Fringes._types.get(k, float)
            ),  # must be callable
            choices=Fringes._choices.get(k, None),
            # default=Fringes._defaults.get(k, None),  # arguments here seem to override parser argument `argparse.SUPPRESS`
            help=Fringes._help[k],
        )
    args = parser.parse_args()

    level = ["WARNING", "INFO", "DEBUG"][min(args.verbose, 2)]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)8s.%(funcName)s: %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    # formatter = logging.Formatter("%(name)8s.%(funcName)s: %(message)s")
    # handler = RichHandler()
    handler.setFormatter(formatter)
    for log in __package__, "__main__":
        logger = logging.getLogger(log)
        logger.addHandler(handler)
        logger.setLevel(level)

    params = {}
    for k, v in vars(args).items():
        if k in Fringes._setters and v is not None:
            params[k] = v
    f = Fringes()
    f._params = params

    def parse_fname(fname: str = "outfile") -> str:
        head, tail = os.path.split(fname)
        path = os.path.abspath(head)
        root, ext = os.path.splitext(tail)
        return os.path.join(path, root)

    if "gui" in args:
        # gui.main(f)
        pass
    elif "infile" not in args:  # encode
        I = f.encode()
        fname = parse_fname(args.outfile)
        np.save(fname, I)
        logger.info(f"Saved encoded fringe pattern sequence to '{args.outfile}'.")
    else:  # decode
        fname = parse_fname(args.infile) + ".npy"
        I = np.load(fname)
        dec = f.decode(I)
        fname = parse_fname(args.outfile)
        np.savez(fname, **dec._asdict())
        logger.info(f"Saved decoded data to '{args.outfile}'.")


if __name__ == "__main__":
    main()
