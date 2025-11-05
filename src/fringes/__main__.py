import argparse
import glob
import importlib
import logging
import os
import sys
import time

import cv2
import numpy as np
# from rich.logging import RichHandler
# from rich_argparse import RichHelpFormatter

from fringes import Fringes, vshape  # todo: gui

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
    "v": "p",  # number of periods
}
SHORT_FLAGS_INVERSE = {v: k for k, v in SHORT_FLAGS.items()}


class MyFormatter(argparse.MetavarTypeHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):  # RichHelpFormatter):
    pass  # https://python-forum.io/thread-28707.html


def base_parser() -> argparse.ArgumentParser:
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
        type=str,
        help="Input file(s). If unspecified, encodes the fringe pattern sequence. Else, it decodes the given input.",
    )
    parser.add_argument("outfile", type=str, help="Output file.")
    parser.add_argument("-s", "--show", action='store_true', help="Show output.")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads to use. Default is all threads. If negative, this many threads will not be used.",
    )
    # todo: parser.add_argument("--gui", action="store_true", help="Start GUI.")
    return parser


def add_fringes_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    for k in sorted(Fringes._setters, key=lambda k: LONG_FLAGS.get(k, k.lower())):  # key=str.lower
        if k == "grid":
            continue  # todo: enable grid

        elif len(k) <= 2:
            flags = (f"-{SHORT_FLAGS.get(k, k)}", f"--{LONG_FLAGS.get(k, k.lower())}")
        else:
            flags = (f"--{LONG_FLAGS.get(k, k.lower())}",)
        parser.add_argument(
            *flags,
            dest=SHORT_FLAGS.get(k, k),
            action=argparse.BooleanOptionalAction if isinstance(Fringes._types.get(k, None), bool) else "store",
            nargs="+" if Fringes._types.get(k, None) is tuple or k == "l" else "?",
            type=(
                type(np.array(Fringes._defaults[k]).item(0))
                if Fringes._types.get(k, None) is tuple
                else Fringes._types.get(k, float)
            ),  # must be callable
            choices=Fringes._choices.get(k, None),
            # default=Fringes._defaults.get(k, None),  # arguments here seem to override parser argument `argparse.SUPPRESS`
            help=Fringes._help[k],
        )
    return parser

def parse_args() -> argparse.Namespace:
    parser = base_parser()
    add_fringes_args(parser)
    return parser.parse_args()


def set_logging(verbosity: int):
    level = ["WARNING", "INFO", "DEBUG"][min(verbosity, 2)]
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)8s.%(funcName)s: %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    # formatter = logging.Formatter("%(name)8s.%(funcName)s: %(message)s")
    # handler = RichHandler()
    handler.setFormatter(formatter)
    for log in __package__, "__main__":
        logger = logging.getLogger(log)
        logger.addHandler(handler)
        logger.setLevel(level)


def configure_fringes(args: argparse.Namespace) -> Fringes:
    params = {}
    for k, v in vars(args).items():
        if SHORT_FLAGS_INVERSE.get(k, k) in Fringes._setters and v is not None:
            params[SHORT_FLAGS_INVERSE.get(k, k)] = v
    f = Fringes()
    f._params = params
    return f


def load_images(flist: list) -> np.ndarray:
    """Load images into imagestack.

    Parameters
    ----------
    flist : list
        Filenames.
        They must end in an image format 'OpenCV' can read.

    Returns
    -------
    I : np.ndarray
        Image stack.

    Raises
    ------
    ValueError
        If images don't have the same shape or dtype.
    """
    t0 = time.perf_counter()

    img = cv2.imread(flist[0], flags=cv2.IMREAD_UNCHANGED)
    I = np.empty((len(flist),) + img.shape, img.dtype)
    I[0] = img
    for t, f in enumerate(flist[1:], start=1):
        img = cv2.imread(f, flags=cv2.IMREAD_UNCHANGED)
        if img.shape != I.shape[1:]:
            raise ValueError(f"Image {flist[t]} has shape {img.shape} but previous ones have shape {I.shape[1:]}.")
        elif img.dtype != I.dtype:
            raise ValueError(f"Image {flist[t]} has dtype {img.dtype} but previous ones have dtype {I.dtype}.")
        else:
            I[t] = img

    I = vshape(I)
    T, Y, X, C = I.shape

    # compensate OpenCV'S BRG(A) color order
    if C == 3:
        I = I[..., ::-1]
    elif C == 4:
        np.rollaxis(I[..., ::-1], -1, -1)  # indices backwards is [3, 2, 1, 0], then rolled is [2, 1, 0, 3]

    logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")
    return I


def save_images(fname: str, I: np.ndarray):
    """Save image stack as images.

    Parameters
    ----------
    fname: str, optional
        File (base) name.
        Trailing numbers are appended before the file extension.
    I : np.ndarray
        Image stack.
    """
    t0 = time.perf_counter()

    I = vshape(I)
    T, Y, X, C = I.shape

    # compensate OpenCV'S BRG(A) color order
    if C == 3:
        I = I[..., ::-1]
    elif C == 4:
        np.rollaxis(I[..., ::-1], -1, -1)  # indices backwards is [3, 2, 1, 0], then rolled is [2, 1, 0, 3]

    x = len(str(T))  # length of digit string
    root, ext = os.path.splitext(fname)
    for t, image in enumerate(I):
        fname = f"{root}{t + 1:0{x}}{ext}"
        cv2.imwrite(fname, image)

    logger.debug(f"{(time.perf_counter() - t0) * 1000:.0f}ms")


def main():
    args = parse_args()
    set_logging(args.verbose)
    f = configure_fringes(args)

    if "gui" in args:
        NotImplementedError  # todo: gui.main(f)
    elif "infile" not in args:  # encode
        # encode
        I = f.encode()

        # save
        root, ext = os.path.splitext(args.outfile)
        if ext.lower() in {".png", ".tif", ".tiff"} and I.dtype.type in {np.uint8, np.uint16}:
            save_images(root + ext, I)
            logger.info(f"Saved encoded fringe pattern sequence to '{root}_*{ext}'.")
        else:
            ext = ".npy"
            np.save(root + ext, I)  # appends '.npy' to fname if missing
            logger.info(f"Saved encoded fringe pattern sequence to '{root + ext}'.")

        # show
        if "show" in args:
            try:
                import pyqtgraph as pg
                pg.setConfigOptions(imageAxisOrder="row-major")
                pg.image(I, title="fringe pattern sequence")
                pg.exec()
            except ImportError:
                logging.error("'pyqtgraph' and its dependencies are not installed.")

    else:  # decode
        # load
        _, ext = os.path.splitext(args.infile)
        if ext == ".npy":
            I = np.load(args.infile)  # load single *.npy file
        elif ext in {".png", ".tif", ".tiff"}:
            flist = glob.glob(args.infile)
            I = load_images(flist)  # load image files
        else:
            raise ValueError(f"File type '{ext}' not supported."
                             f"Must be one of {{{".npy", ".png", ".tif", ".tiff"}}}.")

        # decode
        dec = f.decode(I, threads=getattr(args, "threads", None))

        # save
        root, _ = os.path.splitext(args.outfile)
        np.savez(root + ".npz", **dec._asdict())  # appends '.npz' to fname if missing
        logger.info(f"Saved decoded data to '{root + ".npz"}'.")

        # show
        if "show" in args:
            try:
                import pyqtgraph as pg
                pg.setConfigOptions(imageAxisOrder="row-major")
                pg.image(I, title="fringe pattern sequence")
                pg.image(dec.a, title="brightness")
                pg.image(dec.b, title="modulation")
                pg.image(dec.x, title="coordinate")
                pg.exec()
            except ImportError:
                logger.error("'pyqtgraph' and its dependencies are not installed.")


if __name__ == "__main__":
    main()
