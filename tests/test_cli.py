import glob
import os
import subprocess as sp
import tempfile

from fringes import Fringes
from fringes.__main__ import LONG_FLAGS, SHORT_FLAGS, base_parser


def test_args():
    parser = base_parser()
    a = set(SHORT_FLAGS.get(k, k) for k in Fringes._setters) | set(SHORT_FLAGS.get(k, k) for k in LONG_FLAGS.keys())
    b = set(k.lstrip("-") for k in parser._option_string_actions.keys())
    assert not (a & b)  # no intersection


def test_cli():
    with tempfile.TemporaryDirectory() as tempdir:
        # encode
        sp.run(["fringes", "pattern"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "pattern.npy"))

        sp.run(["fringes", "pattern1.npy"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "pattern1.npy"))

        sp.run(["fringes", "pattern_.png"], cwd=tempdir)
        flist = glob.glob(os.path.join(tempdir, "pattern_*.png"))
        assert len(flist) == Fringes().T

        # decode
        sp.run(["fringes", "-i", "pattern.npy", "decoded"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "decoded.npz"))

        sp.run(["fringes", "-i", "pattern.npy", "decoded1.npz"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "decoded1.npz"))

        sp.run(["fringes", "-i", "pattern_*.png", "decoded2"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "decoded2.npz"))
