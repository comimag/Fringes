import os
import subprocess as sp
import tempfile


def test_cli():
    with tempfile.TemporaryDirectory() as tempdir:
        sp.run("fringes", cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "outfile.npy"))

        sp.run(["fringes", "-o", "pattern.npy"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "pattern.npy"))

        sp.run(["fringes", "-i", "pattern.npy", "-o", "decoded.npz"], cwd=tempdir)
        assert os.path.isfile(os.path.join(tempdir, "decoded.npz"))
