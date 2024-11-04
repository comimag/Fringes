import glob
import importlib
import os

import matplotlib.pyplot as plt


def test_examples(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)  # https://stackoverflow.com/questions/60127165/pytest-test-function-that-creates-plots

    modules = glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "*.py"))
    names = [os.path.basename(f)[:-3] for f in modules if f != "__init__.py"]

    for module in names:
        if module in ["height"]:  # todo: test height example
            continue

        m = importlib.import_module(module)  # on import, the module is run
        assert m.__doc__ is not None, f"Module '{module}' has no docstring."
