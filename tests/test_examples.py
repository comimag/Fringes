import glob
import importlib
import os

import cv2
import matplotlib.pyplot as plt


def test_examples(monkeypatch):
    monkeypatch.setattr(cv2, "waitKey", lambda x: None)
    monkeypatch.setattr(plt, "show", lambda: None)

    for module in [
        os.path.basename(f)[:-3] for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "*.py"))
    ]:

        m = importlib.import_module(module)  # on import, the module is run
        plt.close()
        cv2.destroyAllWindows()

        assert m.__doc__ is not None, f"Module '{module}' has no docstring."
