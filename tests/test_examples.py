import importlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("module", [p.stem for p in (Path(__file__).parent / ".." / "examples").resolve().glob("*.py")])
def test_examples(module, monkeypatch):
    monkeypatch.setattr(cv2, "waitKey", lambda x: None)
    monkeypatch.setattr(plt, "show", lambda: None)

    try:
        m = importlib.import_module(module)  # on import, the module is run
    except AttributeError as e:
        if module == "record":
            pass  # probably OpenCV didnt find any cameras   todo: test OpenCV didnt find any cameras
        else:
            raise e
    else:
        assert m.__doc__ is not None
    finally:
        cv2.destroyAllWindows()
        plt.close()
