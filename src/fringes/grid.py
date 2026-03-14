import logging

import numpy as np

logger = logging.getLogger(__name__)


def img(Y: int = 1024, X: int = 1024, a: float = 0):
    """Image coordinates."""
    yy, xx = np.indices((Y, X))
    return rot(xx, yy, a)


def cart(Y: int = 1024, X: int = 1024, a: float = 0):
    """Cartesian coordinates."""
    x = np.linspace(-X / 2 + 0.5, X / 2 - 0.5, X, endpoint=True)
    y = np.linspace(Y / 2 - 0.5, -Y / 2 + 0.5, Y, endpoint=True)
    xx, yy = np.meshgrid(x, y, copy=False)
    return rot(xx, yy, a)


def pol(Y: int = 1024, X: int = 1024, a: float = 0):
    """Polar coordinates."""
    xx, yy = cart(Y, X)
    pp = np.arctan2(yy, xx) / (2 * np.pi)
    rr = np.sqrt(xx**2 + yy**2) / min(X, Y)
    return rot(pp, rr, a)


def logpol(Y: int = 1024, X: int = 1024, a: float = 0):
    """Log-polar coordinates."""
    pp, rr = pol(Y, X)
    ll = np.log(rr) / (2 * np.pi)
    return rot(pp, ll, a)


def rot(uu, vv, a):
    """Rotate coordinate matrix."""
    if a % 360 == 0:
        return uu, vv

    t = np.deg2rad(a)

    if a % 90 == 0:
        c = np.cos(t)
        s = np.sin(t)
        # R = np.array([[c, -s], [s, c]])
        ur = c * uu - s * vv
        vr = s * uu + c * vv
        # [Y, 2] @ [2, 2] -> []
        # todo: matrix multiplication: @

        # # Coordinate matrix (N points, 2 columns [x, y])
        # coords = np.array([[1, 0], [0, 1], [1, 1]])
        #
        # # Apply rotation: (N, 2) @ (2, 2).T -> (N, 2)
        # rotated_coords = coords.dot(R.T)
        #
        # print(rotated_coords)
        # # Output: [[ 0.  1.] [-1.  0.] [-1.  1.]]
    else:
        ur = uu - vv * np.tan(t)
        vr = uu + vv / np.tan(t)

    return ur, vr


def inner_circ(Y: int = 1024, X: int = 1024):
    """Boolean mask with True values inside inscribed circle."""
    return pol(Y, X)[1] <= 0.5
