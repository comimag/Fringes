import numpy as np
# todo: import numba as nb

# grid = "logpol"
#
# if grid == "img":
#     # img2img
#     u = i
#     v = j
#
#     # img2img
#     i = u
#     j = v
# elif grid == "cart":
#     # img2cart
#     x = i - (X - 1) / 2
#     y = (Y - 1) / 2 - j
#
#     # cart2img
#     i = x + (X - 1) / 2
#     j = (Y - 1) / 2 - y
# elif grid == "pol":
#     # cart2pol
#     p = np.arctan2(y, x) / (2 * np.pi) * (L - 1) / 2
#     r = np.sqrt(x ** 2 + y ** 2)
#
#     # pol2cart
#     x = r * np.cos(p / (L - 1) * 2 * 2 * np.pi)
#     y = r * np.sin(p / (L - 1) * 2 * 2 * np.pi)
# elif grid == "logpol":
#     # pol2logpol
#     p = p
#     l = np.log(r)
#
#     # logpol2pol
#     p = p
#     r = np.exp(l)


def img(Y: int = 720, X: int = 720, a: float = 0):
    # y, x = np.mgrid[:Y, :X]
    y, x = np.indices((Y, X))
    return rot(x, y, a)


def cart(Y: int = 720, X: int = 720, a: float = 0):
    x = np.linspace(-X / 2, X / 2, X, endpoint=True)
    y = np.linspace(Y / 2, -Y / 2, Y, endpoint=True)
    xx, yy = np.meshgrid(x, y)
    return rot(xx, yy, a)


def pol(Y: int = 720, X: int = 720, a: float = 0, centered: bool = True):
    xx, yy = cart(Y, X)
    pp = np.arctan2(yy, xx) / (2 * np.pi) * max(X, Y)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    rr /= np.sqrt((min(X, Y) / max(X, Y)) ** 2)
    return rot(pp, rr, a)


def logpol(Y: int = 720, X: int = 720, a: float = 0, centered: bool = True):
    pp, rr = pol(Y, X)
    L = max(X, Y)
    ll = np.log(rr / L + 1) * L  # with rr in [0, 1] -> ll in [0, 0.693]; important is only the logarithmical progression
    return rot(pp, ll, a)


def rot(uu, vv, a):
    # a *= -1
    if a % 360 == 0:
        return uu, vv

    t = a / 360 * 2 * np.pi  # angle in radians (theta)

    if a % 90 == 0:
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[c, -s], [s, c]])
        # R = np.matrix([[c, -s], [s, c]])
        ur = R[0, 0] * uu + R[0, 1] * vv
        vr = R[1, 0] * uu + R[1, 1] * vv
        # u = np.dot(uu, R)  # todo: matrix multiplication
        # v = np.dot(vv, R)
    else:
        ur = uu - vv * np.tan(t)
        vr = uu + vv / np.tan(t)

        # c = np.cos(t)
        # s = np.sin(t)
        # R = np.array([[c, -s], [s, c]])
        # ur = R[0, 0] * uu + R[0, 1] * vv
        # vr = R[1, 0] * uu + R[1, 1] * vv

    return ur, vr


def innercirc(Y: int = 720, X: int = 720):
    """Mask with area inside inscribed circle."""
    return pol(Y, X)[1] <= min(X, Y) / 2


# todo: coordinate transformations, add angle
def cart2pol(uv, a: float = 0):
    xx = uv[0]
    yy = uv[1]
    pp = np.arctan2(yy, xx)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    return np.stack(rot(pp, rr, a), axis=0)


def pol2cart(uv, a: float = 0):
    pp = uv[0] * 2 * np.pi
    rr = uv[1]
    xx = rr * np.cos(pp)
    yy = rr * np.sin(pp)
    return np.stack(rot(xx, yy, a), axis=0)


def pol2logpol(uv, a: float = 0):
    pp = uv[0]
    rr = uv[1]
    ll = np.log(rr)
    return np.stack(rot(pp, ll, a), axis=0)


def logpol2pol(uv, a: float = 0):
    pp = uv[0]
    ll = uv[1]
    rr = np.exp(ll) - 1
    return np.stack(rot(pp, rr, a), axis=0)


def cart2logpol(uv, a: float = 0):
    xx = uv[0]
    yy = uv[1]
    pp = np.arctan2(yy, xx) / (2 * np.pi)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    ll = np.log(rr)
    return np.stack((pp, ll), axis=0)


def logpol2cart(uv, a: float = 0):
    pp = uv[0] * 2 * np.pi
    ll = uv[1]
    rr = np.exp(ll)
    xx = rr * np.cos(pp)
    yy = rr * np.sin(pp)
    return np.stack((xx, yy), axis=0)


def cart2img(uv, a: float = 0):  # todo: test this
    xc = uv[0]
    yc = uv[1]
    L = min(xc.shape) - 1
    xi = xc * L / 2 + L / 2
    # yj = L / 2 - yc * L / 2
    yj = L / 2 * (1 - yc)
    return np.stack((xi, yj), axis=0)


def img2cart(uv, a: float = 0):
    xi = uv[0]
    yi = uv[1]
    Y, X = xi.shape
    xc = xi
    yc = yi
    return np.stack((xc, yc), axis=0)
