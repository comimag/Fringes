"""Curvature."""

from fringes import Fringes
from fringes.filter import curvature
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'

a, b, x = f.decode(Irec)

s = x  # todo: compute slope from calibrated setup

c = curvature(s)

# show first color channel of result
plt.figure("curvature 'c'")
plt.imshow(c[:, :, 0])
plt.colorbar()
plt.show()
