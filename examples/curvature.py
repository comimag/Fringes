"""Curvature."""

from fringes import Fringes
from fringes.filter import curvature

f = Fringes()

I = f.encode()
Irec = I  # todo: replace this line with recording data (cf. example in 'record.py')
a, b, x = f.decode(Irec)

s = x  # todo: compute slope from calibrated setup
c = curvature(s)

# show result
import matplotlib.pyplot as plt

plt.figure("curvature 'c'")
plt.imshow(c[:, :, 0])  # only first color channel
plt.colorbar()
plt.show()
