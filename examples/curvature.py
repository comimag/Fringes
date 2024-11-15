"""Curvature."""

from fringes import Fringes
from fringes.filter import curvature
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

a, b, x = f.decode(I_rec)

c = curvature(x)

plt.figure("curvature 'c'")
plt.imshow(c[0, :, :, 0])
plt.colorbar()
plt.show()
