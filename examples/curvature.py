"""Curvature."""

import fringes as frng
import matplotlib.pyplot as plt

f = frng.Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

a, b, x = f.decode(I_rec)

c = frng.curvature(x)

plt.figure("curvature 'c'")
plt.imshow(c[0, :, :, 0])
plt.colorbar()
plt.show()
