"""Visibility and Exposure."""

from fringes import Fringes
from fringes.filter import visibility, exposure
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

a, b, x = f.decode(I_rec)

V = visibility(a, b)
E = exposure(a, I_rec)

plt.figure("exposure 'E'")
plt.imshow(E[0, :, :, 0])
plt.colorbar()
plt.figure("visibility 'V'")
plt.imshow(V[0, :, :, 0])
plt.colorbar()
plt.show()
