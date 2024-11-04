"""Visibility and Exposure."""

import fringes as frng
import matplotlib.pyplot as plt

f = frng.Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

res = f.decode(I_rec, verbose=True)

# make use of namedtuple:
a = res.brightness
b = res.modulation
x = res.registration
E = res.exposure
V = res.visibility

plt.figure("exposure 'E'")
plt.imshow(E[0, :, :, 0])
plt.colorbar()
plt.figure("visibility 'V'")
plt.imshow(V[0, :, :, 0])
plt.colorbar()
plt.show()
