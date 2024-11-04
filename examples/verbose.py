"""Decode verbose results."""

import fringes as frng
import matplotlib.pyplot as plt

f = frng.Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

# decode and return additional results:
res = f.decode(I_rec, verbose=True)

# make use of namedtuple:
a = res.brightness
b = res.modulation
x = res.registration
p = res.phase
k = res.order
r = res.residuals
u = res.uncertainty

# display first frame of results
plt.figure("phase 'p'")
plt.imshow(p[0, :, :, 0])
plt.colorbar()
plt.figure("fringe order 'k'")
plt.imshow(k[0, :, :, 0])
plt.colorbar()
plt.figure("residuals 'r'")
plt.imshow(r[0, :, :, 0])
plt.colorbar()
plt.figure("uncertainty 'u'")
plt.imshow(u[0, :, :, 0])
plt.colorbar()
plt.show()
