"""Decode verbose results."""

from fringes import Fringes
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

# decode and return additional results:
dec = f.decode(I_rec, verbose=True)

# make use of namedtuple:
a = dec.brightness
b = dec.modulation
x = dec.registration
p = dec.phase
k = dec.order
r = dec.residuals
u = dec.uncertainty

# display first frame of results
plt.figure("uncertainty 'u'")
plt.imshow(u[0, :, :, 0])
plt.colorbar()
plt.figure("residuals 'r'")
plt.imshow(r[0, :, :, 0])
plt.colorbar()
plt.figure("fringe order 'k'")
plt.imshow(k[0, :, :, 0])
plt.colorbar()
plt.figure("phase 'p'")
plt.imshow(p[0, :, :, 0])
plt.colorbar()
plt.show()
