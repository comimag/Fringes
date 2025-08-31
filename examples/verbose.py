"""Decode verbose results."""

from fringes import Fringes
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'

# decode and return additional results:
dec = f.decode(Irec, verbose=True)

# make use of namedtuple:
a = dec.a
b = dec.b
x = dec.x
p = dec.p
k = dec.k
r = dec.r
u = dec.u

# show first frame and first color channel of results
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
