"""Decode fringe pattern sequence."""

from fringes import Fringes
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

Irec = I  # todo: replace this line with recording data as in 'record.py'

a, b, x = f.decode(Irec)

# show first frame and first color channel of results
plt.figure("coordinate 'x'")
plt.imshow(x[0, :, :, 0])
plt.colorbar()
plt.figure("modulation 'b'")
plt.imshow(b[0, :, :, 0])
plt.colorbar()
plt.figure("brightness 'a'")
plt.imshow(a[0, :, :, 0])
plt.colorbar()
plt.show()
