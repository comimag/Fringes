"""Decode fringe pattern sequence."""

import fringes as frng
import matplotlib.pyplot as plt

f = frng.Fringes()
I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

a, b, x = f.decode(I_rec)

# display first frame  and first color channel of each result
plt.figure("registration 'x'")
plt.imshow(x[0, :, :, 0])
plt.colorbar()
plt.figure("modulation 'b'")
plt.imshow(b[0, :, :, 0])
plt.colorbar()
plt.figure("brightness 'a'")
plt.imshow(a[0, :, :, 0])
plt.colorbar()
plt.show()
