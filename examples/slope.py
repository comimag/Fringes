"""Slope."""

from fringes import Fringes
import matplotlib.pyplot as plt

f = Fringes()

I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

a, b, x = f.decode(I_rec)

s = x  # todo: compute slope from calibrated setup

plt.figure("slope 'y'")
plt.imshow(s[1, :, :, 0])
plt.colorbar()
plt.figure("slope 'x'")
plt.imshow(s[0, :, :, 0])
plt.colorbar()
plt.show()
