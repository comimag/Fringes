"""Slope."""

from fringes import Fringes
import matplotlib.pyplot as plt

f = Fringes()

I = f.encode()

Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'

a, b, x = f.decode(Irec)

s = x  # todo: compute slope from calibrated setup

# show first frame and first color channel of results
plt.figure("slope 's'")
plt.imshow(s[0, :, :, 0])
plt.colorbar()
plt.show()
