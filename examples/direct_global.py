"""Direct and global illumination component.
https://dl.acm.org/doi/abs/10.1145/1179352.1141977"""

import fringes as frng
import matplotlib.pyplot as plt

f = frng.Fringes()
f.v = 99, 100, 101  # spatial frequency 'v' must be high enough but still resolved by the recording camera

I = f.encode()

I_rec = I  # todo: replace this line with recording data as in 'record.py'

res = f.decode(I_rec, verbose=True)

# make use of namedtuple:
a = res.brightness
b = res.modulation
x = res.registration
d = res.direct
g = res.glob

plt.figure("global 'g'")
plt.imshow(g[0, :, :, 0])
plt.colorbar()
plt.figure("direct 'd'")
plt.imshow(d[0, :, :, 0])
plt.colorbar()
plt.show()
