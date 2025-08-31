"""Direct and global illumination component.
https://dl.acm.org/doi/abs/10.1145/1179352.1141977"""

from fringes import Fringes
from fringes.filter import direct, indirect
import matplotlib.pyplot as plt

f = Fringes()
f.v = 99, 100, 101  # spatial frequency 'v' must be high enough but still resolved by the recording camera

I = f.encode()

Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'

a, b, x = f.decode(Irec)

d = direct(a)
g = indirect(a, b)

# show first frame and first color channel of results
plt.figure("global 'g'")
plt.imshow(g[0, :, :, 0])
plt.colorbar()
plt.figure("direct 'd'")
plt.imshow(d[0, :, :, 0])
plt.colorbar()
plt.show()
