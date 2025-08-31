"""Visibility and Exposure."""

from fringes import Fringes
from fringes.filter import visibility, exposure
import matplotlib.pyplot as plt

f = Fringes()
I = f.encode()

Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'

a, b, x = f.decode(Irec)

V = visibility(a, b)
E = exposure(a, Irec)

# show first frame and first color channel of results
plt.figure("exposure 'E'")
plt.imshow(E[0, :, :, 0])
plt.colorbar()
plt.figure("visibility 'V'")
plt.imshow(V[0, :, :, 0])
plt.colorbar()
plt.show()
