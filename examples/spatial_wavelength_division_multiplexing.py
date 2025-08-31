"""Spatial and Wavelength Division Multiplexing.
https://doi.org/10.1364/OE.24.027993
"""

from fringes import Fringes
from matplotlib import pyplot as plt

f = Fringes()
f.X = f.Y = 1024
f.v = 10
f.D = 2
f.K = 1
f.SDM = True
f.N = 3
f.WDM = True

I = f.encode()

# show first frame
plt.imshow(I[0])
plt.show()
