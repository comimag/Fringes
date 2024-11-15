"""Wavelength Division Multiplexing.
https://doi.org/10.1117/1.602151
"""

from fringes import Fringes
from matplotlib import pyplot as plt

f = Fringes()
f.X = f.Y = 1024
f.v = 10
f.N = 3
f.WDM = True

I = f.encode()

plt.imshow(I[0])
plt.show()
