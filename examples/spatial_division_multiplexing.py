"""Spatial Division Multiplexing.
https://doi.org/10.1117/12.816472
"""

from fringes import Fringes
from matplotlib import pyplot as plt

f = Fringes()
f.X = f.Y = 1024
f.v = 10
f.D = 2
f.SDM = True

I = f.encode()

plt.imshow(I[0], cmap="gray")
plt.show()
