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
print(f.T)  # it consists of one frame only -> single shot application

I = f.encode()

# show fringe patterns
plt.figure()
plt.imshow(I[0])  # first frame is only frame
plt.show()
