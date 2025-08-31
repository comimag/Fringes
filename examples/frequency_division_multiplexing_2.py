"""Frequency Division Multiplexing (2 directions, 1 set).
https://doi.org/10.1515/aot-2014-0032
"""

from fringes import Fringes
from matplotlib import pyplot as plt

f = Fringes()
f.X = f.Y = 1024
f.v = 10
f.f = [[1], [2]]
f.D = 2
f.K = 1
f.N = 5  # 2 * max(f.f) + 1
f.FDM = True

I = f.encode()

# show first frame
plt.imshow(I[0], cmap="gray")
plt.show()
