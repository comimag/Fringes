"""Frequency Division Multiplexing (2 directions, 2 sets).
https://publikationen.bibliothek.kit.edu/1000088264
"""

from fringes import Fringes
from matplotlib import pyplot as plt

f = Fringes()
f.X = f.Y = 1024
f.v = [[10, 13],
       [10, 13]]
f.f = [[1, 2],
       [3, 4]]
f.D = 2
f.K = 2
f.N = 9  # 2 * max(f.f) + 1
f.FDM = True

I = f.encode()

plt.imshow(I[0], cmap="gray")
plt.show()
