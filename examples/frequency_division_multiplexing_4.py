"""Frequency Division Multiplexing (2 directions, 2 sets, static pattern).
https://publikationen.bibliothek.kit.edu/1000088264
"""

import fringes as frng
from matplotlib import pyplot as plt

f = frng.Fringes()
f.X = f.Y = 1024
f.v = [[5, 13],
       [7, 11]]
f.f = f.v
f.D = 2
f.K = 2
f.N = 27  # 2 * max(f.f) + 1
f.FDM = True
f.static = True

I = f.encode()

plt.imshow(I[0], cmap="gray")
plt.show()
