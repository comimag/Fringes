"""Frequency Division Multiplexing (2 directions, 2 sets, static pattern).
https://publikationen.bibliothek.kit.edu/1000088264
"""

from fringes import Fringes
from matplotlib import pyplot as plt
import matplotlib.animation as animation

f = Fringes()
f.X = f.Y = 1024
f.v = [[5, 13], [7, 11]]
f.f = f.v
f.D = 2
f.K = 2
f.N = 27  # 2 * max(f.f) + 1
f.FDM = True
f.static = True

I = f.encode()

# show fringe patterns
fig, ax = plt.subplots()
ims = [[ax.imshow(frame, cmap="gray", animated=True)] for frame in I]
ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=1000, repeat=True, blit=True)
plt.show()
