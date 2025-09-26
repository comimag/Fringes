"""Frequency Division Multiplexing (1 direction, 2 sets).
https://doi.org/10.1364/OE.18.005229
"""

from fringes import Fringes
from matplotlib import pyplot as plt
import matplotlib.animation as animation

f = Fringes()
f.X = f.Y = 1024
f.v = 10, 13
f.f = 1, 2
f.D = 1
f.K = 2
f.N = 5  # 2 * max(f.f) + 1
f.FDM = True

I = f.encode()

# show fringe patterns
fig, ax = plt.subplots()
ims = [[ax.imshow(frame, cmap="gray", animated=True)] for frame in I]
ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=1000, repeat=True, blit=True)
plt.show()
