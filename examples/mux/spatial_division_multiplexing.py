"""Spatial Division Multiplexing.
https://doi.org/10.1117/12.816472
"""

from fringes import Fringes
from matplotlib import pyplot as plt
import matplotlib.animation as animation

f = Fringes()
f.X = f.Y = 1024
f.v = 10, 13
f.D = 2
f.SDM = True

I = f.encode()

# show fringe patterns
fig, ax = plt.subplots()
ims = [[ax.imshow(frame, cmap="gray", animated=True)] for frame in I]
ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=1000, repeat=True, blit=True)
plt.show()
