"""Wavelength Division Multiplexing.

https://doi.org/10.1117/1.602151
"""

from fringes import Fringes

f = Fringes()
f.X = f.Y = 1024
f.N = 3
f.WDM = True

I = f.encode()

# show fringe patterns
import matplotlib.animation as animation
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ims = [[ax.imshow(frame, animated=True)] for frame in I]
ani = animation.ArtistAnimation(fig, ims, interval=250, repeat_delay=1000, repeat=True, blit=True)
plt.show()
