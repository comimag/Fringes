"""Direct and global illumination component.
https://dl.acm.org/doi/abs/10.1145/1179352.1141977"""

from fringes import Fringes
from fringes.filter import direct, indirect
import matplotlib.pyplot as plt
from matplotlib import colors

f = Fringes()
f.v = 99, 100, 101  # spatial frequency 'v' must be high enough but still resolved by the recording camera

I = f.encode()
Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'
a, b, x = f.decode(Irec)

d = direct(b)
g = indirect(a, b)

# show results
fig, axs = plt.subplots(nrows=f.K, ncols=f.D)
fig.suptitle("global 'g'")
norm = colors.Normalize(vmin=g.min(), vmax=g.max())
images = []
for i_d in range(f.D):
    axs[0, i_d].set_title(f"{f.indexing[f.axis if f.D == 1 else i_d]}-direction")
    for i in range(f.K):
        if i_d == 0:
            axs[i, i_d].set_ylabel(f"{i}-th set")
        t = i_d * f.K + i
        images.append(axs[i, i_d].imshow(g[t, :, :, 0], norm=norm))  # shows only first color channel
fig.colorbar(images[0], ax=axs)

fig, axs = plt.subplots(nrows=f.K, ncols=f.D)
fig.suptitle("direct 'd'")
norm = colors.Normalize(vmin=d.min(), vmax=d.max())
images = []
for i_d in range(f.D):
    axs[0, i_d].set_title(f"{f.indexing[f.axis if f.D == 1 else i_d]}-direction")
    for i in range(f.K):
        if i_d == 0:
            axs[i, i_d].set_ylabel(f"{i}-th set")
        t = i_d * f.K + i
        images.append(axs[i, i_d].imshow(d[t, :, :, 0], norm=norm))  # shows only first color channel
fig.colorbar(images[0], ax=axs)

plt.show()
