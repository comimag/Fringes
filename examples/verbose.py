"""Decode verbose results."""

from fringes import Fringes
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

f = Fringes()

I = f.encode()
Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'
dec = f.decode(Irec, verbose=True)  # here, a namedtuple is returned (see usage below)

# show verbose results
fig, axs = plt.subplots(ncols=f.D, squeeze=False, num="uncertainty 'u'")
norm = colors.Normalize(vmin=dec.u.min(), vmax=dec.u.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    image = axs[0, d].imshow(dec.u[d, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(ncols=f.D, squeeze=False, num="residuals 'r'")
norm = colors.Normalize(vmin=dec.r.max(), vmax=dec.r.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    image = axs[0, d].imshow(dec.r[d, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(nrows=f.K, ncols=f.D, squeeze=False, num="fringe orders 'k'")
norm = colors.Normalize(vmin=dec.k.min(), vmax=dec.k.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    for i in range(f.K):
        if d == 0:
            axs[i, d].set_ylabel(f"{i}-th set")
        t = d * f.K + i
        image = axs[i, d].imshow(dec.k[t, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(nrows=f.K, ncols=f.D, squeeze=False, num="phase 'p'")
norm = colors.Normalize(vmin=0, vmax=2 * np.pi)
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")  # shows only first color channel
    for i in range(f.K):
        if d == 0:
            axs[i, d].set_ylabel(f"{i}-th set")
        t = d * f.K + i
        image = axs[i, d].imshow(dec.p[t, :, :, 0], norm=norm)
cbar = fig.colorbar(image, ax=axs, ticks=[0, np.pi, 2 * np.pi])
cbar.ax.set_yticklabels(["0", f"$\\pi$", f"$2\\pi$"])

plt.show()
