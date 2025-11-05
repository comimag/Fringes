"""Decode fringe pattern sequence."""

from fringes import Fringes
import matplotlib.pyplot as plt
from matplotlib import colors

f = Fringes()

I = f.encode()
Irec = I  # replace this line with recording data as in 'record.py'
a, b, x = f.decode(Irec)

# show results
fig, axs = plt.subplots(ncols=f.D, squeeze=False, num="coordinate 'x'")
norm = colors.Normalize(vmin=x.min(), vmax=x.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    image = axs[0, d].imshow(x[d, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(nrows=f.K, ncols=f.D, squeeze=False, num="modulation 'b'")
norm = colors.Normalize(vmin=b.min(), vmax=b.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    for i in range(f.K):
        if d == 0:
            axs[i, d].set_ylabel(f"{i}-th set")
        t = d * f.K + i
        image = axs[i, d].imshow(b[t, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(ncols=f.D, squeeze=False, num="brightness 'a'")
norm = colors.Normalize(vmin=a.min(), vmax=a.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    image = axs[0, d].imshow(a[d, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

plt.show()
