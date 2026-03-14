"""Visibility and Exposure."""

from fringes import Fringes
from fringes.filter import exposure, visibility

f = Fringes()

I = f.encode()
Irec = I  # todo: replace this line with recording data (cf. example in 'record.py')
a, b, x = f.decode(Irec)

E = exposure(a, Irec)
V = visibility(a, b)

# show results
import matplotlib.pyplot as plt
from matplotlib import colors

fig, axs = plt.subplots(nrows=f.K, ncols=f.D, squeeze=False, num="visibility 'V'")
norm = colors.Normalize(vmin=V.min(), vmax=V.max())
for d in range(f.D):
    for i in range(f.K):
        if d == 0:
            axs[i, d].set_ylabel(f"{i}-th set")
        t = d * f.K + i
        image = axs[i, d].imshow(V[t, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(ncols=f.D, squeeze=False, num="exposure 'E'")
norm = colors.Normalize(vmin=E.max(), vmax=E.max())
for d in range(f.D):
    axs[0, d].set_title(f"{'yx'[f.axes[d]]}-direction")
    image = axs[0, d].imshow(E[d, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

plt.show()
