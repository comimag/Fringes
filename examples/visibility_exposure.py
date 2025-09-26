"""Visibility and Exposure."""

from fringes import Fringes
from fringes.filter import visibility, exposure
import matplotlib.pyplot as plt
from matplotlib import colors

f = Fringes()

I = f.encode()
Irec = I  # todo: replace this line with the recorded data, cf. example in 'record.py' as in 'record.py'
a, b, x = f.decode(Irec)

E = exposure(a, Irec)
V = visibility(a, b)

# show results
fig, axs = plt.subplots(nrows=f.K, ncols=f.D, squeeze=False, num="visibility 'V'")
norm = colors.Normalize(vmin=V.min(), vmax=V.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    for i in range(f.K):
        if d == 0:
            axs[i, d].set_ylabel(f"{i}-th set")
        t = d * f.K + i
        image = axs[i, d].imshow(V[t, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

fig, axs = plt.subplots(ncols=f.D, squeeze=False, num="exposure 'E'")
norm = colors.Normalize(vmin=E.max(), vmax=E.max())
for d in range(f.D):
    axs[0, d].set_title(f"{f.indexing[f.axis if f.D == 1 else d]}-direction")
    image = axs[0, d].imshow(E[d, :, :, 0], norm=norm)  # shows only first color channel
fig.colorbar(image, ax=axs)

plt.show()
