"""Slope."""

from fringes import Fringes

f = Fringes()

I = f.encode()
Irec = I  # todo: replace this line with the recorded data (cf. example in 'record.py')
a, b, x = f.decode(Irec)

s = x  # todo: compute slope from calibrated setup

# show results
import matplotlib.pyplot as plt
from matplotlib import colors

fig, axs = plt.subplots(ncols=f.D)
fig.suptitle("slope 's'")
norm = colors.Normalize(vmin=s.min(), vmax=s.max())
images = []
for d in range(f.D):
    axs[d].set_title(f"{'yx'[f.axes[d]]}-direction")
    images.append(axs[d].imshow(s[d, :, :, 0], norm=norm))  # shows only first color channel
fig.colorbar(images[0], ax=axs)
plt.show()
