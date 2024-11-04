import matplotlib.pyplot as plt
import numpy as np

Imax = 1
X = 1000
E = np.linspace(0, 1, X)
V = np.linspace(0, 1, X)
EE, VV = np.meshgrid(E, V)
v = np.round(np.sqrt(X))
k = 2 * np.pi * v
x = np.arange(X) / X
p0 = np.pi

I = Imax * EE * (1 + VV * np.cos(k * x[:, None] - p0))

invalid = EE > 1 / (1 + VV)
I[invalid] = np.nan

plt.figure()
plt.imshow(I, cmap="gray", origin="lower")
plt.xticks([0, 999], ["0", "1"])
plt.yticks([0, 999], ["0", "1"])
plt.xlabel("E")
plt.ylabel("V", rotation="horizontal")
plt.show()
