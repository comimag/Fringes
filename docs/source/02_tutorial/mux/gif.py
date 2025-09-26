import imageio
imageio.mimsave("WDM.gif", I, loop=0, fps=4)
imageio.mimsave("SDM.gif", I[..., -1], loop=0, fps=4)