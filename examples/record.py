"""Use the `Fringes` package to encode, record and decode data."""

import logging

# configure logging (this is optional)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s.%(funcName)s: %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger("fringes")
logger.addHandler(handler)
logger.setLevel("INFO")

import cv2
import fringes as frng
import numpy as np

# prepare window in which the fringe pattern sequence will be shown in fullscreen mode
cv2.namedWindow("Fringes", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Fringes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
left, top, width, height = cv2.getWindowImageRect("Fringes")

# prepare camera
camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # turn on autofocus
camera.set(cv2.CAP_PROP_AUTO_WB, 1)  # turn on whitebalance
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # turn on autoexposure

white = np.full((height, width), 255, np.uint8)  # white image
cv2.imshow("Fringes", white)  # display white image
key = cv2.waitKey(250)  # delay time of the screen for displaying the image

for _ in range(10):
    # let camera automatically set focus, whitebalance and exposure
    ret, image = camera.read()

camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn off autofocus
camera.set(cv2.CAP_PROP_AUTO_WB, 0)  # turn off whitebalance
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # turn off autoexposure

# configure and create fringe pattern sequence
f = frng.Fringes()
f.X = width
f.Y = height
I = f.encode()

# allocate empty image stack
I_rec = np.empty((f.T,) + image.shape, image.dtype)

# record
try:
    # record fringe patterns in a loop
    for t in range(f.T):
        # display fringe pattern with index 't' in fullscreen mode
        cv2.imshow("Fringes", I[t])
        key = cv2.waitKey(250)  # delay time of the screen for displaying the image

        # capture reflected fringe pattern; ensure the fringe pattern is not overexposed!
        ret, image = camera.read()  # note: OpenCV has color order 'BGR' (instead of 'RGB') for color images

        if ret:
            # save captured fringe pattern to image stack
            I_rec[t] = image
finally:
    # close window
    cv2.destroyWindow("Fringes")

    # release camera resources
    camera.release()

# analyze recorded fringe pattern sequence
a, b, x = f.decode(I_rec)
