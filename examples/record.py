"""Configure, encode and record fringe patterns using `Fringes` and `OpenCV`."""

import cv2
import numpy as np
from fringes import Fringes

# prepare window (in which the fringe patterns will be shown in fullscreen mode)
cv2.namedWindow("Fringes", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Fringes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
left, top, width, height = cv2.getWindowImageRect("Fringes")

# prepare camera
camera = cv2.VideoCapture(0)

delay = 500  # delay time of the screen until the image is actually shown
white = np.full((height, width), 255, np.uint8)  # white image
cv2.imshow("Fringes", white)  # display white image
cv2.waitKey(delay)  # wait delay time

camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # turn on autofocus
camera.set(cv2.CAP_PROP_AUTO_WB, 1)  # turn on white balance
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # turn on autoexposure

for _ in range(100):
    ret, image = camera.read()  # let camera set focus, white balance and exposure

camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn off autofocus
camera.set(cv2.CAP_PROP_AUTO_WB, 0)  # turn off white balance
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # turn off autoexposure

# configure and encode fringe patterns
f = Fringes()
f.X = width
f.Y = height
I = f.encode()

# record fringe patterns
Irec = np.empty((f.T,) + image.shape, image.dtype)  # allocate empty image stack
try:
    for t in range(f.T):  # record fringe patterns in a loop
        cv2.imshow("Fringes", I[t])  # display fringe pattern in fullscreen mode
        key = cv2.waitKey(delay)  # wait delay time

        ret, image = camera.read()  # capture the fringe pattern (AVOID OVEREXPOSURE !!!)

        if ret:
            Irec[t] = image  # save captured fringe pattern to image stack
finally:
    cv2.destroyWindow("Fringes")  # close window
    camera.release()  # release camera resources

# show recorded fringe patterns
for t, frame in enumerate(Irec, start=1):
    cv2.imshow(f"frame {t}/{f.T}", frame)
    cv2.waitKey(0)
