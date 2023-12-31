import cv2
import numpy as np

ROOT = "./2023/Magic with ruller/Surface restoration/"
H = 3508
W = 2480
TR_SIZE = 20

OVERSCAN_H = H - (H // TR_SIZE) * TR_SIZE
OVERSCAN_W = W - (W // TR_SIZE) * TR_SIZE
TR_AMOUNT_X = int((W - OVERSCAN_W) / TR_SIZE)
TR_AMOUNT_Y = int((H - OVERSCAN_H) / TR_SIZE)

img = np.zeros((H, W, 3))
img[:, :, :] = 255

for y_tr in range(TR_AMOUNT_Y):
    for x_tr in range(TR_AMOUNT_X):
        points = np.array([[OVERSCAN_W/2 + x_tr*TR_SIZE, OVERSCAN_H/2 + y_tr*TR_SIZE], [OVERSCAN_W/2 + (x_tr+1)*TR_SIZE,
                          OVERSCAN_H/2 + y_tr*TR_SIZE], [OVERSCAN_W/2 + x_tr*TR_SIZE, OVERSCAN_H/2 + (y_tr+1)*TR_SIZE]], dtype=int)
        cv2.fillPoly(img, pts=[points], color=(0, 0, 0))

cv2.imshow("out", img)
cv2.waitKey(0)
cv2.imwrite(ROOT + "generated patterns/out_super_small.png", img)
cv2.destroyAllWindows()
