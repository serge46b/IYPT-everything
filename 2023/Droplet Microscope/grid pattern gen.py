import numpy as np
import cv2

ROOT = "./2023/Droplet Microscope/"
H = 3508
W = 2480
LINE_SIZE = 5

LINES_AMOUNT_Y = H // LINE_SIZE // 2 + 1
LINES_AMOUNT_X = W // LINE_SIZE // 2 + 1

# print(LINES_AMOUNT_Y)

img = np.zeros((H, W, 3))
img[:, :, :] = 255

for y_tr in range(LINES_AMOUNT_Y):
    cv2.rectangle(img, (0, y_tr*LINE_SIZE*2),
                  (W, y_tr*LINE_SIZE*2+LINE_SIZE-1), (0, 0, 0), -1)

for x_tr in range(LINES_AMOUNT_X):
    cv2.rectangle(img, (x_tr*LINE_SIZE*2, 0),
                  (x_tr*LINE_SIZE*2+LINE_SIZE-1, H), (0, 0, 0), -1)

cv2.imshow("out", img)
cv2.waitKey(0)
cv2.imwrite(ROOT+"grid patterns/out_grid_5px.png", img)
cv2.destroyAllWindows()
