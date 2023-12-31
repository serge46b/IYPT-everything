import numpy as np
import cv2

CANVAS_W = 500
CANVAS_H = 500


x = 0
while True:
    canvas = np.zeros((CANVAS_W, CANVAS_H, 3))
    cv2.line(canvas, ((x + 1)//2, 0), ((x + 1) //
             2, CANVAS_H - 1), (255, 255, 255), 1)
    cv2.line(canvas, (CANVAS_W - x//2 - 1, 0),
             (CANVAS_W - x//2 - 1, CANVAS_H - 1), (255, 255, 255), 1)
    cv2.imshow("image", canvas)
    key = cv2.waitKey(0)
    if key == 27:
        break
    elif key == ord('d'):
        x += 1
        if x >= CANVAS_W:
            x = CANVAS_W - 1
    elif key == ord('a'):
        x -= 1
        if x <= 0:
            x = 0

cv2.destroyAllWindows()
