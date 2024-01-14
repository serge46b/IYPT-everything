import os
import cv2
import json
import numpy as np
from cv2 import aruco


ROOT = "./Calibration/camera calibration samples/s46b camera 0.8/"
VIDEO_SOURCE = ROOT + "calibration.mp4"
OUT_SV_PATH = ROOT

ARUCO_DICTIONARY = aruco.DICT_4X4_1000

DISPLAY_IMG_HEIGHT = 500  # units - pixels

AUTO_MODE = False
MARKERS_ON_BOARD = 4*5
FRAME_SKIP = 10

# CAMERA_RES_WIDTH = 1280
# CAMERA_RES_HEIGHT = 720
# CAMERA_RES_WIDTH = 1920
# CAMERA_RES_HEIGHT = 1080
# CAMERA_RES_WIDTH = 2400
# CAMERA_RES_HEIGHT = 1028


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
arucoParams = aruco.DetectorParameters()
camera = cv2.VideoCapture(VIDEO_SOURCE)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RES_WIDTH)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RES_HEIGHT)
print("processing...")
pause_flag = False
l_w_frame = 1
while True:
    ret, img = camera.read()
    if not ret:
        print("Stream accidently closed or video file ended")
        break
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        im_gray, aruco_dict, parameters=arucoParams)
    img_aruco = img.copy()
    if len(corners) == 0:
        pass
    else:
        img_aruco = aruco.drawDetectedMarkers(
            img_aruco, corners, ids, (0, 255, 0))
        if AUTO_MODE and len(ids) == MARKERS_ON_BOARD:
            if int(camera.get(cv2.CAP_PROP_POS_FRAMES)) - l_w_frame > FRAME_SKIP:
                l_w_frame = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.imwrite(
                    f"{OUT_SV_PATH}frame {l_w_frame}.png", img)
                print(
                    f"progress {int(l_w_frame*100/camera.get(cv2.CAP_PROP_FRAME_COUNT))}% | frame {l_w_frame} saved        ", end='\r')
        else:
            l_w_frame = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
    q = 0
    if not AUTO_MODE:
        cv2.imshow("markers", resize_with_aspect_ratio(
            img_aruco, height=DISPLAY_IMG_HEIGHT))
        q = cv2.waitKey(1)
    if pause_flag:
        q = cv2.waitKey(0)
        if q == ord("a"):
            print("prev frame")
            camera.set(cv2.CAP_PROP_POS_FRAMES,
                       camera.get(cv2.CAP_PROP_POS_FRAMES) - 2)
            # camera.set(cv2.CAP_PROP_POS_MSEC, camera.get(
            #     cv2.CAP_PROP_POS_MSEC) - 2*(1000/camera.get(cv2.CAP_PROP_FPS)))
            # print(camera.get(cv2.CAP_PROP_POS_FRAMES))
        elif q == ord("d"):
            print("next frame")
        else:
            # camera.set(cv2.CAP_PROP_POS_MSEC, camera.get(
            # cv2.CAP_PROP_POS_MSEC) - (1000/camera.get(cv2.CAP_PROP_FPS)))
            camera.set(cv2.CAP_PROP_POS_FRAMES,
                       camera.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        if q == ord("s"):
            cv2.imwrite(
                f"{OUT_SV_PATH}frame {int(camera.get(cv2.CAP_PROP_POS_FRAMES))}.png", img)
            print(f"saved frame {int(camera.get(cv2.CAP_PROP_POS_FRAMES))}")
    if q == 27:
        print("Requested exit")
        break
    elif q == ord(" ") and not AUTO_MODE:
        pause_flag = not pause_flag
        print("pause" if pause_flag else "play")

camera.release()
cv2.destroyAllWindows()
