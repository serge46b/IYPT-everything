import cv2
import json
import numpy as np
from cv2 import aruco


ROOT = "./2023/magnetic gearbox/"
CALIB_FILE_PATH = ROOT + \
    "../../Calibration/calibration files/Nikon (stock lens).json"
# CALIB_FILE_PATH = ROOT + "../../Calibration/calibration.json"
VIDEO_SOURCE = ROOT + "exp videos/ArUco test 2.MOV"
OUT_SV_PATH = ROOT + \
    (str(VIDEO_SOURCE) if VIDEO_SOURCE is int else VIDEO_SOURCE[VIDEO_SOURCE.rfind(
        '/'):VIDEO_SOURCE.rfind('.')]) + ".csv"

ARUCO_DICTIONARY = aruco.DICT_4X4_1000
MARKER_SIZE = 0.048  # units - meters

DISPLAY_IMG_HEIGHT = 700  # units - pixels
MARKER_AXIS_DISPLAY_LENGTH = 0.05  # units - meters
DEBUG = True

# CAMERA_RES_WIDTH = 1280
# CAMERA_RES_HEIGHT = 720
CAMERA_RES_WIDTH = 1920
CAMERA_RES_HEIGHT = 1080


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
camera = cv2.VideoCapture(VIDEO_SOURCE, apiPreference=cv2.CAP_ANY, params=[
    cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RES_WIDTH,
    cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RES_HEIGHT])
# with open(CALIB_FILE_PATH) as f:
#     loadeddict = json.load(f)
with open(CALIB_FILE_PATH) as f:
    loadeddict = json.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)
h, w = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
    camera.get(cv2.CAP_PROP_FRAME_WIDTH))
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

res_file = open(OUT_SV_PATH, "w")
res_file.write("frame, id, tx, ty, tz, rx, ry, rz\n")

while True:
    ret, img = camera.read()
    if not ret:
        print("Stream accidently closed")
        break
    if DEBUG:
        img_aruco = img.copy()
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        im_gray, aruco_dict, parameters=arucoParams)
    if len(corners) == 0:
        pass
    else:
        if DEBUG:
            img_aruco = aruco.drawDetectedMarkers(
                img_aruco, corners, ids, (0, 255, 0))
        for i in range(len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, newcameramtx,
                                                                       dist)
            # print(tvec[0], rvec[0])
            res_file.write(
                f"{int(camera.get(cv2.CAP_PROP_POS_FRAMES))}, {i}, {tvec[0, 0, 0]}, {tvec[0, 0, 1]}, {tvec[0, 0, 2]}, {rvec[0, 0, 0]}, {rvec[0, 0, 1]}, {rvec[0, 0, 2]}\n")
            if DEBUG:
                img_aruco = cv2.drawFrameAxes(img_aruco, newcameramtx, dist, rvec, tvec,
                                              MARKER_AXIS_DISPLAY_LENGTH)
            # rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], MARKER_SIZE, mtx,
            #                                                            dist)
            # img_aruco = cv2.drawFrameAxes(img_aruco, mtx, dist, rvec, tvec,
            #                               5)
    if DEBUG:
        q = cv2.waitKey(1)
        cv2.imshow("estimation", resize_with_aspect_ratio(
            img_aruco, height=DISPLAY_IMG_HEIGHT))
        if q == 27:
            print("Requested exit")
            break

res_file.close()
cv2.destroyAllWindows()
