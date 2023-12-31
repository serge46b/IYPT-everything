import cv2
import json
import numpy as np
from cv2 import aruco


ROOT = "./2023/Cushion Catapult/"
CALIB_DATA_PATH = ROOT + "../Calibration/calibration.json"
USE_CONFIG = True
CONFIG_PATH = ROOT + "config.json"
USING_CAMERA = True
VIDEO_IN_PATH = ROOT + "videos/test.mp4"

FRAME_AXIS_DCT = aruco.DICT_5X5_1000
FRAME_AXIS_MARKER_SIZE = 5  # Numbers from head, need to be changed
FRAME_AXIS_MARKER_SEP = 2  # Numbers from head, need to be changed
IGNORE_FA_INVIS = True
FOLOWING_ARUCO = False
FOLOWING_DCT = aruco.DICT_5X5_1000
FOLOWING_MARKER_SIZE = 5  # Numbers from head, need to be changed
FOLOWING_MARKER_SEP = 2  # Numbers from head, need to be changed
IGNORE_FW_INVIS = False
FW_BASIC_SIZE = 0.6  # Numbers from head
FW_OBJ_POINTS = np.array([(-FW_BASIC_SIZE/2, FW_BASIC_SIZE/2, 0), (FW_BASIC_SIZE/2, FW_BASIC_SIZE/2, 0),
                          (FW_BASIC_SIZE/2, -FW_BASIC_SIZE/2, 0), (-FW_BASIC_SIZE/2, -FW_BASIC_SIZE/2, 0)]).astype("float32")

DISPLAY_IMG_HEIGHT = 700

TR_OBJ_COLOR_LB = (44, 100, 66)
TR_OBJ_COLOR_HB = (71, 142, 103)
FILTERING_ON = True


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


def detect_objects(image, mtx, dist, detect_dct=None):
    if detect_dct is not None:
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(
            im_gray, detect_dct, parameters=arucoParams)
        # aruco.drawDetectedMarkers(
        #     img_with_drawings, corners, ids, (0, 255, 0))
        rvecs, tvecs = [], []
        for c in corners:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(c, FRAME_AXIS_MARKER_SIZE, mtx,
                                                            dist)
            rvecs.append(rvec)
            tvecs.append(tvec)
        return rvecs, tvecs, corners, ids
    mask = cv2.inRange(image, TR_OBJ_COLOR_LB, TR_OBJ_COLOR_HB)
    # _, mask = cv2.threshold(
    # image[:, :, 1], 200, 255, cv2.THRESH_BINARY)
    if FILTERING_ON:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("mask", mask)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    rects = []
    rvecs, tvecs = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        app_rect = np.array([(x+w, y), (x, y), (x, y+h), (x+w, y+h)])
        rects.append(app_rect)
        _, rvec, tvec = cv2.solvePnP(
            FW_OBJ_POINTS, app_rect.astype("float32"), mtx, dist)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs, rects, None


fa_dict = aruco.getPredefinedDictionary(FRAME_AXIS_DCT)
if FOLOWING_ARUCO:
    fw_dict = aruco.getPredefinedDictionary(FOLOWING_DCT)
arucoParams = aruco.DetectorParameters_create()
# markerLength = 3.75
# markerSeparation = 0.5
video_stream = cv2.VideoCapture(1 if USING_CAMERA else VIDEO_IN_PATH, apiPreference=cv2.CAP_ANY, params=[
    cv2.CAP_PROP_FRAME_WIDTH, 1280,
    cv2.CAP_PROP_FRAME_HEIGHT, 720])
ret, img = video_stream.read()

with open(CALIB_DATA_PATH) as f:
    loadeddict = json.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)
counter = 0
# print(dist)

# ret, img = video_stream.read()
# img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# h, w = img_gray.shape[:2]
w, h = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH), video_stream.get(
    cv2.CAP_PROP_FRAME_HEIGHT)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

fa_rvec = None
fa_tvec = None
while video_stream.isOpened():
    ret, img = video_stream.read()
    if not ret:
        break
    img_aruco = img.copy()
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = im_gray.shape[:2]
    # fa_corners, fa_ids, _ = aruco.detectMarkers(
    #     im_gray, fa_dict, parameters=arucoParams)
    cv2.imshow("original", img)
    fa_rvecs, fa_tvecs, fa_corners, fa_ids = detect_objects(
        img, mtx, dist, fa_dict)
    if len(fa_corners) != 1 and not IGNORE_FA_INVIS:
        q = cv2.waitKey(1)
        if q == 27:
            break
        continue
    fw_rvecs, fw_tvecs, fw_corners, fw_ids = detect_objects(img, mtx, dist)
    if len(fw_corners) < 1 and not IGNORE_FW_INVIS:
        q = cv2.waitKey(1)
        if q == 27:
            break
        continue
    img_with_drawings = img.copy()
    for i in range(len(fw_corners)):
        cv2.rectangle(img_with_drawings,
                      fw_corners[i][1], fw_corners[i][3], (0, 255, 0), 2)
    cv2.drawFrameAxes(img_with_drawings, mtx, dist, fw_rvecs[0], fw_tvecs[0],
                      5)  # axis length 100 can be changed according to your requirement
    # cv2.drawFrameAxes(img_with_drawings, mtx, dist, fa_rvec, fa_tvec,
    #                   5)  # axis length 100 can be changed according to your requirement
    cv2.imshow("img with drawings", img_with_drawings)
    q = cv2.waitKey(1)
    if q == 27:
        break

cv2.destroyAllWindows()
