import cv2
import json
import numpy as np
from cv2 import aruco


ROOT = "./2023/Cushion Catapult/"
CALIB_DATA_PATH = ROOT + "../Calibration/calibration.json"
USE_CONFIG = True
CONFIG_PATH = ROOT + "config.json"
USING_CAMERA = False
# VIDEO_IN_PATH = ROOT + "videos/test.mp4"
VIDEO_IN_PATH = "C:/Users/serg4/Downloads/Phone tr test.mp4"

# FRAME_AXIS_DCT = aruco.DICT_5X5_1000
# FRAME_AXIS_MARKER_SIZE = 5  # Numbers from head, need to be changed
# FRAME_AXIS_MARKER_SEP = 2  # Numbers from head, need to be changed
# IGNORE_FA_INVIS = True
FW_BASIC_SIZE = 0.6  # Numbers from head
FW_OBJ_POINTS = np.array([(-FW_BASIC_SIZE/2, FW_BASIC_SIZE/2, 0), (FW_BASIC_SIZE/2, FW_BASIC_SIZE/2, 0),
                          (FW_BASIC_SIZE/2, -FW_BASIC_SIZE/2, 0), (-FW_BASIC_SIZE/2, -FW_BASIC_SIZE/2, 0)]).astype("float32")

DISPLAY_IMG_HEIGHT = 700

# TR_OBJ_COLOR_LB = (44, 100, 66)
# TR_OBJ_COLOR_HB = (71, 142, 103)
# FILTERING_ON = True


def crop_to_page(img):
    _, th = cv2.threshold(cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)
    # th = cv2.adaptiveThreshold(
    #     im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mx_area = 0
    mx_cnt_idx = 0
    for cnt_idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if mx_area < area:
            mx_area = area
            mx_cnt_idx = cnt_idx
    peri = cv2.arcLength(contours[mx_cnt_idx], True)
    approx = cv2.approxPolyDP(contours[mx_cnt_idx], 0.15 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    cv2.imshow("cropped", img[y:y+h, x:x+w, :])
    # cv2.imshow("th", th)
    # cnt_img = im_gray.copy()
    # cv2.drawContours(cnt_img, [approx], -1, 0, 1)
    # cv2.imshow("cnt", cnt_img)


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

w, h = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH), video_stream.get(
    cv2.CAP_PROP_FRAME_HEIGHT)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

while video_stream.isOpened():
    ret, img = video_stream.read()
    if not ret:
        break
    img_aruco = img.copy()
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # --ACTUAL CODE HERE--
    canny = cv2.Canny(im_gray, 100, 200)
    # th = cv2.adaptiveThreshold(
    #     im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # cv2.imshow("threshold", th)
    cnt, hierarchy = cv2.findContours(
        canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    crop_to_page(img)
    # cv2.imshow("canny", canny)
    # cnt_img = img.copy()
    # cv2.imshow("cnt", cnt_img)
    # --------------------
    cv2.imshow("original", img)
    q = cv2.waitKey(1)
    if q == 27:
        break
    elif q == ord("a"):
        video_stream.set(cv2.CAP_PROP_POS_FRAMES,
                         video_stream.get(cv2.CAP_PROP_POS_FRAMES)-2)
    elif q == ord("d"):
        pass
    else:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES,
                         video_stream.get(cv2.CAP_PROP_POS_FRAMES)-1)

cv2.destroyAllWindows()
