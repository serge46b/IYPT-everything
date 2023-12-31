import cv2
import json
import numpy as np
from cv2 import aruco


ROOT = "./2023/Walker/"
CALIB_FILE_PATH = ROOT + \
    "../../Calibration/calibration files/Nikon (stock lens).json"
# CALIB_FILE_PATH = ROOT + "../../Calibration/calibration.json"
VIDEO_SOURCE = ROOT + "aruco tag exp/videos/AL exp 2.MOV"
TRACED_DATA_IN_PATH = ROOT + "aruco tag exp/results/" + \
    (str(VIDEO_SOURCE) if VIDEO_SOURCE is int else VIDEO_SOURCE[VIDEO_SOURCE.rfind(
        '/'):VIDEO_SOURCE.rfind('.')]) + ".csv"
# OUT_SV_PATH = ROOT + "aruco tag exp/results/" + VIDEO_SOURCE + ".txt"

# ARUCO_DICTIONARY = aruco.DICT_4X4_1000
# MARKER_SIZE = 0.048  # units - meters

DISPLAY_IMG_HEIGHT = 700  # units - pixels
MARKER_AXIS_DISPLAY_LENGTH = 0.05  # units - meters
# DEBUG = False

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


# aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
# arucoParams = aruco.DetectorParameters()
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

trvec_file = open(TRACED_DATA_IN_PATH, "r")
_ = trvec_file.readline()
frame_vecs = {}
for line in trvec_file:
    # params = line.split(', ')
    frame_idx, m_idx, tx, ty, tz, rx, ry, rz = line.split(', ')
    if int(frame_idx) not in frame_vecs.keys():
        frame_vecs[int(frame_idx)] = {}
    frame_vecs[int(frame_idx)][int(m_idx)] = {
        "tvec": np.array([[[float(tx), float(ty), float(tz)]]]), "rvec": np.array([[[float(rx), float(ry), float(rz)]]])}
    # print(frame_vecs[int(frame_idx)].keys())
# print(frame_vecs[2].keys())
pause_flag = False
while True:
    ret, img = camera.read()
    if pause_flag:
        camera.set(cv2.CAP_PROP_POS_FRAMES, int(
            camera.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    if not ret:
        print("Stream accidently closed")
        break
    img_aruco = img.copy()
    cam_pose = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
    if cam_pose not in frame_vecs.keys():
        markers = []
        # print(cam_pose)
        if not pause_flag:
            continue
    else:
        markers = frame_vecs[cam_pose]
    # TODO: write loading ids from file
    for i in markers:
        tvec, rvec = markers[i]["tvec"], markers[i]["rvec"]
        img_aruco = cv2.drawFrameAxes(img_aruco, newcameramtx, dist, rvec, tvec,
                                      MARKER_AXIS_DISPLAY_LENGTH)
    q = cv2.waitKey(1)
    cv2.imshow("estimation", resize_with_aspect_ratio(
        img_aruco, height=DISPLAY_IMG_HEIGHT))
    if q == 27:
        print("Requested exit")
        break
    elif q == ord(" "):
        pause_flag = not pause_flag
        print("pause" if pause_flag else "play")
    elif q == ord("a"):
        print("prev frame")
        camera.set(cv2.CAP_PROP_POS_FRAMES, int(
            camera.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    elif q == ord("d"):
        print("next frame")
        camera.set(cv2.CAP_PROP_POS_FRAMES, int(
            camera.get(cv2.CAP_PROP_POS_FRAMES)) + 1)


cv2.destroyAllWindows()
