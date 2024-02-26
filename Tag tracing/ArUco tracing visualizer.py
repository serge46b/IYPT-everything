import os
import cv2
import json
import numpy as np
from cv2 import aruco


CONFIG_FILE_PATH = "./2023/Walker/ArUco tracing config.json"
# CONFIG_FILE_PATH = "./2023/Magnetic gearbox/exp configs/Nk single wheel.json"
# CONFIG_FILE_PATH = "./2023/Magnetic gearbox/exp configs/s46b ArUco general tracing cfg.json"


if not CONFIG_FILE_PATH:
    CONFIG_FILE_PATH = input("Pass config file path:")
if not os.path.exists(CONFIG_FILE_PATH):
    print("cannot load config file. Exiting...")
    os._exit(0)

with open(CONFIG_FILE_PATH) as cfg_f:
    loaded_cfg = json.load(cfg_f)


ROOT = loaded_cfg["root_path"]
CALIB_FILE_PATH = loaded_cfg["calibration_file_path"]
# CALIB_FILE_PATH = ROOT + "../../Calibration/calibration.json"
VIDEO_SOURCE = ROOT + loaded_cfg["video_in_rpath"]
TRACED_DATA_IN_PATH = ROOT + loaded_cfg["tracing_file_dir_rpath"] + \
    (str(VIDEO_SOURCE) if VIDEO_SOURCE is int else VIDEO_SOURCE[VIDEO_SOURCE.rfind(
        '/'):VIDEO_SOURCE.rfind('.')]) + ".csv"
# OUT_SV_PATH = ROOT + "aruco tag exp/results/" + VIDEO_SOURCE + ".txt"

# ARUCO_DICTIONARY = aruco.DICT_4X4_1000
# MARKER_SIZE = 0.048  # units - meters
GROUND_MARKER_ID = loaded_cfg["root_marker_id"]

DISPLAY_IMG_HEIGHT = 700  # units - pixels
MARKER_AXIS_DISPLAY_LENGTH = 0.025  # units - meters
# DEBUG = False


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


# aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
# arucoParams = aruco.DetectorParameters()
camera = cv2.VideoCapture(VIDEO_SOURCE)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RES_WIDTH)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RES_HEIGHT)
# camera = cv2.VideoCapture(VIDEO_SOURCE, apiPreference=cv2.CAP_ANY, params=[
#     cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RES_WIDTH,
#     cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RES_HEIGHT])
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
    shift_flag = False
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
        if i == GROUND_MARKER_ID:
            img_aruco = cv2.drawFrameAxes(img_aruco, mtx, dist, rvec, tvec,
                                          MARKER_AXIS_DISPLAY_LENGTH)
            continue
        axis_points = np.array([[0, 0, 0], [0, 0, MARKER_AXIS_DISPLAY_LENGTH], [
                               0, MARKER_AXIS_DISPLAY_LENGTH, 0], [MARKER_AXIS_DISPLAY_LENGTH, 0, 0]])
        prj_axis = np.array(cv2.projectPoints(
            axis_points, rvec, tvec, mtx, dist)[0], dtype=int)
        cv2.line(img_aruco, prj_axis[0, 0], prj_axis[1, 0], (255, 255, 0), 3)
        cv2.line(img_aruco, prj_axis[0, 0], prj_axis[2, 0], (0, 255, 255), 3)
        cv2.line(img_aruco, prj_axis[0, 0], prj_axis[3, 0], (255, 0, 255), 3)
    q = cv2.waitKey(1)
    cv2.imshow("estimation", resize_with_aspect_ratio(
        img_aruco, height=DISPLAY_IMG_HEIGHT))
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
    if q == 27:
        print("Requested exit")
        break
    elif q == ord(" "):
        pause_flag = not pause_flag
        print("pause" if pause_flag else "play")


camera.release()
cv2.destroyAllWindows()
