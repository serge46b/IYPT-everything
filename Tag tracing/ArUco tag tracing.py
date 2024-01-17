import os
import cv2
import json
import numpy as np
from cv2 import aruco


CONFIG_FILE_PATH = "./2023/Walker/ArUco tracing config.json"
# CONFIG_FILE_PATH = "./Tag tracing/camera dbg cfg.json"
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
VIDEO_SOURCE = loaded_cfg["video_in_rpath"] if type(loaded_cfg["video_in_rpath"]) == int else (
    ROOT + loaded_cfg["video_in_rpath"])
OUT_SV_PATH = ROOT + loaded_cfg["tracing_file_dir_rpath"] + \
    (str(VIDEO_SOURCE) if type(VIDEO_SOURCE) == int else VIDEO_SOURCE[VIDEO_SOURCE.rfind(
        '/'):VIDEO_SOURCE.rfind('.')]) + ".csv"

DCTS = {"4X4_1000": aruco.DICT_4X4_1000, "5X5_1000": aruco.DICT_5X5_1000}
ARUCO_DICTIONARY = DCTS[loaded_cfg["tracing_marker_dict"]]
ARUCO_GROUND_DICTIONARY = DCTS[loaded_cfg["root_marker_dict"]]
GROUND_MARKER_SIZE = loaded_cfg["root_marker_sz"]
GROUND_MARKER_ID = loaded_cfg["root_marker_id"]
MARKER_SIZE = loaded_cfg["tracing_marker_sz"]  # units - meters
TRACING_MARKERS_IDS = loaded_cfg["tracing_marker_ids"]

DISPLAY_IMG_HEIGHT = 700  # units - pixels
MARKER_AXIS_DISPLAY_LENGTH = 0.025  # units - meters
DEBUG = True

WRITE_VIDEO = True
OUT_VIDEO_SV_PATH = ROOT + loaded_cfg["tracing_file_dir_rpath"] + \
    (str(VIDEO_SOURCE) if type(VIDEO_SOURCE) == int else VIDEO_SOURCE[VIDEO_SOURCE.rfind(
        '/'):VIDEO_SOURCE.rfind('.')]) + ".avi"
FOURCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

# CAMERA_RES_WIDTH = 1280
# CAMERA_RES_HEIGHT = 720
# CAMERA_RES_WIDTH = 1920
# CAMERA_RES_HEIGHT = 1080
# CAMERA_RES_HEIGHT = 2400
# CAMERA_RES_WIDTH = 1028


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


tr_aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
gd_aruco_dict = aruco.getPredefinedDictionary(ARUCO_GROUND_DICTIONARY)
arucoParams = aruco.DetectorParameters()
camera = cv2.VideoCapture(VIDEO_SOURCE)
# camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RES_WIDTH)
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RES_HEIGHT)
if WRITE_VIDEO:
    traced_video_writer = cv2.VideoWriter(OUT_VIDEO_SV_PATH, FOURCC, camera.get(
        cv2.CAP_PROP_FPS), (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))))
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
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

res_file = open(OUT_SV_PATH, "w")
res_file.write("frame, id, tx, ty, tz, rx, ry, rz\n")
print("processing...")
while True:
    ret, img = camera.read()
    if not ret:
        print("Stream accidently closed or video file ended")
        break
    if DEBUG or WRITE_VIDEO:
        img_aruco = img.copy()
    im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # _, im_th = cv2.threshold(
    #     im_gray, 30, 255, cv2.THRESH_BINARY)
    # cv2.imshow("dbg", resize_with_aspect_ratio(
    #     im_th, height=DISPLAY_IMG_HEIGHT))
    # im_norm = cv2.equalizeHist(im_gray)
    gd_corners, gd_ids, rejectedImgPoints = aruco.detectMarkers(
        im_gray, gd_aruco_dict, parameters=arucoParams)
    if len(gd_corners) == 0:
        pass
    else:
        if DEBUG or WRITE_VIDEO:
            img_aruco = aruco.drawDetectedMarkers(
                img_aruco, gd_corners, gd_ids, (255, 255, 255))
        for i in range(len(gd_ids)):
            if gd_ids[i] != GROUND_MARKER_ID:
                continue
            _, rvec, tvec = cv2.solvePnP(np.array([[-GROUND_MARKER_SIZE/2, GROUND_MARKER_SIZE/2, 0], [GROUND_MARKER_SIZE/2, GROUND_MARKER_SIZE/2, 0], [GROUND_MARKER_SIZE/2, -GROUND_MARKER_SIZE/2, 0],
                                                   [-GROUND_MARKER_SIZE/2, -GROUND_MARKER_SIZE/2, 0]]), gd_corners[i][0], mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            res_file.write(
                f"{int(camera.get(cv2.CAP_PROP_POS_FRAMES))}, {gd_ids[i, 0]}, {tvec[0, 0]}, {tvec[1, 0]}, {tvec[2, 0]}, {rvec[0, 0]}, {rvec[1, 0]}, {rvec[2, 0]}\n")
            img_aruco = cv2.drawFrameAxes(img_aruco, mtx, dist, rvec, tvec,
                                          MARKER_AXIS_DISPLAY_LENGTH)
            break
    tr_corners, tr_ids, rejectedImgPoints = aruco.detectMarkers(
        im_gray, tr_aruco_dict, parameters=arucoParams)
    if len(tr_corners) == 0:
        pass
    else:
        if DEBUG or WRITE_VIDEO:
            img_aruco = aruco.drawDetectedMarkers(
                img_aruco, tr_corners, tr_ids, (0, 255, 0))
        for i in range(len(tr_ids)):
            if tr_ids[i] not in TRACING_MARKERS_IDS:
                continue
            # rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], m_size, mtx,
            #                                                            dist)
            # print(corners[i][0])
            _, rvec, tvec = cv2.solvePnP(np.array([[-MARKER_SIZE/2, MARKER_SIZE/2, 0], [MARKER_SIZE/2, MARKER_SIZE/2, 0], [MARKER_SIZE/2, -MARKER_SIZE/2, 0],
                                         [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]]), tr_corners[i][0], mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            # print(rvec, tvec)
            # res_file.write(
            #     f"{int(camera.get(cv2.CAP_PROP_POS_FRAMES))}, {ids[i, 0]}, {tvec[0, 0, 0]}, {tvec[0, 0, 1]}, {tvec[0, 0, 2]}, {rvec[0, 0, 0]}, {rvec[0, 0, 1]}, {rvec[0, 0, 2]}\n")
            res_file.write(
                f"{int(camera.get(cv2.CAP_PROP_POS_FRAMES))}, {tr_ids[i, 0]}, {tvec[0, 0]}, {tvec[1, 0]}, {tvec[2, 0]}, {rvec[0, 0]}, {rvec[1, 0]}, {rvec[2, 0]}\n")
            if DEBUG or WRITE_VIDEO:
                axis_points = np.array([[0, 0, 0], [0, 0, MARKER_AXIS_DISPLAY_LENGTH], [
                    0, MARKER_AXIS_DISPLAY_LENGTH, 0], [MARKER_AXIS_DISPLAY_LENGTH, 0, 0]])
                prj_axis = np.array(cv2.projectPoints(
                    axis_points, rvec, tvec, mtx, dist)[0], dtype=int)
                cv2.line(img_aruco, prj_axis[0, 0],
                         prj_axis[1, 0], (255, 255, 0), 3)
                cv2.line(img_aruco, prj_axis[0, 0],
                         prj_axis[2, 0], (0, 255, 255), 3)
                cv2.line(img_aruco, prj_axis[0, 0],
                         prj_axis[3, 0], (255, 0, 255), 3)
    if WRITE_VIDEO:
        traced_video_writer.write(img_aruco)
    if DEBUG:
        q = cv2.waitKey(1)
        cv2.imshow("estimation", resize_with_aspect_ratio(
            img_aruco, height=DISPLAY_IMG_HEIGHT))
        if q == 27:
            print("Requested exit")
            break

res_file.close()
camera.release()
if WRITE_VIDEO:
    traced_video_writer.release()
cv2.destroyAllWindows()
