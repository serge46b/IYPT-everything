import pandas as pd
import numpy as np
import json
import cv2
import os
# import gc

# SETTINGS
# file names
FILE_NAME = "./16 ball.MOV"
CONFIG_FILE_NAME = "./all config.json"
CROPPED_VIDEO_FILE_NAME = "./16 ball/16 ball cropped.avi"
NORMALIZED_VIDEO_FILE_NAME = "./16 ball/16 ball normalized.avi"
ROTATED_VIDEO_FILE_NAME = "./16 ball/16 ball rotated.avi"
NOCC_VIDEO_FILE_NAME = "./16 ball/16 ball normalized of cluster center.avi"
DATA_FILE_NAME = "./16 ball/16 ball data.xlsx"
# FILE_NAME = "./1 ball.MOV"
# CONFIG_FILE_NAME = "./all config.json"
# CROPPED_VIDEO_FILE_NAME = "./1 ball/2348 cropped.avi"
# NORMALIZED_VIDEO_FILE_NAME = "./1 ball/1 ball normalized.avi"
# ROTATED_VIDEO_FILE_NAME = "./1 ball/1 ball rotated.avi"
# DATA_FILE_NAME = "./1 ball/1 ball data.xlsx"
# FILE_NAME = "./IMG_2348.MOV"
# CONFIG_FILE_NAME = "./all config.json"
# CROPPED_VIDEO_FILE_NAME = "./2348/2348 cropped.avi"
# NORMALIZED_VIDEO_FILE_NAME = "./2348/2348 normalized.avi"
# ROTATED_VIDEO_FILE_NAME = "./2348/2348 rotated.avi"
# DATA_FILE_NAME = "./2348/2348 data.xlsx"
# tracking:
MARKERS_LOWER_BOUND = None
MARKERS_UPPER_BOUND = None
TRACKING_OBJECT_LOWER_BOUND = None
TRACKING_OBJECT_UPPER_BOUND = None
CLUSTER_OBJECTS_LOWER_BOUND = None
CLUSTER_OBJECTS_UPPER_BOUND = None
TRACK_OBJECT = True
TRACK_CLUSTER = True
# video crop settings
ENABLE_CROP = True
ENABLE_AUTO_CROP = False
VIDEO_CROP = None
INITIAL_SHIFT = None
BOX_SIZE = None
# rendering settings
RENDER_CROPPED_VIDEO = True
RENDER_NORMALIZED_VIDEO = True
RENDER_ROTATED_VIDEO = True
RENDER_NOCC_VIDEO = True  # NEW
ROTATION_INITIAL_ANGLE = 90
RENDER_MARKER_TRACE = True
RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO = True
RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO = True
RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO = True
RENDER_TR_OBJECT_TRACE_ON_NOCC_VIDEO = True
RENDER_CL_CENTER_TRACE_ON_CR_VIDEO = True
RENDER_CL_CENTER_TRACE_ON_NR_VIDEO = True
RENDER_CL_CENTER_TRACE_ON_RT_VIDEO = True
RENDER_CL_CENTER_POINT_ON_NOCC_VIDEO = True
RENDER_M_POINT = True
RENDER_CENTER_POINT = True
TRACE_MAX_LENGTH = 200
M_POINT_CIRCLE = None
MARKER_ROTATION_CENTER = None
# color settings
DECOLOR_CROP_VIDEO = False
DECOLOR_NORMALIZED_VIDEO = False
DECOLOR_ROTATED_VIDEO = False
DECOLOR_NOCC_VIDEO = False
CLUSTER_CENTER_POINT_COLOR = (0, 125, 255)
CLUSTER_CENTER_TRACK_COLOR = (96, 176, 252)
TRACKING_OBJECT_POINT_COLOR = (0, 0, 255)
TRACKING_OBJECT_TRACK_COLOR = (0, 255, 255)
MARKER_POINT_COLOR = (255, 0, 0)
MARKER_TRACK_COLOR = (255, 0, 255)
M_POINT_COLOR = (255, 0, 0)
CENTER_COLOR = (0, 0, 255)
# data logging settings
LOG_DATA = True
APPEND_TO_SETTINGS_FILE_IF_NEEDED = True
# for developers
DEBUG = False
USE_EXTERNAL_CONFIG_FILE = True
OVERWRITE_SETTINGS = True
FOURCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

if not RENDER_NORMALIZED_VIDEO and not RENDER_CROPPED_VIDEO and not RENDER_ROTATED_VIDEO:
    import sys
    print("Wrong settings provided!\nRENDER_CROPPED_VIDEO and "
          "RENDER_NORMALIZED_VIDEO and RENDER_ROTATED_VIDEO are equals FALSE\nexiting...")
    sys.exit(0)

if not RENDER_CROPPED_VIDEO and (RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO or RENDER_CL_CENTER_TRACE_ON_CR_VIDEO or RENDER_MARKER_TRACE or RENDER_M_POINT):
    import sys
    print("Wrong settings provided!\nRENDER_CROPPED_VIDEO is FALSE and "
          "RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO or RENDER_MARKER_TRACE or RENDER_M_POINT are TRUE\nexiting...")
    sys.exit(0)

if not RENDER_NORMALIZED_VIDEO and (RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO or RENDER_CL_OBJECT_TRACE_ON_NR_VIDEO):
    import sys
    print("Wrong settings provided!\nRENDER_NORMALIZED_VIDEO is FALSE and "
          "RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO is TRUE\nexiting...")
    sys.exit(0)

if not RENDER_ROTATED_VIDEO and (RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO or RENDER_CL_CENTER_TRACE_ON_RT_VIDEO or RENDER_M_POINT):
    import sys
    print("Wrong settings provided!\nRENDER_ROTATED_VIDEO is FALSE and "
          "RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO or RENDER_CL_CENTER_TRACE_ON_RT_VIDEO or RENDER_M_POINT is TRUE\nexiting...")
    sys.exit(0)

if not RENDER_NOCC_VIDEO and (RENDER_TR_OBJECT_TRACE_ON_NOCC_VIDEO or RENDER_CL_CENTER_POINT_ON_NOCC_VIDEO):
    import sys
    print("Wrong settings provided!\nRENDER_NOCC_VIDEO is FALSE and "
          "RENDER_TR_OBJECT_TRACE_ON_NOCC_VIDEO or RENDER_CL_CENTER_TRACE_ON_NOCC_VIDEO is TRUE\nexiting...")
    sys.exit(0)

if not RENDER_CROPPED_VIDEO and DECOLOR_CROP_VIDEO:
    import sys
    print("Wrong settings provided!\nRENDER_CROPPED_VIDEO is FALSE and "
          "DECOLOR_CROP_VIDEO is TRUE\nexiting...")
    sys.exit(0)

if not RENDER_NORMALIZED_VIDEO and DECOLOR_NORMALIZED_VIDEO:
    import sys
    print("Wrong settings provided!\nRENDER_NORMALIZED_VIDEO is FALSE and "
          "DECOLOR_NORMALIZED_VIDEO is TRUE\nexiting...")
    sys.exit(0)

if not RENDER_ROTATED_VIDEO and DECOLOR_ROTATED_VIDEO:
    import sys
    print("Wrong settings provided!\nRENDER_ROTATED_VIDEO is FALSE and "
          "DECOLOR_ROTATED_VIDEO is TRUE\nexiting...")
    sys.exit(0)

if not RENDER_NOCC_VIDEO and DECOLOR_NOCC_VIDEO:
    import sys
    print("Wrong settings provided!\nRENDER_NOCC_VIDEO is FALSE and "
          "DECOLOR_NOCC_VIDEO is TRUE\nexiting...")
    sys.exit(0)


if not ENABLE_CROP:
    VIDEO_CROP = None


def append_to_settings(setting_name, setting_value):
    with open(CONFIG_FILE_NAME, "r") as config_file:
        settings = json.load(config_file)
    if FILE_NAME not in settings:
        settings[FILE_NAME] = {}
    if setting_name not in settings[FILE_NAME] or OVERWRITE_SETTINGS:
        settings[FILE_NAME][setting_name] = setting_value
    with open(CONFIG_FILE_NAME, "w") as config_file:
        json.dump(settings, config_file)


def read_settings(setting_name):
    with open(CONFIG_FILE_NAME, "r") as config_file:
        settings = json.load(config_file)
    if FILE_NAME not in settings:
        print("no settings for that file found")
        return -1
    now_config = settings[FILE_NAME]
    if setting_name not in now_config:
        print(f"{setting_name} not found in this config file")
        return -1
    return now_config[setting_name]


# load settings
if USE_EXTERNAL_CONFIG_FILE:
    if os.path.exists(CONFIG_FILE_NAME):
        print("loading settings...")
        if MARKERS_LOWER_BOUND is None:
            new_param = read_settings("MARKERS_LOWER_BOUND")
            if new_param != -1:
                MARKERS_LOWER_BOUND = tuple(new_param)
        if MARKERS_UPPER_BOUND is None:
            new_param = read_settings("MARKERS_UPPER_BOUND")
            if new_param != -1:
                MARKERS_UPPER_BOUND = tuple(new_param)
        if TRACKING_OBJECT_LOWER_BOUND is None:
            new_param = read_settings("TRACKING_OBJECT_LOWER_BOUND")
            if new_param != -1:
                TRACKING_OBJECT_LOWER_BOUND = tuple(new_param)
        if TRACKING_OBJECT_UPPER_BOUND is None:
            new_param = read_settings("TRACKING_OBJECT_UPPER_BOUND")
            if new_param != -1:
                TRACKING_OBJECT_UPPER_BOUND = tuple(new_param)
        if CLUSTER_OBJECTS_LOWER_BOUND is None:
            new_param = read_settings("CLUSTER_OBJECTS_LOWER_BOUND")
            if new_param != -1:
                CLUSTER_OBJECTS_LOWER_BOUND = tuple(new_param)
        if CLUSTER_OBJECTS_UPPER_BOUND is None:
            new_param = read_settings("CLUSTER_OBJECTS_UPPER_BOUND")
            if new_param != -1:
                CLUSTER_OBJECTS_UPPER_BOUND = tuple(new_param)
        if VIDEO_CROP is None and ENABLE_CROP:
            new_param = read_settings("VIDEO_CROP")
            if new_param != -1:
                VIDEO_CROP = tuple(new_param)
        if INITIAL_SHIFT is None:
            new_param = read_settings("INITIAL_SHIFT")
            if new_param != -1:
                INITIAL_SHIFT = tuple(new_param)
        if BOX_SIZE is None:
            new_param = read_settings("BOX_SIZE")
            if new_param != -1:
                BOX_SIZE = tuple(new_param)
        if M_POINT_CIRCLE is None:
            new_param = read_settings("M_POINT_CIRCLE")
            if new_param != -1:
                M_POINT_CIRCLE = tuple(new_param)
        if MARKER_ROTATION_CENTER is None:
            new_param = read_settings("MARKER_ROTATION_CENTER")
            if new_param != -1:
                MARKER_ROTATION_CENTER = tuple(new_param)
        print("done")
    else:
        print("can't find settings file for that video!")


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


def crop_to_roi(frame, initial_shift, box_size, marker_cords):
    x, y = marker_cords[0] - \
        initial_shift[0], marker_cords[1] - initial_shift[1]
    w, h = box_size[0], box_size[1]
    return frame.copy()[y:y + h, x:x + w]


def find_initial_shift_and_box_size(capturer, marker_cords, crop_box=None):
    roi = find_video_crop(capturer, crop_box=crop_box, only_first_frame=True)
    # roi = cv2.selectROI(frame)
    box_size = (roi[2], roi[3])
    initial_shift = (marker_cords[0] - roi[0], marker_cords[1] - roi[1])
    return initial_shift, box_size


def find_objects(frame, object_lower_range, object_upper_range):
    mask = cv2.inRange(frame, object_lower_range, object_upper_range)
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("morphology", opening_img)
    contours, hierarchy = cv2.findContours(
        opening_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    center_cords = []
    if len(contours) != 0:
        for c in range(len(contours)):
            m = cv2.moments(contours[c])
            if m["m10"] == 0 or m["m00"] == 0:
                cx = 0
            else:
                cx = int(m["m10"] / m["m00"])
            if m["m01"] == 0 or m["m00"] == 0:
                cy = 0
            else:
                cy = int(m["m01"] / m["m00"])
            center_cords.append([cx, cy])
        return center_cords


def find_markers(frame, marker_lower_range, marker_upper_range):
    center_cords = find_objects(frame, marker_lower_range, marker_upper_range)
    if center_cords is None:
        return
    if len(center_cords) == 2:
        return center_cords


def rotate_and_crop(frame, initial_shift, box_size, marker_cords, degree):
    roi = (marker_cords[0] - INITIAL_SHIFT[0], marker_cords[1] -
           INITIAL_SHIFT[1], BOX_SIZE[0], BOX_SIZE[1])
    rotation_pivot = (roi[0]+(roi[2]//2), roi[1]+(roi[2]//2))
    rotation_matrix = cv2.getRotationMatrix2D(rotation_pivot, degree, 1)
    rotated_frame = cv2.warpAffine(
        frame.copy(), rotation_matrix, frame.shape[1::-1])
    roi_image = crop_to_roi(
        rotated_frame, initial_shift, box_size, marker_cords)
    return roi_image


def find_angle_between_two_lines(crd1, crd2, crd3):
    # avec = [crd2, crd1]
    avec = [crd2[0] - crd1[0], crd2[1] - crd1[1]]
    # cvec = [crd2, crd3]
    cvec = [crd2[0] - crd3[0], crd2[1] - crd3[1]]
    dp = np.dot(avec, cvec)
    avec_mag = np.linalg.norm(avec)
    cvec_mag = np.linalg.norm(cvec)
    rads = np.arccos(dp/(avec_mag*cvec_mag))
    if np.isnan(rads):
        # print("nan")
        rads = 0
    # translating from |0-180| to 0 - 360
    nvec = [crd1[1] - crd2[1], crd2[0] - crd1[0]]
    # nvec = [crd2[0] + crd1[1], crd2[1] - crd1[0]]
    dp = np.dot(cvec, nvec)
    nvec_mag = np.linalg.norm(nvec)
    def_rads = np.arccos(dp/(cvec_mag*nvec_mag))
    if np.isnan(def_rads):
        # print("nan")
        def_rads = 0
    # print(np.degrees(def_rads), end=" ")
    if def_rads > np.pi/2:
        rads = 2*np.pi-rads
    # print(360 - np.degrees(rads))
    return np.degrees(rads)


def find_marker_rotation_center(capturer, find_marker_fn, marker_lb, marker_ub, crop_box=None):
    original_frame_shape = (capturer.get(
        cv2.CAP_PROP_FRAME_HEIGHT), capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    if crop_box is not None:
        original_frame_shape = crop_box[:1:-1]
    capturer.set(cv2.CAP_PROP_POS_FRAMES, 0)
    mxx = 0
    mnx = original_frame_shape[1]
    mxy = 0
    mny = original_frame_shape[0]
    while capturer.isOpened():
        ret, frame = capturer.read()
        if not ret:
            break
        if crop_box is not None:
            frame = frame[crop_box[1]:crop_box[1] + crop_box[3],
                          crop_box[0]:crop_box[0] + crop_box[2], :]
        marker_cords = find_marker_fn(frame, marker_lb, marker_ub)
        if marker_cords is None:
            continue
        marker_cords = marker_cords[0]
        x = marker_cords[0]
        y = marker_cords[1]
        if x > mxx:
            mxx = x
        if x < mnx:
            mnx = x
        if y > mxy:
            mxy = y
        if y < mny:
            mny = y
    return (mxx + mnx) // 2, (mxy + mny) // 2


def find_video_crop_automatically(capturer, marker_lower_range, marker_upper_range, padding=0):
    lt_x = None
    lt_y = None
    rd_x = None
    rd_y = None
    original_frame_shape = (capturer.get(
        cv2.CAP_PROP_FRAME_HEIGHT), capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    capturer.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while capturer.isOpened():
        ret, frame = capturer.read()
        if not ret:
            break
        markers_cords = find_markers(
            frame, marker_lower_range, marker_upper_range)
        if markers_cords is None:
            continue
        left = min(markers_cords[0][0], markers_cords[1][0])
        right = max(markers_cords[0][0], markers_cords[1][0])
        top = min(markers_cords[0][1], markers_cords[1][1])
        down = max(markers_cords[0][1], markers_cords[1][1])
        if lt_x is None:
            lt_x = left
        if lt_y is None:
            lt_y = top
        if rd_x is None:
            rd_x = right
        if rd_y is None:
            rd_y = down
        lt_x = min(left, lt_x)
        lt_y = min(top, lt_y)
        rd_x = max(right, rd_x)
        rd_y = max(down, rd_y)
    if type(padding) != list or type(padding) != tuple:
        padding = (padding, padding)
    x = max(lt_x - padding[0], 0)
    y = max(lt_y - padding[1], 0)
    w = min(rd_x, original_frame_shape[1]) - x + padding[0]
    h = min(rd_y, original_frame_shape[0]) - y + padding[1]
    return tuple((x, y, w, h))


def find_video_crop(capturer, crop_box=None, only_first_frame=False):
    points = [None, None]

    def show_with_points(frame):
        frame = frame.copy()
        if points[0] is not None and points[1] is not None:
            frame = cv2.rectangle(
                frame, points[0], points[1], (0, 255, 255), 1)
        for i in points:
            if i is None:
                continue
            frame = cv2.circle(frame, i, 1, (0, 0, 255), -1)
        cv2.imshow("frame", frame)

    orig_frame_shape = (capturer.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        capturer.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame = capturer.read()[1]
    if crop_box is not None:
        frame = frame[crop_box[1]:crop_box[1] + crop_box[3],
                      crop_box[0]:crop_box[0] + crop_box[2], :]
        orig_frame_shape = (crop_box[3], crop_box[2])
    frame = resize_with_aspect_ratio(frame, height=900)
    point_edits = 0

    def mouse_callback(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN or flag == 1:
            points[point_edits] = (x, y)
            show_with_points(frame)

    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", mouse_callback)
    key = 0
    now_frame = 0
    height = 900 if orig_frame_shape[0] > 900 else None
    # ENTER - 13 D(next frame) A(prev frame) C(del editing point) 1(edit first point) 2(edit last point)
    while key != 27 and not (key == 13 and not (points[0] is None or points[1] is None)):
        capturer.set(cv2.CAP_PROP_POS_FRAMES, now_frame)
        frame = capturer.read()[1]
        if crop_box is not None:
            frame = frame[crop_box[1]:crop_box[1] + crop_box[3],
                          crop_box[0]:crop_box[0] + crop_box[2], :]
        frame = resize_with_aspect_ratio(frame, height=height)
        show_with_points(frame)
        key = cv2.waitKey(0)
        if key == 27 or key == 13:
            cv2.destroyWindow("frame")
        if key == 13:
            roi = [0, 0, 0, 0]
            roi[0] = int(min(points[0][0], points[1][0]) /
                         frame.shape[1] * orig_frame_shape[1])
            roi[1] = int(min(points[0][1], points[1][1]) /
                         frame.shape[0] * orig_frame_shape[0])
            roi[2] = int(abs(points[1][0] - points[0][0]) /
                         frame.shape[1] * orig_frame_shape[1])
            roi[3] = int(abs(points[1][1] - points[0][1]) /
                         frame.shape[0] * orig_frame_shape[0])
            print(f"final results: VIDEO_CROP = {tuple(roi)}")
            return roi
        elif key == ord("d"):
            now_frame += 1
            if now_frame >= capturer.get(cv2.CAP_PROP_FRAME_COUNT):
                now_frame = capturer.get(cv2.CAP_PROP_FRAME_COUNT) - 1
                print("last frame reached")
        elif key == ord("a"):
            now_frame -= 1
            if now_frame < 0:
                now_frame = 0
                print("first frame reached")
        elif key == ord("c"):
            points[point_edits] = None
            print(f"deleted point {point_edits + 1}")
        elif key == ord("1"):
            point_edits = 0
        elif key == ord("2"):
            point_edits = 1


def find_lower_upper_bounds(capturer, obj_amount, crop_box=None):
    colors_and_bounds = {"lower": [0, 0, 0], "upper": [255, 255, 255]}
    color_to_idx = {"red": 2, "green": 1, "blue": 0}
    bound_edit = "lower"
    color_edit = "red"
    now_frame = 0
    inc = 10
    key = 0
    mode = "multiple"
    on_filters = False
    height = 900 if capturer.get(cv2.CAP_PROP_FRAME_HEIGHT) > 900 else None
    vr_fl = False
    # window_title = "original frame|frame: 0|bound: lower|color: red|inc/dec: 10"
    # ENTER - 13 D(next frame) A(prev frame) W(up) S(down) R(edit red) G(edit green) B(edit blue)
    # C(coarse - low precision settings (i/d 10)) F(fine - high precision setting (i/d 1))
    # U(edit upper bound) L(edit lower bound) P(print actual bounds)
    # M(multiple color mode) O(single color mode)
    while key != 27 and key != 13:
        capturer.set(cv2.CAP_PROP_POS_FRAMES, now_frame)
        frame = capturer.read()[1]
        if crop_box is not None:
            frame = frame[crop_box[1]:crop_box[1] + crop_box[3],
                          crop_box[0]:crop_box[0] + crop_box[2], :]
        frame = resize_with_aspect_ratio(frame, height=height)
        cv2.imshow("frame", frame)
        new_title = f"original frame|frame: {now_frame}|bound: {bound_edit}|color: {color_edit}|inc/dec: {inc}"
        cv2.setWindowTitle("frame", new_title)
        if mode == "single":
            mask = cv2.inRange(frame, tuple(
                colors_and_bounds["lower"]), tuple(colors_and_bounds["upper"]))
            if on_filters:
                kernel = np.ones((5, 5), np.uint8)
                opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                opening_img = cv2.morphologyEx(
                    opening_img, cv2.MORPH_OPEN, kernel)
            else:
                opening_img = mask
            cv2.imshow("mask", opening_img)
        elif mode == "multiple":
            r_mask = cv2.inRange(
                frame[:, :, 2], colors_and_bounds["lower"][2], colors_and_bounds["upper"][2])
            g_mask = cv2.inRange(
                frame[:, :, 1], colors_and_bounds["lower"][1], colors_and_bounds["upper"][1])
            b_mask = cv2.inRange(
                frame[:, :, 0], colors_and_bounds["lower"][0], colors_and_bounds["upper"][0])
            final_mask = np.zeros_like(frame)
            final_mask[:, :, 2] = r_mask
            final_mask[:, :, 1] = g_mask
            final_mask[:, :, 0] = b_mask
            cv2.imshow("mask", final_mask)
        key = cv2.waitKey(0)
        if key == 27 or key == 13:
            cv2.destroyWindow("frame")
            cv2.destroyWindow("mask")
        if key == 13:
            print(f"final settings:\n\tLOWER_BOUND = {tuple(colors_and_bounds['lower'])}"
                  f"\n\tUPPER_BOUND = {tuple(colors_and_bounds['upper'])}")
            return tuple(colors_and_bounds["lower"]), tuple(colors_and_bounds["upper"])
        elif key == ord("d"):
            now_frame += inc
            if now_frame >= capturer.get(cv2.CAP_PROP_FRAME_COUNT):
                now_frame = capturer.get(cv2.CAP_PROP_FRAME_COUNT) - 1
                print("last frame reached")
        elif key == ord("a"):
            now_frame -= inc
            if now_frame < 0:
                now_frame = 0
                print("first frame reached")
        elif key == ord("w"):
            colors_and_bounds[bound_edit][color_to_idx[color_edit]] += inc
            if colors_and_bounds[bound_edit][color_to_idx[color_edit]] > 255:
                colors_and_bounds[bound_edit][color_to_idx[color_edit]] = 255
                print("reached maximum color value")
        elif key == ord("s"):
            colors_and_bounds[bound_edit][color_to_idx[color_edit]] -= inc
            if colors_and_bounds[bound_edit][color_to_idx[color_edit]] < 0:
                colors_and_bounds[bound_edit][color_to_idx[color_edit]] = 0
                print("reached minimum color value")
        elif key == ord("r"):
            color_edit = "red"
        elif key == ord("g"):
            color_edit = "green"
        elif key == ord("b"):
            color_edit = "blue"
        elif key == ord("c"):
            inc = 10
        elif key == ord("f"):
            inc = 1
        elif key == ord("u"):
            bound_edit = "upper"
        elif key == ord("l"):
            bound_edit = "lower"
        elif key == ord("p"):
            print(f"actual_settings settings:\n\tLOWER_BOUND = {colors_and_bounds['lower']}"
                  f"\n\tUPPER_BOUND = {colors_and_bounds['upper']}")
        elif key == ord("o"):
            if mode == "multiple":
                mode = "single"
            elif mode == "single":
                mode = "multiple"
        elif key == ord("q"):
            on_filters = not on_filters
        elif key == ord("v"):
            if vr_fl is True:
                print("checking...")
                vr_fl = False
                has_errors = False
                cv2.destroyWindow("frame")
                cv2.destroyWindow("mask")
                capturer.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while capturer.isOpened():
                    ret, frame = capturer.read()
                    if not ret:
                        break
                    objects = find_objects(frame, tuple(
                        colors_and_bounds["lower"]), tuple(colors_and_bounds["upper"]))
                    if objects is None or len(objects) != obj_amount:
                        now_frame = int(capturer.get(
                            cv2.CAP_PROP_POS_FRAMES)) - 1
                        print(f"wrong marker at frame {now_frame}")
                        has_errors = True
                        break
                if not has_errors:
                    print("everything is correct!")
            else:
                print("press V again to verify")
                vr_fl = True
        if key != ord("v"):
            vr_fl = False
        # window_title = new_title


def find_m_point_circle(frame):
    points = [None, None]

    def show_with_points(frame):
        frame = frame.copy()
        if points[0] is not None and points[1] is not None:
            frame = cv2.circle(
                frame, points[0],
                int(np.sqrt((points[1][0] - points[0][0]) **
                    2 + (points[1][1] - points[0][1])**2)),
                (0, 255, 255), 1
            )
            # frame = cv2.rectangle(frame, points[0], points[1], (0, 255, 255), 1)
        for i in points:
            if i is None:
                continue
            frame = cv2.circle(frame, i, 1, (0, 0, 255), -1)
        cv2.imshow("frame", frame)

    height = 900 if frame.shape[0] > 900 else None
    frame = resize_with_aspect_ratio(frame, height=height)
    point_edits = 0

    def mouse_callback(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN or flag == 1:
            points[point_edits] = (x, y)
            show_with_points(frame)

    cv2.imshow("frame", frame)
    cv2.setMouseCallback("frame", mouse_callback)
    key = 0

    while key != 27 and key != 13:
        show_with_points(frame)
        key = cv2.waitKey(0)
        if key == 27 or key == 13:
            cv2.destroyWindow("frame")
        if key == 13:
            circle_param = [None, 0]
            circle_param[0] = points[0]
            circle_param[1] = int(
                np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2))
            print(f"final results: M_POINT_CIRCLE = {tuple(circle_param)}")
            return tuple(circle_param)
        elif key == ord("c"):
            points[point_edits] = None
            print(f"deleted point {point_edits + 1}")
        elif key == ord("1"):
            point_edits = 0
        elif key == ord("2"):
            point_edits = 1


def render_trace(frame, trace_list, object_cords, color, trace_max_len):
    if len(trace_list) == 0:
        trace_list.append(object_cords)
        return frame, trace_list
    new_frame = frame.copy()
    trace_list.append(object_cords)
    if len(trace_list) > trace_max_len and trace_max_len > 0:
        trace_list.pop(0)
    for trace_idx in range(1, len(trace_list)):
        line_brightness = (0.5 + trace_idx / (len(trace_list)-1)
                           * 0.5) if trace_max_len > 0 else 1
        new_color = (int(color[0]*line_brightness), int(color[1]
                     * line_brightness), int(color[2]*line_brightness))
        new_frame = cv2.line(
            new_frame, trace_list[trace_idx-1], trace_list[trace_idx], new_color, 2)
    return new_frame, trace_list


def find_cluster_center(frame, object_lower_bound, object_upper_bound, cluster_lower_bound, cluster_upper_bound):
    mask = cv2.inRange(frame, object_lower_bound, object_upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("morphology", opening_img)
    contours, hierarchy = cv2.findContours(
        opening_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_data = []
    area_sum = 0
    if len(contours) != 0:
        for c in range(len(contours)):
            m = cv2.moments(contours[c])
            if m["m10"] == 0 or m["m00"] == 0:
                cx = 0
            else:
                cx = int(m["m10"] / m["m00"])
            if m["m01"] == 0 or m["m00"] == 0:
                cy = 0
            else:
                cy = int(m["m01"] / m["m00"])
            area = cv2.contourArea(contours[c])
            contour_data.append([cx, cy, area])
            area_sum += area
    mask = cv2.inRange(frame, cluster_lower_bound, cluster_upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("morphology", opening_img)
    contours, hierarchy = cv2.findContours(
        opening_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        for c in range(len(contours)):
            m = cv2.moments(contours[c])
            if m["m10"] == 0 or m["m00"] == 0:
                cx = 0
            else:
                cx = int(m["m10"] / m["m00"])
            if m["m01"] == 0 or m["m00"] == 0:
                cy = 0
            else:
                cy = int(m["m01"] / m["m00"])
            area = cv2.contourArea(contours[c])
            contour_data.append([cx, cy, area])
            area_sum += area
            # contour_data.append([cx, cy, cv2.contourArea(contours[c])])
    # object_cords = find_objects(frame, object_lower_bound, object_upper_bound)
    # cluster_cords = find_objects(frame, cluster_lower_bound, cluster_upper_bound)
    c_x = 0
    c_y = 0
    if len(contour_data) == 0:
        return
    for i in contour_data:
        c_x += i[0] * (i[2] / area_sum)
        c_y += i[1] * (i[2] / area_sum)
        # print(i)
    return int(c_x), int(c_y)
    # if object_cords is not None:
    #     for i in object_cords:
    #         c_x += i[0]
    #         c_y += i[1]
    #     length += len(object_cords)
    # if cluster_cords is not None:
    #     for i in cluster_cords:
    #         c_x += i[0]
    #         c_y += i[1]
    #     length += len(cluster_cords)
    # if length == 0:
    #     return
    # return c_x // length, c_y // length


cap = cv2.VideoCapture(FILE_NAME)
if MARKERS_LOWER_BOUND is None or MARKERS_UPPER_BOUND is None:
    print("find the markers bounds")
    l_bound, u_bound = find_lower_upper_bounds(cap, 2)
    print(f"MARKERS_LOWER_BOUND = {l_bound}"
          f"\nMARKERS_UPPER_BOUND = {u_bound}")
    MARKERS_LOWER_BOUND = l_bound
    MARKERS_UPPER_BOUND = u_bound
    if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
        append_to_settings("MARKERS_LOWER_BOUND", MARKERS_LOWER_BOUND)
        append_to_settings("MARKERS_UPPER_BOUND", MARKERS_UPPER_BOUND)
        print("results saved to config file")
if ENABLE_CROP and not ENABLE_AUTO_CROP and VIDEO_CROP is None:
    print("crop the video, select two points")
    video_crop = find_video_crop(cap)
    VIDEO_CROP = video_crop
    if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
        append_to_settings("VIDEO_CROP", VIDEO_CROP)
        print("results saved to config file")
if ENABLE_AUTO_CROP:
    print("searching perfect video crop")
    video_crop = find_video_crop_automatically(
        cap, MARKERS_LOWER_BOUND, MARKERS_UPPER_BOUND, 100)
    VIDEO_CROP = video_crop
    print("done")
    print(f"found crop box: VIDEO_CROP = {video_crop}")
if TRACK_OBJECT and (TRACKING_OBJECT_LOWER_BOUND is None or TRACKING_OBJECT_UPPER_BOUND is None):
    l_bound, u_bound = find_lower_upper_bounds(cap, 1, VIDEO_CROP)
    print(f"TRACKING_OBJECT_LOWER_BOUND = {l_bound}"
          f"\nTRACKING_OBJECT_UPPER_BOUND = {u_bound}")
    TRACKING_OBJECT_LOWER_BOUND = l_bound
    TRACKING_OBJECT_UPPER_BOUND = u_bound
    if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
        append_to_settings("TRACKING_OBJECT_LOWER_BOUND",
                           TRACKING_OBJECT_LOWER_BOUND)
        append_to_settings("TRACKING_OBJECT_UPPER_BOUND",
                           TRACKING_OBJECT_UPPER_BOUND)
        print("results saved to config file")
if TRACK_CLUSTER and (CLUSTER_OBJECTS_LOWER_BOUND is None or CLUSTER_OBJECTS_UPPER_BOUND is None):
    ball_amount = input("enter amount of non trackable objects: ")
    l_bound, u_bound = find_lower_upper_bounds(cap, ball_amount, VIDEO_CROP)
    print(f"TRACKING_OBJECT_LOWER_BOUND = {l_bound}"
          f"\nTRACKING_OBJECT_UPPER_BOUND = {u_bound}")
    CLUSTER_OBJECTS_LOWER_BOUND = l_bound
    CLUSTER_OBJECTS_UPPER_BOUND = u_bound
    if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
        append_to_settings("CLUSTER_OBJECTS_LOWER_BOUND",
                           CLUSTER_OBJECTS_LOWER_BOUND)
        append_to_settings("CLUSTER_OBJECTS_UPPER_BOUND",
                           CLUSTER_OBJECTS_UPPER_BOUND)
        print("results saved to config file")
if INITIAL_SHIFT is None or BOX_SIZE is None:
    print("select the plate with balls")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tmp_frame = cap.read()[1]
    if ENABLE_CROP:
        tmp_frame = tmp_frame[VIDEO_CROP[1]:VIDEO_CROP[1] +
                              VIDEO_CROP[3], VIDEO_CROP[0]:VIDEO_CROP[0] + VIDEO_CROP[2], :]
    initial_shift, box_size = find_initial_shift_and_box_size(
        cap,
        find_markers(
            tmp_frame,
            MARKERS_LOWER_BOUND,
            MARKERS_UPPER_BOUND
        )[0],
        VIDEO_CROP
    )
    print(f"INITIAL_SHIFT = {initial_shift}\nBOX_SIZE = {box_size}")
    INITIAL_SHIFT = initial_shift
    BOX_SIZE = box_size
    if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
        append_to_settings("INITIAL_SHIFT", INITIAL_SHIFT)
        append_to_settings("BOX_SIZE", BOX_SIZE)
        print("results saved to config file")
if M_POINT_CIRCLE is None and (RENDER_ROTATED_VIDEO or RENDER_M_POINT or RENDER_CENTER_POINT):
    print("select circle for M point")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    tmp_frame = cap.read()[1]
    if ENABLE_CROP:
        tmp_frame = tmp_frame[VIDEO_CROP[1]:VIDEO_CROP[1] +
                              VIDEO_CROP[3], VIDEO_CROP[0]:VIDEO_CROP[0] + VIDEO_CROP[2], :]
    m_point_circle = find_m_point_circle(tmp_frame)
    M_POINT_CIRCLE = m_point_circle
    if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
        append_to_settings("M_POINT_CIRCLE", M_POINT_CIRCLE)
        print("results saved to config file")
if RENDER_ROTATED_VIDEO or RENDER_M_POINT or RENDER_CENTER_POINT:
    if MARKER_ROTATION_CENTER is None:
        print("searching marker rotation center...")
        marker_rotation_center = find_marker_rotation_center(
            cap, find_markers, MARKERS_LOWER_BOUND, MARKERS_UPPER_BOUND, VIDEO_CROP)
        print("done")
        MARKER_ROTATION_CENTER = marker_rotation_center
        if APPEND_TO_SETTINGS_FILE_IF_NEEDED:
            append_to_settings("MARKER_ROTATION_CENTER",
                               MARKER_ROTATION_CENTER)
            print("results saved to config file")
    else:
        marker_rotation_center = MARKER_ROTATION_CENTER
    if ROTATION_INITIAL_ANGLE is None:
        ROTATION_INITIAL_ANGLE = 0
    center_point_x = marker_rotation_center[0] - \
        INITIAL_SHIFT[0] + (BOX_SIZE[0] // 2)
    center_point_y = marker_rotation_center[1] - \
        INITIAL_SHIFT[1] + (BOX_SIZE[1] // 2)
print("writing video...")
crop_video_shape = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH))
if ENABLE_CROP:
    crop_video_shape = VIDEO_CROP[:1:-1]
if RENDER_CROPPED_VIDEO and not DEBUG:
    cropped_video_writer = cv2.VideoWriter(
        CROPPED_VIDEO_FILE_NAME,
        FOURCC,
        cap.get(cv2.CAP_PROP_FPS),
        crop_video_shape[1::-1]
    )
if RENDER_NORMALIZED_VIDEO and not DEBUG:
    normalized_video_writer = cv2.VideoWriter(
        NORMALIZED_VIDEO_FILE_NAME,
        FOURCC,
        cap.get(cv2.CAP_PROP_FPS),
        BOX_SIZE
    )
if RENDER_ROTATED_VIDEO and not DEBUG:
    rotated_video_writer = cv2.VideoWriter(
        ROTATED_VIDEO_FILE_NAME,
        FOURCC,
        cap.get(cv2.CAP_PROP_FPS),
        BOX_SIZE
    )
if RENDER_NOCC_VIDEO and not DEBUG:
    nocc_video_writer = cv2.VideoWriter(
        NOCC_VIDEO_FILE_NAME,
        FOURCC,
        cap.get(cv2.CAP_PROP_FPS),
        BOX_SIZE
    )
if LOG_DATA:
    data = pd.DataFrame()
    config = pd.DataFrame({
        "frame width": crop_video_shape[1],
        "frame height": crop_video_shape[0],
        "frame rate": cap.get(cv2.CAP_PROP_FPS),
        "box size x": BOX_SIZE[0],
        "box size y": BOX_SIZE[1],
        "mm/px": None
    }, index=[0])
# gc.collect()

marker_path = []
object_abs_path = []
object_rel_path = []
object_rel_rt_path = []
object_nocc_path = []
cluster_center_abs_path = []
cluster_center_rel_path = []
cluster_center_rel_rt_path = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if ENABLE_CROP:
        frame = frame[VIDEO_CROP[1]:VIDEO_CROP[1] + VIDEO_CROP[3],
                      VIDEO_CROP[0]:VIDEO_CROP[0] + VIDEO_CROP[2], :]
    # object detection phase
    markers_cords = find_markers(
        frame, MARKERS_LOWER_BOUND, MARKERS_UPPER_BOUND)
    if markers_cords is None:
        continue
    marker_cords = markers_cords[0]
    # video creating phase
    if RENDER_CROPPED_VIDEO:
        crop_frame = frame.copy()
    if RENDER_NORMALIZED_VIDEO:
        norm_frame = crop_to_roi(frame, INITIAL_SHIFT, BOX_SIZE, marker_cords)
    # object detection on normalized video
    if RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO or RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO:
        object_rel_cords = find_objects(
            norm_frame, TRACKING_OBJECT_LOWER_BOUND, TRACKING_OBJECT_UPPER_BOUND)[0]
    if RENDER_CL_CENTER_TRACE_ON_CR_VIDEO or RENDER_CL_CENTER_TRACE_ON_NR_VIDEO:
        cluster_rel_center = find_cluster_center(
            norm_frame, TRACKING_OBJECT_LOWER_BOUND, TRACKING_OBJECT_UPPER_BOUND, CLUSTER_OBJECTS_LOWER_BOUND, CLUSTER_OBJECTS_UPPER_BOUND)
    # rendering rotated video
    if RENDER_ROTATED_VIDEO:
        zero_marker_angle = find_angle_between_two_lines(
            markers_cords[0],
            marker_rotation_center,
            (marker_rotation_center[0] + 1, marker_rotation_center[1])
        )
        init_marker_angle = zero_marker_angle + ROTATION_INITIAL_ANGLE
        rotated_frame = rotate_and_crop(
            frame, INITIAL_SHIFT, BOX_SIZE, markers_cords[0], -init_marker_angle)
    if RENDER_NOCC_VIDEO:
        cc_shift = (cluster_rel_center[0] - BOX_SIZE[0] //
                    2, cluster_rel_center[1] - BOX_SIZE[1]//2)
        nocc_vid_shift = (
            INITIAL_SHIFT[0] - cc_shift[0], INITIAL_SHIFT[1] - cc_shift[1])
        nocc_frame = crop_to_roi(frame, nocc_vid_shift, BOX_SIZE, marker_cords)
    # object detection on rotated frame
    if RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO:
        rev_obj_cords = find_objects(
            rotated_frame, TRACKING_OBJECT_LOWER_BOUND, TRACKING_OBJECT_UPPER_BOUND)[0]
    if RENDER_CL_CENTER_TRACE_ON_RT_VIDEO:
        rev_cl_center = find_cluster_center(rotated_frame, TRACKING_OBJECT_LOWER_BOUND,
                                            TRACKING_OBJECT_UPPER_BOUND, CLUSTER_OBJECTS_LOWER_BOUND, CLUSTER_OBJECTS_UPPER_BOUND)
    # video decoloration
    if DECOLOR_CROP_VIDEO:
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2BGR)
    if DECOLOR_NORMALIZED_VIDEO:
        norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_BGR2GRAY)
        norm_frame = cv2.cvtColor(norm_frame, cv2.COLOR_GRAY2BGR)
    if DECOLOR_ROTATED_VIDEO:
        rotated_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
        rotated_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_GRAY2BGR)
    # visualisation
    if RENDER_M_POINT:
        marker_rev_x = marker_rotation_center[0] - marker_cords[0]
        marker_rev_y = marker_rotation_center[1] - marker_cords[1]
        marker_rev_len = int(np.sqrt(marker_rev_x**2 + marker_rev_y**2))
        k = M_POINT_CIRCLE[1] / marker_rev_len
        marker_rev_x = int(k * marker_rev_x)
        marker_rev_y = int(k * marker_rev_y)
        rev_rotation_centre_x = marker_cords[0] - \
            INITIAL_SHIFT[0] + (BOX_SIZE[0] // 2)
        rev_rotation_centre_y = marker_cords[1] - \
            INITIAL_SHIFT[1] + (BOX_SIZE[1] // 2)
        crop_frame = cv2.circle(
            crop_frame,
            (rev_rotation_centre_x - marker_rev_x,
             rev_rotation_centre_y - marker_rev_y),
            3,
            M_POINT_COLOR,
            -1
        )
        if RENDER_ROTATED_VIDEO:
            empty_frame = np.zeros_like(crop_frame)
            empty_frame = cv2.circle(
                empty_frame,
                (rev_rotation_centre_x - marker_rev_x,
                 rev_rotation_centre_y - marker_rev_y),
                3,
                M_POINT_COLOR,
                -1
            )
            empty_frame = rotate_and_crop(
                empty_frame, INITIAL_SHIFT, BOX_SIZE, markers_cords[0], -init_marker_angle)
            cnd = empty_frame[:, :, 0] > 0
            rotated_frame[cnd] = empty_frame[cnd]
    if RENDER_CENTER_POINT:
        crop_frame = cv2.line(
            crop_frame,
            (center_point_x, center_point_y - 10),
            (center_point_x, center_point_y + 10),
            (0, 0, 255),
            2
        )
        crop_frame = cv2.line(
            crop_frame,
            (center_point_x - 10, center_point_y),
            (center_point_x + 10, center_point_y),
            (0, 0, 255),
            2
        )
    if RENDER_TR_OBJECT_TRACE_ON_NOCC_VIDEO:
        object_nocc_cords = (
            object_rel_cords[0] - cc_shift[0], object_rel_cords[1] - cc_shift[1])
        nocc_frame, object_nocc_path = render_trace(
            nocc_frame,
            object_nocc_path,
            object_nocc_cords,
            TRACKING_OBJECT_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        nocc_frame = cv2.circle(
            nocc_frame, object_nocc_cords, 2, TRACKING_OBJECT_POINT_COLOR, -1)
    if RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO:
        rotated_frame, object_rel_rt_path = render_trace(
            rotated_frame,
            object_rel_rt_path,
            rev_obj_cords,
            TRACKING_OBJECT_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        rotated_frame = cv2.circle(
            rotated_frame, rev_obj_cords, 2, TRACKING_OBJECT_POINT_COLOR, -1)
    if RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO:
        norm_frame, object_rel_path = render_trace(
            norm_frame,
            object_rel_path,
            object_rel_cords,
            TRACKING_OBJECT_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        norm_frame = cv2.circle(
            norm_frame, object_rel_cords, 2, TRACKING_OBJECT_POINT_COLOR, -1)
    if RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO:
        object_abs_cords = (
            marker_cords[0] - INITIAL_SHIFT[0] + object_rel_cords[0],
            marker_cords[1] - INITIAL_SHIFT[1] + object_rel_cords[1]
        )
        crop_frame, object_abs_path = render_trace(
            crop_frame,
            object_abs_path,
            object_abs_cords,
            TRACKING_OBJECT_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        crop_frame = cv2.circle(
            crop_frame, object_abs_cords, 2, TRACKING_OBJECT_POINT_COLOR, -1)
    if RENDER_CL_CENTER_POINT_ON_NOCC_VIDEO:
        nocc_frame_center = (BOX_SIZE[0]//2, BOX_SIZE[1]//2)
        nocc_frame = cv2.circle(
            nocc_frame, nocc_frame_center, 2, CLUSTER_CENTER_POINT_COLOR, -1)
    if RENDER_CL_CENTER_TRACE_ON_RT_VIDEO:
        rotated_frame, cluster_center_rel_rt_path = render_trace(
            rotated_frame,
            cluster_center_rel_rt_path,
            rev_cl_center,
            CLUSTER_CENTER_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        rotated_frame = cv2.circle(
            rotated_frame, rev_cl_center, 2, CLUSTER_CENTER_POINT_COLOR, -1)
    if RENDER_CL_CENTER_TRACE_ON_NR_VIDEO:
        norm_frame, cluster_center_rel_path = render_trace(
            norm_frame,
            cluster_center_rel_path,
            cluster_rel_center,
            CLUSTER_CENTER_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        norm_frame = cv2.circle(
            norm_frame, cluster_rel_center, 2, CLUSTER_CENTER_POINT_COLOR, -1)
    if RENDER_CL_CENTER_TRACE_ON_CR_VIDEO:
        cluster_center_abs_cords = (
            marker_cords[0] - INITIAL_SHIFT[0] + cluster_rel_center[0],
            marker_cords[1] - INITIAL_SHIFT[1] + cluster_rel_center[1]
        )
        crop_frame, cluster_center_abs_path = render_trace(
            crop_frame,
            cluster_center_abs_path,
            cluster_center_abs_cords,
            CLUSTER_CENTER_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        crop_frame = cv2.circle(
            crop_frame, cluster_center_abs_cords, 2, TRACKING_OBJECT_POINT_COLOR, -1)
    if RENDER_MARKER_TRACE:
        crop_frame, marker_path = render_trace(
            crop_frame,
            marker_path,
            marker_cords,
            MARKER_TRACK_COLOR,
            TRACE_MAX_LENGTH
        )
        crop_frame = cv2.circle(crop_frame, marker_cords,
                                2, MARKER_POINT_COLOR, -1)
    # debug and vide writing
    if DEBUG:
        if RENDER_CROPPED_VIDEO:
            cv2.imshow("cropped", crop_frame)
        if RENDER_NORMALIZED_VIDEO:
            cv2.imshow("normalized", norm_frame)
        if RENDER_ROTATED_VIDEO:
            cv2.imshow("rotated", rotated_frame)
        if RENDER_NOCC_VIDEO:
            cv2.imshow("nocc", nocc_frame)
        key = cv2.waitKey(0)
        if key == 27:
            break
    else:
        if RENDER_CROPPED_VIDEO:
            cropped_video_writer.write(crop_frame)
        if RENDER_NORMALIZED_VIDEO:
            normalized_video_writer.write(norm_frame)
        if RENDER_ROTATED_VIDEO:
            rotated_video_writer.write(rotated_frame)
        if RENDER_NOCC_VIDEO:
            nocc_video_writer.write(nocc_frame)
    if LOG_DATA:
        if not (RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO or RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO):
            object_rel_cords = [None, None]
        if not RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO:
            object_abs_cords = [None, None]
        if not RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO:
            rev_obj_cords = [None, None]
        if not (RENDER_CL_CENTER_TRACE_ON_CR_VIDEO or RENDER_CL_CENTER_TRACE_ON_NR_VIDEO):
            cluster_rel_center = [None, None]
        if not RENDER_CL_CENTER_TRACE_ON_CR_VIDEO:
            cluster_center_abs_cords = [None, None]
        if not RENDER_CL_CENTER_TRACE_ON_RT_VIDEO:
            rev_cl_center = [None, None]
        data = pd.concat([data, pd.DataFrame(
            {"frame": cap.get(cv2.CAP_PROP_POS_FRAMES) - 1,
             "marker x": marker_cords[0],
             "marker y": marker_cords[1],
             "object relative x": object_rel_cords[0],
             "object relative y": object_rel_cords[1],
             "object absolute x": object_abs_cords[0],
             "object absolute y": object_abs_cords[1],
             "object rotated x": rev_obj_cords[0],
             "object rotated y": rev_obj_cords[1],
             "cluster center relative x": cluster_rel_center[0],
             "cluster center relative y": cluster_rel_center[1],
             "cluster center absolute x": cluster_center_abs_cords[0],
             "cluster center absolute y": cluster_center_abs_cords[1],
             "cluster center rotated x": rev_cl_center[0],
             "cluster center rotated y": rev_cl_center[1]
             }, index=[0]
        )])

if RENDER_CROPPED_VIDEO and not DEBUG:
    cropped_video_writer.release()
if RENDER_NORMALIZED_VIDEO and not DEBUG:
    normalized_video_writer.release()
if RENDER_ROTATED_VIDEO and not DEBUG:
    rotated_video_writer.release()
cap.release()
cv2.destroyAllWindows()
print("done")
if LOG_DATA:
    print("writing to xlsx file...")
    with pd.ExcelWriter(DATA_FILE_NAME) as writer:
        data.to_excel(writer, sheet_name="data", index=False)
    with pd.ExcelWriter(DATA_FILE_NAME, mode="a") as writer:
        config.to_excel(writer, sheet_name="config", index=False)
    print("done")
