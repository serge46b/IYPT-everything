import numpy as np
import json
import cv2
import os

# SETTINGS
CONFIG_FILE_NAME = "./super config.json"
MARKERS_LOWER_BOUND = None
MARKERS_UPPER_BOUND = None
TRACKING_OBJECT_LOWER_BOUND = None
TRACKING_OBJECT_UPPER_BOUND = None
VIDEO_CROP = None
INITIAL_SHIFT = None
BOX_SIZE = None
M_POINT_CIRCLE = None


def read_settings(setting_name, file_name):
    with open(CONFIG_FILE_NAME, "r") as config_file:
        settings = json.load(config_file)
    if file_name not in settings:
        print("no settings for that file found")
        return -1
    now_config = settings[file_name]
    if setting_name not in now_config:
        print(f"{setting_name} not found in this config file")
        return -1
    return now_config[setting_name]


def video_to_frame_list(capturer):
    frame_list = []
    while capturer.isOpened():
        ret, frame = capturer.read()
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
        if not ret:
            break
        frame_list.append(frame)
        # frame_list = np.append(frame_list, frame)
    return frame_list


def crop_frame_list(frame_list, roi):
    cropped_frame_list = []
    for i in frame_list:
        cropped_frame = i[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        cropped_frame_list.append(cropped_frame)
    return cropped_frame_list


def crop_to_roi(frame, initial_shift, box_size, marker_cords):
    x, y = marker_cords[0] - initial_shift[0], marker_cords[1] - initial_shift[1]
    w, h = box_size[0], box_size[1]
    return frame.copy()[y:y + h, x:x + w]


def find_objects(frame, marker_lower_range, marker_upper_range):
    mask = cv2.inRange(frame, marker_lower_range, marker_upper_range)
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("morphology", opening_img)
    contours, hierarchy = cv2.findContours(opening_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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


err_obj = {}
with open(CONFIG_FILE_NAME, 'r') as config:
    files_and_settings = json.load(config)

for file_name in files_and_settings:
    print(f"working on {file_name}")
    err_obj[file_name] = {"unset": []}
    cap = cv2.VideoCapture(file_name)
    # print("opening video file...")
    # frame_list = video_to_frame_list(cap)
    # print("done")
    if os.path.exists(CONFIG_FILE_NAME):
        print("loading settings...")
        if MARKERS_LOWER_BOUND is None:
            new_param = read_settings("MARKERS_LOWER_BOUND", file_name)
            if new_param != -1:
                MARKERS_LOWER_BOUND = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("MARKERS_LOWER_BOUND")
        if MARKERS_UPPER_BOUND is None:
            new_param = read_settings("MARKERS_UPPER_BOUND", file_name)
            if new_param != -1:
                MARKERS_UPPER_BOUND = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("MARKERS_UPPER_BOUND")
        if TRACKING_OBJECT_LOWER_BOUND is None:
            new_param = read_settings("TRACKING_OBJECT_LOWER_BOUND", file_name)
            if new_param != -1:
                TRACKING_OBJECT_LOWER_BOUND = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("TRACKING_OBJECT_LOWER_BOUND")
        if TRACKING_OBJECT_UPPER_BOUND is None:
            new_param = read_settings("TRACKING_OBJECT_UPPER_BOUND", file_name)
            if new_param != -1:
                TRACKING_OBJECT_UPPER_BOUND = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("TRACKING_OBJECT_UPPER_BOUND")
        if VIDEO_CROP is None:
            new_param = read_settings("VIDEO_CROP", file_name)
            if new_param != -1:
                VIDEO_CROP = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("VIDEO_CROP")
        if INITIAL_SHIFT is None:
            new_param = read_settings("INITIAL_SHIFT", file_name)
            if new_param != -1:
                INITIAL_SHIFT = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("INITIAL_SHIFT")
        if BOX_SIZE is None:
            new_param = read_settings("BOX_SIZE", file_name)
            if new_param != -1:
                BOX_SIZE = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("BOX_SIZE")
        if M_POINT_CIRCLE is None:
            new_param = read_settings("M_POINT_CIRCLE", file_name)
            if new_param != -1:
                M_POINT_CIRCLE = tuple(new_param)
            else:
                err_obj[file_name]["unset"].append("M_POINT_CIRCLE")
        print("done")
    print("checking settings...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if VIDEO_CROP is not None:
            roi = VIDEO_CROP
            frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        # print("checking settings...")
        # for frame_idx in range(len(frame_list)):
        # frame = frame_list[frame_idx]
        if MARKERS_LOWER_BOUND is not None and MARKERS_UPPER_BOUND is not None:
            markers = find_markers(frame, MARKERS_LOWER_BOUND, MARKERS_UPPER_BOUND)
            if markers is None:
                if "wrong markers" not in err_obj[file_name]:
                    err_obj[file_name]["wrong markers"] = []
                err_obj[file_name]["wrong markers"].append(frame_idx)
                print(f"wrong marker at frame {frame_idx} in file '{file_name}'")
        if TRACKING_OBJECT_LOWER_BOUND is not None and TRACKING_OBJECT_UPPER_BOUND is not None:
            tr_obj = find_objects(frame, TRACKING_OBJECT_LOWER_BOUND, TRACKING_OBJECT_UPPER_BOUND)
            if tr_obj is None or len(tr_obj) > 1:
                if "wrong object" not in err_obj[file_name]:
                    err_obj[file_name]["wrong object"] = []
                err_obj[file_name]["wrong object"].append(frame_idx)
                print(f"wrong object at frame {frame_idx} in file '{file_name}'")
        frame_idx += 1
    print("done")


cfg_name = CONFIG_FILE_NAME[CONFIG_FILE_NAME.rfind("/")+1:]
with open(f"./errors in {cfg_name}", "w") as err_file:
    json.dump(err_obj, err_file)
