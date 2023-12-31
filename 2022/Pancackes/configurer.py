import numpy as np
import json
import cv2
import sys
import os
# import gc

# SETTINGS
# file names
FILE_NAME = "./16 ball.MOV"
CONFIG_FILE_NAME = "./super config.json"
# # tracking:
# MARKERS_LOWER_BOUND = None
# MARKERS_UPPER_BOUND = None
# TRACKING_OBJECT_LOWER_BOUND = None
# TRACKING_OBJECT_UPPER_BOUND = None
# TRACK_OBJECT = True
# # video crop settings
# ENABLE_CROP = True
# ENABLE_AUTO_CROP = False
# VIDEO_CROP = None
# INITIAL_SHIFT = None
# BOX_SIZE = None
# # rendering settings
# RENDER_CROPPED_VIDEO = True
# RENDER_NORMALIZED_VIDEO = True
# RENDER_ROTATED_VIDEO = True
# ROTATION_INITIAL_ANGLE = 90
# RENDER_MARKER_TRACE = True
# RENDER_TR_OBJECT_TRACE_ON_CR_VIDEO = True
# RENDER_TR_OBJECT_TRACE_ON_NR_VIDEO = True
# RENDER_TR_OBJECT_TRACE_ON_RT_VIDEO = True
# RENDER_M_POINT = True
# RENDER_CENTER_POINT = True
# TRACE_MAX_LENGTH = 30
# M_POINT_CIRCLE = None
# # color settings
# DECOLOR_CROP_VIDEO = True
# DECOLOR_NORMALIZED_VIDEO = True
# DECOLOR_ROTATED_VIDEO = True
# TRACKING_OBJECT_POINT_COLOR = (0, 0, 255)
# TRACKING_OBJECT_TRACK_COLOR = (0, 255, 255)
# MARKER_POINT_COLOR = (255, 0, 0)
# MARKER_TRACK_COLOR = (255, 0, 255)
# M_POINT_COLOR = (255, 0, 0)
# CENTER_COLOR = (0, 0, 255)
# # data logging settings
# LOG_DATA = True
# APPEND_TO_SETTINGS_FILE_IF_NEEDED = True
# # for developers
# DEBUG = False
# USE_EXTERNAL_CONFIG_FILE = True
# OVERWRITE_SETTINGS = True
# FOURCC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

MARKERS_LOWER_BOUND = None
MARKERS_UPPER_BOUND = None
TRACKING_OBJECT_LOWER_BOUND = None
TRACKING_OBJECT_UPPER_BOUND = None
VIDEO_CROP = None
INITIAL_SHIFT = None
BOX_SIZE = None
M_POINT_CIRCLE = None
MARKER_ROTATION_CENTER = None

ENABLE_CROP = True
ENABLE_AUTO_CROP = False



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
    if VIDEO_CROP is None:
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


def find_initial_shift_and_box_size(frame, marker_cords):
    roi = find_video_crop([frame])
    # roi = cv2.selectROI(frame)
    box_size = (roi[2], roi[3])
    initial_shift = (marker_cords[0] - roi[0], marker_cords[1] - roi[1])
    return initial_shift, box_size


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


def rotate_and_crop(frame, initial_shift, box_size, marker_cords, degree):
    roi = (marker_cords[0] - INITIAL_SHIFT[0], marker_cords[1] - INITIAL_SHIFT[1], BOX_SIZE[0], BOX_SIZE[1])
    rotation_pivot = (roi[0]+(roi[2]//2), roi[1]+(roi[2]//2))
    rotation_matrix = cv2.getRotationMatrix2D(rotation_pivot, degree, 1)
    rotated_frame = cv2.warpAffine(frame.copy(), rotation_matrix, frame.shape[1::-1])
    roi_image = crop_to_roi(rotated_frame, initial_shift, box_size, marker_cords)
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


def find_marker_rotation_center(frame_list, find_marker_fn, marker_lb, marker_ub):
    mxx = 0
    mnx = frame_list[0].shape[1]
    mxy = 0
    mny = frame_list[0].shape[0]
    for frame in frame_list:
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


def find_video_crop_automatically(frame_list, marker_lower_range, marker_upper_range, padding=0):
    lt_x = None
    lt_y = None
    rd_x = None
    rd_y = None
    for i in frame_list:
        markers_cords = find_markers(i, marker_lower_range, marker_upper_range)
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
    w = min(rd_x, frame_list[0].shape[1]) - x
    h = min(rd_y, frame_list[0].shape[0]) - y
    return tuple((x, y, w, h))


def find_video_crop(frame_list):
    points = [None, None]

    def show_with_points(frame):
        frame = frame.copy()
        if points[0] is not None and points[1] is not None:
            frame = cv2.rectangle(frame, points[0], points[1], (0, 255, 255), 1)
        for i in points:
            if i is None:
                continue
            frame = cv2.circle(frame, i, 1, (0, 0, 255), -1)
        cv2.imshow("frame", frame)

    frame = frame_list[0]
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
    height = 900 if frame_list[0].shape[0] > 900 else None
    # ENTER - 13 D(next frame) A(prev frame) C(del editing point) 1(edit first point) 2(edit last point)
    while key != 27 and not (key == 13 and not (points[0] is None or points[1] is None)):
        frame = frame_list[now_frame]
        frame = resize_with_aspect_ratio(frame, height=height)
        show_with_points(frame)
        key = cv2.waitKey(0)
        if key == 27 or key == 13:
            cv2.destroyWindow("frame")
        if key == 13:
            roi = [0, 0, 0, 0]
            roi[0] = int(min(points[0][0], points[1][0]) / frame.shape[1] * frame_list[0].shape[1])
            roi[1] = int(min(points[0][1], points[1][1]) / frame.shape[0] * frame_list[0].shape[0])
            roi[2] = int(abs(points[1][0] - points[0][0]) / frame.shape[1] * frame_list[0].shape[1])
            roi[3] = int(abs(points[1][1] - points[0][1]) / frame.shape[0] * frame_list[0].shape[0])
            print(f"final results: VIDEO_CROP = {tuple(roi)}")
            return roi
        elif key == ord("d"):
            now_frame += 1
            if now_frame >= len(frame_list):
                now_frame = len(frame_list) - 1
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


def find_lower_upper_bounds(frame_list):
    colors_and_bounds = {"lower": [0, 0, 0], "upper": [255, 255, 255]}
    color_to_idx = {"red": 2, "green": 1, "blue": 0}
    bound_edit = "lower"
    color_edit = "red"
    now_frame = 0
    inc = 10
    key = 0
    mode = "multiple"
    on_filters = False
    height = 900 if frame_list[0].shape[0] > 900 else None
    # window_title = "original frame|frame: 0|bound: lower|color: red|inc/dec: 10"
    # ENTER - 13 D(next frame) A(prev frame) W(up) S(down) R(edit red) G(edit green) B(edit blue)
    # C(coarse - low precision settings (i/d 10)) F(fine - high precision setting (i/d 1))
    # U(edit upper bound) L(edit lower bound) P(print actual bounds)
    # M(multiple color mode) O(single color mode)
    while key != 27 and key != 13:
        frame = frame_list[now_frame]
        frame = resize_with_aspect_ratio(frame, height=height)
        cv2.imshow("frame", frame)
        new_title = f"original frame|frame: {now_frame}|bound: {bound_edit}|color: {color_edit}|inc/dec: {inc}"
        cv2.setWindowTitle("frame", new_title)
        if mode == "single":
            mask = cv2.inRange(frame, tuple(colors_and_bounds["lower"]), tuple(colors_and_bounds["upper"]))
            if on_filters:
                kernel = np.ones((5, 5), np.uint8)
                opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
            else:
                opening_img = mask
            cv2.imshow("mask", opening_img)
        elif mode == "multiple":
            r_mask = cv2.inRange(frame[:, :, 2], colors_and_bounds["lower"][2], colors_and_bounds["upper"][2])
            g_mask = cv2.inRange(frame[:, :, 1], colors_and_bounds["lower"][1], colors_and_bounds["upper"][1])
            b_mask = cv2.inRange(frame[:, :, 0], colors_and_bounds["lower"][0], colors_and_bounds["upper"][0])
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
            if now_frame >= len(frame_list):
                now_frame = len(frame_list) - 1
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
        # window_title = new_title


def find_m_point_circle(frame):
    points = [None, None]

    def show_with_points(frame):
        frame = frame.copy()
        if points[0] is not None and points[1] is not None:
            frame = cv2.circle(
                frame, points[0],
                int(np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)),
                (0, 0, 255), 1
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
            circle_param[1] = int(np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2))
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
    if len(trace_list) > trace_max_len:
        trace_list.pop(0)
    for trace_idx in range(1, len(trace_list)):
        line_brightness = 0.5 + trace_idx / (len(trace_list)-1) * 0.5
        new_color = tuple(np.array(color)*line_brightness)
        new_frame = cv2.line(new_frame, trace_list[trace_idx-1], trace_list[trace_idx], new_color, 2)
    return new_frame, trace_list


cap = cv2.VideoCapture(FILE_NAME)
print("opening video file...")
frame_list = video_to_frame_list(cap)
print("done")
if MARKERS_LOWER_BOUND is None or MARKERS_UPPER_BOUND is None:
    print("find the markers bounds")
    l_bound, u_bound = find_lower_upper_bounds(frame_list)
    print(f"MARKERS_LOWER_BOUND = {l_bound}"
          f"\nMARKERS_UPPER_BOUND = {u_bound}")
    MARKERS_LOWER_BOUND = l_bound
    MARKERS_UPPER_BOUND = u_bound
if ENABLE_CROP and not ENABLE_AUTO_CROP and VIDEO_CROP is None:
    print("crop the video, select two points")
    video_crop = find_video_crop(frame_list)
    VIDEO_CROP = video_crop
    append_to_settings("VIDEO_CROP", VIDEO_CROP)
    print("VIDEO_CROP saved to config file")
if ENABLE_AUTO_CROP:
    print("searching perfect video crop")
    video_crop = find_video_crop_automatically(frame_list, MARKERS_LOWER_BOUND, MARKERS_UPPER_BOUND, 100)
    VIDEO_CROP = video_crop
    print("done")
    print(f"found crop box: VIDEO_CROP = {video_crop}")
    append_to_settings("VIDEO_CROP", VIDEO_CROP)
    print("VIDEO_CROP saved to config file")
if ENABLE_CROP:
    print("cropping video...")
    frame_list = crop_frame_list(frame_list, VIDEO_CROP)
    print("done")
if TRACKING_OBJECT_LOWER_BOUND is None or TRACKING_OBJECT_UPPER_BOUND is None:
    l_bound, u_bound = find_lower_upper_bounds(frame_list)
    print(f"TRACKING_OBJECT_LOWER_BOUND = {l_bound}"
          f"\nTRACKING_OBJECT_UPPER_BOUND = {u_bound}")
    TRACKING_OBJECT_LOWER_BOUND = l_bound
    TRACKING_OBJECT_UPPER_BOUND = u_bound

print("checking settings...")
m_has_errors = False
o_has_errors = False
for frame_idx in frame_list:
    frame = frame_list[frame_idx]
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
            m_has_errors = True
    if TRACKING_OBJECT_LOWER_BOUND is not None and TRACKING_OBJECT_UPPER_BOUND is not None:
        tr_obj = find_objects(frame, TRACKING_OBJECT_LOWER_BOUND, TRACKING_OBJECT_UPPER_BOUND)
        if tr_obj is None or len(tr_obj) > 1:
            if "wrong object" not in err_obj[file_name]:
                err_obj[file_name]["wrong object"] = []
            err_obj[file_name]["wrong object"].append(frame_idx)
            print(f"wrong object at frame {frame_idx} in file '{file_name}'")
            o_has_errors = True
    frame_idx += 1
print("done")
if not m_has_errors:
    append_to_settings("MARKERS_LOWER_BOUND", MARKERS_LOWER_BOUND)
    append_to_settings("MARKERS_UPPER_BOUND", MARKERS_UPPER_BOUND)
    print("markers configuration checked and saved")
if not o_has_errors:
    append_to_settings("TRACKING_OBJECT_LOWER_BOUND", TRACKING_OBJECT_LOWER_BOUND)
    append_to_settings("TRACKING_OBJECT_UPPER_BOUND", TRACKING_OBJECT_UPPER_BOUND)
    print("object configuration checked and saved")
if o_has_errors or m_has_errors:
    agreement = input("configuration has errors do you want to save it? (y/n): ")
    if not agreement:
        print("settings, that has errors not saved")
        sys.exit(0)

if INITIAL_SHIFT is None or BOX_SIZE is None:
    print("select the plate with balls")
    initial_shift, box_size = find_initial_shift_and_box_size(
        frame_list[0],
        find_markers(
            frame_list[0],
            MARKERS_LOWER_BOUND,
            MARKERS_UPPER_BOUND
        )[0]
    )
    print(f"INITIAL_SHIFT = {initial_shift}\nBOX_SIZE = {box_size}")
    INITIAL_SHIFT = initial_shift
    BOX_SIZE = box_size
    append_to_settings("INITIAL_SHIFT", INITIAL_SHIFT)
    append_to_settings("BOX_SIZE", BOX_SIZE)
    print("INITIAL_SHIFT and BOX_SIZE saved to config file")
if M_POINT_CIRCLE is None and (RENDER_ROTATED_VIDEO or RENDER_M_POINT or RENDER_CENTER_POINT):
    print("select circle for M point")
    m_point_circle = find_m_point_circle(frame_list[0])
    M_POINT_CIRCLE = m_point_circle
    append_to_settings("M_POINT_CIRCLE", M_POINT_CIRCLE)
    print("M_POINT_CIRCLE saved to config file")
if RENDER_ROTATED_VIDEO or RENDER_M_POINT or RENDER_CENTER_POINT:
    print("searching marker rotation center...")
    marker_rotation_center = find_marker_rotation_center(frame_list, find_markers, MARKERS_LOWER_BOUND, MARKERS_UPPER_BOUND)
    print("done")
    append_to_settings("MARKER_ROTATION_CENTER", MARKER_ROTATION_CENTER)
    print("MARKER_ROTATION_CENTER saved to config file")
