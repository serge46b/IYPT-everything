import pandas as pd
import numpy as np
import cv2
import sys


ROOT = "./2023/magnetic gearbox/"
VIDEO_PATH = ROOT + "exp videos/LANa 1000ms 100mm.mp4"
DATA_SV_PATH = ROOT + "A200delt minmm 240fps result.xlsx"

DISPLAY_IMG_HEIGHT = 700
ENABLE_CROP = True
CROPPING_SHIFT = 50
LOG_DATA = False
IS_CONFIGING = True
DEBUG = True

DRIVEN_WHEELS_AMOUNT = 1
# DRIVER_WHEEL_CFG = ([487, 454], 32, 7)
# DRIVEN_WHEEL_CFGS = [([403, 417], 35, 5)]

# DRIVER_WHEEL_CFG = ([597, 1286], 362, 70)
# DRIVER_WHEEL_COLOR = ((142, 101, 77), (219, 164, 139))
# DRIVEN_WHEEL_CFGS = [([570, 504], 353, 69)]
# DRIVEN_WHEEL_COLOR = [((96, 42, 56), (208, 97, 135))]

DRIVER_WHEEL_CFG = ([471, 1053], 316, 59)
# DRIVER_WHEEL_CFG = None
# DRIVER_WHEEL_CFG = ([600, 1278], 362, 65)
DRIVER_WHEEL_COLOR = None
DRIVEN_WHEEL_CFGS = None
DRIVEN_WHEEL_COLOR = None


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


def find_objects(frame, object_lower_range, object_upper_range, wheel_cfg):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.circle(mask, wheel_cfg[0], wheel_cfg[1] -
               wheel_cfg[2], 1, 1)
    cv2.circle(mask, wheel_cfg[0], wheel_cfg[1] +
               wheel_cfg[2], 1, 1)
    cv2.floodFill(
        mask, None, (wheel_cfg[0][0]-wheel_cfg[1], wheel_cfg[0][1]), 1)
    cutted_frame = cv2.bitwise_and(frame, frame, mask=mask)
    mask = cv2.inRange(cutted_frame, object_lower_range, object_upper_range)
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
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


def find_lower_upper_bounds(capturer, obj_amount, cfg=None):
    colors_and_bounds = {"lower": [0, 0, 0], "upper": [255, 255, 255]}
    color_to_idx = {"red": 2, "green": 1, "blue": 0}
    bound_edit = "lower"
    color_edit = "red"
    now_frame = 0
    inc = 10
    key = 0
    mode = "multiple"
    on_filters = False
    height = DISPLAY_IMG_HEIGHT if capturer.get(
        cv2.CAP_PROP_FRAME_HEIGHT) > DISPLAY_IMG_HEIGHT else None
    vr_fl = False
    crop_box = None if cfg is None else (
        cfg[0][0]-cfg[1]-cfg[2], cfg[0][1]-cfg[1]-cfg[2], (cfg[1]+cfg[2])*2, (cfg[1]+cfg[2])*2)
    # window_title = "original frame|frame: 0|bound: lower|color: red|inc/dec: 10"
    # ENTER - 13 D(next frame) A(prev frame) W(up) S(down) R(edit red) G(edit green) B(edit blue)
    # C(coarse - low precision settings (i/d 10)) F(fine - high precision setting (i/d 1))
    # U(edit upper bound) L(edit lower bound) P(print actual bounds)
    # M(multiple color mode) O(single color mode)
    while key != 27 and key != 13:
        capturer.set(cv2.CAP_PROP_POS_FRAMES, now_frame)
        frame = capturer.read()[1]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        if crop_box is not None:
            frame = frame[crop_box[1]:crop_box[1] + crop_box[3],
                          crop_box[0]:crop_box[0] + crop_box[2], :]
        # frame = resize_with_aspect_ratio(frame, height=height)
        cv2.imshow("frame", resize_with_aspect_ratio(
            frame, height=height))
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
            cv2.imshow("mask", resize_with_aspect_ratio(
                opening_img, height=height))
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
            cv2.imshow("mask", resize_with_aspect_ratio(
                final_mask, height=height))
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
                    # if crop_box is not None:
                    #     frame = frame[crop_box[1]:crop_box[1] + crop_box[3],
                    #                   crop_box[0]:crop_box[0] + crop_box[2], :]
                    objects = find_objects(frame, tuple(
                        colors_and_bounds["lower"]), tuple(colors_and_bounds["upper"]), cfg)
                    if objects is None or len(objects) != obj_amount:
                        now_frame = int(capturer.get(
                            cv2.CAP_PROP_POS_FRAMES)) - 1
                        print(
                            f"wrong marker at frame {now_frame}, objects found: {objects}")
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


def find_marker_circles(frame):
    points = [None, None, None]
    height = DISPLAY_IMG_HEIGHT if frame.shape[0] > DISPLAY_IMG_HEIGHT else None

    def show_with_points(frame):
        show_frame = resize_with_aspect_ratio(frame.copy(), height=height)
        # show_frame = cv2.cvtColor(resize_with_aspect_ratio(
        #     show_frame, height=DISPLAY_IMG_HEIGHT), cv2.COLOR_GRAY2BGR)
        if points[0] is not None and points[1] is not None:
            r = np.sqrt((points[1][0] - points[0][0]) **
                        2 + (points[1][1] - points[0][1])**2)
            cv2.circle(
                show_frame, points[0],
                int(r),
                (0, 255, 255), 1
            )
            if points[2] is not None:
                new_r = np.sqrt((points[2][0] - points[0][0])
                                ** 2 + (points[2][1] - points[0][1])**2)
                cv2.circle(show_frame, points[0], int(new_r), (255, 0, 255), 1)
                cv2.circle(show_frame, points[0], int(
                    2*r-new_r), (255, 0, 255), 1)
        for i in points:
            if i is None:
                continue
            cv2.circle(show_frame, i, 1, (0, 0, 255), -1)
        cv2.imshow("frame", show_frame)
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
            circle_param = [None, 0, 0]
            circle_param[0] = (points[0][0] * frame.shape[0] //
                               height, points[0][1] * frame.shape[0] // height)
            circle_param[1] = int(
                np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2) * frame.shape[0] // height)
            circle_param[2] = int((np.sqrt((points[2][0] - points[0][0])**2 + (
                points[2][1] - points[0][1])**2) * frame.shape[0] // height) - circle_param[1])
            # print(f"final results: M_POINT_CIRCLE = {tuple(circle_param)}")
            return tuple(circle_param)
        elif key == ord("c"):
            points[point_edits] = None
            print(f"deleted point {point_edits + 1}")
        elif key == ord("1"):
            point_edits = 0
        elif key == ord("2"):
            point_edits = 1
        elif key == ord("3"):
            point_edits = 2


def get_roi():
    min_x, min_y = DRIVER_WHEEL_CFG[0][0] - \
        DRIVER_WHEEL_CFG[1], DRIVER_WHEEL_CFG[0][1]-DRIVER_WHEEL_CFG[1]
    max_x, max_y = DRIVER_WHEEL_CFG[0][0] + \
        DRIVER_WHEEL_CFG[1], DRIVER_WHEEL_CFG[0][1]+DRIVER_WHEEL_CFG[1]
    for driven_cfg in DRIVEN_WHEEL_CFGS:
        x, y = driven_cfg[0][0]-driven_cfg[1], driven_cfg[0][1]-driven_cfg[1]
        x2, y2 = driven_cfg[0][0]+driven_cfg[1], driven_cfg[0][1]+driven_cfg[1]
        if min_x > x:
            min_x = x
        if min_y > y:
            min_y = y
        if max_x < x2:
            max_x = x2
        if max_y < y2:
            max_y = y2
    roi = (min_y-CROPPING_SHIFT, max_y+CROPPING_SHIFT,
           min_x-CROPPING_SHIFT, max_x+CROPPING_SHIFT)
    DRIVER_WHEEL_CFG[0][0] -= roi[2]
    DRIVER_WHEEL_CFG[0][1] -= roi[0]
    for idx in range(DRIVEN_WHEELS_AMOUNT):
        DRIVEN_WHEEL_CFGS[idx][0][0] -= roi[2]
        DRIVEN_WHEEL_CFGS[idx][0][1] -= roi[0]
    return roi


def crop_to_roi(frame, roi):
    return frame[roi[0]:roi[1], roi[2]:roi[3], :]


def get_marker_position(frame, wheel_cfg):
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.circle(mask, wheel_cfg[0], wheel_cfg[1] -
               wheel_cfg[2], 1, 1)
    cv2.circle(mask, wheel_cfg[0], wheel_cfg[1] +
               wheel_cfg[2], 1, 1)
    cv2.floodFill(
        mask, None, (wheel_cfg[0][0]-wheel_cfg[1], wheel_cfg[0][1]), 1)
    cutted_frame = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow("DEBUG: cutout", cutted_frame)
    _, th_cl = cv2.threshold(cutted_frame, 50, 255, cv2.THRESH_BINARY)
    _, th = cv2.threshold(cv2.cvtColor(
        th_cl, cv2.COLOR_BGR2GRAY), 254, 255, cv2.THRESH_BINARY)
    # cut_frame_gary = cv2.cvtColor(cutted_frame, cv2.COLOR_BGR2GRAY)
    # _, th = cv2.threshold(cut_frame_gary, 57, 255, cv2.THRESH_BINARY)
    # cv2.imshow("DEBUG: threshold", th)
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("DEBUG: opening", opening_img)
    contours, hierarchy = cv2.findContours(
        opening_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    # cv2.imshow("DEBUG: contours", frame)
    # cv2.waitKey(0)
    if len(contours) != 1:
        return None
    m = cv2.moments(contours[0])
    if m["m00"] == 0:
        cx = 0
        cy = 0
    else:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
    return (cx, cy)


def get_markers_positions(frame):
    centers = []
    centers.append(get_marker_position(frame, DRIVER_WHEEL_CFG))
    for wheel_cfg in DRIVEN_WHEEL_CFGS:
        centers.append(get_marker_position(frame, wheel_cfg))
    return centers


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
    # if def_rads > np.pi/2:
    #     rads = 2*np.pi-rads
    # print(360 - np.degrees(rads))
    return rads


video_stream = cv2.VideoCapture(VIDEO_PATH)
_, frame = video_stream.read()


if IS_CONFIGING:
    print("Select main wheel")
    if DRIVER_WHEEL_CFG is None:
        DRIVER_WHEEL_CFG = find_marker_circles(frame)
    cfg = DRIVER_WHEEL_CFG
    print(f"config: {DRIVER_WHEEL_CFG}")
    if DRIVER_WHEEL_COLOR is None:
        DRIVER_WHEEL_COLOR = find_lower_upper_bounds(
            video_stream, 1, DRIVER_WHEEL_CFG)
    print(f"color cfg: {DRIVER_WHEEL_COLOR}")
    for i in range(DRIVEN_WHEELS_AMOUNT):
        print(f"Select {i+1} driven wheel")
        # cfg = DRIVEN_WHEEL_CFGS[i]
        if DRIVEN_WHEEL_CFGS[i]:
            DRIVEN_WHEEL_CFGS[i] = find_marker_circles(frame)
        cfg = DRIVEN_WHEEL_CFGS[i]
        print(f"config: {cfg}")
        if DRIVEN_WHEEL_COLOR:
            DRIVEN_WHEEL_COLOR[i] = find_lower_upper_bounds(
                video_stream, 1, (cfg[0][0]-cfg[1]-cfg[2], cfg[0][1]-cfg[1]-cfg[2], (cfg[1]+cfg[2])*2, (cfg[1]+cfg[2])*2))
        print(f"color cfg: {DRIVEN_WHEEL_COLOR[i]}")
    sys.exit(0)

if DRIVER_WHEEL_CFG is None or len(DRIVEN_WHEEL_CFGS) != DRIVEN_WHEELS_AMOUNT:
    print("Wrong config provided")
    sys.exit(0)

ROI = get_roi()

crp_frame = crop_to_roi(frame, ROI)
prev_markers_centers = get_markers_positions(crp_frame)
data = pd.DataFrame()
print("processing video file")
while video_stream.isOpened():
    ret, frame = video_stream.read()
    if not ret:
        print("video stream ended")
        break
    crp_frame = crop_to_roi(frame, ROI)
    markers_centers = get_markers_positions(crp_frame)
    # pre_data = {"frame": video_stream.get(cv2.CAP_PROP_POS_FRAMES)}
    # for i in range(len(markers_centers)):
    #     center = markers_centers[i]
    #     prv_center = prev_markers_centers[i]
    #     circle_ceter = DRIVER_WHEEL_CFG[0] if i == 1 else DRIVEN_WHEEL_CFGS[i-1][0]
    #     angle = find_angle_between_two_lines(center, circle_ceter, prv_center)
    #     # print(i, angle)
    #     pre_data[f"gear angle {i}"] = angle
    # if LOG_DATA:
    #     data = pd.concat([data, pd.DataFrame(
    #         pre_data, index=[0])])
    if DEBUG:
        cv2.imshow("img", resize_with_aspect_ratio(
            frame, height=DISPLAY_IMG_HEIGHT))
        cv2.imshow("cropped img", resize_with_aspect_ratio(
            crp_frame, height=DISPLAY_IMG_HEIGHT))
        dbg_frame = crp_frame.copy()
        for center in markers_centers:
            cv2.circle(dbg_frame, center, 2, (0, 0, 255), -1)
        cv2.imshow("centers", resize_with_aspect_ratio(
            dbg_frame, height=DISPLAY_IMG_HEIGHT))
        # cv2.circle(
        #     dbg_frame, DRIVER_WHEEL_CFG[0], DRIVER_WHEEL_CFG[1], (0, 255, 255), 1)
        # cv2.circle(
        #     dbg_frame, DRIVEN_WHEEL_CFGS[0][0], DRIVEN_WHEEL_CFGS[0][1], (0, 255, 255), 1)
        # cv2.imshow("circles", resize_with_aspect_ratio(
        #     dbg_frame, height=DISPLAY_IMG_HEIGHT))
        q = cv2.waitKey(0)
        if q == 27:
            break
    prev_markers_centers = markers_centers

video_stream.release()
cv2.destroyAllWindows()

if LOG_DATA:
    print("writing to .xlsx file...")
    with pd.ExcelWriter(DATA_SV_PATH) as writer:
        data.to_excel(writer, sheet_name="data", index=False)
    print("done")
