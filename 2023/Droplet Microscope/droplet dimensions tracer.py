from glob import glob
import pandas as pd
import numpy as np
import json
import cv2
import sys
import os

ROOT = "./2023/Droplet Microscope/"
IMGS_IN_PATH = ROOT + "droplet dimensions exp/images/"
RESULTS_SAVE_PATH = ROOT + "droplet dimensions exp/results/test_res.xlsx"
BACKUP_FILE_PATH = ROOT + "exp_backup_data.json"

DISPLAY_IMG_HEIGHT = 700

OVERWRITE_SETTINGS = True


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


def append_to_settings(setting_name, setting_value, file_name):
    with open(BACKUP_FILE_PATH, "r") as config_file:
        settings = json.load(config_file)
    if file_name not in settings:
        settings[file_name] = {}
    if setting_name not in settings[file_name] or OVERWRITE_SETTINGS:
        settings[file_name][setting_name] = setting_value
    with open(BACKUP_FILE_PATH, "w") as config_file:
        json.dump(settings, config_file)


def read_settings(setting_name, file_name):
    with open(BACKUP_FILE_PATH, "r") as config_file:
        settings = json.load(config_file)
    if file_name not in settings:
        print("no settings for that file found")
        return -2
    now_config = settings[file_name]
    if setting_name not in now_config:
        print(f"{setting_name} not found in this config file")
        return -1
    return now_config[setting_name]


def get_pp_sec_point(point1: tuple[int], point2: tuple[int], point3: tuple[int]) -> tuple[int]:
    a_sq = ((point2[0]-point3[0])**2) + \
        ((point2[1]-point3[1])**2)
    b_sq = ((point3[0]-point1[0])**2) + \
        ((point3[1]-point1[1])**2)
    dx_c = (point2[0] - point1[0])
    dy_c = (point2[1] - point1[1])
    c_sq = (dx_c**2)+(dy_c**2)
    new_x = point2[0] - (a_sq + c_sq - b_sq)*dx_c//(2*c_sq)
    new_y = point2[1] - (a_sq + c_sq - b_sq)*dy_c//(2*c_sq)
    return (new_x, new_y)


def select_points(win_name, frame, select_dpt=False):
    global zoom_k
    points = [None]*(2 + select_dpt*2)
    point_edits = 0
    img_h, img_w = frame.shape[:-1]
    # img_h = frame.shape[0]
    is_changing_zoom = False
    zoom_k = 9

    def show_with_points(frame):
        show_frame = resize_with_aspect_ratio(
            frame.copy(), height=DISPLAY_IMG_HEIGHT)
        if points[0] and points[1]:
            cv2.line(show_frame, points[0], points[1], (0, 255, 255), 1)
            if select_dpt and points[2]:
                new_x, new_y = get_pp_sec_point(
                    points[0], points[1], points[2])
                cv2.line(show_frame, points[2],
                         (new_x, new_y), (255, 0, 255), 1)
        if points[0] and select_dpt and points[3]:
            dx = points[3][0] - points[0][0]
            dy = points[3][1] - points[0][1]
            cv2.line(
                show_frame, points[0], (points[0][0]+dx*3, points[0][1]+dy*3), (0, 255, 125), 1)
        for i in points:
            if i is None:
                continue
            cv2.circle(show_frame, i, 1, (0, 0, 255), -1)
        cv2.imshow("frame", show_frame)

    # height = 900 if frame.shape[0] > 900 else None
    # frame = resize_with_aspect_ratio(frame, height=DISPLAY_IMG_HEIGHT)
    point_edits = 0

    def mouse_callback(event, x, y, flag, param):
        global zoom_k
        if is_changing_zoom:
            if event == 10:
                zoom_k += -1 * (flag < 0) + (flag > 0)
                if zoom_k < 1:
                    zoom_k = 1
        if event == cv2.EVENT_LBUTTONDOWN or flag == 1:
            points[point_edits] = [x, y]
            show_with_points(frame)
        zoom_area(x, y)

    def zoom_area(x, y):
        x = x*img_h//DISPLAY_IMG_HEIGHT
        y = y*img_h//DISPLAY_IMG_HEIGHT
        zoom_h, zoom_w = img_h // zoom_k // 2, img_w // zoom_k // 2
        n_x, n_y = x, y
        n_x = n_x if n_x - zoom_w >= 0 else zoom_w
        n_x = n_x if n_x + zoom_w <= img_w else img_w - zoom_w
        n_y = n_y if n_y - zoom_h >= 0 else zoom_h
        n_y = n_y if n_y + zoom_h <= img_h else img_h - zoom_h
        zoom_img = frame.copy()
        zoom_img = cv2.circle(zoom_img, (x, y), radius=0,
                              color=(0, 0, 255), thickness=-1)
        zoom_img = zoom_img[n_y-zoom_h:n_y+zoom_h, n_x-zoom_w:n_x+zoom_w]
        cv2.imshow("zoom", resize_with_aspect_ratio(
            zoom_img, height=DISPLAY_IMG_HEIGHT//2))
        cv2.setWindowProperty("zoom", cv2.WND_PROP_TOPMOST, 1)

    cv2.imshow(win_name, resize_with_aspect_ratio(
        frame, height=DISPLAY_IMG_HEIGHT))
    cv2.setMouseCallback(win_name, mouse_callback)
    key = 0

    while key != 27 and key != 13:
        show_with_points(frame)
        key = cv2.waitKey(0)
        if key == 27 or key == 13:
            cv2.destroyWindow(win_name)
            cv2.destroyWindow("zoom")
        if key == 13:
            for i in range(len(points)):
                if not points[i]:
                    return None
                points[i][0] = points[i][0]*img_h//DISPLAY_IMG_HEIGHT
                points[i][1] = points[i][1]*img_h//DISPLAY_IMG_HEIGHT
            return points
        elif key == ord("w") and points[point_edits]:
            points[point_edits][1] -= 1
        elif key == ord("s") and points[point_edits]:
            points[point_edits][1] += 1
        elif key == ord("a") and points[point_edits]:
            points[point_edits][0] -= 1
        elif key == ord("d") and points[point_edits]:
            points[point_edits][0] += 1
        elif key == ord("c"):
            points[point_edits] = None
        elif key == ord("1"):
            point_edits = 0
        elif key == ord("2"):
            point_edits = 1
        elif key == ord("3") and select_dpt:
            point_edits = 2
        elif key == ord("4") and select_dpt:
            point_edits = 3
        elif key == ord("z"):
            is_changing_zoom = not is_changing_zoom


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


backup_found = False
if os.path.exists(BACKUP_FILE_PATH):
    print("found backup file, using it to recover traced data")
    backup_found = True

paths = glob(IMGS_IN_PATH + "*.jpg")
q = 0
p_idx = -1
ready_to_convert = True
if not backup_found:
    with open(BACKUP_FILE_PATH, "w") as backup_file:
        json.dump({}, backup_file)
    print("backup created")
for path_idx in range(len(paths)):
    path = paths[path_idx]
    f_name = path[path.rfind("\\")+1:]
    tracing_propgress = read_settings("TRACING_PRGR", f_name)
    ref_size = read_settings("BCK_REF_SIZE", f_name)
    # print(tracing_propgress, ref_size)
    if tracing_propgress > 2 and ref_size is not None and ref_size > 0:
        continue
    ready_to_convert = False
    if p_idx == -1:
        p_idx = path_idx
    if tracing_propgress < 0:
        append_to_settings("TRACING_PRGR", 0, f_name)
        append_to_settings("BCK_DROPLET_PTS", [], f_name)
        append_to_settings("BCK_REF_PTS", [], f_name)
        append_to_settings("BCK_REF_SIZE", None, f_name)
if ready_to_convert:
    confirmation = input(
        "Your file is ready for conversion. do you want to do it now? (y/n): ")
    if confirmation == "y":
        print("converting...")
        data = pd.DataFrame()
        for path in paths:
            f_name = path[path.rfind("\\")+1:]
            d_pts = read_settings("BCK_DROPLET_PTS", f_name)
            r_pts = read_settings("BCK_REF_PTS", f_name)
            ref_size = read_settings("BCK_REF_SIZE", f_name)
            ref_h = np.sqrt(
                ((r_pts[0][0] - r_pts[1][0])**2)+((r_pts[0][1] - r_pts[1][1])**2))
            r = np.sqrt(((d_pts[0][0] - d_pts[1][0])**2) +
                        ((d_pts[0][1] - d_pts[1][1])**2))
            r_out = r*ref_size/ref_h
            n_pt = get_pp_sec_point(d_pts[0], d_pts[1], d_pts[2])
            h = np.sqrt(((d_pts[2][0] - n_pt[0])**2) +
                        ((d_pts[2][1] - n_pt[1])**2))
            h_out = h*ref_size/ref_h
            theta = find_angle_between_two_lines(d_pts[3], d_pts[0], d_pts[1])
            data = pd.concat([data, pd.DataFrame({
                "file name": f_name,
                "r (mm)": r_out,
                "h (mm)": h_out,
                "theta": theta
            }, index=[0])])
        print("writing to xlsx file...")
        with pd.ExcelWriter(RESULTS_SAVE_PATH) as writer:
            data.to_excel(writer, sheet_name="data", index=False)
        sys.exit(0)
    print("cancelled")
tracing_propgress = 0
ref_size = None
is_hiden = False
while q != 27:
    path = paths[p_idx]
    f_name = path[path.rfind("\\")+1:]
    img = cv2.imread(path)
    visual_img = img.copy()
    tracing_propgress = read_settings("TRACING_PRGR", f_name)
    dpl_pts = read_settings("BCK_DROPLET_PTS", f_name)
    ref_pts = read_settings("BCK_REF_PTS", f_name)
    ref_size = read_settings("BCK_REF_SIZE", f_name)
    print(f"working with '{f_name}'. Tracing progress: {tracing_propgress}.")
    if not dpl_pts or len(dpl_pts) == 0:
        print("Droplet points are unselected")
    else:
        cv2.line(visual_img, dpl_pts[0], dpl_pts[1], (0, 255, 255), 2)
        s_pts = get_pp_sec_point(dpl_pts[0], dpl_pts[1], dpl_pts[2])
        cv2.line(visual_img, dpl_pts[2], s_pts, (255, 0, 255), 2)
        for pt in dpl_pts:
            cv2.circle(visual_img, pt, 2, (0, 0, 255), -1)
    if not ref_pts or len(ref_pts) == 0:
        print("Reference points are unselected")
    else:
        cv2.line(visual_img, ref_pts[0], ref_pts[1], (0, 255, 255), 2)
        for pt in ref_pts:
            cv2.circle(visual_img, pt, 2, (0, 0, 255), -1)
    if ref_size is None:
        print("Reference size is not specified")
    cv2.imshow("frame", resize_with_aspect_ratio(
        img if is_hiden else visual_img, height=DISPLAY_IMG_HEIGHT))
    cv2.setWindowTitle(
        "frame", f"file: {p_idx+1}/{len(paths)}. Tracing progress: {tracing_propgress}. main window")
    q = cv2.waitKey(0)
    if q == 13:
        p_idx += 1
        if p_idx >= len(paths):
            p_idx = len(paths) - 1
    elif q == 8:
        p_idx -= 1
        if p_idx < 0:
            p_idx = 0
    elif q == ord("r"):
        upd_ref_size = int(
            input("Enter reference size in mm (-1 to cancel): "))
        if upd_ref_size != -1:
            ref_size = upd_ref_size
            append_to_settings("BCK_REF_SIZE", upd_ref_size, f_name)
            print("Reference size saved")
        else:
            print("Operation canceled")
    elif q == ord("d"):
        print("Select droplet points")
        if tracing_propgress >= 2:
            tracing_propgress -= 2
        cv2.setWindowTitle(
            "frame", f"file: {p_idx+1}/{len(paths)}. Tracing progress: {tracing_propgress}. droplet point selection")
        dpl_pts = select_points("frame", img, True)
        if dpl_pts is not None:
            append_to_settings("BCK_DROPLET_PTS", dpl_pts, f_name)
            print("Results saved")
            tracing_propgress += 2
            append_to_settings("TRACING_PRGR", tracing_propgress, f_name)
        else:
            print("Operation canceled")
    elif q == ord("p"):
        print("Select reference points")
        if tracing_propgress % 2 == 1:
            tracing_propgress -= 1
        cv2.setWindowTitle(
            "frame", f"file: {p_idx+1}/{len(paths)}. Tracing progress: {tracing_propgress}. reference points")
        ref_pts = select_points("frame", img)
        if ref_pts is not None:
            append_to_settings("BCK_REF_PTS", ref_pts, f_name)
            print("Results saved")
            tracing_propgress += 1
            append_to_settings("TRACING_PRGR", tracing_propgress, f_name)
        else:
            print("Operation canceled")
    elif q == ord("h"):
        is_hiden = not is_hiden
        print("Hiden"*is_hiden + "shown"*(not is_hiden))
print("Exiting...")

cv2.destroyAllWindows()
