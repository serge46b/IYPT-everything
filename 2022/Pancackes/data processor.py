import pandas as pd
import numpy as np
import cv2

# SETTINGS
# file name
FILE_NAME = "./2348/2348 data.xlsx"
VIDEO_FILE_NAME = "./2348/2348 cropped.avi"
# timing
ST_FRAME_IDX = 73
END_FRAME_IDX = 1016
# Logging
# APPEND_CALCULATIONS_TO_FILE = True

cap = cv2.VideoCapture(VIDEO_FILE_NAME)


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
    return 360 - np.degrees(rads)


def find_T(data_df, circle_center_crd, x_y_props, st_frame=0, end_frame=None):
    if end_frame is None:
        end_frame = len(data_df.iloc) - 1
    # box_size_x = circle_bbox[0]
    # box_size_y = circle_bbox[1]
    # box_center_x = box_size_x//2
    # box_center_y = box_size_y//2
    box_center_x = circle_center_crd[0]
    box_center_y = circle_center_crd[1]

    revolution_x = None
    revolution_y = None
    prev_angle = None
    lst_rev_frame = 0
    T = None
    rev_n = 0
    flag = False
    for i in data_df.iloc:
        if i["frame"] < st_frame or i["frame"] > end_frame:
            continue
        obj_rel_x, obj_rel_y = i[x_y_props[0]], i[x_y_props[1]]
        if revolution_x is None or revolution_y is None:
            revolution_x = obj_rel_x
            revolution_y = obj_rel_y
            continue
        angle = find_angle_between_two_lines(
            (revolution_x, revolution_y),
            (box_center_x, box_center_y),
            (obj_rel_x, obj_rel_y)
        )
        if prev_angle is None:
            prev_angle = 0
        if 135 < angle < 235:
            flag = True
        if abs(prev_angle - angle) > 300 and flag:
            flag = False
            rev_n += 1
            # print(rev_n)
            new_T = i["frame"] - lst_rev_frame
            revolution_x = obj_rel_x
            revolution_y = obj_rel_y
            lst_rev_frame = i["frame"]
            if T is None:
                T = new_T
                continue
            T += new_T
        # print(prev_angle, angle)
        prev_angle = angle
        # frame = frames_list[i["frame"]].copy()
        # frame = cv2.line(frame, (box_center_y-revolution_y+box_center_x, box_center_y+revolution_x-box_center_x), (box_center_x, box_center_y), (255, 255, 0), 1)  # 90
        # frame = cv2.line(frame, (revolution_x, revolution_y), (box_center_x, box_center_y), (255, 0, 255), 1)
        # frame = cv2.line(frame, (box_center_x, box_center_y), (obj_rel_x, obj_rel_y), (255, 0, 255), 1)
        # frame = cv2.circle(frame, (revolution_x, revolution_y), 3, (255, 0, 0), -1)
        # frame = cv2.circle(frame, (obj_rel_x, obj_rel_y), 3, (255, 0, 0), -1)
        # frame = cv2.circle(frame, (box_center_x, box_center_y), 3, (255, 0, 0), -1)
        # cv2.imshow("frame", frame)
        # cv2.waitKey(0)
    return T / rev_n


def find_circle_roi(data_df, x_y_props, st_frame=0, end_frame=None):
    mxx = 0
    mnx = data_df[x_y_props[0]][0]
    mxy = 0
    mny = data_df[x_y_props[1]][0]
    for data in data_df:
        if data["frame"] < st_frame or data["frame"] > end_frame:
            continue
        x = data[x_y_props[0]]
        y = data[x_y_props[1]]
        if x > mxx:
            mxx = x
        if x < mnx:
            mnx = x
        if y > mxy:
            mxy = y
        if y < mny:
            mny = y
    return mnx, mny, mxx-mnx, mxy-mny


def find_st_end_frame(frame_list):
    st_end_frame_idx = {"start": 0, "end": len(frame_list) - 1}
    editing = "start"
    now_frame = 0
    inc = 10
    key = 0
    height = 900 if frame_list[0].shape[0] > 900 else None
    # ENTER - 13 D(next frame) A(prev frame) W(up)
    # C(coarse - low precision settings (i/d 10)) F(fine - high precision setting (i/d 1))
    # S(edit start frame) E(edit end frame)
    while key != 27 and key != 13:
        frame = frame_list[now_frame]
        frame = resize_with_aspect_ratio(frame, height=height)
        cv2.imshow("frame", frame)
        cv2.setWindowTitle("frame", f"original frame|editing: {editing} frame idx|inc: {inc}")
        st_end_frame_idx[editing] = now_frame
        key = cv2.waitKey(0)
        if key == 27 or key == 13:
            cv2.destroyWindow("frame")
            cv2.destroyWindow("mask")
        if key == 13:
            print(f"final settings:\n\tST_FRAME_IDX = {st_end_frame_idx['start']}"
                  f"\n\tEND_FRAME_IDX = {st_end_frame_idx['end']}")
            return st_end_frame_idx['start'], st_end_frame_idx['end']
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
        elif key == ord("c"):
            inc = 10
        elif key == ord("f"):
            inc = 1
        elif key == ord("s"):
            editing = "start"
            now_frame = st_end_frame_idx[editing]
        elif key == ord("e"):
            editing = "end"
            now_frame = st_end_frame_idx[editing]


data = pd.read_excel(FILE_NAME, sheet_name="data")
config = pd.read_excel(FILE_NAME, sheet_name="config")

if ST_FRAME_IDX is None or END_FRAME_IDX is None:
    print("opening video file...")
    frames_list = video_to_frame_list(cap)
    cap.release()
    print("done")
    print("select video start and end frames")
    ST_FRAME_IDX, END_FRAME_IDX = find_st_end_frame(frames_list)

print(find_circle_roi(data, ("object rotated relative x", "object rotated relative y")))

# ball_period = find_T(
#     data,
#     (config["box size x"][0] // 2, config["box size y"][0] // 2),
#     ("object relative x", "object relative y"),
#     ST_FRAME_IDX,
#     END_FRAME_IDX
# )
# print(ball_period)
# mmxx = 0
# mmnx = config["frame width"][0]
# mmxy = 0
# mmny = config["frame height"][0]
#
# for i in data.iloc:
#     if i["frame"] < ST_FRAME_IDX or i["frame"] > END_FRAME_IDX:
#         continue
#     x = i["marker x"]
#     y = i["marker y"]
#     if x > mmxx:
#         mmxx = x
#     if x < mmnx:
#         mmnx = x
#     if y > mmxy:
#         mmxy = y
#     if y < mmny:
#         mmny = y
# marker_center_crd = ((mmxx + mmnx) // 2, (mmxy + mmny) // 2)
# # print(marker_center_crd)
# marker_period = find_T(
#     data,
#     marker_center_crd,
#     ("marker x", "marker y"),
#     ST_FRAME_IDX,
#     END_FRAME_IDX
# )
# print(marker_period)
