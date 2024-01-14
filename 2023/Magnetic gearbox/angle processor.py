import os
import json
import numpy as np
import pandas as pd

# CONFIG_FILE_PATH = "./2023/Walker/ArUco tracing config.json"
CONFIG_FILE_PATH = "./2023/Magnetic gearbox/exp configs/s46b ArUco general tracing cfg.json"


if not CONFIG_FILE_PATH:
    CONFIG_FILE_PATH = input("Pass config file path:")
if not os.path.exists(CONFIG_FILE_PATH):
    print("cannot load config file. Exiting...")
    os._exit(0)

with open(CONFIG_FILE_PATH) as cfg_f:
    loaded_cfg = json.load(cfg_f)


ROOT = loaded_cfg["root_path"]
VIDEO_SOURCE = ROOT + loaded_cfg["video_in_rpath"]
FILE_IN_PATH = ROOT + loaded_cfg["tracing_file_dir_rpath"] + \
    (str(VIDEO_SOURCE) if VIDEO_SOURCE is int else VIDEO_SOURCE[VIDEO_SOURCE.rfind(
        '/'):VIDEO_SOURCE.rfind('.')]) + ".csv"
FILE_OUT_PATH = ROOT + loaded_cfg["tracing_file_dir_rpath"] + "results processed/" + \
    FILE_IN_PATH[FILE_IN_PATH.rfind(
        '/')+1:FILE_IN_PATH.rfind(".")] + " processed.xlsx"


GROUND_MARKER_ID = loaded_cfg["root_marker_id"]
USING_G_MARKER = GROUND_MARKER_ID != -1


def find_angle_between_3_pts(pt1: tuple[float, float], pt2: tuple[float, float], pt3: tuple[float, float], angle0to360=True):
    avec = [pt2[0] - pt1[0], pt2[1] - pt1[1]]
    cvec = [pt2[0] - pt3[0], pt2[1] - pt3[1]]
    dp = np.dot(avec, cvec)
    avec_mag = np.linalg.norm(avec)
    cvec_mag = np.linalg.norm(cvec)
    rads = np.arccos(dp/(avec_mag*cvec_mag))
    if np.isnan(rads):
        rads = 0
    nvec = [pt1[1] - pt2[1], pt2[0] - pt1[0]]
    dp = np.dot(cvec, nvec)
    nvec_mag = np.linalg.norm(nvec)
    def_rads = np.arccos(dp/(cvec_mag*nvec_mag))
    if np.isnan(def_rads):
        def_rads = 0
    angle = 0
    if angle0to360:
        if def_rads > np.pi/2:
            rads = 2*np.pi-rads
        angle = 360 - np.degrees(rads)
    else:
        angle = np.degrees(rads)*(-1 if def_rads > np.pi/2 else 1)
    return angle


print("loading data...")
in_df = pd.read_csv(FILE_IN_PATH, sep=", ", engine='python')
clmns = ["frame"]
for i in loaded_cfg["tracing_marker_ids"]:
    # clmns.append(f"tx{i}")
    # clmns.append(f"ty{i}")
    clmns.append(f"alpha{i}")
    clmns.append(f"d_alpha{i}")
out_df = pd.DataFrame(columns=clmns)


print("processing...")
prev_f_idx = in_df["frame"][0]
ids_found = False
add_dct = {"frame": prev_f_idx}
mn_x, mx_x = {}, {}
mn_y, mx_y = {}, {}
pts = {}
for index in in_df.index:
    frame_idx = in_df["frame"][index]
    m_id = in_df["id"][index]
    x = in_df[f"tx"][index]
    y = in_df[f"ty"][index]
    if not m_id in pts:
        pts[m_id] = {}
    pts[m_id][frame_idx-1] = (x, y)
    if not m_id in mn_x or x < mn_x[m_id]:
        mn_x[m_id] = x
    if not m_id in mn_y or y < mn_y[m_id]:
        mn_y[m_id] = y
    if not m_id in mx_x or x > mx_x[m_id]:
        mx_x[m_id] = x
    if not m_id in mx_y or y > mx_y[m_id]:
        mx_y[m_id] = y


n = {idx: 0 for idx in pts}
prev_angl = {idx: 0 for idx in pts}
for pt_id in range(len(pts[m_id])-1):
    for idx in pts:
        x1 = mn_x[idx]
        y1 = mn_y[idx]
        x2 = mx_x[idx]
        y2 = mx_y[idx]
        c_x = (x1+x2)/2
        c_y = (y1+y2)/2
        try:
            pt1 = pts[idx][pt_id]
        except KeyError:
            continue
        try:
            pt2 = pts[idx][pt_id+1]
        except KeyError:
            continue
        abs_angle = find_angle_between_3_pts(
            (c_x, c_y-1), (c_x, c_y), pt1) * (1 if idx == 0 else -1)
        if abs(prev_angl[idx] - abs_angle) > 180:
            n[idx] += 1
        d_alpha = find_angle_between_3_pts(
            pt1, (c_x, c_y), pt2, False)
        add_dct["frame"] = pt_id+1
        add_dct[f"alpha{idx}"] = abs_angle + n[idx]*360
        add_dct[f"d_alpha{idx}"] = d_alpha
        prev_angl[idx] = abs_angle
    out_df = pd.concat([out_df, pd.DataFrame(add_dct, index=[0])])

print("writing data...")
with pd.ExcelWriter(FILE_OUT_PATH) as writer:
    out_df.to_excel(writer, sheet_name="data", index=False)
print("Done!")
