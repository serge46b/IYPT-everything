import os
import cv2
import json
import numpy as np
import pandas as pd

CONFIG_FILE_PATH = "./2023/Walker/ArUco tracing config.json"
# CONFIG_FILE_PATH = "./2023/Magnetic gearbox/exp configs/s46b ArUco general tracing cfg.json"


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

print("loading data...")
in_df = pd.read_csv(FILE_IN_PATH, sep=", ", engine='python')
clmns = ["frame"]
for i in loaded_cfg["tracing_marker_ids"]:
    clmns.append(f"tx{i}")
    clmns.append(f"ty{i}")
    clmns.append(f"tz{i}")
    clmns.append(f"rx{i}")
    clmns.append(f"ry{i}")
    clmns.append(f"rz{i}")
out_df = pd.DataFrame(columns=clmns)

print("processing...")
prev_f_idx = in_df["frame"][0]
ids_found = False
add_dct = {"frame": prev_f_idx}
processing_queue = []
g_marker_idx = -1
for index in in_df.index:
    frame_idx = in_df["frame"][index]
    if ids_found or frame_idx != prev_f_idx:
        if USING_G_MARKER and g_marker_idx != -1:
            # dbg_img = np.zeros((640, 640, 3), dtype="uint8")
            g_tvec = np.array(
                [in_df["tx"][g_marker_idx], in_df["ty"][g_marker_idx], in_df["tz"][g_marker_idx]])
            g_rvec = np.array(
                [in_df["rx"][g_marker_idx], in_df["ry"][g_marker_idx], in_df["rz"][g_marker_idx]])
            g_rmat = cv2.Rodrigues(g_rvec)[0]
            for prc_idx in processing_queue:
                m_id = in_df["id"][prc_idx]
                m_tvec = np.array(
                    [in_df["tx"][prc_idx], in_df["ty"][prc_idx], in_df["tz"][prc_idx]])
                m_rvec = np.array(
                    [in_df["rx"][prc_idx], in_df["ry"][prc_idx], in_df["rz"][prc_idx]])
                m_rmat = cv2.Rodrigues(m_rvec)[0]
                prc_tvec = (m_tvec-g_tvec).dot(g_rmat)
                prc_rmat = g_rmat.dot(np.linalg.inv(m_rmat))
                # rotated_vec = np.array([0, 20, 0]).dot(prc_rmat)
                # cv2.line(dbg_img, (320, 320), (int(
                #     320 + rotated_vec[1]), int(320 - rotated_vec[2])), (255, 255, int(64*(m_id-1))), 1)
                # cv2.circle(dbg_img, (int(320 + prc_tvec[0]*320), int(
                #     320 - prc_tvec[1]*320)), 2, (255, int(64*(m_id-1)), int(255 + prc_tvec[2]*255)), -1)
                add_dct[f"tx{m_id}"] = prc_tvec[0]
                add_dct[f"ty{m_id}"] = prc_tvec[1]
                add_dct[f"tz{m_id}"] = prc_tvec[2]
                prc_rvec = cv2.Rodrigues(prc_rmat)[0]
                add_dct[f"rx{m_id}"] = prc_rvec[0, 0]
                add_dct[f"ry{m_id}"] = prc_rvec[1, 0]
                add_dct[f"rz{m_id}"] = prc_rvec[2, 0]

            # cv2.imshow("debug", dbg_img)
            # cv2.waitKey(25)
        out_df = pd.concat([out_df, pd.DataFrame(add_dct, index=[0])])
        add_dct = {"frame": frame_idx}
        processing_queue = []
        g_marker_idx = -1
    m_id = in_df["id"][index]
    if not USING_G_MARKER:
        add_dct[f"tx{m_id}"] = in_df["tx"][index]
        add_dct[f"ty{m_id}"] = in_df["ty"][index]
        add_dct[f"tz{m_id}"] = in_df["tz"][index]
        add_dct[f"rx{m_id}"] = in_df["rx"][index]
        add_dct[f"ry{m_id}"] = in_df["ry"][index]
        add_dct[f"rz{m_id}"] = in_df["rz"][index]
        prev_f_idx = frame_idx
        continue
    if m_id != GROUND_MARKER_ID:
        processing_queue.append(index)
    else:
        g_marker_idx = index
    prev_f_idx = frame_idx

print("writing data...")
with pd.ExcelWriter(FILE_OUT_PATH) as writer:
    out_df.to_excel(writer, sheet_name="data", index=False)
print("Done!")
# cv2.destroyAllWindows()
