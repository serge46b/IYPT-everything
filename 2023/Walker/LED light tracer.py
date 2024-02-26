import os
import cv2
import json


CONFIG_FILE_PATH = "./2023/Walker/LED exp/led tracing config.json"

if not CONFIG_FILE_PATH:
    CONFIG_FILE_PATH = input("Pass config file path:")
if not os.path.exists(CONFIG_FILE_PATH):
    print("cannot load config file. Exiting...")
    os._exit(0)

with open(CONFIG_FILE_PATH) as cfg_f:
    loaded_cfg = json.load(cfg_f)
