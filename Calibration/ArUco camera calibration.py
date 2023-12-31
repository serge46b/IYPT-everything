import cv2
from cv2 import aruco
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob


ROOT = "./Calibration/"
CALIB_FILE_PATH = ROOT + "/camera calibration samples/Nikon (stock lens).json"
CALIB_IMG_ARRAY_PATH = ROOT + "camera calibration samples/AAZinkevich camera/"

CALIB_ARUCO_DCT = aruco.DICT_4X4_1000
MARKER_SIZE = 0.048  # units - meters
MARKER_SEP = 0.006  # units - meters
BOARD_SIZE_X = 4
BOARD_SIZE_Y = 5
GNERATE_BOARD_MODE = False


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


# Set path to the images
# calib_imgs_path = root + "imgs/"

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary(CALIB_ARUCO_DCT)

# Provide length of the marker's side
# markerLength = 4  # Here, measurement unit is centimetre.

# Provide separation between markers
# markerSeparation = 0.5   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard(
    (BOARD_SIZE_X, BOARD_SIZE_Y), MARKER_SIZE, MARKER_SEP, aruco_dict)

if GNERATE_BOARD_MODE:
    img = board.generateImage((2480, 3508), marginSize=50)
    cv2.imwrite(ROOT + "cb.png", img)
    print(
        f"Board generated and placed in path: {ROOT + 'cb.png'}\nTo calibrate camera take pictures of the board at different angles\nand switch GENERATE_BOARD_MODE parameter to False.")
    import sys
    sys.exit(0)
print("Starting calibration, collecting images.")

arucoParams = aruco.DetectorParameters()

img_list = []
calib_fnms = glob(CALIB_IMG_ARRAY_PATH + "*.jpg")
print('Using ...', end='')
for idx, fn in enumerate(calib_fnms):
    print(idx, '', end='')
    img = Image.open(fn)
    img.load()
    img_format = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img_list.append(img_format)
    h, w, c = img_format.shape
print('Calibration images')

counter, corners_list, id_list = [], [], []
first = True
for im in tqdm(img_list):
    img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        img_gray, aruco_dict, parameters=arucoParams)
    # cv2.imshow('adsd',img_gray)
    # cv2.waitKey(0)
    if first == True:
        corners_list = corners
        id_list = ids
        first = False
    else:
        corners_list = np.vstack((corners_list, corners))
        id_list = np.vstack((id_list, ids))
    counter.append(len(ids))
print('Found {} unique markers'.format(np.unique(ids)))

counter = np.array(counter)
print("Calibrating camera .... Please wait...")
ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(
    corners_list, id_list, counter, board, img_gray.shape, None, None)

print("Camera matrix: \n", mtx,
      "\n distortion coefficients: \n", dist, "\nEverything is stord in path: '" + CALIB_FILE_PATH + "'")
data = {'camera_matrix': np.asarray(
    mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open(CALIB_FILE_PATH, "w") as f:
    json.dump(data, f)
# with open(CALIB_FILE_PATH, "w") as f:
#     yaml.dump(data, f)
