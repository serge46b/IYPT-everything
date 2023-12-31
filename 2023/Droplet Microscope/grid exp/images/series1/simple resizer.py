import cv2
from glob import glob

ROOT = "./2023/Droplet Microscope/grid exp/images/series1/"


for path in glob(ROOT+"*jpg"):
    f_name = path[path.rfind("\\")+1:]
    img = cv2.imread(path)
    crpd_img = img[1000:2100, :, :]
    cv2.imwrite(ROOT+"crpd/"+f_name, crpd_img)
