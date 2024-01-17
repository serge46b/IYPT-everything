from typing import Annotated, Literal
from dataclasses import dataclass
from numpy import typing as npt
import pandas as pd
import numpy as np
import cv2

ROOT = "./2023/Walker/"
IN_FILE_PATH = ROOT + \
    "ArUco tag exp/results/results processed/d4 a1 test rotated processed.xlsx"
MARKER_PAIR_IDS = (8, 14)


@dataclass
class np3Dvec:
    vec: Annotated[npt.NDArray[np.float32], Literal[3]]

    def __init__(self, array):
        self.vec = np.array(array)

    @property
    def length(self):
        return np.sqrt(self.vec[0]**2 + self.vec[1]**2 + self.vec[2]**2)

    @property
    def x(self):
        return self.vec[0]

    @property
    def y(self):
        return self.vec[1]

    @property
    def z(self):
        return self.vec[2]


@dataclass
class np3Drotation(np3Dvec):
    def __init__(self, array):
        super().__init__(array)

    @property
    def mtx(self):
        return cv2.Rodrigues(self.vec)[0]


@dataclass
class Marker:
    id: int
    crd: np3Dvec
    rot: np3Drotation


def find_angle(world_vector1: np3Dvec, marker1: Marker, world_vector2: np3Dvec, marker2: Marker) -> np.float32:
    rotated_vec1 = world_vector1.vec.dot(marker1.rot.mtx)
    rotated_vec2 = world_vector2.vec.dot(marker2.rot.mtx)
    rads = np.arccos(np.dot(rotated_vec1, rotated_vec2) /
                     (np.linalg.norm(rotated_vec1)*np.linalg.norm(rotated_vec2)))
    # if np.isnan(rads):
    #     rads = 0
    angle = np.degrees(rads)
    # nvec = [pt1[1] - pt2[1], pt2[0] - pt1[0]]
    # dp = np.dot(cvec, nvec)
    # nvec_mag = np.linalg.norm(nvec)
    # def_rads = np.arccos(dp/(cvec_mag*nvec_mag))
    # if np.isnan(def_rads):
    #     def_rads = 0
    # angle = 0
    # if angle0to360:
    #     if def_rads > np.pi/2:
    #         rads = 2*np.pi-rads
    #     angle = 360 - np.degrees(rads)
    # else:
    #     angle = np.degrees(rads)*(-1 if def_rads > np.pi/2 else 1)
    return angle


in_df = pd.read_excel(IN_FILE_PATH)
for index in in_df.index:
    Y_vec1 = np3Dvec([0, 0, 1])
    m1_id = MARKER_PAIR_IDS[0]
    m1 = Marker(m1_id, np3Dvec([in_df[f"tx{m1_id}"][index], in_df[f"ty{m1_id}"][index], in_df[f"tz{m1_id}"][index]]), np3Drotation([
                in_df[f"rx{m1_id}"][index], in_df[f"ry{m1_id}"][index], in_df[f"rz{m1_id}"][index]]))
    m2_id = MARKER_PAIR_IDS[1]
    Y_vec2 = np3Dvec([1, 0, 0])
    m2 = Marker(m2_id, np3Dvec([in_df[f"tx{m2_id}"][index], in_df[f"ty{m2_id}"][index], in_df[f"tz{m2_id}"][index]]), np3Drotation([
                in_df[f"rx{m2_id}"][index], in_df[f"ry{m2_id}"][index], in_df[f"rz{m2_id}"][index]]))
    angle = find_angle(Y_vec1, m1, Y_vec2, m2)
    if not np.isnan(angle):
        print(angle)
    else:
        print()
