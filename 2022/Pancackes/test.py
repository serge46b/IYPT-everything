import numpy as np
# import math
import cv2
#
#
# class clockwise_angle_and_distance():
#     '''
#     A class to tell if point is clockwise from origin or not.
#     This helps if one wants to use sorted() on a list of points.
#
#     Parameters
#     ----------
#     point : ndarray or list, like [x, y]. The point "to where" we g0
#     self.origin : ndarray or list, like [x, y]. The center around which we go
#     refvec : ndarray or list, like [x, y]. The direction of reference
#
#     use:
#         instantiate with an origin, then call the instance during sort
#     reference:
#     https://stackoverflow.com/questions/41855695/sorting-list-of-two-dimensional-coordinates-by-clockwise-angle-using-python
#
#     Returns
#     -------
#     angle
#
#     distance
#
#
#     '''
#
#     def __init__(self, origin):
#         self.origin = origin
#
#     def __call__(self, point, refvec=[0, 1]):
#         if self.origin is None:
#             raise NameError("clockwise sorting needs an origin. Please set origin.")
#         # Vector between point and the origin: v = p - o
#         vector = [point[0] - self.origin[0], point[1] - self.origin[1]]
#         # Length of vector: ||v||
#         lenvector = np.linalg.norm(vector[0] - vector[1])
#         # If length is zero there is no angle
#         if lenvector == 0:
#             return -math.pi, 0
#         # Normalize vector: v/||v||
#         normalized = [vector[0] / lenvector, vector[1] / lenvector]
#         dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
#         diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
#         angle = math.atan2(diffprod, dotprod)
#         # Negative angles represent counter-clockwise angles so we need to
#         # subtract them from 2*pi (360 degrees)
#         if angle < 0:
#             return 2 * math.pi + angle, lenvector
#         # I return first the angle because that's the primary sorting criterium
#         # but if two vectors have the same angle then the shorter distance
#         # should come first.
#         return angle, lenvector
#
#
img = cv2.imread("./test.jpg")
# cv2.imshow("orig", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("thresh", th)
contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnt_img = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 1)
cv2.imshow("cnts", cnt_img)
cnt_mask = cv2.drawContours(np.zeros((img.shape[0], img.shape[1])), contours, -1, 1, -1)
cv2.imshow("cnt mask", cnt_mask)
# morph_img = cv2.morphologyEx(cnt_mask, cv2.MORPH_CLOSE, np.ones((20, 20), np.uint8))
# cv2.imshow("morph img", morph_img)
# # all_cnt = np.vstack(contours)
# # print(len(all_cnt), len(contours))
# # hull = cv2.convexHull(ctr)
# # hull_list = []
# # for i in range(len(contours)):
# #     hull = cv2.convexHull(contours[i])
# #     hull_list.append(hull)
# # hull_img = cv2.drawContours(cnt_img.copy(), hull, -1, (0, 255, 0), 1)
# # cv2.imshow("hull", hull_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np
#
#
# def find_if_close(cnt1, cnt2):
#     row1, row2 = cnt1.shape[0], cnt2.shape[0]
#     for i in range(row1):
#         for j in range(row2):
#             dist = np.linalg.norm(cnt1[i]-cnt2[j])
#             if abs(dist) < 50:
#                 return True
#             elif i == row1-1 and j == row2-1:
#                 return False
#
#
# img = cv2.imread('test.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, 0)
# contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
#
# LENGTH = len(contours)
# status = np.zeros((LENGTH, 1))
# print("distance calculation")
# for i, cnt1 in enumerate(contours):
#     x = i
#     if i != LENGTH-1:
#         for j, cnt2 in enumerate(contours[i+1:]):
#             x = x+1
#             dist = find_if_close(cnt1, cnt2)
#             if dist:
#                 val = min(status[i], status[x])
#                 status[x] = status[i] = val
#             else:
#                 if status[x] == status[i]:
#                     status[x] = i+1
#
# unified = []
# maximum = int(max(status))+1
# print("contours calculation")
# for i in range(maximum):
#     pos = np.where(status == i)[0]
#     if pos.size != 0:
#         cont = np.vstack(contours[i] for i in pos)
#         hull = cv2.convexHull(cont)
#         unified.append(hull)
#
# cv2.drawContours(img, unified, -1, (0, 255, 0), 2)
# cv2.drawContours(thresh, unified, -1, 255, -1)
#
# cv2.imshow("img", img)
