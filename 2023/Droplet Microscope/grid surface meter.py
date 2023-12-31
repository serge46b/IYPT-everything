from glob import glob
import pandas as pd
import numpy as np
import sys
import cv2


ROOT = "./2023/Droplet Microscope/"
IMAGE_IN_PATH = ROOT + "grid exp/images/different cams/tim CoolCam/n dp.jpg"
TRACED_IMG_SV_PATH = ROOT + "grid exp/results/different cams/tim CoolCam/"
TRACED_DATA_SV_PATH = ROOT + "grid exp/results/none.xlsx"

DISPLAY_IMG_HEIGHT = 800
MIN_CNT_AREA = 20
THRESHOLD_V = 70

DECOLOR_OUT_RECTS = False
CROP_TO_DROPLET = True
ROI_MARGIN = 500

SAVE_IMG = True
SAVE_DATA = False
CREATE_DIAG_SHEET = False
SAVE_OUT_IDXS = False


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


def find_droplet_bounding_circle(frame, only_drop=False):
    points = [None] * (3 - only_drop)
    height = DISPLAY_IMG_HEIGHT if frame.shape[0] > DISPLAY_IMG_HEIGHT else frame.shape[0]

    def show_with_points(frame):
        show_frame = resize_with_aspect_ratio(frame.copy(), height=height)
        # show_frame = cv2.cvtColor(resize_with_aspect_ratio(
        #     show_frame, height=DISPLAY_IMG_HEIGHT), cv2.COLOR_GRAY2BGR)
        if points[0] is not None and points[1] is not None:
            cv2.circle(
                show_frame, points[0],
                int(np.sqrt((points[1][0] - points[0][0]) **
                    2 + (points[1][1] - points[0][1])**2)),
                (0, 255, 255), 1
            )
            if points[2] is not None and not only_drop:
                cv2.circle(show_frame, points[0],
                           int(np.sqrt((points[2][0] - points[0][0]) **
                                       2 + (points[2][1] - points[0][1])**2)),
                           (0, 125, 255), 1
                           )
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
            circle_param[0] = [points[0][0] * frame.shape[0] //
                               height, points[0][1] * frame.shape[0] // height]
            circle_param[1] = int(
                np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2) * frame.shape[0] // height)
            if not only_drop:
                circle_param[2] = int(
                    np.sqrt((points[2][0] - points[0][0])**2 + (points[2][1] - points[0][1])**2) * frame.shape[0] // height)
            return tuple(circle_param)
        elif key == ord("c"):
            points[point_edits] = None
            print(f"deleted point {point_edits + 1}")
        elif key == ord("1"):
            point_edits = 0
        elif key == ord("2"):
            point_edits = 1
        elif key == ord("3") and not only_drop:
            point_edits = 2


def get_diag_by_b_circle(frame, b_circle):
    point = [None, None]
    height = DISPLAY_IMG_HEIGHT if frame.shape[0] > DISPLAY_IMG_HEIGHT else frame.shape[0]
    bc_x = b_circle[0][0]*height//frame.shape[0]
    bc_y = b_circle[0][1]*height//frame.shape[0]
    bc_r = b_circle[1]*height//frame.shape[0]

    def show_with_points(frame):
        show_frame = resize_with_aspect_ratio(frame.copy(), height=height)
        # show_frame = cv2.cvtColor(resize_with_aspect_ratio(
        #     show_frame, height=DISPLAY_IMG_HEIGHT), cv2.COLOR_GRAY2BGR)
        cv2.circle(show_frame, (bc_x, bc_y), 1, (255, 0, 0), -1)
        cv2.circle(show_frame, (bc_x, bc_y), bc_r, (0, 255, 255), 1)
        if point[0] is not None and point[1] is not None:
            dx = (point[0] - bc_x) * 2
            dy = (point[1] - bc_y) * 2
            cv2.line(
                show_frame, (point[0] - dx, point[1] - dy), point, (0, 125, 255), 1)
            cv2.circle(show_frame, point, 1, (0, 0, 255), -1)
        cv2.imshow("frame", show_frame)

    def mouse_callback(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN or flag == 1:
            l = np.sqrt((bc_x - x)**2 + (bc_y - y)**2)
            point[0] = bc_x + \
                int((bc_x - x)*bc_r//l)
            point[1] = bc_y + \
                int((bc_y - y)*bc_r//l)
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
            line_param = [None, None]
            pt_x = point[0] * frame.shape[0]//height
            pt_y = point[1] * frame.shape[0]//height
            dx = (point[0] - bc_x) * 2 * frame.shape[0]//height
            dy = (point[1] - bc_y) * 2 * frame.shape[0]//height
            line_param[0] = (pt_x, pt_y)
            line_param[1] = ((pt_x - dx),
                             (pt_y - dy))
            return tuple(line_param)
        elif key == ord("c"):
            point = [None, None]


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


def split_contours_parameters(contours):
    bboxes = []
    centers = []
    areas = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_CNT_AREA:
            continue
        m = cv2.moments(c)
        if m['m00'] == 0:
            cx = 0
            cy = 0
        else:
            cx = m['m10']//m['m00']
            cy = m['m01']//m['m00']
        bboxes.append(cv2.boundingRect(c))
        centers.append((cx, cy))
        areas.append(area)
    return bboxes, centers, areas


def split_in_and_out_cells(bboxes, dpt_b_circle, frame_shape):
    # dbg_image = np.zeros_like(dbg_img)
    # cv2.circle(dbg_image, dpt_b_circle[0], dpt_b_circle[1], (0, 0, 255), 3)
    in_cells = []
    g_cells = []
    out_cells = []
    circle_x, circle_y = dpt_b_circle[0]
    r = dpt_b_circle[1]
    r2 = dpt_b_circle[2]
    for idx in range(len(bboxes)):
        # area = areas[idx]
        x, y, w, h = bboxes[idx]
        c_x, c_y = x+w//2, y+h//2
        # dist = np.sqrt((circle_x-x)**2+(circle_y-y)**2)
        dist = np.sqrt((circle_x-c_x)**2+(circle_y-c_y)**2)
        diag = np.sqrt(w**2 + h**2)
        # if dist < r:
        #     in_cells.append(idx)
        #     continue
        # if dist < r + np.sqrt(w**2 + h**2):
        #     dist_l = np.sqrt((circle_x-(x+w))**2+(circle_y-y)**2)
        #     dist_d = np.sqrt((circle_x-x)**2+(circle_y-(y+h))**2)
        #     if dist_l < r or dist_d < r:
        #         in_cells.append(idx)
        #         continue
        # if dist < r2:
        #     g_cells.append(idx)
        #     continue
        # if dist < r2 + np.sqrt(w**2 + h**2):
        #     dist_l = np.sqrt((circle_x-(x+w))**2+(circle_y-y)**2)
        #     dist_d = np.sqrt((circle_x-x)**2+(circle_y-(y+h))**2)
        #     if dist_l < r2 or dist_d < r2:
        #         g_cells.append(idx)
        #         continue
        if dist < r - diag/2:
            in_cells.append(idx)
            continue
        if dist < r2 - diag/2:
            g_cells.append(idx)
            continue
        n_dist = np.sqrt(w**2 + h**2)
        if c_x - n_dist < 0 or c_x + n_dist > frame_shape[1] or c_y - n_dist < 0 or c_y + n_dist > frame_shape[0]:
            g_cells.append(idx)
            continue
        out_cells.append(idx)
    #     cv2.rectangle(dbg_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # cv2.imshow("DEBUG", resize_with_aspect_ratio(
    #     dbg_image, height=DISPLAY_IMG_HEIGHT))
    # cv2.imshow("...", dbg_image)
    # cv2.waitKey(0)
    return in_cells, g_cells, out_cells


def get_avg_aout_area(out_idxs, areas):
    avg_area = 0
    for idx in out_idxs:
        avg_area += areas[idx]
    return avg_area/len(out_idxs)


def get_deviations(avg_out_area, areas, in_idxs, calc_idxs):
    deviations = {}
    max_deviation = 0
    min_deviation = np.sqrt(areas[calc_idxs[0]]/avg_out_area)
    abs_min_deviation = min_deviation
    for idx in calc_idxs:
        area = areas[idx]
        deviation = np.sqrt(area/avg_out_area)
        if deviation < abs_min_deviation:
            abs_min_deviation = deviation
        deviations[idx] = deviation
    mr_deviation = 0
    for in_idx in in_idxs:
        deviation = deviations[in_idx]
        mr_deviation += deviation
        if deviation > max_deviation:
            max_deviation = deviation
        if deviation < min_deviation:
            min_deviation = deviation
    return deviations, abs_min_deviation, min_deviation, max_deviation, mr_deviation / len(in_idxs)


def get_norm_deviations(deviations, min_deviation, max_deviation):
    norm_deviations = {}
    for dev in deviations:
        nm_dev = (deviations[dev]-min_deviation)/(max_deviation-min_deviation)
        nm_dev = 1 if nm_dev > 1 else (0 if nm_dev < 0 else nm_dev)
        norm_deviations[dev] = nm_dev
    return norm_deviations


def blue_purple_red_color_fn(value, no_dev_val):
    r = int(255*(value*(1/no_dev_val) if value < no_dev_val else 1))
    b = int(255*(1 if value < no_dev_val else (1/(1-no_dev_val))*(1-value)))
    return (b, 0, r)


def LGBT_color_fn(value, no_dev_val):
    # b = int(255*(0 if value < no_dev_val else (1 if value > no_dev_val +
    #         (1-no_dev_val)/2 else 2*(value-no_dev_val)/(1-no_dev_val))))
    # g = int(255*(2*value/no_dev_val if value < no_dev_val/2 else (1 if value <
    #         no_dev_val + (1-no_dev_val)/2 else 2*(1-value)/(1-no_dev_val))))
    # r = int(255*(2*(no_dev_val-value)/no_dev_val if no_dev_val/2 < value < no_dev_val else ((2*value -
    #         no_dev_val-1)/(1-no_dev_val) if value > no_dev_val + (1-no_dev_val)/2 else value <= (no_dev_val/2))))
    b = int(255*(4*(value-no_dev_val)/(1-no_dev_val) if no_dev_val <
            value < no_dev_val+(1-no_dev_val)/4 else value > no_dev_val))
    g = int(255*(2*value/no_dev_val if value < no_dev_val/2 else (2*(no_dev_val+1-2*value)/(1-no_dev_val)
            if no_dev_val+(1-no_dev_val)/4 < value < no_dev_val+(1-no_dev_val)/2 else value <= no_dev_val+(1-no_dev_val)/4)))
    r = int(255*(2*(no_dev_val-value)/no_dev_val if no_dev_val/2 < value < no_dev_val else ((2*value -
            no_dev_val-1)/(1-no_dev_val) if value > no_dev_val+(1-no_dev_val)/2 else value < no_dev_val)))
    return (b, g, r)


def draw_deviation_map(img, bboxes, deviations, color_fn, no_dev_val, out_idxs, g_idxs, in_idxs):
    out_img = img.copy()
    for out_idx in out_idxs:
        dev = deviations[out_idx]
        x, y, w, h = bboxes[out_idx]
        color = color_fn(dev, no_dev_val) if not DECOLOR_OUT_RECTS else (
            255, 255, 255)
        cv2.rectangle(out_img, (x, y), (x+w, y+h),
                      color, -1)
    for g_idx in g_idxs:
        x, y, w, h = bboxes[g_idx]
        color = (255, 255, 255)
        cv2.rectangle(out_img, (x, y), (x+w, y+h),
                      color, -1)
    for in_idx in in_idxs:
        dev = deviations[in_idx]
        x, y, w, h = bboxes[in_idx]
        cv2.rectangle(out_img, (x, y), (x+w, y+h),
                      color_fn(dev, no_dev_val), -1)
    # for idx in range(len(deviations)):
    #     dev = deviations[idx]
    #     x, y, w, h = bboxes[idx]
    #     # print(dev)
    #     cv2.rectangle(out_img, (x, y), (x+w, y+h),
    #                   color_fn(dev, no_dev_val), -1)
    return out_img


def generate_color_grad(img_shape, color_fn, no_dev_val, mn_dev, mx_dev):
    cl_grd_image = np.zeros((img_shape[0], img_shape[1]//5, 3), dtype="uint8")
    grd_width = cl_grd_image.shape[1]//3
    grd_margin_right = grd_width
    grd_height = int(cl_grd_image.shape[0]*0.8)
    grd_margn_top = cl_grd_image.shape[0]//10
    for line_idx in range(grd_height):
        color = color_fn(1-line_idx/grd_height, no_dev_val)
        cv2.line(cl_grd_image, (grd_margin_right, grd_margn_top+line_idx),
                 (grd_margin_right+grd_width, grd_margn_top+line_idx), color, 1)
    cv2.putText(cl_grd_image, f'{mx_dev:.{2}f}', (grd_margin_right+grd_width+10, grd_margn_top),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cl_grd_image, f'{mn_dev:.{2}f}', (grd_margin_right+grd_width+10, grd_margn_top +
                grd_height), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    center_y = int(grd_height*(1-no_dev_val))
    print(center_y)
    cv2.line(cl_grd_image, (grd_margin_right, grd_margn_top+center_y),
             (grd_margin_right+grd_width, grd_margn_top+center_y), (255, 255, 255), 2)
    cv2.putText(cl_grd_image, "1", (grd_margin_right+grd_width+10, grd_margn_top +
                center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return cl_grd_image


# ----debug color fn section----
# debug_img = np.zeros((400, 400, 3)).astype("uint8")
# for i in range(100):
#     color = LGBT_color_fn(i/100, 0.5)
#     debug_img[:, :] = color
#     print(i, color)
#     cv2.imshow("DEBUG", debug_img)
#     q = cv2.waitKey(0)
#     if q == 27:
#         break
# cv2.destroyAllWindows()
# sys.exit(0)

paths = [IMAGE_IN_PATH]
if IMAGE_IN_PATH.endswith("/"):
    paths = glob(IMAGE_IN_PATH + "*.jpg")
if SAVE_DATA:
    excel_writer = pd.ExcelWriter(TRACED_DATA_SV_PATH)
for path in paths:
    sl_idx = path.rfind("\\") if path.rfind("\\") > 0 else path.rfind("/")
    f_name = path[sl_idx+1:]
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img_gray, THRESHOLD_V, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    opening_img = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    cv2.imshow("threshold", resize_with_aspect_ratio(
        opening_img, height=DISPLAY_IMG_HEIGHT))
    # droplet_b_circle = find_droplet_bounding_circle(img)
    droplet_b_circle = find_droplet_bounding_circle(
        cv2.cvtColor(opening_img, cv2.COLOR_GRAY2BGR))
    # cv2.imshow("...", th)
    # edges = cv2.Canny(img_gray, 100, 300)
    # cv2.imshow("edges", resize_with_aspect_ratio(edges, height=DISPLAY_IMG_HEIGHT))
    # cv2.imshow("...", edges)
    if CROP_TO_DROPLET:
        x, y = droplet_b_circle[0]
        x -= droplet_b_circle[2]+ROI_MARGIN
        y -= droplet_b_circle[2]+ROI_MARGIN
        sz = (droplet_b_circle[1]+ROI_MARGIN)*2
        opening_img = opening_img[y:y+sz, x:x+sz]
        img = img[y:y+sz, x:x+sz, :]
        droplet_b_circle[0][0] -= x
        droplet_b_circle[0][1] -= y
        # deviation_map_img = deviation_map_img[y:y+sz, x:x+sz, :]
    contours, hierarchy = cv2.findContours(
        opening_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    dbg_img = img.copy()
    cv2.drawContours(dbg_img, contours, -1, (0, 255, 0), 1)
    cv2.imshow("contours", resize_with_aspect_ratio(
        dbg_img, height=DISPLAY_IMG_HEIGHT))
    # cv2.imshow("...", dbg_img)
    bboxes, centers, areas = split_contours_parameters(contours)
    if not droplet_b_circle:
        break
    in_idxs, g_idxs, out_idxs = split_in_and_out_cells(
        bboxes, droplet_b_circle, opening_img.shape)
    avg_out_area = get_avg_aout_area(out_idxs, areas)
    deviations, a_mn_dev, mn_dev, mx_dev, mr_dev = get_deviations(
        avg_out_area, areas, in_idxs, in_idxs+out_idxs)
    print(mx_dev, mn_dev, mr_dev)
    norm_deviations = get_norm_deviations(deviations, a_mn_dev, mx_dev)
    no_deviation_val = (1-a_mn_dev)/(mx_dev-a_mn_dev)
    deviation_map_img = draw_deviation_map(
        np.zeros_like(img), bboxes, norm_deviations, LGBT_color_fn, no_deviation_val, out_idxs, g_idxs, in_idxs)

    cv2.imshow("deviation map", resize_with_aspect_ratio(
        deviation_map_img, height=DISPLAY_IMG_HEIGHT))
    # deviation_img_blured = cv2.GaussianBlur(deviation_map_img, (199, 199), 0)
    # cv2.imshow("deviation map blured", resize_with_aspect_ratio(
    #     deviation_img_blured, height=DISPLAY_IMG_HEIGHT))

    cl_grd_img = generate_color_grad(
        img.shape, LGBT_color_fn, no_deviation_val, a_mn_dev, mx_dev)
    cv2.imshow("cropped", resize_with_aspect_ratio(
        img, height=DISPLAY_IMG_HEIGHT))
    # cv2.imshow("cl_grad", cl_grd_img)
    # cv2.waitKey(0)
    stacked_image = cv2.hconcat([deviation_map_img, cl_grd_img])
    cv2.imshow("deviation with grade", resize_with_aspect_ratio(
        stacked_image, height=DISPLAY_IMG_HEIGHT))

    q = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if q == 27:
        break
    if SAVE_IMG and q != ord("c"):
        # print(TRACED_IMG_SV_PATH + "traced " + f_name)
        cv2.imwrite(TRACED_IMG_SV_PATH + "traced " + f_name, stacked_image)
        print("image saved")
    if SAVE_DATA and q != ord("c"):
        w = h = droplet_b_circle[1]
        x = droplet_b_circle[0][0] - w//2
        y = droplet_b_circle[0][1] - h//2

        print("Preprocessing data. Preprocessing #1: 0%", end="\r")
        data = pd.DataFrame()
        for i_idx in range(len(in_idxs)):
            prgr = i_idx//len(in_idxs)
            if prgr % 5 == 0:
                print(
                    f"Preprocessing data. Preprocessing in idxs: {prgr}%", end="\r")
            idx = in_idxs[i_idx]
            data = pd.concat([data, pd.DataFrame({
                "idx": idx,
                "area": areas[idx],
                "centerX": (centers[idx][0]-x)/w,
                "centerY": (centers[idx][1]-y)/h,
                "deviation": deviations[idx],
                "is_in": True
            }, index=[0])])
        if SAVE_OUT_IDXS:
            for o_idx in range(len(out_idxs)):
                prgr = o_idx/len(out_idxs)*100
                if o_idx % 10 == 0:
                    print(
                        f"Preprocessing data. Preprocessing out_idxs: {prgr:3.2f}%  ", end="\r")
                idx = out_idxs[o_idx]
                data = pd.concat([data, pd.DataFrame({
                    "idx": idx,
                    "area": areas[idx],
                    "centerX": (centers[idx][0]-x)/w,
                    "centerY": (centers[idx][1]-y)/h,
                    "deviation": deviations[idx],
                    "is_in": False
                }, index=[0])])
        if CREATE_DIAG_SHEET:
            diag_df = pd.DataFrame()
            ln = get_diag_by_b_circle(
                deviation_map_img, droplet_b_circle)
            ln_l = np.sqrt((ln[0][0] - ln[1][0])**2 + (ln[0][1] - ln[1][1])**2)
            new_in = []
            new_g = g_idxs.copy()
            for i_idx in range(len(in_idxs)):
                prgr = i_idx//len(in_idxs)
                if prgr % 5 == 0:
                    print(
                        f"Preprocessing data. Preprocessing in idxs: {prgr}%", end="\r")
                idx = in_idxs[i_idx]
                ppts = get_pp_sec_point(ln[0], ln[1], centers[idx])
                dist = np.sqrt((ppts[0] - centers[idx][0])
                               ** 2 + (ppts[1] - centers[idx][1])**2)
                if dist > np.sqrt((bboxes[idx][0] - centers[idx][0])**2 + (bboxes[idx][1] - centers[idx][1])**2)*1.5:
                    new_g.append(idx)
                    continue
                # if dist > bboxes[idx][2]/2:
                #     new_g.append(idx)
                #     print(dist, bboxes[idx][2]/2)
                #     continue
                new_in.append(idx)
                r = np.sqrt((ln[0][0] - ppts[0])**2 +
                            (ln[0][1] - ppts[1])**2)/ln_l
                diag_df = pd.concat([diag_df, pd.DataFrame({
                    "idx": idx,
                    "dst": r,
                    "deviation": deviations[idx]
                }, index=[0])])
            dbg_map_img = draw_deviation_map(
                np.zeros_like(img), bboxes, norm_deviations, LGBT_color_fn, no_deviation_val, out_idxs, new_g, new_in)
            cv2.imshow("debug map", resize_with_aspect_ratio(
                dbg_map_img, height=DISPLAY_IMG_HEIGHT))
            q = cv2.waitKey(0)
            if q == 13:
                print("saving diag...                                                 ")
                diag_df.to_excel(
                    excel_writer, sheet_name=f"diag of {f_name}", index=False)
        print("Saving data...                                                 ")
        data.to_excel(
            excel_writer, sheet_name=f"data of {f_name}", index=False)

        # angl_mx = 0
        # scs = 0
        # for angle in range(3600):
        #     angle_rad = angle * 2 * np.pi / 3600

        print("data saved")
if SAVE_DATA:
    excel_writer.close()
cv2.destroyAllWindows()
