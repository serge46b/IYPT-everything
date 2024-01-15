from dataclasses import dataclass
import pandas as pd
import numpy as np
import cv2


ROOT = "./2023/Droplet Microscope/"
OUT_SV_PATH = ROOT + "modeling results/test result.xlsx"


@dataclass
class Function:
    _fn: callable
    params: dict

    def __init__(self, function: callable, params: dict):
        self._fn = function
        self.params = params

    @property
    def fn(self):
        return self._fn(self.params)

    def fn_value(self, x: float):
        return self._fn(self.params)(x)


@dataclass
class LineFunction(Function):
    def __init__(self, k, m):
        self.params = {"k": k, "m": m}
        self._fn = lambda dct: (lambda x: dct["k"]*x + dct["m"])

    @property
    def k(self):
        return self.params["k"]

    @k.setter
    def k(self, value):
        self.params["k"] = value

    @property
    def m(self):
        return self.params["m"]

    @m.setter
    def m(self, value):
        self.params["m"] = value


@dataclass
class ParabolaFunction(Function):
    def __init__(self, a, b, c):
        self.params = {"a": a, "b": b, "c": c}
        self._fn = lambda dct: (
            lambda x: dct["a"]*x**2 + dct["b"]*x + dct["c"])
        self._derivative = lambda dct: (lambda x: 2*dct["a"]*x + dct["b"])

    @property
    def dv(self):
        return self._derivative(self.params)

    def dv_value(self, x: float):
        return self._derivative(self.params)(x)

    @property
    def a(self):
        return self.params["a"]

    @a.setter
    def a(self, value):
        self.params["a"] = value

    @property
    def b(self):
        return self.params["b"]

    @b.setter
    def b(self, value):
        self.params["b"] = value

    @property
    def c(self):
        return self.params["c"]

    @c.setter
    def c(self, value):
        self.params["c"] = value

    @property
    def x1(self):
        return (-self.b + np.sqrt(self.b**2 - 4 * self.a*self.c))/(2*self.a)

    @property
    def x2(self):
        return (-self.b - np.sqrt(self.b**2 - 4 * self.a*self.c))/(2*self.a)


# DROPLET_FUNCTION = ParabolaFunction(-1, 2, 3)
DROPLET_FUNCTION = ParabolaFunction(-0.296, 1.037, -0.195)

D = (DROPLET_FUNCTION.x1, DROPLET_FUNCTION.x2)
print(D)
# D = (-1, 3)
H = (0, 2)
SHIFT = 1
N = 1.33

H_LENS = 1.4
H_VIEWER = 1.6

IMG_SHAPE = (640, int(640*(D[1]-D[0]+2*SHIFT)/(H[1]-H[0]+2*SHIFT)), 3)

X_RW = D[1]+SHIFT - D[0]+SHIFT
X_K = IMG_SHAPE[1]/X_RW
Y_RH = H[1]+SHIFT - H[0]+SHIFT
Y_K = -IMG_SHAPE[0]/Y_RH


@dataclass
class Point:
    _r_x: float
    _r_y: float
    _img_x: int
    _img_y: int

    def __init__(self, r_x: float | None = None, r_y: float | None = None, img_x: int | None = None, img_y: int | None = None):
        self._img_x = self._rx2ix(r_x) if r_x is not None else img_x
        self._img_y = self._ry2iy(r_y) if r_y is not None else img_y
        self._r_x = self._ix2rx(img_x) if img_x is not None else r_x
        self._r_y = self._iy2ry(img_y) if img_y is not None else r_y
        assert self._img_x is not None
        assert self._img_y is not None
        assert self._r_x is not None
        assert self._r_y is not None
        # print(self._r_x, self._r_y, self._img_x, self._img_y)

    def _rx2ix(_, rx):
        return int((rx - D[0] + SHIFT)*X_K)

    def _ry2iy(_, ry):
        return int((ry - H[0] + SHIFT)*Y_K) + IMG_SHAPE[0]

    def _ix2rx(_, ix):
        return ix/X_K + D[0] - SHIFT

    def _iy2ry(_, iy):
        return (iy - IMG_SHAPE[0])/Y_K + H[0] - SHIFT

    @property
    def rx(self):
        return self._r_x

    @rx.setter
    def rx(self, value):
        self._r_x = value
        self._img_x = self._rx2ix(value)

    @property
    def ry(self):
        return self._r_y

    @ry.setter
    def ry(self, value):
        self._r_y = value
        self._img_y = self._ry2iy(value)

    @property
    def ix(self):
        return self._img_x

    @ix.setter
    def ix(self, value):
        self._img_x = value
        self._r_x = self._ix2rx(value)

    @property
    def iy(self):
        return self._img_y

    @iy.setter
    def iy(self, value):
        self._img_y = value
        self._r_y = self._iy2ry(value)

    @property
    def r_crd(self):
        return (self.rx, self.ry)

    @r_crd.setter
    def r_crd(self, value: tuple[float, float]):
        self.rx = value[0]
        self.ry = value[1]

    @property
    def i_crd(self):
        return (self.ix, self.iy)

    @i_crd.setter
    def i_crd(self, value: tuple[float, float]):
        self.ix = value[0]
        self.iy = value[1]


@dataclass
class Object:
    crd: Point
    width: float
    color: tuple[int, int, int]

    @property
    def crd_end(self):
        return Point(r_x=self.crd.rx+self.width, r_y=self.crd.ry)


VIEWER_LINE = LineFunction(0, H_VIEWER)
DESK_LINE = LineFunction(0, H[0])
OBJECT_SHIFT_STEP = 0.03  # in digits after comma
CALCULATION_PRECISION = 5  # in digits after comma
OBJECT_WIDTH = 0.03
OBJECT = Object(Point(r_x=D[0], r_y=H[0]), OBJECT_WIDTH, (0, 0, 255))
LENSE_CENTER = Point(r_x=(D[1]-D[0]+2*SHIFT)/2+D[0]-SHIFT, r_y=H_LENS)


COLOR_DROPLET = (255, 125, 0)
COLOR_DESK = (255, 255, 255)
COLOR_LENS = (255, 255, 255)
COLOR_VIEWER = (255, 255, 255)


def draw_function(img: np.ndarray, function: Function, color: tuple[int, int, int], d: tuple[float, float] = (D[0]-SHIFT, D[1]+SHIFT)) -> None:
    for img_x in range(IMG_SHAPE[1]):
        drw_pt = Point(img_x=img_x, r_y=0)
        if drw_pt.rx < d[0]:
            continue
        if drw_pt.rx > d[1]:
            break
        drw_pt.ry = function.fn_value(drw_pt.rx)
        # print(drw_pt.ry)
        cv2.circle(img, drw_pt.i_crd, 0, color, -1)


def draw_lens(img):
    x1, y1 = Point(r_x=D[0]-SHIFT, r_y=H_LENS).i_crd
    x2, y2 = Point(r_x=D[1]+SHIFT, r_y=H_LENS).i_crd
    cv2.line(img, (x1, y1), (x2, y2), COLOR_LENS, 1)
    cv2.line(img, (x1, y1), (x1 + 20, y1 + 10), COLOR_LENS, 1)
    cv2.line(img, (x1, y1), (x1 + 20, y1 - 10), COLOR_LENS, 1)
    cv2.line(img, (x2, y2), (x2 - 20, y1 + 10), COLOR_LENS, 1)
    cv2.line(img, (x2, y2), (x2 - 20, y1 - 10), COLOR_LENS, 1)


def draw_object(img, object: Object) -> None:
    cv2.line(img, object.crd.i_crd, object.crd_end.i_crd, object.color)


def generate_line_function_by_pts(pt1: Point, pt2: Point) -> LineFunction:
    k = (pt2.ry-pt1.ry)/(pt2.rx-pt1.rx)
    m = pt1.ry - k*pt1.rx
    return LineFunction(k, m)


def find_intersection_of_2_lines(line1: LineFunction, line2: LineFunction) -> Point:
    x = (line2.m - line1.m)/(line1.k - line2.k)
    y = line1.k*x + line1.m
    return Point(r_x=x, r_y=y)


def find_intersection_of_parabola_line(parabola: ParabolaFunction, line: LineFunction) -> Point:
    a, b, c = parabola.params.values()
    k, m = line.params.values()
    x1 = (k - b + np.sqrt((b-k)**2 - 4*a*(c-m)))/(2*a)
    y1 = k*x1 + m
    x2 = (k - b - np.sqrt((b-k)**2 - 4*a*(c-m)))/(2*a)
    y2 = k*x2 + m
    return Point(r_x=x1 if y1 > y2 else x2, r_y=y1 if y1 > y2 else y2)


def find_obj_projection(obj: Object, img=None) -> Object:
    obj_pt1_lense_line = generate_line_function_by_pts(obj.crd, LENSE_CENTER)
    pt1_prj = find_intersection_of_2_lines(
        obj_pt1_lense_line, VIEWER_LINE)
    if img is not None:
        draw_function(img, obj_pt1_lense_line, (255, 0, 128),
                      (obj.crd.rx, pt1_prj.rx))
    obj_pt2_lense_line = generate_line_function_by_pts(
        obj.crd_end, LENSE_CENTER)
    pt2_prj = find_intersection_of_2_lines(
        obj_pt2_lense_line, VIEWER_LINE)
    if img is not None:
        draw_function(img, obj_pt2_lense_line, (255, 0, 128),
                      (obj.crd_end.rx, pt2_prj.rx))
    return Object(pt2_prj, np.sqrt((pt2_prj.rx-pt1_prj.rx)**2 + (pt2_prj.ry-pt1_prj.ry)**2), (0, 128, 128))


def refract_ray(droplet: ParabolaFunction, line: LineFunction, img=None):
    intersection_pt = find_intersection_of_parabola_line(
        droplet, line)
    if img is not None:
        cv2.line(img, find_intersection_of_2_lines(
            line, VIEWER_LINE).i_crd, intersection_pt.i_crd, (0, 255, 0))
        cv2.circle(img, intersection_pt.i_crd, 2, (0, 0, 255), -1)
    dpt_f_dv_v = droplet.dv_value(intersection_pt.rx)
    tangent = LineFunction(dpt_f_dv_v, droplet.fn_value(
        intersection_pt.rx) - dpt_f_dv_v*intersection_pt.rx)
    if img is not None:
        draw_function(img, tangent, (0, 128, 255))
    normal = LineFunction(-1/tangent.k, intersection_pt.rx /
                          tangent.k + intersection_pt.ry)
    if img is not None:
        draw_function(img, normal, (255, 255, 0))
    angle = np.arctan((line.k-normal.k)/(1+line.k*normal.k))
    snelius_angle = np.arcsin(np.sin(angle)/N)
    new_k = (normal.k+np.tan(snelius_angle))/(1-normal.k*np.tan(snelius_angle))
    refracted_ray = LineFunction(
        new_k, intersection_pt.ry - new_k*intersection_pt.rx)
    if img is not None:
        cv2.line(img, intersection_pt.i_crd, find_intersection_of_2_lines(
            refracted_ray, LineFunction(0, H[0])).i_crd, (0, 255, 0))
    return refracted_ray


def find_parabola_tangent_by_point(parabola: ParabolaFunction, point: Point) -> [LineFunction, LineFunction]:
    a, b, c = parabola.params.values()
    kx, ky = point.r_crd
    x1 = (a*kx + np.sqrt(a**2*kx**2 - a*(ky - c - b*kx)))/a
    x2 = (a*kx - np.sqrt(a**2*kx**2 - a*(ky - c - b*kx)))/a
    k1 = parabola.dv_value(x1)
    m1 = parabola.fn_value(x1) - k1*x1
    k2 = parabola.dv_value(x2)
    m2 = parabola.fn_value(x2) - k2*x2
    return LineFunction(k1, m1), LineFunction(k2, m2)


# Visual preparation
img = np.zeros(IMG_SHAPE)
draw_function(img, DROPLET_FUNCTION, COLOR_DROPLET, D)
draw_function(img, VIEWER_LINE, COLOR_VIEWER)
draw_lens(img)
draw_function(img, DESK_LINE, COLOR_DESK)
# -----------Parabola variant---------
# Get edge points
tg1, tg2 = find_parabola_tangent_by_point(
    DROPLET_FUNCTION, LENSE_CENTER)
# draw_function(img, tg1, (255, 255, 255))
# draw_function(img, tg2, (255, 255, 255))
edge1_point = find_intersection_of_2_lines(tg1, VIEWER_LINE)
edge2_point = find_intersection_of_2_lines(tg2, VIEWER_LINE)
# Get desk edge points
ray1 = generate_line_function_by_pts(Point(
    r_x=edge1_point.rx-1/10**CALCULATION_PRECISION, r_y=edge1_point.ry), LENSE_CENTER)
refracted_ray1 = refract_ray(DROPLET_FUNCTION, ray1)
# draw_function(img, refracted_ray1, (255, 0, 255))
desk_edge1_point = find_intersection_of_2_lines(refracted_ray1, DESK_LINE)
ray2 = generate_line_function_by_pts(Point(
    r_x=edge2_point.rx+1/10**CALCULATION_PRECISION, r_y=edge2_point.ry), LENSE_CENTER)
refracted_ray2 = refract_ray(DROPLET_FUNCTION, ray2)
# draw_function(img, refracted_ray2, (255, 0, 255))
desk_edge2_point = find_intersection_of_2_lines(refracted_ray2, DESK_LINE)
print("processing...")
out_df = pd.DataFrame()
OBJECT.crd.rx = desk_edge1_point.rx
q = 0
while OBJECT.crd_end.rx < desk_edge2_point.rx:
    print(
        f"progress: {(OBJECT.crd_end.rx-D[0]) / (D[1]-D[0]):,.2f}%", end="\r")
    projected_obj_xs = [None, None]
    dbg_img = img.copy()
    draw_object(dbg_img, OBJECT)
    for crd_idx, searching_x in enumerate([OBJECT.crd.rx, OBJECT.crd_end.rx]):
        x = (edge1_point.rx + edge2_point.rx)/2 + 0.00001
        prev_x = edge1_point.rx
        while True:
            dbg_img2 = dbg_img.copy()
            ray = generate_line_function_by_pts(
                Point(r_x=x, r_y=H_VIEWER), LENSE_CENTER)
            dpt_intersect_point = find_intersection_of_parabola_line(
                DROPLET_FUNCTION, ray)
            refracted_ray = refract_ray(DROPLET_FUNCTION, ray)
            ray_intersection_point = find_intersection_of_2_lines(
                refracted_ray, DESK_LINE)
            diff = round(searching_x, CALCULATION_PRECISION) - \
                round(ray_intersection_point.rx, CALCULATION_PRECISION)
            if diff == 0:
                break
            elif diff > 0:
                x, prev_x = x - abs(prev_x-x)/2, x
            elif diff < 0:
                x, prev_x = x + abs(prev_x-x)/2, x
            draw_function(dbg_img2, ray, (0, 255, 255), (min(
                x, dpt_intersect_point.rx), max(x, dpt_intersect_point.rx)))
            draw_function(dbg_img2, refracted_ray, (0, 255, 255), (min(dpt_intersect_point.rx,
                                                                       ray_intersection_point.rx), max(dpt_intersect_point.rx, ray_intersection_point.rx)))
            cv2.imshow("debug", dbg_img2)
            q = cv2.waitKey(30)
            if q == 27:
                break
        if q == 27:
            break
        projected_obj_xs[crd_idx] = x
    if q == 27:
        print()
        print("Aborted")
        break
    prj_obj = find_obj_projection(OBJECT)
    prj_obj_no_dpt_size = abs(prj_obj.crd.rx-prj_obj.crd_end.rx)
    prj_obj_with_dpt_size = abs(projected_obj_xs[0] - projected_obj_xs[1])
    out_df = pd.concat([out_df, pd.DataFrame(
        {"x": OBJECT.crd.rx, "magnification": prj_obj_with_dpt_size/prj_obj_no_dpt_size}, index=[0])])
    OBJECT.crd.rx += OBJECT_SHIFT_STEP
print()
if q != 27:
    print("Writing...")
    with pd.ExcelWriter(OUT_SV_PATH) as writer:
        out_df.to_excel(writer, "data")
cv2.destroyAllWindows()
