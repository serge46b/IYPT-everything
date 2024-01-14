from dataclasses import dataclass
import numpy as np
import cv2


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


DROPLET_FUNCTION = ParabolaFunction(-1, 2, 3)
D = (DROPLET_FUNCTION.x1, DROPLET_FUNCTION.x2)
# print(D)
# D = (-1, 3)
H = (0, 10)
SHIFT = 1
N = 1.33

H_LENS = 8
H_VIEWER = 10

IMG_SHAPE = (640, 320, 3)

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


# DROPLET_FUNCTION = Function(lambda dct: (
    # lambda x: dct["a"]*x**2 + dct["b"]*x + dct["c"]), {"a": -1, "b": 2, "c": 3})
# DROPLET_FUNCTION = ParabolaFunction(-1, 2, 3)
VIEWER_LINE = LineFunction(0, H_VIEWER)
OBJECT = Object(Point(r_x=D[0], r_y=H[0]), 1, (0, 0, 255))
LENSE_CENTER = Point(r_x=(D[1]-D[0]+2*SHIFT)/2+D[0]-SHIFT, r_y=H_LENS)
# def DROPLET_FUNCTION(x):
#     A = -1
#     B = 2
#     C = 3
#     return A*x**2 + B*x + C


# def VIEWER_FUNCTION(x):
#     return H_VIEWER


COLOR_DROPLET = (255, 125, 0)
COLOR_DESK = (255, 255, 255)
COLOR_LENS = (255, 255, 255)
COLOR_VIEWER = (255, 255, 255)


def transfer_crd_real2img(crd: tuple[float, float]) -> tuple[int, int]:
    x = int((crd[0] - D[0] + SHIFT)/X_K)
    y = IMG_SHAPE[0] - int((crd[1] - H[0] + SHIFT)/(-Y_K)) - 1
    return (x, y)


def transfer_crd_img2real(crd: tuple[int, int]) -> tuple[float, float]:
    x = crd[0]*X_K + D[0] - SHIFT
    # y = crd[1]*X_K + H[0] - SHIFT
    return (x, 0)


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


# draw_object(img, OBJECT)
# main_obj_projection = find_obj_projection(OBJECT, img)
# draw_object(img, main_obj_projection)
for test_x in range(5001, 15000, 6):
    img = np.zeros(IMG_SHAPE)
    draw_function(img, DROPLET_FUNCTION, COLOR_DROPLET, D)
    draw_function(img, VIEWER_LINE, COLOR_VIEWER)
    draw_lens(img)
    cv2.line(img, Point(r_x=D[0]-SHIFT, r_y=H[0]).i_crd,
             Point(r_x=D[1]+SHIFT, r_y=H[0]).i_crd, COLOR_DESK, 1)
    ray = generate_line_function_by_pts(
        Point(test_x/10000, H_VIEWER), LENSE_CENTER)
    # draw_function(img, ray, (0, 255, 255))
    refracted_ray = refract_ray(DROPLET_FUNCTION, ray, img)
    # draw_function(img, refracted_ray, (255, 0, 255))
    cv2.imshow("function", img)
    q = cv2.waitKey(1)
    if q == 27:
        break
cv2.destroyAllWindows()
