from typing import TypeVar
import cv2

import math

from typing import Any, Generator, Generic, Type, TypeVar, overload
import numpy as np
import cv2


from numpy.typing import NDArray


class HoughParams:
    minRadius: int
    maxRadius: int
    param1: int
    param2: int
    method: int
    dp: float

    def __init__(
        self,
        minRadius: int,
        maxRadius: int,
        param1: int,
        param2: int,
        method: int = cv2.HOUGH_GRADIENT,
        dp: float = 1.0,
    ):
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.param1 = param1
        self.param2 = param2
        self.method = method
        self.dp = dp

    def keys(self):
        return ("minRadius", "maxRadius", "param1", "param2", "method", "dp")

    def __getitem__(self, key):
        if key not in self.keys():
            raise KeyError(f"Invalid key: {key}")
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self.keys():
            raise KeyError(f"Invalid key: {key}")
        setattr(self, key, value)

    def copy(self) -> "HoughParams":
        return HoughParams(
            self.minRadius,
            self.maxRadius,
            self.param1,
            self.param2,
            self.method,
            self.dp,
        )

    def __str__(self) -> str:
        return (
            f"HoughParams(minRadius={self.minRadius}, maxRadius={self.maxRadius}, "
            f"param1={self.param1}, param2={self.param2}, method={self.method}, dp={self.dp})"
        )


CordType = TypeVar("CordType", int, float)


class Point(Generic[CordType]):
    _x: CordType
    _y: CordType

    def __init__(self, x: CordType, y: CordType):
        self._x = x
        self._y = y

    @property
    def x(self) -> CordType:
        return self._x

    @property
    def y(self) -> CordType:
        return self._y

    def __hash__(self) -> int:
        return hash((self._x, self._y))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Point):
            return False
        return self._x == other._x and self._y == other._y

    def __add__(self, delta: "Vector[CordType]") -> "Point[CordType]":
        return Point(self.x + delta.x, self.y + delta.y)

    @overload
    def __sub__(self, other: "Point[CordType]") -> "Vector[CordType]": ...
    @overload
    def __sub__(self, other: "Vector[CordType]") -> "Point[CordType]": ...
    def __sub__(
        self, other: "Point[CordType] | Vector[CordType]"
    ) -> "Vector[CordType] | Point[CordType]":
        if isinstance(other, Vector):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Vector(self.x - other.x, self.y - other.y)

    dataType2 = TypeVar("dataType2", int, float)

    def as_type(self, dType: Type[dataType2]) -> "Point[dataType2]":
        return Point(dType(self.x), dType(self.y))


ORIGIN = Point(0.0, 0.0)
ORIGIN_I = Point(0, 0)


class Boundary(Generic[CordType]):
    _min_x: CordType
    _max_x: CordType
    _min_y: CordType
    _max_y: CordType

    def __init__(
        self, min_x: CordType, max_x: CordType, min_y: CordType, max_y: CordType
    ):
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y

    def __hash__(self) -> int:
        return hash((self._min_x, self._max_x, self._min_y, self._max_y))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Boundary):
            return False
        return (
            self._min_x == other._min_x
            and self._max_x == other._max_x
            and self._min_y == other._min_y
            and self._max_y == other._max_y
        )

    def bound(self, point: "Point[CordType]") -> "BoundedPoint[CordType]":
        x = point.x
        y = point.y
        if x < self._min_x:
            x = self._min_x
        if x > self._max_x:
            x = self._max_x
        if y < self._min_y:
            y = self._min_y
        if y > self._max_y:
            y = self._max_y
        bp = BoundedPoint()
        bp._x = x
        bp._y = y
        bp._boundary = self
        return bp


class BoundedPoint(Generic[CordType]):
    _x: CordType
    _y: CordType
    _boundary: Boundary[CordType]

    @property
    def x(self) -> CordType:
        return self._x

    @property
    def y(self) -> CordType:
        return self._y

    def __hash__(self) -> int:
        return hash((self._x, self._y, self._boundary))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BoundedPoint):
            return False
        return (
            self._x == other._x
            and self._y == other._y
            and self._boundary == other._boundary
        )

    def __add__(self, delta: "Vector[CordType]") -> "BoundedPoint[CordType]":
        p = Point(self._x + delta.x, self._y + delta.y)
        return self._boundary.bound(p)

    @overload
    def __sub__(self, other: "BoundedPoint[CordType]") -> "Vector[CordType]": ...
    @overload
    def __sub__(self, other: "Vector[CordType]") -> "BoundedPoint[CordType]": ...
    def __sub__(
        self, other: "BoundedPoint[CordType] | Vector[CordType]"
    ) -> "Vector[CordType] | BoundedPoint[CordType]":
        if isinstance(other, Vector):
            p = Point(self.x - other.x, self.y - other.y)
            return self._boundary.bound(p)
        else:
            if self._boundary != other._boundary:
                raise Exception(
                    "Cannot subtract BoundedPoints with different boundaries"
                )
            return Vector(self.x - other.x, self.y - other.y)


class PointArray:
    def __init__(self, arr: NDArray, safe: bool = True):
        assert arr.ndim == 2 and arr.shape[1] == 2
        if safe:
            self._arr = arr.astype(np.float64, copy=True)
        else:
            self._arr = arr

    def __iter__(self) -> Generator[Point, Any, None]:
        for x, y in self._arr:
            yield Point(x, y)

    def __add__(self, vector: "Vector[float] | VectorArray") -> "PointArray":
        if isinstance(vector, VectorArray):
            return PointArray(self._arr + vector._arr, safe=False)
        else:
            return PointArray(self._arr + np.array([[vector.x, vector.y]]), safe=False)

    def __sub__(self, other: "Point[float] | PointArray") -> "VectorArray":
        if isinstance(other, Point):
            return VectorArray(self._arr - np.array([[other.x, other.y]]), safe=False)
        else:
            return VectorArray(self._arr - other._arr, safe=False)

    def __len__(self) -> int:
        return self._arr.shape[0]

    @property
    def x(self) -> NDArray[np.float64]:
        return self._arr[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self._arr[:, 1]

    def filter(self, mask: NDArray[np.bool_]) -> "PointArray":
        return PointArray(self._arr[mask], safe=False)


class Vector(Generic[CordType]):
    _x: CordType
    _y: CordType
    dataType2 = TypeVar("dataType2", int, float)

    def __init__(self, x: CordType, y: CordType):
        self._x = x
        self._y = y

    @property
    def x(self) -> CordType:
        return self._x

    @property
    def y(self) -> CordType:
        return self._y

    def __hash__(self) -> int:
        return hash((self._x, self._y))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Vector):
            return False
        return self._x == other._x and self._y == other._y

    @staticmethod
    def from_ndarray(arr: NDArray, dataType: Type[dataType2]) -> "Vector[dataType2]":
        assert arr.ndim == 1 and arr.shape[0] == 2
        return Vector(dataType(arr[0]), dataType(arr[1]))

    def as_type(self, dType: Type[dataType2]) -> "Vector[dataType2]":
        return Vector(dType(self.x), dType(self.y))

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalize(self) -> "Vector[float]":
        n = self.norm() + 1e-6
        return Vector(self.x / n, self.y / n)

    def __add__(self, other: "Vector[CordType]") -> "Vector[CordType]":
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector[CordType]") -> "Vector[CordType]":
        return Vector(self.x - other.x, self.y - other.y)

    @overload
    def __mul__(self, other: float) -> "Vector[float]": ...
    @overload
    def __mul__(self, other: "Vector[CordType]") -> CordType: ...

    def __mul__(self, other: "float | Vector[CordType]") -> "float | Vector[float]":
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        else:
            return Vector(self.x * other, self.y * other)

    def __div__(self, other: float) -> "Vector":
        return Vector(self.x / other, self.y / other)


ZERO = Vector(0.0, 0.0)
ZERO_I = Vector(0, 0)


class VectorArray:
    """
    为了节省性能，VectorArray直接使用
    """

    def __init__(self, arr: NDArray, safe: bool = True):
        assert arr.ndim == 2 and arr.shape[1] == 2
        if safe:
            self._arr = arr.astype(np.float64, copy=True)
        else:
            self._arr = arr

    def norms(self) -> NDArray[np.float64]:
        return np.hypot(self._arr[:, 0], self._arr[:, 1])

    def normalize(self) -> "VectorArray":
        norms = self.norms() + 1e-6
        return VectorArray(self._arr / norms[:, np.newaxis], safe=False)

    @overload
    def __mul__(self, other: float | NDArray[np.floating]) -> "VectorArray": ...
    @overload
    def __mul__(self, other: "VectorArray | Vector[float]") -> NDArray[np.float64]: ...
    def __mul__(
        self, other: "float | NDArray[np.floating] | VectorArray | Vector[float]"
    ) -> "NDArray[np.float64] | VectorArray":
        if isinstance(other, float):
            return VectorArray(self._arr * other, safe=False)
        elif isinstance(other, np.ndarray):
            assert other.ndim == 1 and other.shape[0] == self._arr.shape[0]
            return VectorArray(self._arr * other[:, np.newaxis], safe=False)

        elif isinstance(other, Vector):
            return self._arr[:, 0] * other.x + self._arr[:, 1] * other.y
        elif isinstance(other, VectorArray):
            return (
                self._arr[:, 0] * other._arr[:, 0] + self._arr[:, 1] * other._arr[:, 1]
            )
        raise TypeError("Unsupported type for multiplication")

    @overload
    def __add__(self, other: "VectorArray | Vector[float]") -> "VectorArray": ...
    @overload
    def __add__(self, other: "PointArray | Point[float]") -> "PointArray": ...
    def __add__(
        self, other: "VectorArray | Vector[float] | PointArray | Point[float]"
    ) -> "VectorArray | PointArray":
        if isinstance(other, Vector):
            return VectorArray(self._arr + np.array([[other.x, other.y]]), safe=False)
        elif isinstance(other, VectorArray):
            return VectorArray(self._arr + other._arr, safe=False)
        elif isinstance(other, Point):
            return PointArray(self._arr + np.array([[other.x, other.y]]), safe=False)
        elif isinstance(other, PointArray):
            return PointArray(self._arr + other._arr, safe=False)
        raise TypeError("Unsupported type for addition")

    def __sub__(self, other: "VectorArray | Vector[float]") -> "VectorArray":
        if isinstance(other, Vector):
            return VectorArray(self._arr - np.array([[other.x, other.y]]), safe=False)
        else:
            return VectorArray(self._arr - other._arr, safe=False)

    def __div__(self, other: float) -> "VectorArray":
        return VectorArray(self._arr / other, safe=False)

    def __iter__(self) -> Generator[Vector[float], Any, None]:
        for x, y in self._arr:
            yield Vector(x, y)

    def __len__(self) -> int:
        return self._arr.shape[0]

    @property
    def x(self) -> NDArray[np.float64]:
        return self._arr[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self._arr[:, 1]

    def filter(self, mask: NDArray[np.bool_]) -> "VectorArray":
        return VectorArray(self._arr[mask], safe=False)


class Circle:
    _center: Point[float]
    _radius: float

    def __init__(self, x: float, y: float, radius: float):
        assert isinstance(x, float)
        self._center = Point(x, y)
        self._radius = radius

    @property
    def x(self) -> float:
        return self._center.x

    @property
    def y(self) -> float:
        return self._center.y

    @property
    def radius(self) -> float:
        return self._radius

    def __hash__(self) -> int:
        return hash((self._center, self._radius))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Circle):
            return False
        return self._center == other._center and self._radius == other._radius

    @staticmethod
    def from_ndarray(arr: NDArray) -> "Circle":
        assert arr.ndim == 1 and arr.shape[0] == 3
        return Circle(float(arr[0]), float(arr[1]), float(arr[2]))

    def scale(self, factor: float) -> "Circle":
        return Circle(
            self._center.x * factor, self._center.y * factor, self._radius * factor
        )

    def shift(self, v: Vector[float]) -> "Circle":
        return Circle(self._center.x + v.x, self._center.y + v.y, self._radius)

    @property
    def center(self) -> Point[float]:
        return self._center

    def __str__(self) -> str:
        return f"Circle(x={self.x}, y={self.y}, radius={self.radius})"


class ROI:
    _start_point: Point[int]
    _width: int
    _height: int
    score: float = 0.0

    def __init__(self, x: int, y: int, width: int, height: int):
        self._start_point = Point(x, y)
        self._width = width
        self._height = height

    @property
    def x(self) -> int:
        return self._start_point.x

    @property
    def y(self) -> int:
        return self._start_point.y

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def __hash__(self) -> int:
        return hash((self._start_point, self._width, self._height))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ROI):
            return False
        return (
            self._start_point == other._start_point
            and self._width == other._width
            and self._height == other._height
        )

    @property
    def size(self) -> Vector[int]:
        return Vector(self._width, self._height)

    @property
    def center(self) -> Point[float]:
        return Point(
            self._start_point.x + self._width / 2.0,
            self._start_point.y + self._height / 2.0,
        )

    @property
    def start_point(self) -> Point[int]:
        return self._start_point
