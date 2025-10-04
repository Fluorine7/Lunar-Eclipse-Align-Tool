# utils_common.py
from abc import ABC
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import os
from typing import TypeVar
import numpy as np
import cv2


from numpy.typing import NDArray

from lunar_eclipse_align.utils.data_types import Circle
from lunar_eclipse_align.utils.constants import DEBUG


def get_memory_usage_mb():
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def to_display_rgb(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=32)
def soft_disk_mask(
    height: int,
    width: int,
    circle: Circle,
    inner: float,
    outer: float,
) -> NDArray[np.float32]:

    dist = distance_map(width, height, circle)
    m = np.zeros((height, width), np.float32)
    r_in = circle.radius * inner
    r_out = circle.radius * outer

    band = ring_mask(width, height, circle, inner=inner, outer=outer)

    t = (dist[band] - r_in) / ((r_out - r_in) + 1e-6)
    m[band] = 0.5 * (1 + np.cos(np.pi * t))
    m[dist <= r_in] = 1.0
    return m


# TODO: a better way to optimize it is to build a class Detection to wrap these stuffs
# and cache all the intermediate ingredients inside an object
# they will be dropped as the detection ends
@lru_cache(maxsize=32)
def distance_map(width: int, height: int, circle: Circle) -> NDArray[np.float32]:
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((X - circle.x) ** 2 + (Y - circle.y) ** 2)
    return dist


@lru_cache(maxsize=32)
def ring_mask(
    width: int, height: int, circle: Circle, inner: float, outer: float
) -> NDArray[np.bool]:
    dist = distance_map(width, height, circle)
    return (dist >= circle.radius * inner) & (dist <= circle.radius * outer)


def touches_border(width: int, height: int, circle: Circle, margin: int = 5) -> bool:
    return (
        (circle.x - circle.radius < margin)
        or (circle.y - circle.radius < margin)
        or (circle.x + circle.radius > width - margin)
        or (circle.y + circle.radius > height - margin)
    )


NUMBER = TypeVar("NUMBER", int, float)


def clip(min_value: NUMBER, value: NUMBER, max_value: NUMBER) -> NUMBER:
    return max(min_value, min(value, max_value))
