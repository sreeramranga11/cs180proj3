from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    imageio = None  # type: ignore


ArrayLike = np.ndarray
Point = Tuple[float, float]


@dataclass
class CorrespondenceSet:
    # stores corresponding points between two images

    points_a: np.ndarray
    points_b: np.ndarray

    def __post_init__(self) -> None:
        # Ensure always work with numpy arrays so later math stays simple.
        self.points_a = _validate_points(self.points_a)
        self.points_b = _validate_points(self.points_b)
        if self.points_a.shape != self.points_b.shape:
            raise ValueError("Point arrays must have the same shape")
        if self.points_a.shape[1] != 2:
            raise ValueError("Point arrays must have shape (N, 2)")

    @property
    def count(self) -> int:
        return self.points_a.shape[0]

    def as_float(self) -> "CorrespondenceSet":
        return CorrespondenceSet(
            points_a=self.points_a.astype(np.float64),
            points_b=self.points_b.astype(np.float64),
        )

    def to_homogeneous(self) -> Tuple[np.ndarray, np.ndarray]:
        # create homogeneous coordinates on demand.
        ones = np.ones((self.count, 1), dtype=np.float64)
        return (
            np.hstack([self.points_a.astype(np.float64), ones]),
            np.hstack([self.points_b.astype(np.float64), ones]),
        )


class CorrespondenceIO:
    # helpers for reading and writing data

    @staticmethod
    def from_lists(points_a: Iterable[Point], points_b: Iterable[Point]) -> CorrespondenceSet:
        # Accept plain Python sequences to make manual annotations convenient.
        return CorrespondenceSet(
            points_a=np.asarray(list(points_a), dtype=np.float64),
            points_b=np.asarray(list(points_b), dtype=np.float64),
        )

    @staticmethod
    def from_file(path: Path) -> CorrespondenceSet:
        data = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(
                "Correspondence file must have four columns: xa, ya, xb, yb"
            )
        # Split columns into the two point sets.
        points_a = data[:, :2]
        points_b = data[:, 2:]
        return CorrespondenceSet(points_a=points_a, points_b=points_b)

    @staticmethod
    def to_file(correspondences: CorrespondenceSet, path: Path) -> None:
        data = np.hstack([correspondences.points_a, correspondences.points_b])
        np.savetxt(path, data, delimiter=",", fmt="%.6f")


def _validate_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Points must be a 2D array")
    if array.shape[0] < 4:
        raise ValueError("At least four points are required")
    # No copy occurs if the array already matches the dtype, so this is cheap.
    return array


def load_image(path: Path) -> ArrayLike:
    if imageio is None:
        raise RuntimeError("imageio is required for image loading but is not installed")
    return imageio.imread(path)


def save_image(path: Path, image: ArrayLike) -> None:
    if imageio is None:
        raise RuntimeError("imageio is required for image saving but is not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


__all__ = [
    "CorrespondenceSet",
    "CorrespondenceIO",
    "load_image",
    "save_image",
]
