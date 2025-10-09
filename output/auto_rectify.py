"""Automatic rectification helpers for generating assignment deliverables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np

from .feature_matching import HarrisCornerDetector
from .homography import compute_homography
from .io_utils import CorrespondenceIO, load_image
from .warping import warp_image_bilinear, warp_image_nearest_neighbor

ArrayLike = np.ndarray


def order_points(points: np.ndarray) -> np.ndarray:
    """Return points ordered as (top-left, top-right, bottom-right, bottom-left)."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError("Expected four 2D points")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()

    ordered = np.zeros((4, 2), dtype=np.float64)
    ordered[0] = pts[np.argmin(s)]  # top-left has smallest sum
    ordered[2] = pts[np.argmax(s)]  # bottom-right has largest sum
    ordered[1] = pts[np.argmin(diff)]  # top-right has smallest difference
    ordered[3] = pts[np.argmax(diff)]  # bottom-left has largest difference
    return ordered


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def is_reasonable_rectangle(points: np.ndarray, min_size: float = 40.0) -> bool:
    ordered = order_points(points)
    edges = np.vstack([ordered, ordered[0]])
    vectors = np.diff(edges, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)

    if np.any(lengths < min_size):
        return False

    cos_angles = []
    for i in range(4):
        u = vectors[i]
        v = vectors[(i + 1) % 4]
        denom = np.linalg.norm(u) * np.linalg.norm(v) + 1e-8
        cos_theta = float(np.dot(u, v) / denom)
        cos_angles.append(abs(cos_theta))
    return max(cos_angles) < 0.25  # within ~75-105 degrees


def find_rectangular_quad(points: np.ndarray, image_shape: Tuple[int, int], seed: int = 1234) -> np.ndarray:
    if points.shape[0] < 4:
        raise RuntimeError("Insufficient points to find a rectangle")

    rng = np.random.default_rng(seed)
    h, w = image_shape
    best_area = 0.0
    best_quad: np.ndarray | None = None

    # Reject corners hugging the image boundary to avoid picking the outer frame.
    margin_y = 0.05 * h
    margin_x = 0.05 * w

    valid_mask = (
        (points[:, 0] > margin_y)
        & (points[:, 0] < h - margin_y)
        & (points[:, 1] > margin_x)
        & (points[:, 1] < w - margin_x)
    )
    candidates = points[valid_mask]
    if candidates.shape[0] < 4:
        candidates = points

    for _ in range(5000):
        sample_idx = rng.choice(candidates.shape[0], size=4, replace=False)
        quad = candidates[sample_idx]
        if not is_reasonable_rectangle(quad):
            continue
        area = polygon_area(order_points(quad))
        if area > best_area:
            best_area = area
            best_quad = quad

    if best_quad is None:
        raise RuntimeError("Failed to locate a rectangular region for rectification")
    return order_points(best_quad)


def compute_destination_rectangle(points: np.ndarray) -> np.ndarray:
    ordered = order_points(points)
    width_top = np.linalg.norm(ordered[1] - ordered[0])
    width_bottom = np.linalg.norm(ordered[2] - ordered[3])
    height_left = np.linalg.norm(ordered[3] - ordered[0])
    height_right = np.linalg.norm(ordered[2] - ordered[1])

    width = int(round(max(width_top, width_bottom)))
    height = int(round(max(height_left, height_right)))
    width = max(width, 50)
    height = max(height, 50)

    dst = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float64,
    )
    return dst


@dataclass
class AutomaticRectifier:
    corner_detector: HarrisCornerDetector = field(default_factory=lambda: HarrisCornerDetector(num_features=1200))

    def plan_rectification(self, image: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        keypoints = self.corner_detector.detect(image)
        quad = find_rectangular_quad(keypoints, image.shape[:2])
        dst = compute_destination_rectangle(quad)
        # Convert (row, col) to (x, y)
        src_points = np.column_stack([quad[:, 1], quad[:, 0]])
        dst_points = dst
        return src_points, dst_points

    def rectify(self, image_path: Path, output_dir: Path, name: str) -> Tuple[Path, Path, Path]:
        image = load_image(image_path)
        src, dst = self.plan_rectification(image)
        correspondences = CorrespondenceIO.from_lists(src, dst)
        H = compute_homography(correspondences)

        bilinear, _, _ = warp_image_bilinear(image, H)
        nearest, _, _ = warp_image_nearest_neighbor(image, H)

        output_dir.mkdir(parents=True, exist_ok=True)
        bilinear_path = output_dir / f"{name}_rectified_bilinear.png"
        nearest_path = output_dir / f"{name}_rectified_nearest.png"
        CorrespondenceIO.to_file(correspondences, output_dir / f"{name}_rectification_points.csv")
        np.savetxt(output_dir / f"{name}_rectification_homography.txt", H, fmt="%.6f")

        from .io_utils import save_image

        save_image(bilinear_path, bilinear)
        save_image(nearest_path, nearest)

        overlay_path = output_dir / f"{name}_rectangle_overlay.png"
        save_rectangle_overlay(image, src, overlay_path)
        return bilinear_path, nearest_path, overlay_path


def save_rectangle_overlay(image: ArrayLike, src_points: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    quad = np.vstack([src_points, src_points[0]])
    ax.plot(quad[:, 0], quad[:, 1], "-", color="lime", linewidth=2)
    ax.scatter(src_points[:, 0], src_points[:, 1], color="red", s=30)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


__all__ = ["AutomaticRectifier", "save_rectangle_overlay", "order_points"]
