# Feature detection and matching helpers used by the deliverables pipeline

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .homography import compute_homography, apply_homography
from .io_utils import CorrespondenceIO, CorrespondenceSet

ArrayLike = np.ndarray
PointArray = np.ndarray


# Image processing primitives
def to_grayscale(image: ArrayLike) -> np.ndarray:
    # Convert an RGB or grayscale image to float32 grayscale in [0, 1]

    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3 and image.shape[2] == 3:
        image_f = image.astype(np.float32)
        # Standard luminance weights from ITU-R BT.601.
        gray = 0.2989 * image_f[..., 0] + 0.5870 * image_f[..., 1] + 0.1140 * image_f[..., 2]
    else:
        raise ValueError("Unsupported image shape for grayscale conversion")
    gray -= gray.min()
    ptp = np.ptp(gray)
    if ptp > 0:
        gray /= ptp
    return gray.astype(np.float32)


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("Gaussian kernel size must be odd")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d)
    return kernel.astype(np.float32)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Perform a 2d convolution with reflective padding

    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    # Einsum keeps the implementation compact and reasonably fast.
    convolved = np.einsum("ij,xyij->xy", kernel, windows)
    return convolved.astype(np.float32)


# Harris corner detector and simple patch descriptors
@dataclass
class HarrisCornerDetector:
    num_features: int = 1200
    k: float = 0.04
    window_size: int = 5
    min_distance: int = 5

    def detect(self, image: ArrayLike) -> np.ndarray:
        gray = to_grayscale(image)

        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32) / 8.0
        sobel_y = sobel_x.T
        ix = convolve2d(gray, sobel_x)
        iy = convolve2d(gray, sobel_y)

        gauss = gaussian_kernel(self.window_size, sigma=self.window_size / 3)
        ix2 = convolve2d(ix * ix, gauss)
        iy2 = convolve2d(iy * iy, gauss)
        ixy = convolve2d(ix * iy, gauss)

        det = ix2 * iy2 - ixy**2
        trace = ix2 + iy2
        response = det - self.k * (trace**2)

        response = np.where(response > 0, response, 0.0)
        threshold = max(0.005 * response.max(), 1e-6)
        response[response < threshold] = 0

        # Non-maximum suppression: greedily pick the strongest corners while blanking out a radius around each accepted point.
        coords = np.argwhere(response > 0)
        strengths = response[coords[:, 0], coords[:, 1]]
        order = np.argsort(strengths)[::-1]

        suppressed = np.zeros_like(response, dtype=bool)
        selected = []
        radius = self.min_distance

        for idx in order:
            y, x = coords[idx]
            if suppressed[y, x]:
                continue
            selected.append((y, x))
            y0 = max(0, y - radius)
            y1 = min(response.shape[0], y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(response.shape[1], x + radius + 1)
            suppressed[y0:y1, x0:x1] = True
            if len(selected) >= self.num_features:
                break

        return np.asarray(selected, dtype=np.float32)


@dataclass
class PatchDescriptorExtractor:
    patch_size: int = 21
    num_cells: int = 4
    num_bins: int = 8

    def extract(self, image: ArrayLike, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.patch_size % 2 == 0:
            raise ValueError("Patch size must be odd so it has a well-defined center")

        gray = to_grayscale(image)
        half = self.patch_size // 2
        descriptor_dim = self.num_cells * self.num_cells * self.num_bins

        valid_mask = (
            (keypoints[:, 0] >= half)
            & (keypoints[:, 0] < gray.shape[0] - half)
            & (keypoints[:, 1] >= half)
            & (keypoints[:, 1] < gray.shape[1] - half)
        )

        candidate_points = keypoints[valid_mask]
        if candidate_points.size == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, descriptor_dim), dtype=np.float32),
            )

        cell_edges = np.linspace(0, self.patch_size, self.num_cells + 1, dtype=int)
        descriptors: list[np.ndarray] = []
        accepted_points: list[Tuple[float, float]] = []

        for y, x in candidate_points:
            y0 = int(y) - half
            y1 = int(y) + half + 1
            x0 = int(x) - half
            x1 = int(x) + half + 1
            patch = gray[y0:y1, x0:x1].astype(np.float32)
            if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                continue

            gy, gx = np.gradient(patch)
            magnitude = np.hypot(gx, gy)
            if float(magnitude.sum()) <= 1e-6:
                continue
            orientation = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)

            histograms = []
            for i in range(self.num_cells):
                for j in range(self.num_cells):
                    y_start, y_stop = cell_edges[i], cell_edges[i + 1]
                    x_start, x_stop = cell_edges[j], cell_edges[j + 1]

                    cell_mag = magnitude[y_start:y_stop, x_start:x_stop]
                    cell_ori = orientation[y_start:y_stop, x_start:x_stop]
                    if cell_mag.size == 0:
                        histograms.append(np.zeros(self.num_bins, dtype=np.float32))
                        continue

                    bins = np.zeros(self.num_bins, dtype=np.float32)
                    bin_idx = np.floor((cell_ori / (2 * np.pi)) * self.num_bins).astype(int)
                    bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
                    np.add.at(bins, bin_idx.ravel(), cell_mag.ravel())
                    histograms.append(bins)

            descriptor = np.concatenate(histograms).astype(np.float32, copy=False)
            norm = np.linalg.norm(descriptor)
            if norm > 1e-6:
                descriptor = descriptor / norm
                descriptor = np.sqrt(descriptor)
                norm = np.linalg.norm(descriptor)
                if norm > 1e-6:
                    descriptor /= norm

            descriptors.append(descriptor)
            accepted_points.append((y, x))

        if not descriptors:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, descriptor_dim), dtype=np.float32),
            )

        return (
            np.asarray(accepted_points, dtype=np.float32),
            np.vstack(descriptors).astype(np.float32, copy=False),
        )


def match_descriptors(
    descriptors_a: np.ndarray,
    descriptors_b: np.ndarray,
    ratio: float = 0.8,
    require_consistency: bool = True,
) -> np.ndarray:
    if descriptors_a.size == 0 or descriptors_b.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    def _normalize_rows(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-12)
        return arr / safe_norms

    a_norm = _normalize_rows(descriptors_a.astype(np.float32, copy=False))
    b_norm = _normalize_rows(descriptors_b.astype(np.float32, copy=False))

    similarity = np.clip(a_norm @ b_norm.T, -1.0, 1.0)
    distances = np.sqrt(np.maximum(2.0 - 2.0 * similarity, 0.0))

    order = np.argsort(distances, axis=1)
    best_idx = order[:, 0]
    best_dist = distances[np.arange(distances.shape[0]), best_idx]

    if distances.shape[1] > 1:
        second_idx = order[:, 1]
        second_dist = distances[np.arange(distances.shape[0]), second_idx]
    else:
        second_dist = np.full_like(best_dist, np.inf)

    ratios = np.divide(
        best_dist,
        second_dist,
        out=np.full_like(best_dist, np.inf),
        where=second_dist > 1e-12,
    )
    candidate_mask = ratios < ratio

    if require_consistency:
        best_a_for_b = np.argmin(distances, axis=0)
        mutual_mask = best_a_for_b[best_idx] == np.arange(distances.shape[0])
        candidate_mask &= mutual_mask

    if not np.any(candidate_mask):
        return np.empty((0, 2), dtype=np.int64)

    selected_indices = np.where(candidate_mask)[0]
    selected = np.stack(
        [selected_indices, best_idx[selected_indices]],
        axis=1,
    )

    order_by_distance = np.argsort(best_dist[selected_indices])
    return selected[order_by_distance].astype(np.int64, copy=False)


# RANSAC based homography estimation
@dataclass
class RansacHomographyEstimator:
    num_iterations: int = 4000
    inlier_threshold: float = 2.0
    random_seed: int = 42

    def estimate(self, points_a: PointArray, points_b: PointArray) -> Tuple[np.ndarray, CorrespondenceSet]:
        if points_a.shape[0] != points_b.shape[0]:
            raise ValueError("Point arrays must have the same length")
        if points_a.shape[0] < 4:
            raise ValueError("At least four correspondences are required for RANSAC")

        rng = np.random.default_rng(self.random_seed)
        best_inliers = None
        best_H = None

        for _ in range(self.num_iterations):
            sample_idx = rng.choice(points_a.shape[0], size=4, replace=False)
            sample = CorrespondenceSet(points_a=points_a[sample_idx], points_b=points_b[sample_idx])
            H_candidate = compute_homography(sample)

            projected = apply_homography(points_a, H_candidate)
            errors = np.linalg.norm(projected - points_b, axis=1)
            inliers = errors < self.inlier_threshold
            inlier_count = int(np.sum(inliers))

            if inlier_count < 4:
                continue

            if best_inliers is None or inlier_count > int(np.sum(best_inliers)):
                best_inliers = inliers
                best_H = H_candidate

        if best_inliers is None or best_H is None:
            raise RuntimeError("RANSAC failed to find a valid homography")

        inlier_corr = CorrespondenceSet(points_a=points_a[best_inliers], points_b=points_b[best_inliers])
        refined_H = compute_homography(inlier_corr)
        return refined_H, inlier_corr


# feature matching helper
@dataclass
class FeatureMatcher:
    corner_detector: HarrisCornerDetector = field(default_factory=HarrisCornerDetector)
    descriptor_extractor: PatchDescriptorExtractor = field(default_factory=PatchDescriptorExtractor)
    estimator: RansacHomographyEstimator = field(default_factory=RansacHomographyEstimator)

    def match_images(self, image_a: ArrayLike, image_b: ArrayLike) -> Tuple[np.ndarray, CorrespondenceSet]:
        keypoints_a = self.corner_detector.detect(image_a)
        keypoints_b = self.corner_detector.detect(image_b)

        pts_a, desc_a = self.descriptor_extractor.extract(image_a, keypoints_a)
        pts_b, desc_b = self.descriptor_extractor.extract(image_b, keypoints_b)

        matches = match_descriptors(desc_a, desc_b)
        if matches.shape[0] < 4:
            raise RuntimeError("Not enough matches were found between the images")

        matched_pts_a = pts_a[matches[:, 0]]
        matched_pts_b = pts_b[matches[:, 1]]

        # Convert from (row, col) to Cartesian (x, y) ordering expected elsewhere.
        points_a = np.column_stack([matched_pts_a[:, 1], matched_pts_a[:, 0]])
        points_b = np.column_stack([matched_pts_b[:, 1], matched_pts_b[:, 0]])

        H, inlier_corr = self.estimator.estimate(points_a, points_b)
        return H, inlier_corr

    def match_from_paths(self, path_a: Path, path_b: Path) -> Tuple[np.ndarray, CorrespondenceSet]:
        from .io_utils import load_image

        image_a = load_image(path_a)
        image_b = load_image(path_b)
        return self.match_images(image_a, image_b)


# Visualization helpers
def visualize_correspondences(
    image_a: ArrayLike,
    image_b: ArrayLike,
    correspondences: CorrespondenceSet,
    output_path: Path,
    max_lines: int = 80,
) -> None:
    import matplotlib.pyplot as plt

    if image_a.ndim == 2:
        image_a = np.repeat(image_a[..., None], 3, axis=2)
    if image_b.ndim == 2:
        image_b = np.repeat(image_b[..., None], 3, axis=2)

    height = max(image_a.shape[0], image_b.shape[0])
    width = image_a.shape[1] + image_b.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.float32)

    canvas[: image_a.shape[0], : image_a.shape[1]] = image_a.astype(np.float32) / 255.0
    canvas[: image_b.shape[0], image_a.shape[1] :] = image_b.astype(np.float32) / 255.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    ax.axis("off")

    num_points = min(correspondences.count, max_lines)
    pts_a = correspondences.points_a[:num_points]
    pts_b = correspondences.points_b[:num_points]

    for (xa, ya), (xb, yb) in zip(pts_a, pts_b):
        xb_shifted = xb + image_a.shape[1]
        ax.plot([xa, xb_shifted], [ya, yb], "-", linewidth=0.5, color="yellow")
        ax.scatter([xa, xb_shifted], [ya, yb], s=10, c="red")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_correspondences(
    correspondences: CorrespondenceSet,
    output_dir: Path,
    name: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}_correspondences.csv"
    CorrespondenceIO.to_file(correspondences, csv_path)

    from .homography import construct_dlt_matrix

    A = construct_dlt_matrix(correspondences)
    system_path = output_dir / f"{name}_system_matrix.txt"
    np.savetxt(system_path, A, fmt="%.6f", delimiter=",")
    return csv_path, system_path


__all__ = [
    "FeatureMatcher",
    "HarrisCornerDetector",
    "PatchDescriptorExtractor",
    "RansacHomographyEstimator",
    "visualize_correspondences",
    "save_correspondences",
]
