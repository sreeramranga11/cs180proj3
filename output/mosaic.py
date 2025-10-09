from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

from .debug import DebugLogger
from .homography import apply_homography, compute_homography
from .io_utils import CorrespondenceSet
from .warping import compute_warp_bounds, warp_image_bilinear

ArrayLike = np.ndarray


class FeatherBlendStrategy(ABC):
    # blending warped images

    @abstractmethod
    def compute_weights(self, mask: np.ndarray) -> np.ndarray:
        '''return blending weights given binary mask'''


@dataclass
class LinearFeatherBlendStrategy(FeatherBlendStrategy):
    # Linear feathering from image center to boundaries

    def compute_weights(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("Mask must be 2D")
        ys, xs = np.indices(mask.shape)
        center_y = (mask.shape[0] - 1) / 2.0
        center_x = (mask.shape[1] - 1) / 2.0
        # Distances from the center let us fade contributions near the edge.
        distances = np.sqrt((ys - center_y) ** 2 + (xs - center_x) ** 2)
        max_distance = distances[mask].max() if mask.any() else 1.0
        weights = 1.0 - (distances / max_distance)
        weights[~mask] = 0.0
        return weights


@dataclass
class LaplacianPyramidBlendStrategy(FeatherBlendStrategy):
    # Two-level Laplacian pyramid blending

    num_levels: int = 2

    def compute_weights(self, mask: np.ndarray) -> np.ndarray:
        import scipy.ndimage as ndi

        weights = mask.astype(np.float64)
        sigma = max(1.0, self.num_levels)
        # Smooth the mask so the transition between images is gradual.
        weights = ndi.gaussian_filter(weights, sigma=sigma)
        weights /= weights.max() if weights.max() > 0 else 1.0
        return weights


@dataclass
class MosaicImage:
    image: ArrayLike
    mask: np.ndarray
    weights: np.ndarray
    offset: Tuple[int, int]


@dataclass
class MosaicBuilder:
    reference_shape: Tuple[int, int, int]
    logger: DebugLogger | None = None
    blend_strategy: FeatherBlendStrategy = field(default_factory=LinearFeatherBlendStrategy)

    def _initialize_canvas(self, reference_image: ArrayLike) -> MosaicImage:
        height, width = reference_image.shape[:2]
        # Start the mosaic using the reference image as-is.
        canvas = MosaicImage(
            image=reference_image.astype(np.float32),
            mask=np.ones((height, width), dtype=bool),
            weights=np.ones((height, width), dtype=np.float32),
            offset=(0, 0),
        )
        return canvas

    def build(
        self,
        reference_image: ArrayLike,
        additional_images: Sequence[ArrayLike],
        correspondences: Sequence[CorrespondenceSet],
    ) -> ArrayLike:
        if len(additional_images) != len(correspondences):
            raise ValueError("Each additional image must have a correspondence set")

        canvas = self._initialize_canvas(reference_image)

        for idx, (image, corr) in enumerate(zip(additional_images, correspondences)):
            H = compute_homography(corr)
            # Warp into the reference frame so we can blend directly.
            bounds = compute_warp_bounds(image.shape, H, sample_points=corr.points_a)
            warped, mask, (min_x, min_y) = warp_image_bilinear(image, H, bounds=bounds)
            mask = self._apply_overlap_constraint(mask, corr, H, (min_x, min_y))
            weights = self.blend_strategy.compute_weights(mask)

            canvas = self._expand_canvas(canvas, warped.shape, (min_x, min_y))
            self._paste(canvas, warped, mask, weights, (min_x, min_y))

            if self.logger is not None:
                self.logger.log(
                    "Added image to mosaic",
                    index=idx,
                    bounds=(min_x, min_y, min_x + warped.shape[1] - 1, min_y + warped.shape[0] - 1),
                )
                self.logger.save_image(warped, name=f"mosaic_warped_{idx}")

        normalized = self._normalize(canvas)
        return normalized.astype(reference_image.dtype)

    def _expand_canvas(
        self,
        canvas: MosaicImage,
        warped_shape: Tuple[int, int, int] | Tuple[int, int],
        warped_origin: Tuple[int, int],
    ) -> MosaicImage:
        h, w = canvas.image.shape[:2]
        offset_x, offset_y = canvas.offset

        global_min_x = -offset_x
        global_min_y = -offset_y
        global_max_x = global_min_x + w - 1
        global_max_y = global_min_y + h - 1

        warped_height, warped_width = warped_shape[:2]
        min_x, min_y = warped_origin
        max_x = min_x + warped_width - 1
        max_y = min_y + warped_height - 1

        new_global_min_x = min(global_min_x, min_x)
        new_global_min_y = min(global_min_y, min_y)
        new_global_max_x = max(global_max_x, max_x)
        new_global_max_y = max(global_max_y, max_y)

        if (
            new_global_min_x == global_min_x
            and new_global_min_y == global_min_y
            and new_global_max_x == global_max_x
            and new_global_max_y == global_max_y
        ):
            return canvas

        new_width = new_global_max_x - new_global_min_x + 1
        new_height = new_global_max_y - new_global_min_y + 1

        expanded_image = np.zeros((new_height, new_width, canvas.image.shape[2]), dtype=canvas.image.dtype)
        expanded_mask = np.zeros((new_height, new_width), dtype=bool)
        expanded_weights = np.zeros((new_height, new_width), dtype=canvas.weights.dtype)

        start_x = global_min_x - new_global_min_x
        start_y = global_min_y - new_global_min_y

        expanded_image[start_y:start_y + h, start_x:start_x + w] = canvas.image
        expanded_mask[start_y:start_y + h, start_x:start_x + w] = canvas.mask
        expanded_weights[start_y:start_y + h, start_x:start_x + w] = canvas.weights

        canvas.image = expanded_image
        canvas.mask = expanded_mask
        canvas.weights = expanded_weights
        canvas.offset = (-new_global_min_x, -new_global_min_y)
        return canvas

    def _paste(
        self,
        canvas: MosaicImage,
        image: ArrayLike,
        mask: np.ndarray,
        weights: np.ndarray,
        origin: Tuple[int, int],
    ) -> None:
        offset_x, offset_y = canvas.offset
        min_x, min_y = origin
        start_x = min_x + offset_x
        start_y = min_y + offset_y

        h, w = image.shape[:2]
        slice_y = slice(start_y, start_y + h)
        slice_x = slice(start_x, start_x + w)

        weight_expanded = weights[..., None]
        # Accumulate weighted colors so overlapping regions blend smoothly.
        canvas.image[slice_y, slice_x] += image * weight_expanded
        canvas.weights[slice_y, slice_x] += weights
        canvas.mask[slice_y, slice_x] |= mask

    def _normalize(self, canvas: MosaicImage) -> ArrayLike:
        epsilon = 1e-8
        # Avoid division by zero when the weights sum to zero.
        result = canvas.image / (canvas.weights[..., None] + epsilon)
        result[~canvas.mask] = 0
        return result

    def _apply_overlap_constraint(
        self,
        mask: np.ndarray,
        correspondences: CorrespondenceSet,
        H: np.ndarray,
        origin: Tuple[int, int],
    ) -> np.ndarray:
        """Limit the warped contribution to the region supported by correspondences."""

        if correspondences.count < 3:
            return mask

        warped_points = apply_homography(correspondences.points_a, H)
        valid = np.isfinite(warped_points).all(axis=1)
        warped_points = warped_points[valid]
        if warped_points.shape[0] < 3:
            return mask

        hull = _convex_hull(warped_points)
        if hull.shape[0] < 3:
            return mask

        expanded_hull = _expand_polygon(hull, mask.shape)
        rect_mask = _rectangular_mask(mask.shape, origin, expanded_hull)
        polygon_mask = _polygon_mask(mask.shape, origin, expanded_hull)
        combined_mask = rect_mask | polygon_mask
        if not combined_mask.any():
            return mask
        return mask & combined_mask


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a set of 2D points using the monotonic chain."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return pts

    # Sort lexicographically (y first to stabilize vertical structures).
    order = np.lexsort((pts[:, 0], pts[:, 1]))
    sorted_pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[np.ndarray] = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.vstack((lower[:-1], upper[:-1]))
    return hull


def _expand_polygon(polygon: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
    # Expand polygon radially to admit nearby regions when masking

    if polygon.shape[0] < 3:
        return polygon

    height, width = mask_shape
    extent = np.ptp(polygon, axis=0)
    extent_x = max(float(extent[0]), 1.0)
    extent_y = max(float(extent[1]), 1.0)

    ratio_x = width / extent_x
    ratio_y = height / extent_y
    ratio = max(ratio_x, ratio_y)

    expansion_cap = 8.0
    expansion_factor = min(expansion_cap, max(1.5, ratio * 1.15))

    center = polygon.mean(axis=0, keepdims=True)
    offsets = polygon - center
    expanded = center + offsets * expansion_factor

    max_extent = max(extent_x, extent_y)
    padding = max_extent * 0.35 + 40.0
    max_padding = max(height, width) * 0.5
    padding = float(min(padding, max_padding))

    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-6, norms, 1.0)
    expanded += offsets * (padding / safe_norms)
    return expanded


def _rectangular_mask(shape: Tuple[int, int], origin: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    # Create axis aligned mask that bounds polygon

    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if polygon.shape[0] < 1:
        mask[:] = True
        return mask

    min_corner = polygon.min(axis=0)
    max_corner = polygon.max(axis=0)

    origin_x, origin_y = origin
    x0 = int(np.floor(min_corner[0] - origin_x))
    y0 = int(np.floor(min_corner[1] - origin_y))
    x1 = int(np.ceil(max_corner[0] - origin_x))
    y1 = int(np.ceil(max_corner[1] - origin_y))

    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))

    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _polygon_mask(shape: Tuple[int, int], origin: Tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Rasterize a polygon into a boolean mask anchored at the provided origin."""

    height, width = shape
    if polygon.shape[0] < 3:
        return np.ones((height, width), dtype=bool)

    min_x, min_y = origin
    poly = polygon - np.array([min_x, min_y], dtype=np.float64)

    xs = np.arange(width, dtype=np.float64)[None, :]
    ys = np.arange(height, dtype=np.float64)[:, None]

    inside = np.zeros((height, width), dtype=bool)
    x0 = poly[:, 0]
    y0 = poly[:, 1]
    num_vertices = len(poly)

    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        xi, yi = x0[i], y0[i]
        xj, yj = x0[j], y0[j]
        # Ray casting: flip inside flag on edge crossings.
        intersects = ((yi > ys) != (yj > ys)) & (
            xs
            < (xj - xi) * (ys - yi) / (yj - yi + 1e-12) + xi
        )
        inside ^= intersects

    return inside


__all__ = [
    "MosaicBuilder",
    "FeatherBlendStrategy",
    "LinearFeatherBlendStrategy",
    "LaplacianPyramidBlendStrategy",
]
