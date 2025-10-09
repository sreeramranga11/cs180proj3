# Image warping primitives

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .homography import apply_homography

ArrayLike = np.ndarray


def _ensure_channel_dim(image: ArrayLike) -> Tuple[np.ndarray, bool]:
    if image.ndim == 2:
        # Record that we need to squeeze the result later while adding a dummy
        # axis so the math works for both grayscale and color images.
        return image[..., None], True
    return image, False


def _restore_channel_dim(image: np.ndarray, squeezed: bool) -> np.ndarray:
    # Undo the temporary channel axis if we started with a 2D image.
    return image[..., 0] if squeezed else image


def compute_output_bounds(shape: Tuple[int, int], H: np.ndarray) -> Tuple[int, int, int, int]:
    # compute bounding box of warped image coordinates

    height, width = shape[:2]
    corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [0, height - 1],
            [width - 1, height - 1],
        ],
        dtype=np.float64,
    )
    warped = apply_homography(corners, H)
    valid = np.isfinite(warped).all(axis=1)
    if not np.any(valid):
        raise RuntimeError("Warp produced no finite coordinates")
    warped = warped[valid]

    # Use integer bounds so the caller can allocate an array of the right size.
    min_x = np.floor(np.min(warped[:, 0])).astype(int)
    max_x = np.ceil(np.max(warped[:, 0])).astype(int)
    min_y = np.floor(np.min(warped[:, 1])).astype(int)
    max_y = np.ceil(np.max(warped[:, 1])).astype(int)
    return min_x, max_x, min_y, max_y


def inverse_warp_coordinates(
    output_shape: Tuple[int, int],
    H: np.ndarray,
    offset: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    # Compute inverse warped coordinates for each pixel in output

    height, width = output_shape
    xs = np.arange(width) + offset[0]
    ys = np.arange(height) + offset[1]
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    H_inv = np.linalg.inv(H)
    # Feed the grid through the inverse so each output pixel knows where to
    # sample from in the source image.
    source_coords = apply_homography(flat_coords, H_inv)
    return source_coords[:, 0].reshape(output_shape), source_coords[:, 1].reshape(output_shape)


def compute_warp_bounds(
    image_shape: Tuple[int, int, int] | Tuple[int, int],
    H: np.ndarray,
    sample_points: Optional[np.ndarray] = None,
    margin: Optional[int] = None,
    max_extent: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    #Compute a bounded output region for warping to limit memory usage. keeps crashing

    height, width = image_shape[:2]
    if margin is None:
        margin = int(max(height, width) * 0.75)
    if max_extent is None:
        max_extent = max(height, width) * 2

    base_min_x, base_max_x, base_min_y, base_max_y = compute_output_bounds(image_shape, H)
    base_width = base_max_x - base_min_x + 1
    base_height = base_max_y - base_min_y + 1

    center_x = (base_min_x + base_max_x) / 2.0
    center_y = (base_min_y + base_max_y) / 2.0
    region_width = base_width
    region_height = base_height

    if sample_points is not None and sample_points.size > 0:
        warped = apply_homography(sample_points, H)
        valid = np.isfinite(warped).all(axis=1)
        if np.any(valid):
            region = warped[valid]
            region_min = np.floor(region.min(axis=0)).astype(int)
            region_max = np.ceil(region.max(axis=0)).astype(int)
            region_width = max(region_max[0] - region_min[0] + 1, 1)
            region_height = max(region_max[1] - region_min[1] + 1, 1)
            center_x = float(region[:, 0].mean())
            center_y = float(region[:, 1].mean())

    width_limit = min(base_width, max_extent)
    height_limit = min(base_height, max_extent)

    width_needed = int(region_width + 2 * margin)
    height_needed = int(region_height + 2 * margin)

    width_final = min(width_limit, max(width_needed, min(width_limit, region_width + margin)))
    height_final = min(height_limit, max(height_needed, min(height_limit, region_height + margin)))

    width_final = max(1, width_final)
    height_final = max(1, height_final)

    min_x = int(np.floor(center_x - width_final / 2.0))
    max_x = min_x + width_final - 1
    if min_x < base_min_x:
        min_x = base_min_x
        max_x = min_x + width_final - 1
    if max_x > base_max_x:
        max_x = base_max_x
        min_x = max_x - width_final + 1

    min_y = int(np.floor(center_y - height_final / 2.0))
    max_y = min_y + height_final - 1
    if min_y < base_min_y:
        min_y = base_min_y
        max_y = min_y + height_final - 1
    if max_y > base_max_y:
        max_y = base_max_y
        min_y = max_y - height_final + 1

    return int(min_x), int(max_x), int(min_y), int(max_y)


def warp_image_nearest_neighbor(
    image: ArrayLike, H: np.ndarray, bounds: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[ArrayLike, np.ndarray, Tuple[int, int]]:
    # Warp an image using nearest neighbor interpolation

    if bounds is None:
        min_x, max_x, min_y, max_y = compute_output_bounds(image.shape, H)
    else:
        min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    xs, ys = inverse_warp_coordinates((height, width), H, offset=(min_x, min_y))

    image_ch, squeezed = _ensure_channel_dim(image)
    sampled = np.zeros((height, width, image_ch.shape[2]), dtype=image.dtype)
    mask = np.zeros((height, width), dtype=bool)

    finite = np.isfinite(xs) & np.isfinite(ys)
    xs_safe = np.where(finite, xs, 0.0)
    ys_safe = np.where(finite, ys, 0.0)

    x_nn = np.round(xs_safe).astype(int)
    y_nn = np.round(ys_safe).astype(int)

    # Guard against sampling outside the input image.
    valid = finite & (x_nn >= 0) & (x_nn < image.shape[1]) & (y_nn >= 0) & (y_nn < image.shape[0])
    mask[valid] = True
    sampled[valid] = image_ch[y_nn[valid], x_nn[valid]]

    return _restore_channel_dim(sampled, squeezed), mask, (min_x, min_y)


def warp_image_bilinear(
    image: ArrayLike, H: np.ndarray, bounds: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[ArrayLike, np.ndarray, Tuple[int, int]]:
    # Warp an image using bilinear interpolation

    if bounds is None:
        min_x, max_x, min_y, max_y = compute_output_bounds(image.shape, H)
    else:
        min_x, max_x, min_y, max_y = bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    xs, ys = inverse_warp_coordinates((height, width), H, offset=(min_x, min_y))

    image_ch, squeezed = _ensure_channel_dim(image)
    accumulator_dtype = np.float32 if not np.issubdtype(image.dtype, np.floating) else image.dtype
    sampled = np.zeros((height, width, image_ch.shape[2]), dtype=accumulator_dtype)
    mask = np.zeros((height, width), dtype=bool)

    finite = np.isfinite(xs) & np.isfinite(ys)
    xs_safe = np.where(finite, xs, 0.0)
    ys_safe = np.where(finite, ys, 0.0)

    x0 = np.floor(xs_safe).astype(int)
    y0 = np.floor(ys_safe).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Only pixels whose four neighbors exist contribute to the bilinear result.
    valid = (
        finite
        & (x0 >= 0)
        & (x1 < image.shape[1])
        & (y0 >= 0)
        & (y1 < image.shape[0])
    )

    wx = xs_safe - x0
    wy = ys_safe - y0

    def clip_coords(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Clamp coordinates so advanced indexing never walks out of bounds.
        return (
            np.clip(x, 0, image.shape[1] - 1),
            np.clip(y, 0, image.shape[0] - 1),
        )

    x0c, y0c = clip_coords(x0, y0)
    x1c, y1c = clip_coords(x1, y1)

    Ia = image_ch[y0c, x0c]
    Ib = image_ch[y0c, x1c]
    Ic = image_ch[y1c, x0c]
    Id = image_ch[y1c, x1c]

    wx = wx[..., None]
    wy = wy[..., None]

    # Perform the usual bilinear blend using the fractional offsets.
    interpolated = (
        Ia * (1 - wx) * (1 - wy)
        + Ib * wx * (1 - wy)
        + Ic * (1 - wx) * wy
        + Id * wx * wy
    )
    if interpolated.dtype != sampled.dtype:
        interpolated = interpolated.astype(sampled.dtype, copy=False)

    sampled[valid] = interpolated[valid]
    mask[valid] = True

    if np.issubdtype(image.dtype, np.integer):
        sampled = np.clip(sampled, 0, np.iinfo(image.dtype).max).astype(image.dtype)
    else:
        sampled = sampled.astype(image.dtype)

    return _restore_channel_dim(sampled, squeezed), mask, (min_x, min_y)


__all__ = [
    "warp_image_nearest_neighbor",
    "warp_image_bilinear",
    "compute_output_bounds",
    "inverse_warp_coordinates",
    "compute_warp_bounds",
]
