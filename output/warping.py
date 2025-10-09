# Image warping primitives

from __future__ import annotations

from typing import Tuple

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


def warp_image_nearest_neighbor(
    image: ArrayLike, H: np.ndarray
) -> Tuple[ArrayLike, np.ndarray, Tuple[int, int]]:
    # Warp an image using nearest neighbor interpolation

    min_x, max_x, min_y, max_y = compute_output_bounds(image.shape, H)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    xs, ys = inverse_warp_coordinates((height, width), H, offset=(min_x, min_y))

    image_ch, squeezed = _ensure_channel_dim(image)
    sampled = np.zeros((height, width, image_ch.shape[2]), dtype=image.dtype)
    mask = np.zeros((height, width), dtype=bool)

    x_nn = np.round(xs).astype(int)
    y_nn = np.round(ys).astype(int)

    # Guard against sampling outside the input image.
    valid = (
        (x_nn >= 0)
        & (x_nn < image.shape[1])
        & (y_nn >= 0)
        & (y_nn < image.shape[0])
    )
    mask[valid] = True
    sampled[valid] = image_ch[y_nn[valid], x_nn[valid]]

    return _restore_channel_dim(sampled, squeezed), mask, (min_x, min_y)


def warp_image_bilinear(
    image: ArrayLike, H: np.ndarray
) -> Tuple[ArrayLike, np.ndarray, Tuple[int, int]]:
    # Warp an image using bilinear interpolation

    min_x, max_x, min_y, max_y = compute_output_bounds(image.shape, H)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    xs, ys = inverse_warp_coordinates((height, width), H, offset=(min_x, min_y))

    image_ch, squeezed = _ensure_channel_dim(image)
    sampled = np.zeros((height, width, image_ch.shape[2]), dtype=np.float64)
    mask = np.zeros((height, width), dtype=bool)

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Only pixels whose four neighbors exist contribute to the bilinear result.
    valid = (
        (x0 >= 0)
        & (x1 < image.shape[1])
        & (y0 >= 0)
        & (y1 < image.shape[0])
    )

    wx = xs - x0
    wy = ys - y0

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
]
