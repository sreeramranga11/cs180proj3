from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .debug import DebugLogger
from .homography import compute_homography
from .io_utils import CorrespondenceSet
from .warping import warp_image_bilinear

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
    blend_strategy: FeatherBlendStrategy = LinearFeatherBlendStrategy()

    def _initialize_canvas(self, reference_image: ArrayLike) -> MosaicImage:
        height, width = reference_image.shape[:2]
        # Start the mosaic using the reference image as-is.
        canvas = MosaicImage(
            image=reference_image.astype(np.float64),
            mask=np.ones((height, width), dtype=bool),
            weights=np.ones((height, width), dtype=np.float64),
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
            warped, mask, (min_x, min_y) = warp_image_bilinear(image, H)
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
        h = canvas.image.shape[0]
        w = canvas.image.shape[1]
        offset_x, offset_y = canvas.offset

        warped_height, warped_width = warped_shape[:2]
        min_x, min_y = warped_origin
        max_x = min_x + warped_width - 1
        max_y = min_y + warped_height - 1

        new_min_x = min(0, min_x + offset_x)
        new_min_y = min(0, min_y + offset_y)
        new_max_x = max(w - 1, max_x + offset_x)
        new_max_y = max(h - 1, max_y + offset_y)

        new_width = new_max_x - new_min_x + 1
        new_height = new_max_y - new_min_y + 1

        if new_width == w and new_height == h and new_min_x == 0 and new_min_y == 0:
            return canvas

        # Allocate a larger canvas and copy the existing mosaic into place.
        expanded_image = np.zeros((new_height, new_width, canvas.image.shape[2]))
        expanded_mask = np.zeros((new_height, new_width), dtype=bool)
        expanded_weights = np.zeros((new_height, new_width), dtype=np.float64)

        start_x = offset_x - new_min_x
        start_y = offset_y - new_min_y

        expanded_image[start_y:start_y + h, start_x:start_x + w] = canvas.image
        expanded_mask[start_y:start_y + h, start_x:start_x + w] = canvas.mask
        expanded_weights[start_y:start_y + h, start_x:start_x + w] = canvas.weights

        canvas.image = expanded_image
        canvas.mask = expanded_mask
        canvas.weights = expanded_weights
        canvas.offset = (-new_min_x, -new_min_y)
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


__all__ = [
    "MosaicBuilder",
    "FeatherBlendStrategy",
    "LinearFeatherBlendStrategy",
    "LaplacianPyramidBlendStrategy",
]
