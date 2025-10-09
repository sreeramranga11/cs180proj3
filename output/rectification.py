# Rectification helper built on top of the core warping utils

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .debug import DebugLogger
from .homography import compute_homography
from .io_utils import CorrespondenceIO
from .warping import warp_image_bilinear

ArrayLike = np.ndarray


def rectify_image(
    image_path: Path,
    src_points: Iterable[Tuple[float, float]],
    dst_points: Iterable[Tuple[float, float]],
    output_path: Path | None = None,
    logger: DebugLogger | None = None,
) -> ArrayLike:
    # rectify a single image given source and destination correspondences

    from .io_utils import load_image, save_image

    image = load_image(image_path)
    correspondences = CorrespondenceIO.from_lists(src_points, dst_points)
    H = compute_homography(correspondences)

    if logger is not None:
        # Stash the inputs so we can inspect the rectification later on.
        logger.save_correspondences(src_points, dst_points, name="rectification")
        logger.save_matrix(H, name="rectification_homography")

    warped, mask, _ = warp_image_bilinear(image, H)

    if output_path is not None:
        save_image(output_path, warped)
        if logger is not None:
            # log when files were written
            logger.log("Rectified image saved", output=str(output_path))

    return warped


__all__ = ["rectify_image"]
