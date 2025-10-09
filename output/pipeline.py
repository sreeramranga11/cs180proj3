from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from .debug import DebugLogger
from .homography import compute_homography
from .io_utils import CorrespondenceIO, CorrespondenceSet, load_image, save_image
from .mosaic import MosaicBuilder
from .warping import warp_image_bilinear, warp_image_nearest_neighbor

ArrayLike = np.ndarray


@dataclass
class RectificationExample:
    source_image: Path
    src_points: Sequence[Tuple[float, float]]
    dst_points: Sequence[Tuple[float, float]]
    output_path: Path


@dataclass
class MosaicExample:
    reference_image: Path
    additional_images: Sequence[Path]
    correspondences: Sequence[CorrespondenceSet]
    output_path: Path


class MosaicingPipeline:
    # Coordinates the stages of the mosaicing assignment

    def __init__(self, debug_root: Path | None = None) -> None:
        # Centralize logging so every step can drop artifacts in one place.
        self.logger = DebugLogger(root=debug_root)

    def compute_homography_from_file(self, path: Path) -> np.ndarray:
        correspondences = CorrespondenceIO.from_file(path)
        H = compute_homography(correspondences)
        self.logger.save_matrix(H, name=path.stem)
        self.logger.log("Computed homography", source=str(path), determinant=float(np.linalg.det(H)))
        return H

    def warp_with_both_interpolations(
        self, image_path: Path, H: np.ndarray, output_dir: Path
    ) -> dict[str, Path]:
        image = load_image(image_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run both interpolation schemes so we can compare their artifacts.
        nn_image, _, _ = warp_image_nearest_neighbor(image, H)
        bilinear_image, _, _ = warp_image_bilinear(image, H)

        nn_path = output_dir / f"{image_path.stem}_nn.png"
        bilinear_path = output_dir / f"{image_path.stem}_bilinear.png"

        save_image(nn_path, nn_image)
        save_image(bilinear_path, bilinear_image)

        self.logger.save_image(nn_image, name=f"{image_path.stem}_nn")
        self.logger.save_image(bilinear_image, name=f"{image_path.stem}_bilinear")

        return {"nearest": nn_path, "bilinear": bilinear_path}

    def build_mosaic(self, example: MosaicExample) -> Path:
        reference = load_image(example.reference_image)
        additional = [load_image(path) for path in example.additional_images]

        builder = MosaicBuilder(reference_shape=reference.shape, logger=self.logger)
        mosaic = builder.build(reference, additional, example.correspondences)
        save_image(example.output_path, mosaic)
        self.logger.save_image(mosaic, name=example.output_path.stem)
        self.logger.log(
            "Saved mosaic", output=str(example.output_path), num_images=len(example.additional_images) + 1
        )
        return example.output_path

    def rectify(self, example: RectificationExample) -> Path:
        image = load_image(example.source_image)
        correspondences = CorrespondenceIO.from_lists(example.src_points, example.dst_points)
        H = compute_homography(correspondences)
        # Rectification is just another warp; reuse the bilinear helper.
        rectified, _, _ = warp_image_bilinear(image, H)
        save_image(example.output_path, rectified)
        self.logger.save_matrix(H, name=f"rectify_{example.source_image.stem}")
        self.logger.save_image(rectified, name=f"rectified_{example.source_image.stem}")
        self.logger.log("Rectified image", source=str(example.source_image), output=str(example.output_path))
        return example.output_path


__all__ = [
    "MosaicingPipeline",
    "MosaicExample",
    "RectificationExample",
]
