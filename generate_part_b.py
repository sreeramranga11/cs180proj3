"""Generate Part B deliverables (automatic feature pipeline)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from generate_deliverables import DATASETS
from output.feature_matching import FeatureMatcher, visualize_correspondences
from output.io_utils import CorrespondenceIO, CorrespondenceSet, load_image, save_image
from output.mosaic import MosaicBuilder


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_corner_overlay(image: np.ndarray, points: np.ndarray, path: Path, title: str) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.scatter(points[:, 1], points[:, 0], s=8, c="#ff4d4d", linewidths=0.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_descriptor_grid(descriptors: np.ndarray, path: Path, max_patches: int = 25) -> None:
    ensure_dir(path.parent)
    if descriptors.shape[0] == 0:
        plt.figure(figsize=(4, 3))
        plt.text(0.5, 0.5, "No descriptors", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return

    num = min(max_patches, descriptors.shape[0])
    patches = descriptors[:num].reshape(num, 8, 8)
    cols = min(5, num)
    rows = int(np.ceil(num / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = np.atleast_2d(axes)
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx >= num:
            continue
        patch = patches[idx]
        ax.imshow(patch, cmap="gray", interpolation="nearest")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def process_dataset(dataset: str, reference_index: int, output_root: Path) -> None:
    filenames = DATASETS[dataset]
    images_dir = Path("images")
    images = [load_image(images_dir / name) for name in filenames]

    from output.feature_matching import HarrisCornerDetector, PatchDescriptorExtractor, RansacHomographyEstimator, match_descriptors
    import numpy as np
    
    class CustomFeatureMatcher(FeatureMatcher):
        def match_images(self, image_a, image_b):
            keypoints_a = self.corner_detector.detect(image_a)
            keypoints_b = self.corner_detector.detect(image_b)

            pts_a, desc_a = self.descriptor_extractor.extract(image_a, keypoints_a)
            pts_b, desc_b = self.descriptor_extractor.extract(image_b, keypoints_b)

            matches = match_descriptors(desc_a, desc_b, ratio=0.85, require_consistency=True, max_distance=0.6)
            if matches.shape[0] < 4:
                raise RuntimeError("Not enough matches were found between the images")

            matched_pts_a = pts_a[matches[:, 0]]
            matched_pts_b = pts_b[matches[:, 1]]

            # Convert from (row, col) to Cartesian (x, y) ordering expected elsewhere.
            points_a = np.column_stack([matched_pts_a[:, 1], matched_pts_a[:, 0]])
            points_b = np.column_stack([matched_pts_b[:, 1], matched_pts_b[:, 0]])

            H, inlier_corr = self.estimator.estimate(points_a, points_b)
            return H, inlier_corr
    
    corner_detector = HarrisCornerDetector(num_features=2000, edge_discard=20, anms_oversample=8000)
    descriptor_extractor = PatchDescriptorExtractor(patch_size=8, window_size=40)
    estimator = RansacHomographyEstimator(num_iterations=8000, inlier_threshold=2.0, random_seed=42)
    matcher = CustomFeatureMatcher(corner_detector=corner_detector, descriptor_extractor=descriptor_extractor, estimator=estimator)
    reference_image = images[reference_index]

    # Corner overlays
    raw_corners, anms_corners = matcher.corner_detector.detect(reference_image, return_all=True)
    corners_dir = output_root / "corners" / dataset
    save_corner_overlay(reference_image, raw_corners, corners_dir / "corners_raw.png", "Harris corners (top strength)")
    save_corner_overlay(reference_image, anms_corners, corners_dir / "corners_anms.png", "Corners after ANMS")

    # Descriptors grid
    _, descriptors = matcher.descriptor_extractor.extract(reference_image, anms_corners)
    save_descriptor_grid(descriptors, output_root / "descriptors" / dataset / "descriptor_grid.png")

    # Matching + correspondences
    correspondences: list[CorrespondenceSet] = []
    matches_dir = output_root / "matches" / dataset
    ensure_dir(matches_dir)

    reference_path = images_dir / filenames[reference_index]
    for idx, name in enumerate(filenames):
        if idx == reference_index:
            continue
        image_path = images_dir / name
        image = images[idx]
        H, corr = matcher.match_images(image, reference_image)
        correspondences.append(corr)

        pair_name = f"{Path(name).stem}_to_{Path(filenames[reference_index]).stem}"
        visualize_correspondences(image, reference_image, corr, matches_dir / f"{pair_name}.png")
        corr_dir = output_root / "correspondences" / dataset
        ensure_dir(corr_dir)
        CorrespondenceIO.to_file(corr, corr_dir / f"{pair_name}.csv")

    # Automatic mosaic
    auto_dir = output_root / "mosaics"
    ensure_dir(auto_dir)
    from output.mosaic import LaplacianPyramidBlendStrategy
    builder = MosaicBuilder(reference_shape=reference_image.shape, blend_strategy=LaplacianPyramidBlendStrategy(num_levels=3))
    additional_images = [images[idx] for idx in range(len(images)) if idx != reference_index]
    mosaic = builder.build(reference_image, additional_images, correspondences)
    save_image(auto_dir / f"{dataset}_automatic.png", mosaic)


def copy_manual_mosaics(datasets: Iterable[str], output_root: Path) -> None:
    from shutil import copy2

    manual_dir = Path("deliverables/mosaics")
    target_dir = output_root / "mosaics"
    ensure_dir(target_dir)
    for dataset in datasets:
        src = manual_dir / f"{dataset}_mosaic.png"
        if src.exists():
            dst = target_dir / f"{dataset}_manual.png"
            copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Part B automatic stitching deliverables")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["street", "stairway", "classroom"],
        choices=sorted(DATASETS.keys()),
        help="Datasets to process (default: street stairway classroom)",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=1,
        help="Index of reference image within each triplet (default: 1)",
    )
    args = parser.parse_args()

    output_root = Path("3b_deliverables")
    ensure_dir(output_root)

    for dataset in args.datasets:
        process_dataset(dataset, args.reference_index, output_root)

    copy_manual_mosaics(args.datasets, output_root)


if __name__ == "__main__":
    main()
