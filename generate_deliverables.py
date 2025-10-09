# Command line entry point that generates all mosaicing assignment deliverables

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from output.auto_rectify import AutomaticRectifier
from output.feature_matching import FeatureMatcher, save_correspondences, visualize_correspondences
from output.io_utils import CorrespondenceSet, load_image, save_image
from output.pipeline import MosaicExample, MosaicingPipeline
from output.warping import warp_image_bilinear, warp_image_nearest_neighbor


DATASETS: Dict[str, List[str]] = {
    "bedroom": ["bedroom1.jpg", "bedroom2.jpg", "bedroom3.jpg"],
    "stairway": ["stairway1.jpg", "stairway2.jpg", "stairway3.jpg"],
    "street": ["street1.jpg", "street2.jpg", "street3.jpg"],
}

RECTIFICATION_IMAGES = ["bedroom1.jpg", "stairway1.jpg", "street1.jpg"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Directory containing source photographs")
    parser.add_argument("--output-dir", type=Path, default=Path("deliverables"), help="Directory to store generated artifacts")
    parser.add_argument(
        "--reference-index",
        type=int,
        default=1,
        help="Index (0-based) of the reference image within each triplet. Defaults to the middle image.",
    )
    return parser


def ensure_output_structure(root: Path) -> Dict[str, Path]:
    subdirs = {
        "root": root,
        "mosaics": root / "mosaics",
        "correspondences": root / "correspondences",
        "warps": root / "warps",
        "rectifications": root / "rectifications",
        "sources": root / "source_images",
        "metadata": root / "metadata",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def copy_source_images(images_dir: Path, sources_dir: Path) -> None:
    import shutil

    for dataset, filenames in DATASETS.items():
        dataset_dir = sources_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src = images_dir / filename
            dst = dataset_dir / filename
            if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(src, dst)


def match_images_for_dataset(
    dataset: str,
    filenames: List[str],
    images_dir: Path,
    matcher: FeatureMatcher,
    reference_index: int,
    dirs: Dict[str, Path],
) -> List[CorrespondenceSet]:
    if not 0 <= reference_index < len(filenames):
        raise ValueError(
            f"Reference index {reference_index} is out of range for dataset '{dataset}'"
        )

    reference_path = images_dir / filenames[reference_index]
    reference_image = load_image(reference_path)

    correspondences: List[CorrespondenceSet] = []
    dataset_corr_dir = dirs["correspondences"] / dataset
    dataset_corr_dir.mkdir(parents=True, exist_ok=True)

    for idx, filename in enumerate(filenames):
        if idx == reference_index:
            continue
        image_path = images_dir / filename
        image = load_image(image_path)

        print(f"Matching {filename} -> {filenames[reference_index]}")
        H, corr = matcher.match_images(image, reference_image)
        correspondences.append(corr)

        pair_name = f"{Path(filename).stem}_to_{Path(filenames[reference_index]).stem}"
        save_correspondences(corr, dataset_corr_dir, pair_name)
        viz_path = dataset_corr_dir / f"{pair_name}_visualization.png"
        visualize_correspondences(image, reference_image, corr, viz_path)
        np.savetxt(dataset_corr_dir / f"{pair_name}_homography.txt", H, fmt="%.6f")

        warp_dir = dirs["warps"] / dataset / Path(filename).stem
        warp_dir.mkdir(parents=True, exist_ok=True)
        nn_image, _, _ = warp_image_nearest_neighbor(image, H)
        bilinear_image, _, _ = warp_image_bilinear(image, H)
        save_image(warp_dir / "nearest_neighbor.png", nn_image)
        save_image(warp_dir / "bilinear.png", bilinear_image)

    return correspondences


def build_mosaics(
    pipeline: MosaicingPipeline,
    matcher: FeatureMatcher,
    dirs: Dict[str, Path],
    images_dir: Path,
    reference_index: int,
) -> None:
    for dataset, filenames in DATASETS.items():
        print(f"\nProcessing dataset: {dataset}")
        correspondences = match_images_for_dataset(dataset, filenames, images_dir, matcher, reference_index, dirs)

        mosaic_dir = dirs["mosaics"]
        mosaic_dir.mkdir(parents=True, exist_ok=True)
        output_path = mosaic_dir / f"{dataset}_mosaic.png"

        example = MosaicExample(
            reference_image=images_dir / filenames[reference_index],
            additional_images=[images_dir / fname for idx, fname in enumerate(filenames) if idx != reference_index],
            correspondences=correspondences,
            output_path=output_path,
        )
        pipeline.build_mosaic(example)

        metadata = {
            "dataset": dataset,
            "reference": filenames[reference_index],
            "additional_images": [fname for idx, fname in enumerate(filenames) if idx != reference_index],
            "mosaic_path": str(output_path),
        }
        meta_path = dirs["metadata"] / f"{dataset}_mosaic.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)


def run_rectifications(rectifier: AutomaticRectifier, dirs: Dict[str, Path], images_dir: Path) -> None:
    rect_dir = dirs["rectifications"]
    rect_dir.mkdir(parents=True, exist_ok=True)

    for filename in RECTIFICATION_IMAGES:
        image_path = images_dir / filename
        name = Path(filename).stem
        print(f"Rectifying {filename}")
        try:
            rectifier.rectify(image_path, rect_dir / name, name)
        except RuntimeError as exc:
            print(f"  [warning] Failed to rectify {filename}: {exc}")


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    dirs = ensure_output_structure(args.output_dir)
    copy_source_images(args.images_dir, dirs["sources"])

    pipeline = MosaicingPipeline(debug_root=dirs["root"] / "debug")
    matcher = FeatureMatcher()
    rectifier = AutomaticRectifier()

    build_mosaics(pipeline, matcher, dirs, args.images_dir, args.reference_index)
    run_rectifications(rectifier, dirs, args.images_dir)

    print("\nDeliverables generated in", args.output_dir.resolve())


if __name__ == "__main__":
    main()
