# Command line entry point that generates all mosaicing assignment deliverables

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from output.auto_rectify import AutomaticRectifier
from output.feature_matching import FeatureMatcher, save_correspondences, visualize_correspondences
from output.io_utils import CorrespondenceIO, CorrespondenceSet, load_image, save_image
from output.pipeline import MosaicExample, MosaicingPipeline
from output.warping import compute_warp_bounds, warp_image_bilinear, warp_image_nearest_neighbor


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
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        help="Subset of datasets to process. Defaults to all datasets.",
    )
    parser.add_argument(
        "--skip-mosaics",
        action="store_true",
        help="Skip mosaic generation (useful when only rectifications are needed).",
    )
    parser.add_argument(
        "--skip-rectifications",
        action="store_true",
        help="Skip automatic rectifications (useful when only mosaics are needed).",
    )
    parser.add_argument(
        "--rectification-images",
        nargs="+",
        help="Specific image filenames to rectify. Defaults to the standard set.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", "match", "mosaic", "rectify"],
        default=["all"],
        help="Pipeline stages to run. Combine values for fine-grained control (default: all stages).",
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


def copy_source_images(images_dir: Path, sources_dir: Path, datasets: Sequence[str]) -> None:
    import shutil

    for dataset in datasets:
        filenames = DATASETS[dataset]
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
        bounds = compute_warp_bounds(image.shape, H, sample_points=corr.points_a)
        nn_image, _, _ = warp_image_nearest_neighbor(image, H, bounds=bounds)
        bilinear_image, _, _ = warp_image_bilinear(image, H, bounds=bounds)
        save_image(warp_dir / "nearest_neighbor.png", nn_image)
        save_image(warp_dir / "bilinear.png", bilinear_image)

    return correspondences


def load_saved_correspondences(
    dataset: str,
    filenames: Sequence[str],
    reference_index: int,
    dirs: Dict[str, Path],
) -> List[CorrespondenceSet]:
    dataset_corr_dir = dirs["correspondences"] / dataset
    correspondences: List[CorrespondenceSet] = []
    for idx, filename in enumerate(filenames):
        if idx == reference_index:
            continue
        pair_name = f"{Path(filename).stem}_to_{Path(filenames[reference_index]).stem}"
        path = dataset_corr_dir / f"{pair_name}_correspondences.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Correspondence file {path} is missing. Run the match stage before building mosaics."
            )
        correspondences.append(CorrespondenceIO.from_file(path))
    return correspondences


def build_mosaic_for_dataset(
    dataset: str,
    correspondences: Sequence[CorrespondenceSet],
    pipeline: MosaicingPipeline,
    dirs: Dict[str, Path],
    images_dir: Path,
    reference_index: int,
) -> None:
    print(f"\nBuilding mosaic for dataset: {dataset}")
    filenames = DATASETS[dataset]
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


def run_rectifications(
    rectifier: AutomaticRectifier,
    dirs: Dict[str, Path],
    images_dir: Path,
    rectification_images: Sequence[str],
) -> None:
    rect_dir = dirs["rectifications"]
    rect_dir.mkdir(parents=True, exist_ok=True)

    for filename in rectification_images:
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

    selected_datasets = args.datasets if args.datasets is not None else list(DATASETS.keys())
    rectification_images = args.rectification_images
    if rectification_images is None:
        if args.datasets:
            rectification_images = [DATASETS[name][0] for name in selected_datasets if DATASETS[name]]
        else:
            rectification_images = RECTIFICATION_IMAGES

    stage_set = set(args.stages)
    if "all" in stage_set:
        stage_set = {"match", "mosaic", "rectify"}

    dirs = ensure_output_structure(args.output_dir)
    copy_source_images(args.images_dir, dirs["sources"], selected_datasets)

    run_match = "match" in stage_set
    run_mosaic = "mosaic" in stage_set and not args.skip_mosaics
    run_rectifications_flag = "rectify" in stage_set and not args.skip_rectifications

    match_results: Dict[str, Sequence[CorrespondenceSet]] = {}

    if run_match:
        matcher = FeatureMatcher()
        for dataset in selected_datasets:
            print(f"\nProcessing dataset: {dataset}")
            correspondences = match_images_for_dataset(
                dataset,
                DATASETS[dataset],
                args.images_dir,
                matcher,
                args.reference_index,
                dirs,
            )
            match_results[dataset] = correspondences

    if run_mosaic:
        pipeline = MosaicingPipeline(debug_root=dirs["root"] / "debug")
        for dataset in selected_datasets:
            if dataset in match_results:
                correspondences = match_results[dataset]
            else:
                correspondences = load_saved_correspondences(
                    dataset,
                    DATASETS[dataset],
                    args.reference_index,
                    dirs,
                )
                print(f"\nLoaded saved correspondences for dataset: {dataset}")
            build_mosaic_for_dataset(
                dataset,
                correspondences,
                pipeline,
                dirs,
                args.images_dir,
                args.reference_index,
            )
    elif "mosaic" in stage_set and args.skip_mosaics:
        print("Skipping mosaic generation (--skip-mosaics).")

    if "rectify" in stage_set:
        if run_rectifications_flag and rectification_images:
            rectifier = AutomaticRectifier()
            run_rectifications(rectifier, dirs, args.images_dir, rectification_images)
        elif args.skip_rectifications:
            print("Skipping rectifications (--skip-rectifications).")
        else:
            print("No rectification images selected; skipping rectifications.")

    print("\nDeliverables generated in", args.output_dir.resolve())


if __name__ == "__main__":
    main()
