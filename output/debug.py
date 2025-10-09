# Debugging utils for the mosaicing backend

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass
class DebugLogger:
    #Writes debug info and artifacts to disk. Supports custom root and context metadata

    root: Optional[Path] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        default_root = Path(__file__).resolve().parent.parent / "debug"
        self.root = Path(self.root) if self.root is not None else default_root
        self.root.mkdir(parents=True, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    def _write_json(self, data: Dict[str, Any], filename: str) -> Path:
        # Keep the JSON dumps tidy so they are easy to read in a text editor.
        path = self.root / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def log(self, message: str, **metadata: Any) -> Path:
        # Write a log entry containing a message and metadata

        payload = {
            "timestamp": self._timestamp(),
            "message": message,
            "context": self.context,
            "metadata": metadata,
        }
        filename = f"log_{self._timestamp()}.json"
        return self._write_json(payload, filename)

    def save_matrix(self, matrix: Iterable[Iterable[float]], name: str) -> Path:
        # matrix (e.g., homography) to a JSON file

        payload = {
            "timestamp": self._timestamp(),
            "context": self.context,
            "matrix": [list(map(float, row)) for row in matrix],
        }
        filename = f"matrix_{name}_{self._timestamp()}.json"
        return self._write_json(payload, filename)

    def save_correspondences(
        self,
        points_a: Iterable[Iterable[float]],
        points_b: Iterable[Iterable[float]],
        name: str,
    ) -> Path:
        # save corresponding points for debugging

        payload = {
            "timestamp": self._timestamp(),
            "context": self.context,
            "points_a": [list(map(float, pt)) for pt in points_a],
            "points_b": [list(map(float, pt)) for pt in points_b],
        }
        filename = f"correspondences_{name}_{self._timestamp()}.json"
        return self._write_json(payload, filename)

    def save_image(self, image, name: str) -> Optional[Path]:
        # save an image array for debugging

        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            return None

        filename = f"image_{name}_{self._timestamp()}.png"
        path = self.root / filename
        imageio.imwrite(path, image)
        return path


__all__ = ["DebugLogger"]
