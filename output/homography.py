from __future__ import annotations

import numpy as np
from numpy.linalg import svd

from .io_utils import CorrespondenceSet


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # norm points to improve numerical stability

    # Move the coordinates so that the centroid sits at the origin.
    centroid = points.mean(axis=0)
    shifted = points - centroid

    # Scaling the coordinates to unit average distance keeps the linear
    # system well conditioned and matches the textbook DLT derivation.
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2.0) / mean_dist if mean_dist > 0 else 1.0

    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    # Promote to homogeneous coordinates so that we can apply the similarity
    # transform with a single matrix multiplication.
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized = (transform @ homogeneous.T).T
    return normalized, transform


def construct_dlt_matrix(correspondences: CorrespondenceSet) -> np.ndarray:
    """Construct the linear system for DLT."""

    pts_a = correspondences.points_a
    pts_b = correspondences.points_b
    num_points = correspondences.count

    A = []
    for i in range(num_points):
        x, y = pts_a[i]
        xp, yp = pts_b[i]
        # Each correspondence contributes two rows describing the mapping for
        # x and y respectively. The layout matches the standard DLT matrix.
        A.append([0, 0, 0, -x, -y, -1, yp * x, yp * y, yp])
        A.append([x, y, 1, 0, 0, 0, -xp * x, -xp * y, -xp])
    return np.asarray(A, dtype=np.float64)


def compute_homography(correspondences: CorrespondenceSet) -> np.ndarray:
    # compute projective transform mapping points_a to points_b

    correspondences = correspondences.as_float()
    pts_a_norm, Ta = normalize_points(correspondences.points_a)
    pts_b_norm, Tb = normalize_points(correspondences.points_b)
    # Work in normalized coordinates when forming the linear system.
    normalized = CorrespondenceSet(pts_a_norm[:, :2], pts_b_norm[:, :2])

    A = construct_dlt_matrix(normalized)
    _, _, vh = svd(A)
    h = vh[-1, :]
    H_norm = h.reshape(3, 3)

    # Undo the normalization to recover the homography in the original
    # coordinate system. Finally, scale so that H[2, 2] is 1 for consistency.
    H = np.linalg.inv(Tb) @ H_norm @ Ta
    H /= H[2, 2]
    return H


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    #Apply homography to Cartesian coordinates

    # Convert to homogeneous coordinates before applying the transform.
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    mapped = (H @ homogeneous.T).T
    # Bring the coordinates back to Cartesian form via perspective divide.
    w = mapped[:, 2]
    result = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.abs(w) > 1e-12
    if np.any(valid):
        result[valid] = mapped[valid, :2] / w[valid, None]
    return result


__all__ = [
    "compute_homography",
    "normalize_points",
    "construct_dlt_matrix",
    "apply_homography",
]
