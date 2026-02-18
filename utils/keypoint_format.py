from __future__ import annotations

import numpy as np

POSE_POINTS = 25
FACE_POINTS = 70
HAND_POINTS = 21
TOTAL_POINTS = POSE_POINTS + FACE_POINTS + HAND_POINTS + HAND_POINTS
FEATURE_DIM = TOTAL_POINTS * 3


def flatten_points(points: np.ndarray, use_confidence: bool = True) -> np.ndarray:
    if use_confidence:
        return points.reshape(-1).astype(np.float32)
    return points[:, :2].reshape(-1).astype(np.float32)


def normalize_openpose_like(points: np.ndarray) -> np.ndarray:
    """
    Normalize OpenPose-like 2D skeleton:
    - Keep canonical topology/order unchanged.
    - Translate XY around body center (neck preferred).
    - Scale XY by shoulder distance (fallback: bbox diagonal).
    - Preserve confidence channel.
    """
    out = points.astype(np.float32).copy()
    if out.shape != (TOTAL_POINTS, 3):
        raise ValueError(f"Expected shape {(TOTAL_POINTS, 3)}, got {out.shape}")

    xy = out[:, :2]
    conf = out[:, 2]
    valid = conf > 0.0
    if not np.any(valid):
        return out

    neck_idx = 1
    r_shoulder_idx = 2
    l_shoulder_idx = 5

    if conf[neck_idx] > 0:
        center = xy[neck_idx]
    elif conf[r_shoulder_idx] > 0 and conf[l_shoulder_idx] > 0:
        center = (xy[r_shoulder_idx] + xy[l_shoulder_idx]) * 0.5
    else:
        center = np.mean(xy[valid], axis=0)

    scale = 0.0
    if conf[r_shoulder_idx] > 0 and conf[l_shoulder_idx] > 0:
        scale = float(np.linalg.norm(xy[r_shoulder_idx] - xy[l_shoulder_idx]))
    if scale < 1e-6:
        mins = np.min(xy[valid], axis=0)
        maxs = np.max(xy[valid], axis=0)
        scale = float(np.linalg.norm(maxs - mins))
    if scale < 1e-6:
        scale = 1.0

    xy[valid] = (xy[valid] - center) / scale
    out[:, :2] = xy
    return out

