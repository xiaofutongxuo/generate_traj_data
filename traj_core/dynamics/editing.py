"""Geometry editing helpers for saved pseudo-GT trajectories."""

from __future__ import annotations

import numpy as np

from .metrics import as_xyz


def editable_trajectory_keyframes(
    num_steps: int,
    interval: int = 8,
    include_start: bool = False,
) -> list[int]:
    """Return sparse future-frame handles for direct trajectory editing."""
    total = max(0, int(num_steps))
    if total <= 0:
        return []
    step = max(1, int(interval))
    start = 0 if include_start else min(step, total - 1)
    indices = list(range(start, total, step))
    last = total - 1
    if last not in indices:
        indices.append(last)
    return sorted(set(int(idx) for idx in indices if 0 <= int(idx) < total))


def deform_trajectory_by_keyframe_drag(
    points_xyz: np.ndarray,
    frame_idx: int,
    target_xy: tuple[float, float],
    influence_radius_frames: int = 10,
    lock_start: bool = True,
) -> np.ndarray:
    """Move one keyframe and smoothly distribute the displacement over nearby frames."""
    xyz = as_xyz(points_xyz)
    if len(xyz) == 0:
        return xyz.copy()

    idx = int(np.clip(frame_idx, 0, len(xyz) - 1))
    target = np.asarray(target_xy, dtype=np.float64).reshape(2)
    edited = xyz.copy()
    displacement = target - edited[idx, :2]
    if not np.all(np.isfinite(displacement)):
        return edited

    radius = max(1.0, float(influence_radius_frames))
    frame_distance = np.abs(np.arange(len(edited), dtype=np.float64) - float(idx))
    weights = np.exp(-0.5 * (frame_distance / radius) ** 2)
    if lock_start and len(weights):
        weights[0] = 0.0
    if weights[idx] <= 1e-9:
        weights[idx] = 1.0
    weights = weights / weights[idx]

    edited[:, :2] += weights[:, None] * displacement[None, :]
    edited[idx, :2] = target
    if lock_start:
        edited[0] = xyz[0]
    edited[:, 2] = 0.0
    return edited
