"""Cluster-center trajectory helpers for the enhanced trajectory GUI."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

from .constants import *
from .math_utils import *
from .speed_utils import *

def _prepare_cluster_preview_trajectory(
    xyz: np.ndarray,
    initial_speed_mps: Optional[float],
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> Optional[np.ndarray]:
    """Apply t0-speed anchoring and curvature smoothing to a cluster preview path."""
    speed_limited = _acceleration_limited_resample_path(
        xyz,
        dt_seconds=dt_seconds,
        initial_speed_mps=initial_speed_mps,
        anchor_initial_speed=True,
    )
    if speed_limited is None:
        return None

    for passes in range(CLUSTER_CURVATURE_SMOOTH_PASSES, 0, -1):
        curvature_smoothed = _smooth_curvature_preserving_ends(speed_limited, passes=passes)
        final_limited = _acceleration_limited_resample_path(
            curvature_smoothed,
            dt_seconds=dt_seconds,
            initial_speed_mps=initial_speed_mps,
            anchor_initial_speed=True,
        )
        if final_limited is not None:
            return final_limited
    return speed_limited

def _trajectory_quality_diagnostics(
    xyz: np.ndarray,
    dt_seconds: float = TRAJ_DT_SECONDS,
    acceleration_mps2: Optional[np.ndarray] = None,
) -> dict[str, object]:
    """Check acceleration threshold and position jump problems from ego-local xyz."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        return {
            "bad_accel_indices": [],
            "jump_indices": [],
            "max_accel": 0.0,
            "min_accel": 0.0,
            "max_step_m": 0.0,
            "max_step_speed": 0.0,
            "ok": True,
        }

    step_distance, speed, scalar_accel = _scalar_speed_acceleration_from_xy(
        points[:, :2],
        dt_seconds,
    )
    if acceleration_mps2 is not None:
        source_accel = np.asarray(acceleration_mps2, dtype=np.float64).reshape(-1)
        if len(source_accel) == len(points) and np.isfinite(source_accel).all():
            scalar_accel = source_accel
    bad_accel = np.flatnonzero(
        (scalar_accel < TRAJ_ACCEL_MIN_MPS2) | (scalar_accel > TRAJ_ACCEL_MAX_MPS2)
    )
    jumps = np.flatnonzero(
        (step_distance > TRAJ_MAX_POSITION_STEP_M) | (speed > TRAJ_MAX_STEP_SPEED_MPS)
    )
    return {
        "bad_accel_indices": [int(idx) for idx in bad_accel.tolist()],
        "jump_indices": [int(idx) for idx in jumps.tolist()],
        "max_accel": float(np.nanmax(scalar_accel)) if len(scalar_accel) else 0.0,
        "min_accel": float(np.nanmin(scalar_accel)) if len(scalar_accel) else 0.0,
        "max_step_m": float(np.nanmax(step_distance)) if len(step_distance) else 0.0,
        "max_step_speed": float(np.nanmax(speed)) if len(speed) else 0.0,
        "ok": len(bad_accel) == 0 and len(jumps) == 0,
    }

def _cluster_drag_candidate(
    original_xyz: np.ndarray,
    target_xy: tuple[float, float],
    dt_seconds: float = TRAJ_DT_SECONDS,
    initial_speed_mps: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Temporarily reshape a cluster center to a dragged endpoint if dynamics stay sane."""
    original = np.asarray(original_xyz, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] < 4 or original.shape[1] < 2:
        return None

    original_xy = original[:, :2]
    target = np.asarray(target_xy, dtype=np.float64)
    if not np.isfinite(target).all() or target[0] < -0.25:
        return None

    endpoint_delta = target - original_xy[-1]
    path_steps = np.linalg.norm(
        np.diff(np.vstack([np.zeros((1, 2), dtype=np.float64), original_xy]), axis=0),
        axis=1,
    )
    path_length = float(path_steps.sum())
    max_endpoint_delta = max(8.0, min(22.0, 0.65 * max(path_length, 1.0)))
    if float(np.linalg.norm(endpoint_delta)) > max_endpoint_delta:
        return None

    n_steps = original_xy.shape[0]
    t = np.linspace(0.0, 1.0, n_steps)
    smooth = 3.0 * t * t - 2.0 * t * t * t
    knot_t = np.linspace(0.0, 1.0, 7)
    initial_knot_offsets = np.outer(3.0 * knot_t * knot_t - 2.0 * knot_t * knot_t * knot_t, endpoint_delta)
    interior_initial = initial_knot_offsets[1:-1].reshape(-1)

    original_speed, original_accel, original_curvature = _trajectory_dynamics_from_xy(
        original_xy,
        dt_seconds,
    )

    def candidate_from_vars(values: np.ndarray) -> np.ndarray:
        knot_offsets = np.zeros((len(knot_t), 2), dtype=np.float64)
        knot_offsets[1:-1] = values.reshape(-1, 2)
        knot_offsets[-1] = endpoint_delta
        offsets = np.column_stack([
            np.interp(t, knot_t, knot_offsets[:, 0]),
            np.interp(t, knot_t, knot_offsets[:, 1]),
        ])
        return original_xy + offsets

    def objective(values: np.ndarray) -> float:
        candidate_xy = candidate_from_vars(values)
        speed, accel, curvature = _trajectory_dynamics_from_xy(candidate_xy, dt_seconds)
        if not np.isfinite(candidate_xy).all():
            return 1e12
        accel_cost = np.mean((accel - original_accel) ** 2)
        curvature_cost = np.mean((curvature - original_curvature) ** 2)
        deformation = candidate_xy - original_xy
        deviation_cost = np.mean(deformation * deformation)
        speed_cost = np.mean((speed - original_speed) ** 2) * 0.02
        return (
            CLUSTER_DRAG_ACCELERATION_WEIGHT * accel_cost
            + CLUSTER_DRAG_CURVATURE_WEIGHT * curvature_cost
            + CLUSTER_DRAG_DEVIATION_WEIGHT * deviation_cost
            + speed_cost
        )

    optimized_values = interior_initial
    if minimize is not None:
        margin = max(3.0, float(np.linalg.norm(endpoint_delta)) * 0.75)
        lower = np.minimum(0.0, endpoint_delta) - margin
        upper = np.maximum(0.0, endpoint_delta) + margin
        bounds = [(float(lower[idx % 2]), float(upper[idx % 2])) for idx in range(len(interior_initial))]
        result = minimize(
            objective,
            interior_initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 30, "ftol": 1e-4, "maxls": 12},
        )
        if result.success and np.isfinite(result.fun):
            optimized_values = np.asarray(result.x, dtype=np.float64)

    candidate_xy = candidate_from_vars(optimized_values)
    if not np.isfinite(candidate_xy).all():
        return None
    candidate_prelimit = original.copy()
    candidate_prelimit[:, :2] = candidate_xy
    if candidate_prelimit.shape[1] > 2:
        candidate_prelimit[:, 2] = 0.0
    candidate_limited = _prepare_cluster_preview_trajectory(
        candidate_prelimit,
        initial_speed_mps=initial_speed_mps,
        dt_seconds=dt_seconds,
    )
    if candidate_limited is None:
        return None
    candidate_xy = candidate_limited[:, :2]
    if float(np.linalg.norm(candidate_xy[-1] - target)) > 0.2:
        return None
    if np.nanmin(candidate_xy[:, 0]) < -1.0:
        return None

    speed, accel, curvature = _trajectory_dynamics_from_xy(candidate_xy, dt_seconds)
    _, _, scalar_accel = _scalar_speed_acceleration_from_xy(candidate_xy, dt_seconds)
    max_speed = float(np.nanmax(speed)) if len(speed) else 0.0
    max_accel = float(np.nanmax(np.linalg.norm(accel, axis=1))) if len(accel) else 0.0
    max_curvature = float(np.nanmax(np.abs(curvature))) if len(curvature) else 0.0
    orig_max_speed = float(np.nanmax(original_speed)) if len(original_speed) else 0.0
    orig_max_accel = float(np.nanmax(np.linalg.norm(original_accel, axis=1))) if len(original_accel) else 0.0
    orig_max_curvature = float(np.nanmax(np.abs(original_curvature))) if len(original_curvature) else 0.0

    if max_speed > max(28.0, orig_max_speed * 1.8 + 4.0):
        return None
    if (
        np.nanmin(scalar_accel) < TRAJ_ACCEL_MIN_MPS2 - 0.2
        or np.nanmax(scalar_accel) > TRAJ_ACCEL_MAX_MPS2 + 0.2
    ):
        return None
    if max_accel > max(65.0, orig_max_accel * 2.4 + 8.0):
        return None
    if max_curvature > max(1.25, orig_max_curvature * 3.0 + 0.08):
        return None

    return candidate_limited

__all__ = [name for name in globals() if (name.startswith("_") and not name.startswith("__")) or name.isupper()]
