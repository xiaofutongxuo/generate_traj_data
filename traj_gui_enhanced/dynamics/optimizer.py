"""Conservative pseudo-GT trajectory repair utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..math_utils import _smooth_curvature_preserving_ends
from ..speed_utils import _acceleration_limited_resample_path
from .diagnostics import DynamicsDiagnostics, diagnose_trajectory_dynamics
from .limits import DynamicsLimits
from .metrics import as_xyz


@dataclass(frozen=True)
class DynamicsOptimizationResult:
    xyz: np.ndarray
    ok: bool
    diagnostics: DynamicsDiagnostics
    original_diagnostics: DynamicsDiagnostics
    message: str


def _diagnostic_score(diagnostics: DynamicsDiagnostics, limits: DynamicsLimits) -> float:
    metrics = diagnostics.metrics
    score = 0.0
    score += max(0.0, metrics["max_speed_mps"] / limits.max_speed_mps - 1.0) * 10.0
    score += max(0.0, metrics["max_step_m"] / limits.max_step_m - 1.0) * 10.0
    score += max(0.0, metrics["max_accel_mps2"] / limits.max_accel_mps2 - 1.0) * 5.0
    score += max(0.0, metrics["min_accel_mps2"] / limits.min_accel_mps2 - 1.0) * 5.0
    score += max(0.0, metrics["max_abs_jerk_mps3"] / limits.max_jerk_mps3 - 1.0) * 2.0
    score += max(0.0, metrics["max_abs_curvature_1pm"] / limits.max_curvature_1pm - 1.0) * 8.0
    score += sum(len(indices) for indices in diagnostics.violations.values())
    return float(score)


def _candidate_geometry(points_xyz: np.ndarray, passes: int) -> np.ndarray:
    if passes <= 0:
        candidate = points_xyz.copy()
    else:
        candidate = _smooth_curvature_preserving_ends(points_xyz, passes=passes)
    candidate[:, 2] = 0.0
    return candidate


def optimize_pseudo_gt_trajectory(
    points_xyz: np.ndarray,
    limits: DynamicsLimits | None = None,
    max_smoothing_passes: int = 24,
) -> DynamicsOptimizationResult:
    """Smooth and acceleration-limit one generated trajectory.

    The function is intentionally source-agnostic: callers should only pass non-GT rows.
    """
    limits = limits or DynamicsLimits()
    xyz = as_xyz(points_xyz)
    original_diagnostics = diagnose_trajectory_dynamics(xyz, limits=limits)
    best_xyz = xyz.copy()
    best_diagnostics = original_diagnostics
    best_score = _diagnostic_score(best_diagnostics, limits)

    if original_diagnostics.ok:
        return DynamicsOptimizationResult(
            xyz=best_xyz,
            ok=True,
            diagnostics=best_diagnostics,
            original_diagnostics=original_diagnostics,
            message="trajectory already within dynamics limits",
        )

    if len(xyz) < 2:
        return DynamicsOptimizationResult(
            xyz=best_xyz,
            ok=best_diagnostics.ok,
            diagnostics=best_diagnostics,
            original_diagnostics=original_diagnostics,
            message="trajectory has fewer than two points",
        )

    for passes in range(0, max(0, int(max_smoothing_passes)) + 1):
        geometry = _candidate_geometry(xyz, passes=passes)
        limited = _acceleration_limited_resample_path(
            geometry,
            dt_seconds=limits.dt_seconds,
        )
        candidate = limited if limited is not None else geometry
        candidate[-1] = xyz[-1]
        endpoint_error = float(np.linalg.norm(candidate[-1, :2] - xyz[-1, :2]))
        if endpoint_error > limits.endpoint_tolerance_m:
            continue

        diagnostics = diagnose_trajectory_dynamics(candidate, limits=limits)
        max_change = float(np.nanmax(np.linalg.norm(candidate[:, :2] - xyz[:, :2], axis=1)))
        score = _diagnostic_score(diagnostics, limits)
        score += max(0.0, max_change - limits.max_position_change_m) * 2.0
        if score < best_score:
            best_xyz = candidate
            best_diagnostics = diagnostics
            best_score = score
        if diagnostics.ok:
            return DynamicsOptimizationResult(
                xyz=candidate,
                ok=True,
                diagnostics=diagnostics,
                original_diagnostics=original_diagnostics,
                message=f"optimized with {passes} smoothing passes",
            )

    return DynamicsOptimizationResult(
        xyz=best_xyz,
        ok=best_diagnostics.ok,
        diagnostics=best_diagnostics,
        original_diagnostics=original_diagnostics,
        message="returned best available conservative trajectory",
    )
