"""Diagnostics for generated trajectory dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .limits import DynamicsLimits
from .metrics import trajectory_kinematics


@dataclass(frozen=True)
class DynamicsDiagnostics:
    ok: bool
    violations: dict[str, list[int]]
    metrics: dict[str, float]


def _violation_indices(mask: np.ndarray) -> list[int]:
    return [int(idx) for idx in np.flatnonzero(mask)]


def diagnose_trajectory_dynamics(
    points_xyz: np.ndarray,
    limits: DynamicsLimits | None = None,
) -> DynamicsDiagnostics:
    """Return violation indices and summary metrics for a trajectory."""
    limits = limits or DynamicsLimits()
    kin = trajectory_kinematics(points_xyz, limits=limits)
    violations: dict[str, list[int]] = {}

    speed_hits = _violation_indices(kin.speed > limits.max_speed_mps)
    if speed_hits:
        violations["speed"] = speed_hits

    step_hits = _violation_indices(kin.step_distance > limits.max_step_m)
    if step_hits:
        violations["step"] = step_hits

    accel_hits = _violation_indices(
        (kin.scalar_acceleration > limits.max_accel_mps2)
        | (kin.scalar_acceleration < limits.min_accel_mps2)
    )
    if accel_hits:
        violations["acceleration"] = accel_hits

    jerk_hits = _violation_indices(np.abs(kin.scalar_jerk) > limits.max_jerk_mps3)
    if jerk_hits:
        violations["jerk"] = jerk_hits

    curvature_hits = _violation_indices(np.abs(kin.curvature) > limits.max_curvature_1pm)
    if curvature_hits:
        violations["curvature"] = curvature_hits

    metrics = {
        "max_speed_mps": float(np.nanmax(kin.speed)) if len(kin.speed) else 0.0,
        "max_step_m": float(np.nanmax(kin.step_distance)) if len(kin.step_distance) else 0.0,
        "min_accel_mps2": float(np.nanmin(kin.scalar_acceleration)) if len(kin.scalar_acceleration) else 0.0,
        "max_accel_mps2": float(np.nanmax(kin.scalar_acceleration)) if len(kin.scalar_acceleration) else 0.0,
        "max_abs_jerk_mps3": float(np.nanmax(np.abs(kin.scalar_jerk))) if len(kin.scalar_jerk) else 0.0,
        "max_abs_curvature_1pm": float(np.nanmax(np.abs(kin.curvature))) if len(kin.curvature) else 0.0,
    }
    return DynamicsDiagnostics(ok=not violations, violations=violations, metrics=metrics)
