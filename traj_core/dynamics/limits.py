"""Configurable limits for pseudo-GT trajectory dynamics."""

from __future__ import annotations

from dataclasses import dataclass

from traj_core.constants import (
    SPEED_UNSMOOTH_JERK_MPS3,
    TRAJ_ACCEL_MAX_MPS2,
    TRAJ_ACCEL_MIN_MPS2,
    TRAJ_DT_SECONDS,
    TRAJ_MAX_POSITION_STEP_M,
    TRAJ_MAX_STEP_SPEED_MPS,
)


@dataclass(frozen=True)
class DynamicsLimits:
    """Thresholds used to diagnose and conservatively repair generated trajectories."""

    dt_seconds: float = TRAJ_DT_SECONDS
    max_speed_mps: float = TRAJ_MAX_STEP_SPEED_MPS
    max_step_m: float = TRAJ_MAX_POSITION_STEP_M
    min_accel_mps2: float = TRAJ_ACCEL_MIN_MPS2
    max_accel_mps2: float = TRAJ_ACCEL_MAX_MPS2
    max_jerk_mps3: float = SPEED_UNSMOOTH_JERK_MPS3
    max_curvature_1pm: float = 1.0
    endpoint_tolerance_m: float = 0.25
    max_position_change_m: float = 6.0

    def __post_init__(self) -> None:
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        if self.max_step_m <= 0:
            raise ValueError("max_step_m must be positive")
        if self.max_accel_mps2 < self.min_accel_mps2:
            raise ValueError("max_accel_mps2 must be >= min_accel_mps2")
        if self.max_curvature_1pm <= 0:
            raise ValueError("max_curvature_1pm must be positive")
