"""Low-level trajectory dynamics metrics shared by GUI editing code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traj_core.constants import TRAJ_DT_SECONDS
from traj_core.math_utils import _trajectory_dynamics_from_xy
from .limits import DynamicsLimits


@dataclass(frozen=True)
class TrajectoryKinematics:
    xyz: np.ndarray
    step_distance: np.ndarray
    speed: np.ndarray
    scalar_acceleration: np.ndarray
    scalar_jerk: np.ndarray
    velocity: np.ndarray
    vector_acceleration: np.ndarray
    curvature: np.ndarray


def as_xyz(points_xyz: np.ndarray) -> np.ndarray:
    """Return a finite float64 Nx3 trajectory array."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("trajectory must be a 2D array")
    if points.shape[1] < 2:
        raise ValueError("trajectory must have at least x/y columns")
    if points.shape[1] == 2:
        z = np.zeros((len(points), 1), dtype=np.float64)
        points = np.hstack([points, z])
    else:
        points = points[:, :3].copy()
    if not np.all(np.isfinite(points)):
        raise ValueError("trajectory contains non-finite values")
    return points


def trajectory_kinematics(
    points_xyz: np.ndarray,
    limits: DynamicsLimits | None = None,
) -> TrajectoryKinematics:
    """Compute per-frame speed, acceleration, jerk and curvature."""
    limits = limits or DynamicsLimits()
    xyz = as_xyz(points_xyz)
    num_steps = len(xyz)
    if num_steps == 0:
        empty = np.zeros(0, dtype=np.float64)
        empty_vec = np.zeros((0, 3), dtype=np.float64)
        return TrajectoryKinematics(
            xyz=xyz,
            step_distance=empty,
            speed=empty,
            scalar_acceleration=empty,
            scalar_jerk=empty,
            velocity=empty_vec,
            vector_acceleration=empty_vec,
            curvature=empty,
        )

    dt_seconds = float(limits.dt_seconds)
    previous = np.vstack([np.zeros((1, 3), dtype=np.float64), xyz[:-1]])
    delta = xyz - previous
    step_distance = np.linalg.norm(delta[:, :2], axis=1)
    speed = step_distance / dt_seconds
    velocity = delta / dt_seconds

    scalar_acceleration = np.zeros(num_steps, dtype=np.float64)
    vector_acceleration = np.zeros_like(velocity)
    if num_steps > 1:
        scalar_acceleration[1:] = np.diff(speed) / dt_seconds
        vector_acceleration[1:] = np.diff(velocity, axis=0) / dt_seconds

    scalar_jerk = np.zeros(num_steps, dtype=np.float64)
    if num_steps > 1:
        scalar_jerk[1:] = np.diff(scalar_acceleration) / dt_seconds

    _speed, _acceleration, curvature = _trajectory_dynamics_from_xy(xyz[:, :2], dt_seconds)
    return TrajectoryKinematics(
        xyz=xyz,
        step_distance=step_distance,
        speed=speed,
        scalar_acceleration=scalar_acceleration,
        scalar_jerk=scalar_jerk,
        velocity=velocity,
        vector_acceleration=vector_acceleration,
        curvature=curvature,
    )


def trajectory_components_from_xyz(
    points_xyz: np.ndarray,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> dict[str, np.ndarray]:
    """Recompute parquet trajectory columns from xyz points."""
    xyz = as_xyz(points_xyz)
    num_steps = len(xyz)
    if num_steps == 0:
        empty = np.zeros(0, dtype=np.float64)
        return {
            "x": empty,
            "y": empty,
            "z": empty,
            "vx": empty,
            "vy": empty,
            "vz": empty,
            "qx": empty,
            "qy": empty,
            "qz": empty,
            "qw": empty,
            "curvature": empty,
        }

    previous = np.vstack([np.zeros((1, 3), dtype=np.float64), xyz[:-1]])
    velocity = (xyz - previous) / float(dt_seconds)
    speed_xy = np.linalg.norm(velocity[:, :2], axis=1)
    yaws = np.arctan2(velocity[:, 1], velocity[:, 0])
    if num_steps > 1:
        yaws = np.unwrap(yaws)
        stationary = speed_xy < 1e-3
        moving_indices = np.where(~stationary)[0]
        if len(moving_indices) == 0:
            yaws[:] = 0.0
        else:
            first_moving = int(moving_indices[0])
            yaws[:first_moving] = yaws[first_moving]
            for idx in range(first_moving + 1, num_steps):
                if stationary[idx]:
                    yaws[idx] = yaws[idx - 1]

    return {
        "x": xyz[:, 0].copy(),
        "y": xyz[:, 1].copy(),
        "z": xyz[:, 2].copy(),
        "vx": velocity[:, 0],
        "vy": velocity[:, 1],
        "vz": velocity[:, 2],
        "qx": np.zeros(num_steps, dtype=np.float64),
        "qy": np.zeros(num_steps, dtype=np.float64),
        "qz": np.sin(yaws / 2.0),
        "qw": np.cos(yaws / 2.0),
        "curvature": np.gradient(yaws) if num_steps else np.zeros(0, dtype=np.float64),
    }
