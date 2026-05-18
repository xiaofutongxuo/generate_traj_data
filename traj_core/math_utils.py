"""Pure geometry, color, and trajectory math helpers for the enhanced GUI."""

from __future__ import annotations

import colorsys
from typing import Optional

import numpy as np
import pandas as pd

from traj_core.constants import *

def _rgb_to_hex(color: tuple[int, int, int]) -> str:
    r, g, b = [int(np.clip(channel, 0, 255)) for channel in color]
    return f"#{r:02x}{g:02x}{b:02x}"

def _brighten_rgb(color: tuple[int, int, int], amount: float = 0.35) -> tuple[int, int, int]:
    """Brighten an RGB color while preserving its hue."""
    return tuple(
        int(round(channel + (255 - channel) * amount))
        for channel in color
    )

def _trajectory_base_color(index: int, total_count: int) -> tuple[int, int, int]:
    """Return a stable, distinct RGB color for a trajectory index."""
    if total_count <= len(TRAJ_COLORS):
        return TRAJ_COLORS[index % len(TRAJ_COLORS)]

    hue = (index % max(total_count, 1)) / max(float(total_count), 1.0)
    rgb = colorsys.hsv_to_rgb(hue, 0.78, 0.92)
    return tuple(int(round(channel * 255)) for channel in rgb)

def _row_t0_us(row) -> int:
    """Return t0 for new parquet rows, with fallback for older outputs."""
    if "t0_us" in row and not pd.isna(row["t0_us"]):
        return int(row["t0_us"])
    timestamps = row["timestamp"]
    if len(timestamps) == 0:
        return 0
    return int(timestamps[0]) - 100_000

def _de_casteljau(control_points: np.ndarray, t_values: np.ndarray) -> np.ndarray:
    """Evaluate an arbitrary-degree Bezier curve."""
    points = np.asarray(control_points, dtype=np.float64)
    curves = []
    for t in t_values:
        work = points.copy()
        for level in range(1, len(points)):
            work[:-level] = (1.0 - t) * work[:-level] + t * work[1:len(points) - level + 1]
        curves.append(work[0])
    return np.asarray(curves, dtype=np.float64)

def _sample_cubic_bezier_chain(
    waypoints: np.ndarray,
    samples_per_segment: int = 96,
    start_tangent: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample a smooth cubic Bezier chain that passes through every waypoint."""
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if len(waypoints) < 2:
        return waypoints

    tangents = np.zeros_like(waypoints)
    for idx in range(len(waypoints)):
        if idx == 0:
            if start_tangent is not None:
                tangents[idx] = np.asarray(start_tangent, dtype=np.float64)
            else:
                tangents[idx] = waypoints[1] - waypoints[0]
        elif idx == len(waypoints) - 1:
            tangents[idx] = waypoints[idx] - waypoints[idx - 1]
        else:
            tangents[idx] = 0.5 * (waypoints[idx + 1] - waypoints[idx - 1])
        tangents[idx, 2] = 0.0

    segments = []
    t_values = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    for idx in range(len(waypoints) - 1):
        p0 = waypoints[idx]
        p3 = waypoints[idx + 1]
        c1 = p0 + tangents[idx] / 3.0
        c2 = p3 - tangents[idx + 1] / 3.0
        one_minus_t = 1.0 - t_values[:, None]
        t = t_values[:, None]
        segment = (
            one_minus_t ** 3 * p0
            + 3.0 * one_minus_t ** 2 * t * c1
            + 3.0 * one_minus_t * t ** 2 * c2
            + t ** 3 * p3
        )
        segments.append(segment)

    dense = np.vstack(segments + [waypoints[-1:]])
    dense[:, 2] = 0.0
    return dense

def _resample_curve_by_distance(
    dense_points: np.ndarray,
    num_steps: int,
    initial_speed_mps: float,
    dt_seconds: float,
) -> np.ndarray:
    """Resample a dense path using a smooth distance schedule."""
    dense_points = np.asarray(dense_points, dtype=np.float64)
    if len(dense_points) < 2:
        return dense_points

    seg_lengths = np.linalg.norm(np.diff(dense_points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(dense_points[:1], num_steps, axis=0)

    horizon_s = max(num_steps * dt_seconds, dt_seconds)
    avg_speed = total_length / horizon_s
    start_speed = float(np.clip(initial_speed_mps, 0.2, max(avg_speed * 2.0, 0.2)))
    end_speed = max(0.2, 2.0 * avg_speed - start_speed)

    times = np.arange(1, num_steps + 1, dtype=np.float64) * dt_seconds
    accel = (end_speed - start_speed) / horizon_s
    target_dist = start_speed * times + 0.5 * accel * times * times
    target_dist = np.maximum.accumulate(target_dist)
    target_dist *= total_length / max(float(target_dist[-1]), 1e-6)
    target_dist = np.clip(target_dist, 0.0, total_length)

    sampled = np.empty((num_steps, dense_points.shape[1]), dtype=np.float64)
    for dim in range(dense_points.shape[1]):
        sampled[:, dim] = np.interp(target_dist, cumulative, dense_points[:, dim])
    return sampled

def _smooth_xy(points_xyz: np.ndarray, passes: int = 2) -> np.ndarray:
    """Apply a light binomial smoothing pass while preserving endpoints."""
    smoothed = np.asarray(points_xyz, dtype=np.float64).copy()
    if len(smoothed) < 5:
        return smoothed
    for _ in range(passes):
        prev = smoothed.copy()
        smoothed[1:-1, :2] = 0.25 * prev[:-2, :2] + 0.5 * prev[1:-1, :2] + 0.25 * prev[2:, :2]
    smoothed[:, 2] = 0.0
    return smoothed

def _smooth_curvature_preserving_ends(points_xyz: np.ndarray, passes: int = CLUSTER_CURVATURE_SMOOTH_PASSES) -> np.ndarray:
    """Smooth cluster geometry while preserving the first future point and endpoint."""
    smoothed = np.asarray(points_xyz, dtype=np.float64).copy()
    if smoothed.ndim != 2 or smoothed.shape[0] < 6 or smoothed.shape[1] < 2:
        return smoothed

    first = smoothed[0].copy()
    last = smoothed[-1].copy()
    for _ in range(max(0, int(passes))):
        prev = smoothed.copy()
        smoothed[1:-1, :2] = (
            0.20 * prev[:-2, :2]
            + 0.60 * prev[1:-1, :2]
            + 0.20 * prev[2:, :2]
        )
        smoothed[0] = first
        smoothed[-1] = last
    if smoothed.shape[1] > 2:
        smoothed[:, 2] = 0.0
    return smoothed

def _point_at_path_fraction(points_xyz: np.ndarray, fraction: float) -> Optional[np.ndarray]:
    """Return the point at a normalized arc-length fraction along a polyline."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if len(points) == 0:
        return None
    if len(points) == 1:
        return points[0].copy()

    seg_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return points[0].copy()

    target = float(np.clip(fraction, 0.0, 1.0)) * total_length
    out = np.empty(points.shape[1], dtype=np.float64)
    for dim in range(points.shape[1]):
        out[dim] = np.interp(target, cumulative, points[:, dim])
    return out

def _nearest_path_fraction(points_xyz: np.ndarray, query_xy: tuple[float, float]) -> Optional[float]:
    """Find the nearest normalized arc-length fraction on a polyline."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if len(points) < 2:
        return None

    query = np.asarray(query_xy, dtype=np.float64)
    seg_vecs = np.diff(points[:, :2], axis=0)
    seg_lengths = np.linalg.norm(seg_vecs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return 0.0

    best_dist = None
    best_along = 0.0
    for idx, (p0, vec, seg_len) in enumerate(zip(points[:-1, :2], seg_vecs, seg_lengths)):
        if seg_len <= 1e-6:
            continue
        t = float(np.clip(np.dot(query - p0, vec) / (seg_len * seg_len), 0.0, 1.0))
        projected = p0 + t * vec
        dist = float(np.linalg.norm(query - projected))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_along = float(cumulative[idx] + t * seg_len)

    return best_along / total_length

def _interp_polyline_by_distance(
    dense_points: np.ndarray,
    cumulative: np.ndarray,
    target_distances: np.ndarray,
) -> np.ndarray:
    """Interpolate polyline points at target arc-length distances."""
    sampled = np.empty((len(target_distances), dense_points.shape[1]), dtype=np.float64)
    for dim in range(dense_points.shape[1]):
        sampled[:, dim] = np.interp(target_distances, cumulative, dense_points[:, dim])
    return sampled

def _trajectory_dynamics_from_xy(
    xy: np.ndarray,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return speed, vector acceleration, and signed curvature for ego-local xy."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    delta = np.diff(xy_from_origin, axis=0)
    step_distance = np.linalg.norm(delta, axis=1)
    speed = step_distance / dt_seconds

    velocity = delta / dt_seconds
    acceleration = np.zeros_like(velocity)
    if len(velocity) > 1:
        acceleration[1:] = np.diff(velocity, axis=0) / dt_seconds

    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    moving = speed > 1e-3
    if moving.any():
        first = int(np.flatnonzero(moving)[0])
        yaw[:first] = yaw[first]
        for idx in range(first + 1, len(yaw)):
            if not moving[idx]:
                yaw[idx] = yaw[idx - 1]
        yaw = np.unwrap(yaw)
    else:
        yaw[:] = 0.0

    curvature = np.zeros_like(speed)
    if len(speed) > 1:
        curvature[1:] = np.diff(yaw) / np.maximum(step_distance[1:], 1e-6)
        curvature[~np.isfinite(curvature)] = 0.0
    return speed, acceleration, curvature

def _scalar_speed_acceleration_from_xy(
    xy: np.ndarray,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return step distance, scalar speed, and scalar acceleration per future point."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    step_distance = np.linalg.norm(np.diff(xy_from_origin, axis=0), axis=1)
    speed = step_distance / dt_seconds
    acceleration = np.zeros_like(speed)
    if len(speed) > 1:
        acceleration[1:] = np.diff(speed) / dt_seconds
    return step_distance, speed, acceleration

__all__ = [name for name in globals() if (name.startswith("_") and not name.startswith("__")) or name.isupper()]
