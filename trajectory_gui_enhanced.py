#!/usr/bin/env python3
"""Enhanced GUI tool for visualizing VLM-generated trajectories with camera projections.

Features:
- Multiple camera views (FL, FC, FR, RL, RC, RR)
- RGB color correction
- 3D trajectory projection onto camera images
- Bird's eye view of trajectories

Usage:
    python trajectory_gui_enhanced.py --data_root /path/to/train_data --output_dir /path/to/output
"""

import argparse
from collections import OrderedDict
import colorsys
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

os.environ.setdefault("GENERATE_TRAJ_USE_TORCH", "0")
if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix/X1").exists():
    os.environ["DISPLAY"] = ":1"

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_dataset_names, get_clip_stems_from_dataset, load_data, get_t0_candidates
from calibration_loader import load_calibration_for_segment
from visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

# Camera name to index mapping
CAMERA_IDX_TO_NAME = {0: "FL", 1: "FC", 2: "FR", 3: "RL", 4: "RC", 5: "RR"}
CAMERA_NAME_TO_IDX = {v: k for k, v in CAMERA_IDX_TO_NAME.items()}

# Trajectory colors (RGB)
TRAJ_COLORS = [
    (231, 76, 60),    # Red
    (52, 152, 219),   # Blue
    (155, 89, 182),   # Purple
    (243, 156, 18),  # Orange
    (26, 188, 156),   # Teal
    (233, 30, 99),    # Pink
]

GT_COLOR_RGB = (255, 221, 87)
GT_COLOR_HEX = "#ffdd57"
MANUAL_TRAJ_COLOR_RGB = (0, 212, 255)
MANUAL_TRAJ_COLOR_HEX = "#00d4ff"
CLUSTER_TRAJ_COLOR_RGB = (255, 170, 0)
CLUSTER_TRAJ_COLOR_HEX = "#ffaa00"
FUTURE_TRAJ_STEPS = 64
TRAJ_DT_SECONDS = 0.1
CLUSTER_ENDPOINT_HIT_RADIUS_PX = 18
CLUSTER_DRAG_ACCELERATION_WEIGHT = 1.0
CLUSTER_DRAG_CURVATURE_WEIGHT = 80.0
CLUSTER_DRAG_DEVIATION_WEIGHT = 0.03
CLUSTER_CURVATURE_SMOOTH_PASSES = 4
TRAJ_ACCEL_MIN_MPS2 = -6.0
TRAJ_ACCEL_MAX_MPS2 = 2.0
TRAJ_ACCEL_LIMIT_MARGIN_MPS2 = 0.01
TRAJ_MAX_STEP_SPEED_MPS = 15.0
TRAJ_MAX_POSITION_STEP_M = TRAJ_MAX_STEP_SPEED_MPS * TRAJ_DT_SECONDS
STOP_SPEED_THRESHOLD_MPS = 0.1
STOP_MIN_FRAMES = 5
STOP_MARKER_RADIUS_PX = 5
HOVER_FRAME_COLOR_RGB = (46, 255, 139)
HOVER_FRAME_COLOR_HEX = "#2eff8b"
SPEED_UNSMOOTH_ACCEL_MPS2 = 12.0
SPEED_UNSMOOTH_JERK_MPS3 = 80.0
SPEED_GT_DIFF_TOLERANCE_MPS = 3.0
SPEED_EDIT_LOCAL_RADIUS_FRAMES = 4
SPEED_EDIT_MIN_MPS = 0.0
SPEED_EDIT_MAX_MPS = TRAJ_MAX_STEP_SPEED_MPS
CLUSTER_CATEGORY_FILES = {
    "left": "left.txt",
    "right": "right.txt",
    "s_curve": "s_curve.txt",
    "straight": "straight.txt",
    "stop": "stop.txt",
}
CLUSTER_CATEGORY_ORDER = ["stop", "straight", "left", "right", "s_curve"]
AUTO_OPTIMIZE_GT_ON_LOAD = True
GT_SPEED_OPTIMIZED_COLUMN = "gt_speed_auto_optimized"


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


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def project_3d_to_2d(points_xyz, pose_xyz, pose_quat, intrinsics, extrinsics, image_size=(1920, 1280)):
    """Project 3D world points to 2D image coordinates.
    
    Args:
        points_xyz: (N, 3) array of 3D points in ego frame [x=forward, y=left, z=up]
        pose_xyz: (3,) ego vehicle position (unused for local points)
        pose_quat: (4,) ego vehicle quaternion (unused for local points)
        intrinsics: camera intrinsics dict with fx, fy, cx, cy
        extrinsics: camera extrinsics dict with 'T_bev_to_camera' (3x4 matrix)
        image_size: (width, height) tuple
    
    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    import numpy as np
    
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    img_w, img_h = image_size
    
    # Scale intrinsics to actual image size
    scale_x = img_w / intrinsics.get('image_width', img_w)
    scale_y = img_h / intrinsics.get('image_height', img_h)
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y
    
    # Get transformation matrix
    T = extrinsics['T_bev_to_camera']
    R = T[:, :3]
    t = T[:, 3]
    
    # Convert ego frame to BEV frame
    # Ego: x=forward, y=left
    # BEV: x=east, y=north
    # Forward (ego_x) = North (bev_y), Left (ego_y) = West (-bev_x)
    # So: bev_x = -ego_y, bev_y = ego_x
    x_ego, y_ego, z_ego = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    x_bev = -y_ego
    y_bev = x_ego
    z_bev = z_ego
    p_bev = np.stack([x_bev, y_bev, z_bev], axis=1)
    
    # Transform BEV to camera frame
    p_cam = (R @ p_bev.T + t.reshape(3, 1)).T  # (N, 3)
    
    # Project to image
    u_coords = []
    v_coords = []
    valid = []
    
    for i in range(len(p_cam)):
        x, y, z = p_cam[i]
        if z > 0.1:  # Point is in front of camera
            u = fx * x / z + cx
            v = fy * y / z + cy
            # Check if within image bounds
            if 0 <= u < img_w and 0 <= v < img_h:
                u_coords.append(u)
                v_coords.append(v)
                valid.append(True)
            else:
                u_coords.append(-1)
                v_coords.append(-1)
                valid.append(False)
        else:
            u_coords.append(-1)
            v_coords.append(-1)
            valid.append(False)
    
    return np.array(list(zip(u_coords, v_coords))), np.array(valid)


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


def _speed_profile_from_trajectory(
    x_coords,
    y_coords,
    z_coords=None,
    vx=None,
    vy=None,
    vz=None,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> np.ndarray:
    """Return one scalar speed value per future trajectory frame."""
    x = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    y = np.asarray(y_coords, dtype=np.float64).reshape(-1)
    if len(x) == 0 or len(y) != len(x):
        return np.zeros(0, dtype=np.float64)

    velocity_parts = []
    for values in (vx, vy, vz):
        if values is None:
            velocity_parts = []
            break
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(arr) != len(x) or not np.isfinite(arr).all():
            velocity_parts = []
            break
        velocity_parts.append(arr)

    if velocity_parts:
        return np.linalg.norm(np.column_stack(velocity_parts), axis=1)

    if z_coords is None:
        z = np.zeros_like(x)
    else:
        z = np.asarray(z_coords, dtype=np.float64).reshape(-1)
        if len(z) != len(x):
            z = np.zeros_like(x)
    points = np.column_stack([x, y, z])
    prev = np.vstack([np.zeros((1, 3), dtype=np.float64), points[:-1]])
    return np.linalg.norm(points - prev, axis=1) / max(float(dt_seconds), 1e-6)


def _detect_stop_segments(
    speed: np.ndarray,
    threshold_mps: float = STOP_SPEED_THRESHOLD_MPS,
    min_frames: int = STOP_MIN_FRAMES,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> list[dict]:
    """Find low-speed runs long enough to be treated as stop actions."""
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    if len(speed) == 0:
        return []

    stopped = np.isfinite(speed) & (speed < float(threshold_mps))
    segments = []
    start = None
    for idx, is_stopped in enumerate(stopped):
        if is_stopped and start is None:
            start = idx
        elif not is_stopped and start is not None:
            end = idx - 1
            frames = end - start + 1
            if frames >= int(min_frames):
                segments.append({
                    "start": start,
                    "end": end,
                    "frames": frames,
                    "duration_s": frames * float(dt_seconds),
                    "mean_speed_mps": float(np.nanmean(speed[start:end + 1])),
                })
            start = None

    if start is not None:
        end = len(speed) - 1
        frames = end - start + 1
        if frames >= int(min_frames):
            segments.append({
                "start": start,
                "end": end,
                "frames": frames,
                "duration_s": frames * float(dt_seconds),
                "mean_speed_mps": float(np.nanmean(speed[start:end + 1])),
            })
    return segments


def _speed_smoothness_diagnostics(
    speed: np.ndarray,
    reference_speed: Optional[np.ndarray] = None,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> dict:
    """Return whether a speed curve has abrupt acceleration or jerk changes."""
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    valid = np.isfinite(speed)
    if len(speed) < 4 or not valid.all():
        return {
            "ok": False,
            "reason": "速度不够平滑",
            "max_abs_accel": 0.0,
            "max_abs_jerk": 0.0,
        }

    acceleration = np.diff(speed) / max(float(dt_seconds), 1e-6)
    jerk = np.diff(acceleration) / max(float(dt_seconds), 1e-6)
    max_abs_accel = float(np.nanmax(np.abs(acceleration))) if len(acceleration) else 0.0
    max_abs_jerk = float(np.nanmax(np.abs(jerk))) if len(jerk) else 0.0
    mean_gt_diff = None
    if reference_speed is not None:
        ref = np.asarray(reference_speed, dtype=np.float64).reshape(-1)
        if len(ref) >= 2 and np.isfinite(ref).all():
            x_src = np.linspace(0.0, 1.0, len(ref))
            x_dst = np.linspace(0.0, 1.0, len(speed))
            ref_interp = np.interp(x_dst, x_src, ref)
            mean_gt_diff = float(np.nanmean(np.abs(speed - ref_interp)))

    gt_like = (
        mean_gt_diff is not None
        and mean_gt_diff <= SPEED_GT_DIFF_TOLERANCE_MPS
    )
    has_spike = (
        max_abs_accel > SPEED_UNSMOOTH_ACCEL_MPS2
        or max_abs_jerk > SPEED_UNSMOOTH_JERK_MPS3
    )
    # Curves close to GT are tolerated unless the spike is very large.
    very_large_spike = (
        max_abs_accel > SPEED_UNSMOOTH_ACCEL_MPS2 * 4.0
        or max_abs_jerk > SPEED_UNSMOOTH_JERK_MPS3 * 4.0
    )
    ok = (
        not has_spike
        or (gt_like and not very_large_spike)
    )
    return {
        "ok": bool(ok),
        "reason": "" if ok else "速度不够平滑",
        "max_abs_accel": max_abs_accel,
        "max_abs_jerk": max_abs_jerk,
        "mean_gt_diff": mean_gt_diff,
    }


def _resample_xyz_by_speed_profile(
    original_xyz: np.ndarray,
    speed: np.ndarray,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> Optional[np.ndarray]:
    """Resample an existing path so point density follows the edited speed profile."""
    original = np.asarray(original_xyz, dtype=np.float64)
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    if original.ndim != 2 or original.shape[0] < 2 or len(speed) != len(original):
        return None

    speed = np.clip(speed, SPEED_EDIT_MIN_MPS, SPEED_EDIT_MAX_MPS)
    target = np.cumsum(speed * float(dt_seconds))
    if len(target) == 0 or not np.isfinite(target).all() or float(target[-1]) <= 1e-6:
        return None

    path = np.vstack([np.zeros((1, original.shape[1]), dtype=np.float64), original])
    step = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return None

    # Keep the original endpoint, while changing the density pattern along the path.
    target = target * (total_length / float(target[-1]))
    unique_mask = np.concatenate([[True], np.diff(cumulative) > 1e-8])
    sampled = _interp_polyline_by_distance(
        path[unique_mask],
        cumulative[unique_mask],
        np.clip(target, 0.0, total_length),
    )
    if sampled.shape[1] > 2:
        sampled[:, 2] = original[:, 2]
    return sampled


def _smooth_speed_profile(speed: np.ndarray, passes: int = 2) -> np.ndarray:
    """Smooth speed spikes while preserving broad speed trends."""
    smoothed = np.asarray(speed, dtype=np.float64).reshape(-1).copy()
    if len(smoothed) < 5:
        return np.clip(smoothed, SPEED_EDIT_MIN_MPS, SPEED_EDIT_MAX_MPS)
    smoothed[~np.isfinite(smoothed)] = 0.0
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    for _ in range(max(1, int(passes))):
        padded = np.pad(smoothed, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed = _enforce_speed_acceleration_limits(smoothed)
    return np.clip(smoothed, SPEED_EDIT_MIN_MPS, SPEED_EDIT_MAX_MPS)


def _enforce_speed_acceleration_limits(
    speed: np.ndarray,
    min_accel_mps2: float = TRAJ_ACCEL_MIN_MPS2,
    max_accel_mps2: float = TRAJ_ACCEL_MAX_MPS2,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> np.ndarray:
    """Clamp every speed transition so scalar acceleration stays in range."""
    limited = np.asarray(speed, dtype=np.float64).copy()
    if len(limited) == 0:
        return limited
    limited[~np.isfinite(limited)] = 0.0
    limited = np.clip(limited, 0.0, TRAJ_MAX_STEP_SPEED_MPS)
    min_delta = float(min_accel_mps2) * dt_seconds
    max_delta = float(max_accel_mps2) * dt_seconds

    for _ in range(8):
        for idx in range(1, len(limited)):
            lower = max(0.0, limited[idx - 1] + min_delta)
            upper = min(TRAJ_MAX_STEP_SPEED_MPS, limited[idx - 1] + max_delta)
            limited[idx] = float(np.clip(limited[idx], lower, upper))
        for idx in range(len(limited) - 2, -1, -1):
            lower = max(0.0, limited[idx + 1] - max_delta)
            upper = min(TRAJ_MAX_STEP_SPEED_MPS, limited[idx + 1] - min_delta)
            limited[idx] = float(np.clip(limited[idx], lower, upper))
    return limited


def _enforce_speed_acceleration_limits_with_fixed(
    speed: np.ndarray,
    fixed_speed: np.ndarray,
    min_accel_mps2: float = TRAJ_ACCEL_MIN_MPS2,
    max_accel_mps2: float = TRAJ_ACCEL_MAX_MPS2,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> np.ndarray:
    """Clamp speed transitions while preserving finite fixed-speed entries."""
    limited = np.asarray(speed, dtype=np.float64).copy()
    fixed = np.asarray(fixed_speed, dtype=np.float64).reshape(-1)
    if len(limited) != len(fixed):
        raise ValueError("fixed_speed must match speed length")
    if len(limited) == 0:
        return limited

    fixed_mask = np.isfinite(fixed)
    limited[~np.isfinite(limited)] = 0.0
    limited = np.clip(limited, 0.0, TRAJ_MAX_STEP_SPEED_MPS)
    limited[fixed_mask] = np.clip(fixed[fixed_mask], 0.0, TRAJ_MAX_STEP_SPEED_MPS)
    min_delta = float(min_accel_mps2) * dt_seconds
    max_delta = float(max_accel_mps2) * dt_seconds

    for _ in range(12):
        limited[fixed_mask] = np.clip(fixed[fixed_mask], 0.0, TRAJ_MAX_STEP_SPEED_MPS)
        for idx in range(1, len(limited)):
            lower = max(0.0, limited[idx - 1] + min_delta)
            upper = min(TRAJ_MAX_STEP_SPEED_MPS, limited[idx - 1] + max_delta)
            if not fixed_mask[idx]:
                limited[idx] = float(np.clip(limited[idx], lower, upper))
        limited[fixed_mask] = np.clip(fixed[fixed_mask], 0.0, TRAJ_MAX_STEP_SPEED_MPS)
        for idx in range(len(limited) - 2, -1, -1):
            lower = max(0.0, limited[idx + 1] - max_delta)
            upper = min(TRAJ_MAX_STEP_SPEED_MPS, limited[idx + 1] - min_delta)
            if not fixed_mask[idx]:
                limited[idx] = float(np.clip(limited[idx], lower, upper))
    limited[fixed_mask] = np.clip(fixed[fixed_mask], 0.0, TRAJ_MAX_STEP_SPEED_MPS)
    return limited


def _acceleration_limited_resample_path(
    xyz: np.ndarray,
    dt_seconds: float = TRAJ_DT_SECONDS,
    initial_speed_mps: Optional[float] = None,
    anchor_initial_speed: bool = False,
) -> Optional[np.ndarray]:
    """Resample a path with per-point acceleration clipped to configured limits."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None

    xy = points[:, :2]
    xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    seg_lengths = np.linalg.norm(np.diff(xy_from_origin, axis=0), axis=1)
    total_length = float(seg_lengths.sum())
    if total_length <= 1e-6:
        output = points.copy()
        output[:, :2] = 0.0
        if output.shape[1] > 2:
            output[:, 2] = 0.0
        return output

    desired_speed = seg_lengths / dt_seconds
    best_speed = None
    best_error = float("inf")
    lo = -TRAJ_MAX_STEP_SPEED_MPS
    hi = TRAJ_MAX_STEP_SPEED_MPS
    for _ in range(32):
        bias = (lo + hi) * 0.5
        speed_candidate = desired_speed + bias
        if initial_speed_mps is None or not np.isfinite(initial_speed_mps):
            limited_speed = _enforce_speed_acceleration_limits(
                speed_candidate,
                min_accel_mps2=TRAJ_ACCEL_MIN_MPS2 + TRAJ_ACCEL_LIMIT_MARGIN_MPS2,
                max_accel_mps2=TRAJ_ACCEL_MAX_MPS2 - TRAJ_ACCEL_LIMIT_MARGIN_MPS2,
            )
        else:
            speed_seed = np.concatenate([[float(initial_speed_mps)], speed_candidate])
            fixed_speed = np.full(len(speed_seed), np.nan, dtype=np.float64)
            fixed_speed[0] = float(initial_speed_mps)
            if anchor_initial_speed and len(fixed_speed) > 1:
                fixed_speed[1] = float(initial_speed_mps)
            speed_with_initial = _enforce_speed_acceleration_limits_with_fixed(
                speed_seed,
                fixed_speed,
                min_accel_mps2=TRAJ_ACCEL_MIN_MPS2 + TRAJ_ACCEL_LIMIT_MARGIN_MPS2,
                max_accel_mps2=TRAJ_ACCEL_MAX_MPS2 - TRAJ_ACCEL_LIMIT_MARGIN_MPS2,
            )
            limited_speed = speed_with_initial[1:]
        distance = float(limited_speed.sum() * dt_seconds)
        error = abs(distance - total_length)
        if error < best_error:
            best_error = error
            best_speed = limited_speed
        if distance < total_length:
            lo = bias
        else:
            hi = bias

    if best_speed is None or best_error > max(0.25, total_length * 0.02):
        return None

    target_distances = np.cumsum(best_speed * dt_seconds)
    if len(target_distances) != len(points):
        return None
    target_distances = np.maximum.accumulate(np.clip(target_distances, 0.0, total_length))
    if abs(float(target_distances[-1]) - total_length) <= max(0.25, total_length * 0.02):
        target_distances[-1] = total_length
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    dense_points = np.vstack([
        np.zeros((1, points.shape[1]), dtype=np.float64),
        points.copy(),
    ])
    output = _interp_polyline_by_distance(dense_points, cumulative, target_distances)
    if output.shape[1] > 2:
        output[:, 2] = 0.0

    _, speed, scalar_accel = _scalar_speed_acceleration_from_xy(output[:, :2], dt_seconds)
    if initial_speed_mps is not None and np.isfinite(initial_speed_mps) and len(speed):
        initial_accel = (speed[0] - float(initial_speed_mps)) / max(float(dt_seconds), 1e-6)
        scalar_accel[0] = initial_accel
    if (
        np.nanmin(scalar_accel) < TRAJ_ACCEL_MIN_MPS2 - 0.2
        or np.nanmax(scalar_accel) > TRAJ_ACCEL_MAX_MPS2 + 0.2
    ):
        return None
    return output


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


def _allocate_segment_frames(segment_lengths: list[float], total_frames: int) -> list[int]:
    """Allocate moving frames across path segments by distance."""
    counts = [0] * len(segment_lengths)
    if total_frames <= 0 or not segment_lengths:
        return counts

    positive = [
        idx for idx, length in enumerate(segment_lengths)
        if float(length) > 1e-6
    ]
    if not positive:
        counts[0] = total_frames
        return counts

    if total_frames <= len(positive):
        largest = sorted(positive, key=lambda idx: segment_lengths[idx], reverse=True)
        for idx in largest[:total_frames]:
            counts[idx] = 1
        return counts

    for idx in positive:
        counts[idx] = 1

    remaining = total_frames - len(positive)
    total_length = sum(float(segment_lengths[idx]) for idx in positive)
    raw = [
        remaining * float(segment_lengths[idx]) / max(total_length, 1e-6)
        for idx in positive
    ]
    floors = [int(np.floor(value)) for value in raw]
    for idx, extra in zip(positive, floors):
        counts[idx] += extra

    leftover = remaining - sum(floors)
    remainders = sorted(
        zip(positive, raw),
        key=lambda item: item[1] - np.floor(item[1]),
        reverse=True,
    )
    for idx, _value in remainders[:leftover]:
        counts[idx] += 1

    return counts


def _eased_segment_distances(
    start_dist: float,
    end_dist: float,
    frame_count: int,
    slow_start: bool,
    slow_end: bool,
) -> np.ndarray:
    """Generate non-uniform arc-length samples for one moving segment."""
    if frame_count <= 0:
        return np.empty((0,), dtype=np.float64)

    if slow_start and slow_end:
        u = np.arange(1, frame_count + 1, dtype=np.float64) / (frame_count + 1)
        eased = 3.0 * u * u - 2.0 * u * u * u
    elif slow_start:
        u = np.arange(1, frame_count + 1, dtype=np.float64) / frame_count
        eased = u * u
    elif slow_end:
        u = np.arange(1, frame_count + 1, dtype=np.float64) / (frame_count + 1)
        eased = 1.0 - (1.0 - u) * (1.0 - u)
    else:
        u = np.arange(1, frame_count + 1, dtype=np.float64) / frame_count
        eased = u

    return start_dist + (end_dist - start_dist) * eased


def _resample_curve_with_stops(
    dense_points: np.ndarray,
    num_steps: int,
    initial_speed_mps: float,
    dt_seconds: float,
    stop_points: list[dict],
) -> np.ndarray:
    """Resample a path with discrete stop frames and smooth decel/accel around stops."""
    dense_points = np.asarray(dense_points, dtype=np.float64)
    if len(dense_points) < 2:
        return dense_points

    seg_lengths = np.linalg.norm(np.diff(dense_points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return np.repeat(dense_points[:1], num_steps, axis=0)

    valid_stops = []
    total_hold_steps = 0
    for stop in stop_points:
        try:
            duration_s = float(stop.get("duration_s", 0.0))
            fraction = float(stop.get("fraction", 0.0))
        except (TypeError, ValueError):
            continue
        hold_steps = int(round(duration_s / dt_seconds))
        if hold_steps <= 0:
            continue
        valid_stops.append({
            "fraction": float(np.clip(fraction, 0.0, 1.0)),
            "distance": float(np.clip(fraction, 0.0, 1.0)) * total_length,
            "duration_s": duration_s,
            "hold_steps": hold_steps,
        })
        total_hold_steps += hold_steps

    if not valid_stops:
        return _resample_curve_by_distance(dense_points, num_steps, initial_speed_mps, dt_seconds)

    max_hold_steps = max(0, num_steps - 2)
    if total_hold_steps > max_hold_steps:
        scale = max_hold_steps / max(float(total_hold_steps), 1.0)
        total_hold_steps = 0
        for stop in valid_stops:
            stop["hold_steps"] = max(1, int(round(stop["hold_steps"] * scale)))
            total_hold_steps += stop["hold_steps"]
        while total_hold_steps > max_hold_steps and valid_stops:
            largest = max(valid_stops, key=lambda item: item["hold_steps"])
            largest["hold_steps"] -= 1
            total_hold_steps -= 1
            valid_stops = [stop for stop in valid_stops if stop["hold_steps"] > 0]

    valid_stops.sort(key=lambda item: item["fraction"])
    moving_steps = max(0, num_steps - total_hold_steps)

    segment_bounds = [0.0] + [float(stop["distance"]) for stop in valid_stops] + [total_length]
    segment_lengths = [
        max(0.0, segment_bounds[idx + 1] - segment_bounds[idx])
        for idx in range(len(segment_bounds) - 1)
    ]
    segment_frames = _allocate_segment_frames(segment_lengths, moving_steps)

    target_distances = []
    for segment_idx, stop in enumerate(valid_stops):
        start_dist = segment_bounds[segment_idx]
        end_dist = segment_bounds[segment_idx + 1]
        target_distances.extend(
            _eased_segment_distances(
                start_dist,
                end_dist,
                segment_frames[segment_idx],
                slow_start=segment_idx > 0,
                slow_end=True,
            ).tolist()
        )
        target_distances.extend([float(stop["distance"])] * int(stop["hold_steps"]))

    final_segment_idx = len(valid_stops)
    target_distances.extend(
        _eased_segment_distances(
            segment_bounds[-2],
            segment_bounds[-1],
            segment_frames[final_segment_idx],
            slow_start=bool(valid_stops),
            slow_end=False,
        ).tolist()
    )

    target_distances = np.asarray(target_distances, dtype=np.float64)
    if len(target_distances) > num_steps:
        target_distances = target_distances[:num_steps]
    elif len(target_distances) < num_steps:
        fill_value = total_length if len(target_distances) == 0 else float(target_distances[-1])
        target_distances = np.concatenate([
            target_distances,
            np.full(num_steps - len(target_distances), fill_value, dtype=np.float64),
        ])

    target_distances = np.clip(target_distances, 0.0, total_length)
    sampled = _interp_polyline_by_distance(dense_points, cumulative, target_distances)
    if len(sampled) > num_steps:
        sampled = sampled[:num_steps]
    elif len(sampled) < num_steps:
        sampled = np.vstack([sampled, np.repeat(sampled[-1:], num_steps - len(sampled), axis=0)])
    sampled[:, 2] = 0.0
    return sampled


def _resample_curve_with_final_stop(
    dense_points: np.ndarray,
    num_steps: int,
    initial_speed_mps: float,
    dt_seconds: float,
    duration_s: float,
) -> Optional[np.ndarray]:
    """Resample a path so the final frames remain stopped at the endpoint."""
    dense_points = np.asarray(dense_points, dtype=np.float64)
    if len(dense_points) < 2:
        return None

    seg_lengths = np.linalg.norm(np.diff(dense_points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        output = np.repeat(dense_points[-1:], num_steps, axis=0)
        output[:, 2] = 0.0
        return output

    hold_steps = int(round(float(duration_s) / dt_seconds))
    hold_steps = int(np.clip(hold_steps, 1, max(1, num_steps - 3)))
    moving_steps = num_steps - hold_steps
    if moving_steps < 3:
        return None

    moving_base = _resample_curve_by_distance(
        dense_points,
        moving_steps,
        initial_speed_mps,
        dt_seconds,
    )
    moving_xy0 = np.vstack([np.zeros((1, 2), dtype=np.float64), moving_base[:, :2]])
    desired_speed = np.linalg.norm(np.diff(moving_xy0, axis=0), axis=1) / dt_seconds
    desired_speed = np.concatenate([
        desired_speed,
        np.zeros(hold_steps, dtype=np.float64),
    ])
    fixed_speed = np.full(num_steps, np.nan, dtype=np.float64)
    fixed_speed[-hold_steps:] = 0.0

    best_speed = None
    best_error = float("inf")
    lo = -TRAJ_MAX_STEP_SPEED_MPS
    hi = TRAJ_MAX_STEP_SPEED_MPS
    for _ in range(36):
        bias = (lo + hi) * 0.5
        biased = desired_speed.copy()
        biased[:moving_steps] += bias
        speed = _enforce_speed_acceleration_limits_with_fixed(
            biased,
            fixed_speed,
            min_accel_mps2=TRAJ_ACCEL_MIN_MPS2 + TRAJ_ACCEL_LIMIT_MARGIN_MPS2,
            max_accel_mps2=TRAJ_ACCEL_MAX_MPS2 - TRAJ_ACCEL_LIMIT_MARGIN_MPS2,
            dt_seconds=dt_seconds,
        )
        distance = float(speed.sum() * dt_seconds)
        error = abs(distance - total_length)
        if error < best_error:
            best_error = error
            best_speed = speed
        if distance < total_length:
            lo = bias
        else:
            hi = bias

    if best_speed is None or best_error > max(0.25, total_length * 0.02):
        return None

    target_distances = np.cumsum(best_speed * dt_seconds)
    target_distances = np.maximum.accumulate(np.clip(target_distances, 0.0, total_length))
    if abs(float(target_distances[-1]) - total_length) <= max(0.25, total_length * 0.02):
        target_distances[moving_steps - 1:] = total_length
        target_distances = np.maximum.accumulate(target_distances)
    sampled = _interp_polyline_by_distance(dense_points, cumulative, target_distances)
    sampled[-hold_steps:] = dense_points[-1]
    sampled[:, 2] = 0.0

    diagnostics = _trajectory_quality_diagnostics(sampled, dt_seconds)
    if not bool(diagnostics.get("ok", False)):
        return None
    return sampled


class TrajectoryViewerEnhanced:
    """Enhanced GUI for viewing VLM-generated trajectories with camera projections."""
    
    def __init__(
        self,
        data_root: str,
        output_dir: str,
        calibration_dir: str,
        cameras: list[str] = None,
        start_index: Optional[int] = None,
        start_dataset: str = "",
        start_clip: str = "",
        start_t0: Optional[int] = None,
        restore_last: bool = True,
        gt_only: bool = False,
        gt_stride_frames: int = 3,
    ):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.calibration_dir = Path(calibration_dir)
        self.gt_only = bool(gt_only)
        self.gt_stride_frames = max(1, int(gt_stride_frames))
        self.viewer_state_file = self.output_dir / ".trajectory_gui_state.json"
        # Default: show RL, FC, RR
        self.cameras = cameras or ["RL", "FC", "RR"]
        self.current_cam_for_projection = "FC"
        self.bev_canvas_width = 560
        self.bev_canvas_height = 700
        self.speed_canvas_width = self.bev_canvas_width
        self.speed_canvas_height = 180
        self.bev_forward_scale = 6.2
        self.bev_lateral_scale = 10.0
        self.bev_origin = (self.bev_canvas_width / 2, self.bev_canvas_height - 65)
        self.stop_marker_hitboxes = []
        self.stop_tooltip_items = []
        self.speed_hover_frame_idx = None
        self.speed_hover_source = None
        self.speed_plot_rect = None
        self.gt_speed_plot_rect = None
        self.speed_edit_active = False
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = None
        self.speed_edit_original_traj = None
        self.speed_edit_original_xyz = None
        self.speed_edit_speed = None
        self.speed_edit_last_frame = None
        self.pred_speed_action_frame = None
        self.gt_speed_action_frame = None
        self.gt_stop_action_frame = None
        self.gt_edit_active = False
        self.gt_edit_mode = None
        self.gt_edit_original_xyz = None
        self.gt_edit_preview_xyz = None
        self.gt_stop_frame_idx = None
        self.trajectory_smoothness = {}
        self.traj_list_tooltip = None
        
        # Load dataset info
        self.datasets = get_dataset_names(str(self.data_root))
        self.samples = self._load_sample_index(
            start_dataset=start_dataset,
            start_clip=start_clip,
        )
        
        self.samples.sort(key=lambda x: (x[0], x[1], x[2]))
        
        if not self.samples:
            if self.gt_only:
                raise ValueError(f"No GT samples found in {self.data_root}")
            raise ValueError(f"No trajectory files found in {self.output_dir}")
        
        self.current_idx = self._resolve_start_index(
            start_index=start_index,
            start_dataset=start_dataset,
            start_clip=start_clip,
            start_t0=start_t0,
            restore_last=restore_last,
        )
        self.trajectories = []
        self.trajectory_states = {}
        self.current_traj_idx = 0
        self.image_tk = {cam: None for cam in self.cameras}
        self.cot_index = self._load_cot_index()
        self.manual_points_file = self.output_dir / "manual_points.json"
        self.manual_line_points_index = self._load_manual_line_points_index()
        self.manual_camera_line_points_index = self._load_manual_camera_line_points_index()
        self.manual_stop_points_index = self._load_manual_stop_points_index()
        self.manual_line_points = []
        self.manual_camera_line_points = []
        self.manual_stop_points = []
        self.manual_point_actions = []
        self.manual_line_points_dirty = False
        self.manual_camera_line_points_dirty = False
        self.manual_stop_points_dirty = False
        self.bezier_cluster_center_ids = self._load_bezier_cluster_center_ids()
        self.cluster_center_library = self._load_cluster_center_library()
        self.cluster_preview_traj = None
        self.cluster_preview_record = None
        self.cluster_preview_is_edited = False
        self.cluster_category_var = None
        self.cluster_choice_var = None
        self.cluster_category_combo = None
        self.cluster_choice_combo = None
        self.camera_display_meta = {}
        self.draw_line_enabled = False
        self.draw_line_var = None
        self.stop_duration_seconds = 2.0
        self.stop_duration_var = None
        self.drag_state = None
        self.visual_data_cache = OrderedDict()
        self.visual_data_cache_limit = 8
        self.camera_base_images = {}
        self.gt_future_mode = "raw"
        self.samples_by_dataset = {}
        self.clips_by_dataset = {}
        self.t0_by_dataset_clip = {}
        for sample_dataset, sample_clip, sample_t0 in self.samples:
            self.samples_by_dataset.setdefault(sample_dataset, []).append(
                (sample_clip, int(sample_t0))
            )
            self.clips_by_dataset.setdefault(sample_dataset, [])
            if sample_clip not in self.clips_by_dataset[sample_dataset]:
                self.clips_by_dataset[sample_dataset].append(sample_clip)
            self.t0_by_dataset_clip.setdefault((sample_dataset, sample_clip), []).append(int(sample_t0))
        for clips in self.clips_by_dataset.values():
            clips.sort()
        for t0_values in self.t0_by_dataset_clip.values():
            t0_values.sort()
        self.dataset_var = None
        self.clip_var = None
        self.t0_var = None
        self.dataset_combo = None
        self.clip_combo = None
        self.t0_combo = None
        
        # Current camera frame for projection (default to FC)
        self.current_cam_for_projection = "FC"
        
        # Load initial sample
        self._load_sample(self.current_idx)
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Trajectory Viewer (Enhanced)")
        self.root.geometry("1900x1150")
        self.root.configure(bg="#2b2b2b")
        
        self._create_widgets()
        self._update_display()
        
        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_sample())
        self.root.bind("<Right>", lambda e: self._next_sample())
        self.root.bind("<Up>", lambda e: self._prev_traj())
        self.root.bind("<Down>", lambda e: self._next_traj())
        self.root.bind("<Delete>", lambda e: self._delete_traj())
        self.root.bind("<BackSpace>", lambda e: self._delete_traj())
        self.root.bind("<Control-s>", lambda e: self._save_results())
        self.root.bind("<q>", lambda e: self.root.quit())
        self.root.bind("<Tab>", lambda e: self._toggle_projection_camera())
        self.root.bind("<KeyPress-minus>", lambda e: self._cycle_selected_cluster_center(-1))
        self.root.bind("<KeyPress-underscore>", lambda e: self._cycle_selected_cluster_center(-1))
        self.root.bind("<KeyPress-plus>", lambda e: self._cycle_selected_cluster_center(1))
        self.root.bind("<KeyPress-equal>", lambda e: self._cycle_selected_cluster_center(1))
        
        self.root.mainloop()

    def _load_sample_index(self, start_dataset: str = "", start_clip: str = "") -> list[tuple[str, str, int]]:
        """Build the navigable sample list from GT or generated trajectory files."""
        if self.gt_only:
            return self._load_gt_sample_index(start_dataset=start_dataset, start_clip=start_clip)
        return self._load_generated_sample_index()

    def _load_generated_sample_index(self) -> list[tuple[str, str, int]]:
        samples: list[tuple[str, str, int]] = []
        for dataset_name in self.datasets:
            dataset_output = self.output_dir / dataset_name
            if not dataset_output.exists():
                continue

            for parquet_file in dataset_output.glob("*.egomotion.parquet"):
                clip_stem = str(parquet_file.name).replace(".egomotion.parquet", "")
                try:
                    df = pd.read_parquet(parquet_file)
                    if len(df) > 0:
                        if "t0_us" in df.columns:
                            for t0_us in sorted(df["t0_us"].dropna().astype("int64").unique()):
                                samples.append((dataset_name, clip_stem, int(t0_us)))
                        else:
                            samples.append((dataset_name, clip_stem, _row_t0_us(df.iloc[0])))
                except Exception as e:
                    print(f"Warning: Could not load {parquet_file}: {e}")
        samples.sort(key=lambda x: (x[0], x[1], x[2]))
        return samples

    def _load_gt_sample_index(self, start_dataset: str = "", start_clip: str = "") -> list[tuple[str, str, int]]:
        """Build GT-only samples directly from data-egomotion at a fixed frame stride."""
        dataset_filter = start_dataset.strip()
        clip_filter = start_clip.strip()
        samples: list[tuple[str, str, int]] = []

        for dataset_name in self.datasets:
            if dataset_filter and dataset_name != dataset_filter:
                continue
            dataset_path = self.data_root / dataset_name
            egomotion_dir = dataset_path / "data-egomotion"
            if not egomotion_dir.exists():
                continue

            clip_stems = get_clip_stems_from_dataset(dataset_path)
            if not clip_stems:
                clip_stems = sorted(
                    path.name.replace(".egomotion.parquet", "")
                    for path in egomotion_dir.glob("*.egomotion.parquet")
                )

            for clip_stem in clip_stems:
                if clip_filter and clip_stem != clip_filter:
                    continue
                ego_file = egomotion_dir / f"{clip_stem}.egomotion.parquet"
                if not ego_file.exists():
                    continue
                try:
                    df = pd.read_parquet(ego_file, columns=["timestamp"])
                except Exception as e:
                    print(f"Warning: Could not load GT timestamps {ego_file}: {e}")
                    continue
                timestamps = df["timestamp"].to_numpy(dtype=np.int64)
                # load_data defaults to 16 history steps and 64 future steps.
                lo = 16
                hi = len(timestamps) - 65
                if lo > hi:
                    continue
                for idx in range(lo, hi + 1, self.gt_stride_frames):
                    samples.append((dataset_name, clip_stem, int(timestamps[idx])))

        samples.sort(key=lambda x: (x[0], x[1], x[2]))
        print(
            f"GT-only sample index: {len(samples)} samples "
            f"(stride={self.gt_stride_frames} frames)"
        )
        return samples

    def _load_cot_index(self):
        """Load CoT sidecar records keyed by dataset, clip, t0, sample."""
        cot_file = self.output_dir / "cot.jsonl"
        cot_index = {}
        if not cot_file.exists():
            return cot_index

        try:
            with open(cot_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (
                        row.get("dataset_name", ""),
                        row.get("clip_id", ""),
                        int(row.get("t0_us", 0)),
                        int(row.get("sample_idx", 0)),
                    )
                    cot_index[key] = row.get("cot", "")
        except Exception as e:
            print(f"Warning: Could not load CoT sidecar {cot_file}: {e}")
        return cot_index

    def _load_viewer_state(self) -> dict:
        """Load the last viewed sample state."""
        if not self.viewer_state_file.exists():
            return {}
        try:
            with open(self.viewer_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state if isinstance(state, dict) else {}
        except Exception as e:
            print(f"Warning: Could not load viewer state {self.viewer_state_file}: {e}")
            return {}

    def _save_viewer_state(self) -> None:
        """Persist the current sample so the next GUI launch can resume here."""
        if not (0 <= self.current_idx < len(self.samples)):
            return
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        state = {
            "sample_index": int(self.current_idx),
            "dataset_name": dataset_name,
            "clip_id": clip_stem,
            "t0_us": int(t0_us),
        }
        try:
            self.viewer_state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.viewer_state_file.with_suffix(".json.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.write("\n")
            tmp_file.replace(self.viewer_state_file)
        except Exception as e:
            print(f"Warning: Could not save viewer state {self.viewer_state_file}: {e}")

    def _find_sample_index(
        self,
        dataset_name: str = "",
        clip_stem: str = "",
        t0_us: Optional[int] = None,
    ) -> Optional[int]:
        """Find a sample by dataset, clip, and optionally t0."""
        dataset_name = dataset_name.strip()
        clip_stem = clip_stem.strip()
        for idx, (sample_dataset, sample_clip, sample_t0) in enumerate(self.samples):
            if dataset_name and sample_dataset != dataset_name:
                continue
            if clip_stem and sample_clip != clip_stem:
                continue
            if t0_us is not None and int(sample_t0) != int(t0_us):
                continue
            return idx
        return None

    def _resolve_start_index(
        self,
        start_index: Optional[int],
        start_dataset: str,
        start_clip: str,
        start_t0: Optional[int],
        restore_last: bool,
    ) -> int:
        """Resolve the initial sample index from CLI arguments or saved state."""
        has_manual_target = (
            start_index is not None
            or bool(start_dataset.strip())
            or bool(start_clip.strip())
            or start_t0 is not None
        )

        if start_index is not None:
            return int(np.clip(int(start_index) - 1, 0, len(self.samples) - 1))

        if has_manual_target:
            found = self._find_sample_index(start_dataset, start_clip, start_t0)
            if found is not None:
                return found
            print(
                "Warning: Requested start sample not found; falling back to first sample "
                f"(dataset={start_dataset!r}, clip={start_clip!r}, t0={start_t0!r})"
            )
            return 0

        if restore_last:
            state = self._load_viewer_state()
            found = self._find_sample_index(
                str(state.get("dataset_name", "")),
                str(state.get("clip_id", "")),
                int(state["t0_us"]) if "t0_us" in state else None,
            )
            if found is not None:
                return found
            if "sample_index" in state:
                return int(np.clip(int(state["sample_index"]), 0, len(self.samples) - 1))

        return 0

    def _base_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        return self.output_dir / dataset_name / f"{clip_stem}.egomotion.parquet"

    def _active_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        return self._base_traj_file(dataset_name, clip_stem)

    def _is_gt_trajectory(self, traj: dict, fallback_index: int = -1) -> bool:
        source = str(traj.get("source", "")).lower()
        sample_idx = int(traj.get("sample_idx", fallback_index))
        return source == "gt" or (not source and sample_idx == 0)

    def _first_editable_trajectory_index(self) -> int:
        for idx, traj in enumerate(self.trajectories):
            if not self._is_gt_trajectory(traj, idx):
                return idx
        return 0

    def _optimized_gt_xyz(self, gt_xyz: np.ndarray) -> Optional[np.ndarray]:
        gt = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
        if len(gt) < 2:
            return None
        speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        smoothed = _smooth_speed_profile(speed, passes=3)
        sampled = _resample_xyz_by_speed_profile(gt, smoothed)
        if sampled is None:
            return None
        limited = _acceleration_limited_resample_path(sampled)
        if limited is not None:
            sampled = limited
        return sampled.astype(np.float32)

    def _auto_optimize_gt_row_in_loaded_df(
        self,
        df: pd.DataFrame,
        t0_us: int,
    ) -> pd.DataFrame:
        if self.gt_only or not AUTO_OPTIMIZE_GT_ON_LOAD or df.empty:
            return df

        if "t0_us" in df.columns:
            current_mask = df["t0_us"].astype("int64") == int(t0_us)
        else:
            current_mask = pd.Series(True, index=df.index)

        if "source" in df.columns:
            gt_mask = df["source"].astype(str).str.lower() == "gt"
        elif "sample_idx" in df.columns:
            gt_mask = df["sample_idx"].astype("int64") == 0
        else:
            return df

        matched = df.index[current_mask & gt_mask].tolist()
        if not matched:
            return df
        row_idx = matched[0]

        if GT_SPEED_OPTIMIZED_COLUMN in df.columns:
            try:
                if bool(df.at[row_idx, GT_SPEED_OPTIMIZED_COLUMN]):
                    return df
            except (TypeError, ValueError):
                pass

        row = df.loc[row_idx]
        try:
            gt_xyz = np.column_stack([
                np.asarray(row["x"], dtype=np.float64),
                np.asarray(row["y"], dtype=np.float64),
                np.asarray(row["z"], dtype=np.float64),
            ])
        except Exception as exc:
            print(f"Warning: could not auto-optimize GT speed for t0={int(t0_us)}: {exc}")
            return df

        optimized = self._optimized_gt_xyz(gt_xyz)
        if optimized is None:
            return df

        df = df.copy()
        if GT_SPEED_OPTIMIZED_COLUMN not in df.columns:
            df[GT_SPEED_OPTIMIZED_COLUMN] = False
        components = self._trajectory_components_from_xyz(optimized)
        for key, values in components.items():
            if key in df.columns:
                df.at[row_idx, key] = np.asarray(values).tolist()
        df.at[row_idx, GT_SPEED_OPTIMIZED_COLUMN] = True
        return df

    def _load_cluster_center_library(self) -> dict[str, list[dict]]:
        """Load editable cluster center library from four category txt files."""
        kmeans_dir = Path(__file__).resolve().parent / "k_means"
        library = {category: [] for category in CLUSTER_CATEGORY_ORDER}

        def _append_record(category: str, center_id: int, count: int, traj: np.ndarray, source: str):
            is_bezier_added = int(center_id) in self.bezier_cluster_center_ids.get(category, set())
            final_right = -float(traj[-1, 1])
            label_prefix = "B" if is_bezier_added else "C"
            label = (
                f"{label_prefix}{int(center_id):02d} n={int(count)} "
                f"right={final_right:.1f} fwd={float(traj[-1, 0]):.1f}"
            )
            library[category].append({
                "id": int(center_id),
                "label": label,
                "trajectory": traj.astype(np.float32),
                "source": source,
                "count": int(count),
                "category": category,
                "is_bezier_added": is_bezier_added,
            })

        def _parse_category_file(path: Path, category: str):
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts[0] != "CENTER" or len(parts) < 4:
                        continue
                    try:
                        center_id = int(parts[1])
                        count = int(parts[2])
                        xy = np.asarray(parts[3:], dtype=np.float64).reshape(-1, 2)
                    except ValueError:
                        continue
                    traj = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
                    _append_record(category, center_id, count, traj, path.name)

        category_files_exist = False
        for category, filename in CLUSTER_CATEGORY_FILES.items():
            path = kmeans_dir / filename
            if path.exists():
                category_files_exist = True
            _parse_category_file(path, category)
        if category_files_exist:
            for records in library.values():
                records.sort(key=lambda item: int(item["id"]))
            return library

        def _parse_result_file(path: Path, mode: str):
            if not path.exists():
                return
            headers = {}
            centers = {}
            counts = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        payload = line[1:].strip()
                        if "=" in payload:
                            key, value = payload.split("=", 1)
                            headers[key.strip()] = value.strip()
                        continue
                    parts = line.split()
                    if parts[0] != "CENTER":
                        continue
                    cluster_id = int(parts[1])
                    counts[cluster_id] = int(parts[2])
                    xy = np.asarray(parts[3:], dtype=np.float64).reshape(-1, 2)
                    traj = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
                    centers[cluster_id] = traj

            straight_clusters = int(headers.get("straight_clusters", "0") or 0)
            for cluster_id, traj in sorted(centers.items()):
                final_right = -float(traj[-1, 1])
                if mode == "stop":
                    category = "stop"
                elif straight_clusters and cluster_id < straight_clusters:
                    category = "straight"
                else:
                    category = "right" if final_right >= 0.0 else "left"
                _append_record(category, cluster_id, counts.get(cluster_id, 0), traj, path.name)

        _parse_result_file(kmeans_dir / "kmeans_results.txt", mode="main")
        _parse_result_file(kmeans_dir / "kmeans_stop_results.txt", mode="stop")
        for category, records in library.items():
            if records:
                self._write_cluster_category_file(category, records)
        return library

    def _cluster_category_file(self, category: str) -> Path:
        filename = CLUSTER_CATEGORY_FILES[category]
        return Path(__file__).resolve().parent / "k_means" / filename

    def _bezier_cluster_center_meta_file(self) -> Path:
        return Path(__file__).resolve().parent / "k_means" / "bezier_centers.json"

    def _load_bezier_cluster_center_ids(self) -> dict[str, set[int]]:
        """Load ids for centers created by Save Bezier Center."""
        ids = {category: set() for category in CLUSTER_CATEGORY_FILES}
        path = self._bezier_cluster_center_meta_file()
        if not path.exists():
            return ids

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load Bezier center metadata {path}: {e}")
            return ids

        raw_centers = data.get("centers", {}) if isinstance(data, dict) else {}
        for category, values in raw_centers.items():
            if category not in ids or not isinstance(values, list):
                continue
            for value in values:
                try:
                    ids[category].add(int(value))
                except (TypeError, ValueError):
                    continue
        return ids

    def _write_bezier_cluster_center_ids(self) -> None:
        """Persist ids for centers created by Save Bezier Center."""
        path = self._bezier_cluster_center_meta_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "centers": {
                category: sorted(int(value) for value in values)
                for category, values in sorted(self.bezier_cluster_center_ids.items())
                if values
            },
        }
        tmp_file = path.with_suffix(".json.tmp")
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        tmp_file.replace(path)

    def _write_cluster_category_file(self, category: str, records: list[dict]) -> None:
        path = self._cluster_category_file(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("# cluster_centers_xy_v1\n")
            f.write(f"# category={category}\n")
            f.write("# ego-local xy: x=forward, y=left; plot right=-y\n")
            f.write("# columns: CENTER center_id count x0 y0 ... x63 y63\n")
            for record in sorted(records, key=lambda item: int(item["id"])):
                traj = np.asarray(record["trajectory"], dtype=np.float64)
                values = " ".join(f"{v:.6f}" for v in traj[:, :2].reshape(-1))
                f.write(f"CENTER {int(record['id'])} {int(record.get('count', 1))} {values}\n")

    def _load_manual_line_points_index(self) -> dict[tuple[str, str, int], list[dict]]:
        """Load saved manual BEV line points keyed by dataset, clip, and t0."""
        if not self.manual_points_file.exists():
            return {}

        try:
            with open(self.manual_points_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manual line points {self.manual_points_file}: {e}")
            return {}

        records = data.get("samples", []) if isinstance(data, dict) else data
        points_index = {}
        for record in records:
            try:
                key = (
                    record.get("dataset_name", ""),
                    record.get("clip_id", ""),
                    int(record.get("t0_us", 0)),
                )
                points = []
                for point in record.get("line_points", []):
                    points.append({
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                        "z": float(point.get("z", 0.0)),
                    })
                if points:
                    points_index[key] = points
            except Exception as e:
                print(f"Warning: Skipping invalid manual line point record: {e}")
        return points_index

    def _load_manual_camera_line_points_index(self) -> dict[tuple[str, str, int], list[dict]]:
        """Load saved manual image line points keyed by dataset, clip, and t0."""
        if not self.manual_points_file.exists():
            return {}

        try:
            with open(self.manual_points_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manual camera line points {self.manual_points_file}: {e}")
            return {}

        records = data.get("samples", []) if isinstance(data, dict) else data
        points_index = {}
        for record in records:
            try:
                key = (
                    record.get("dataset_name", ""),
                    record.get("clip_id", ""),
                    int(record.get("t0_us", 0)),
                )
                points = []
                for point in record.get("camera_line_points", []):
                    points.append({
                        "camera": str(point.get("camera", "")),
                        "u": float(point.get("u", 0.0)),
                        "v": float(point.get("v", 0.0)),
                    })
                if points:
                    points_index[key] = points
            except Exception as e:
                print(f"Warning: Skipping invalid manual camera line point record: {e}")
        return points_index

    def _load_manual_stop_points_index(self) -> dict[tuple[str, str, int], list[dict]]:
        """Load saved stop markers keyed by dataset, clip, and t0."""
        if not self.manual_points_file.exists():
            return {}

        try:
            with open(self.manual_points_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manual stop points {self.manual_points_file}: {e}")
            return {}

        records = data.get("samples", []) if isinstance(data, dict) else data
        stop_index = {}
        for record in records:
            try:
                key = (
                    record.get("dataset_name", ""),
                    record.get("clip_id", ""),
                    int(record.get("t0_us", 0)),
                )
                stops = []
                for stop in record.get("stop_points", []):
                    stops.append({
                        "fraction": float(np.clip(float(stop.get("fraction", 0.0)), 0.0, 1.0)),
                        "duration_s": float(np.clip(float(stop.get("duration_s", 2.0)), 0.1, 6.0)),
                    })
                if stops:
                    stop_index[key] = sorted(stops, key=lambda item: item["fraction"])
            except Exception as e:
                print(f"Warning: Skipping invalid manual stop point record: {e}")
        return stop_index

    def _current_manual_points_key(self) -> tuple[str, str, int]:
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        return dataset_name, clip_stem, int(t0_us)

    def _current_sample_key(self) -> tuple[str, str, int]:
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        return dataset_name, clip_stem, int(t0_us)

    def _write_manual_points_index(self):
        records = []
        all_keys = (
            set(self.manual_line_points_index)
            | set(self.manual_camera_line_points_index)
            | set(self.manual_stop_points_index)
        )
        for dataset_name, clip_stem, t0_us in sorted(all_keys):
            line_points = self.manual_line_points_index.get((dataset_name, clip_stem, t0_us), [])
            camera_line_points = self.manual_camera_line_points_index.get((dataset_name, clip_stem, t0_us), [])
            stop_points = self.manual_stop_points_index.get((dataset_name, clip_stem, t0_us), [])
            if not line_points and not camera_line_points and not stop_points:
                continue
            records.append({
                "dataset_name": dataset_name,
                "clip_id": clip_stem,
                "t0_us": int(t0_us),
                "line_points": [
                    {
                        "x": float(point["x"]),
                        "y": float(point["y"]),
                        "z": float(point.get("z", 0.0)),
                    }
                    for point in line_points
                ],
                "camera_line_points": [
                    {
                        "camera": point["camera"],
                        "u": float(point["u"]),
                        "v": float(point["v"]),
                    }
                    for point in camera_line_points
                ],
                "stop_points": [
                    {
                        "fraction": float(np.clip(float(point["fraction"]), 0.0, 1.0)),
                        "duration_s": float(max(0.1, point["duration_s"])),
                    }
                    for point in stop_points
                ],
            })

        self.manual_points_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.manual_points_file.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "samples": records}, f, indent=2)
            f.write("\n")
        tmp_file.replace(self.manual_points_file)
    
    def _load_sample(self, idx: int):
        """Load a sample by index."""
        if idx < 0 or idx >= len(self.samples):
            return
        
        self.current_idx = idx
        dataset_name, clip_stem, t0_us = self.samples[idx]
        self.gt_future_mode = "raw"
        self.speed_hover_frame_idx = None
        self.speed_hover_source = None
        self._cancel_speed_edit(redraw=False)
        self._cancel_gt_speed_edit(redraw=False)
        
        self.trajectories = []
        if not self.gt_only:
            # Load trajectories from parquet
            traj_file = self._active_traj_file(dataset_name, clip_stem)
            df = pd.read_parquet(traj_file)
            optimized_df = self._auto_optimize_gt_row_in_loaded_df(df, int(t0_us))
            if optimized_df is not df:
                optimized_df.to_parquet(traj_file, index=False)
                df = optimized_df
            if "t0_us" in df.columns:
                df = df[df["t0_us"].astype("int64") == int(t0_us)]

            for row_idx, (_, row) in enumerate(df.iterrows()):
                sample_idx = int(row["sample_idx"]) if "sample_idx" in row else row_idx
                cot_key = (dataset_name, clip_stem, int(t0_us), sample_idx)
                self.trajectories.append({
                    "sample_idx": sample_idx,
                    "source": str(row.get("source", "")) if hasattr(row, "get") else "",
                    "timestamp": row["timestamp"],
                    "x": np.array(row["x"]),
                    "y": np.array(row["y"]),
                    "z": np.array(row["z"]),
                    "qx": np.array(row["qx"]),
                    "qy": np.array(row["qy"]),
                    "qz": np.array(row["qz"]),
                    "qw": np.array(row["qw"]),
                    "vx": np.array(row["vx"]),
                    "vy": np.array(row["vy"]),
                    "vz": np.array(row["vz"]),
                    "curvature": np.array(row["curvature"]),
                    "cot": self.cot_index.get(cot_key, ""),
                })
        
        # Initialize states (all kept by default)
        self.trajectory_states = {i: True for i in range(len(self.trajectories))}
        self.current_traj_idx = self._first_editable_trajectory_index()
        self.manual_line_points = [
            dict(point)
            for point in self.manual_line_points_index.get((dataset_name, clip_stem, int(t0_us)), [])
        ]
        self.manual_camera_line_points = [
            dict(point)
            for point in self.manual_camera_line_points_index.get((dataset_name, clip_stem, int(t0_us)), [])
        ]
        self.manual_stop_points = [
            dict(point)
            for point in self.manual_stop_points_index.get((dataset_name, clip_stem, int(t0_us)), [])
        ]
        self.manual_point_actions = []
        self.manual_line_points_dirty = False
        self.manual_camera_line_points_dirty = False
        self.manual_stop_points_dirty = False
        sample_key = (dataset_name, clip_stem, int(t0_us))
        self.camera_base_images = {}
        
        # Load original data for visualization
        try:
            cached_visual = self.visual_data_cache.get(sample_key)
            if cached_visual is not None:
                self.visual_data_cache.move_to_end(sample_key)
                self.conv_data, self.calibration = cached_visual
            else:
                # Load only the one frame displayed by the GUI. Inference uses 4 frames,
                # but this viewer only renders frames[cam_idx, 0].
                self.conv_data = load_data(
                    str(self.data_root),
                    clip_stem,
                    dataset_name,
                    t0_us=t0_us,
                    num_frames=1,
                    target_image_hw=(1080, 1920),
                    cameras=self.cameras,
                )
                
                # Load calibration for this segment
                calib_dataset = dataset_name.replace('_converted', '')
                self.calibration = load_calibration_for_segment(
                    str(self.calibration_dir),
                    calib_dataset,
                    clip_stem,
                )
                self.visual_data_cache[sample_key] = (self.conv_data, self.calibration)
                while len(self.visual_data_cache) > self.visual_data_cache_limit:
                    self.visual_data_cache.popitem(last=False)
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
            self.conv_data = None
            self.calibration = None
        self._save_viewer_state()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar
        title_frame = tk.Frame(main_frame, bg="#2b2b2b")
        title_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            title_frame, text="Trajectory Viewer",
            font=("Arial", 16, "bold"), fg="white", bg="#2b2b2b",
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.nav_label = tk.Label(
            title_frame, text="",
            font=("Arial", 12), fg="#888888", bg="#2b2b2b",
        )
        self.nav_label.pack(side=tk.RIGHT)

        jump_frame = tk.Frame(main_frame, bg="#2b2b2b")
        jump_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Label(
            jump_frame, text="Dataset",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.dataset_var,
            values=self.datasets,
            state="readonly",
            width=30,
        )
        self.dataset_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_combo_selected)

        tk.Label(
            jump_frame, text="Clip",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.clip_var = tk.StringVar()
        self.clip_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.clip_var,
            state="readonly",
            width=24,
        )
        self.clip_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.clip_combo.bind("<<ComboboxSelected>>", self._on_clip_combo_selected)

        tk.Label(
            jump_frame, text="t0",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.t0_var = tk.StringVar()
        self.t0_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.t0_var,
            state="readonly",
            width=22,
        )
        self.t0_combo.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            jump_frame, text="Jump", command=self._jump_to_selected_sample,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            jump_frame, text="Current", command=self._sync_sample_selectors,
        ).pack(side=tk.LEFT)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg="#2b2b2b")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Bird's eye view
        left_frame = tk.Frame(content_frame, bg="#1e1e1e")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(
            left_frame, text="Bird's Eye View",
            font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e",
        ).pack(pady=5)
        
        self.traj_canvas = tk.Canvas(
            left_frame, width=self.bev_canvas_width, height=self.bev_canvas_height,
            bg="#1e1e1e", highlightthickness=0,
        )
        self.traj_canvas.pack(padx=5, pady=5)
        self.traj_canvas.bind("<Button-1>", self._on_canvas_click)
        self.traj_canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.traj_canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.traj_canvas.bind("<Button-3>", self._on_canvas_right_down)
        self.traj_canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.traj_canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.traj_canvas.bind("<Motion>", self._on_traj_canvas_motion)
        self.traj_canvas.bind("<Leave>", lambda _event: self._hide_stop_tooltip())

        pred_speed_header = tk.Frame(left_frame, bg="#1e1e1e")
        pred_speed_header.pack(fill=tk.X, pady=(8, 2))
        tk.Label(
            pred_speed_header, text="Diversity Speed Profile",
            font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e",
        ).pack(side=tk.LEFT)
        tk.Button(
            pred_speed_header,
            text="优化速度曲线",
            command=self._optimize_pred_speed_curve,
            bg="#34495e",
            fg="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=(0, 5))
        self.speed_canvas = tk.Canvas(
            left_frame,
            width=self.speed_canvas_width,
            height=self.speed_canvas_height,
            bg="#171717",
            highlightthickness=0,
        )
        self.speed_canvas.pack(padx=5, pady=(0, 5))
        self.speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "pred"))
        self.speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("pred", None))
        self.pred_speed_action_frame = tk.Frame(left_frame, bg="#1e1e1e")
        tk.Button(
            self.pred_speed_action_frame,
            text="接受",
            command=self._save_speed_edit,
            bg="#1f8f5f",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.pred_speed_action_frame,
            text="取消",
            command=lambda: self._cancel_speed_edit(redraw=True),
            bg="#7f8c8d",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)

        self.gt_speed_header_frame = tk.Frame(left_frame, bg="#1e1e1e")
        self.gt_speed_header_frame.pack(fill=tk.X, pady=(6, 2))
        tk.Label(
            self.gt_speed_header_frame, text="GT Speed Profile",
            font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e",
        ).pack(side=tk.LEFT)
        tk.Button(
            self.gt_speed_header_frame,
            text="优化速度曲线",
            command=self._optimize_gt_speed_curve,
            bg="#5d6d7e",
            fg="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=(0, 5))
        tk.Button(
            self.gt_speed_header_frame,
            text="停车添加",
            command=self._start_gt_stop_add,
            bg="#8e3b35",
            fg="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=(0, 5))
        self.gt_speed_canvas = tk.Canvas(
            left_frame,
            width=self.speed_canvas_width,
            height=self.speed_canvas_height,
            bg="#171717",
            highlightthickness=0,
        )
        self.gt_speed_canvas.pack(padx=5, pady=(0, 5))
        self.gt_speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "gt"))
        self.gt_speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("gt", None))
        self.gt_speed_canvas.bind("<Button-1>", self._on_gt_speed_canvas_click)
        self.gt_speed_action_frame = tk.Frame(left_frame, bg="#1e1e1e")
        tk.Button(
            self.gt_speed_action_frame,
            text="接受",
            command=self._save_gt_speed_edit,
            bg="#1f8f5f",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.gt_speed_action_frame,
            text="取消",
            command=lambda: self._cancel_gt_speed_edit(redraw=True),
            bg="#7f8c8d",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)
        self.gt_stop_action_frame = tk.Frame(left_frame, bg="#1e1e1e")
        tk.Button(
            self.gt_stop_action_frame,
            text="保存",
            command=self._save_gt_speed_edit,
            bg="#1f8f5f",
            fg="white",
            padx=14,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.gt_stop_action_frame,
            text="取消",
            command=lambda: self._cancel_gt_speed_edit(redraw=True),
            bg="#7f8c8d",
            fg="white",
            padx=14,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.gt_stop_action_frame,
            text="撤回",
            command=self._undo_gt_stop_add,
            bg="#a56a22",
            fg="white",
            padx=14,
        ).pack(side=tk.LEFT, padx=4)
        
        # Middle panel - Camera images with projection
        middle_frame = tk.Frame(content_frame, bg="#2b2b2b")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10)
        
        # Camera selection
        cam_select_frame = tk.Frame(middle_frame, bg="#2b2b2b")
        cam_select_frame.pack(fill=tk.X)
        
        tk.Label(
            cam_select_frame, text="Projection Camera:",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=5)
        
        self.cam_var = tk.StringVar(value=self.current_cam_for_projection)
        for cam in ["FL", "FC", "FR", "RL", "RC", "RR"]:
            tk.Radiobutton(
                cam_select_frame, text=cam, variable=self.cam_var, value=cam,
                command=lambda c=cam: self._set_projection_camera(c),
                bg="#2b2b2b", fg="white", selectcolor="#444444",
                activebackground="#2b2b2b", activeforeground="white",
            ).pack(side=tk.LEFT, padx=2)
        
        # Camera image labels. Rear cameras sit side-by-side above the larger FC view.
        self.camera_labels = {}

        def create_camera_panel(parent, cam, side=tk.TOP, expand=False):
            cam_frame = tk.Frame(parent, bg="#1e1e1e")
            cam_frame.pack(side=side, expand=expand, padx=4, pady=4)

            tk.Label(
                cam_frame, text=cam,
                font=("Arial", 10, "bold"), fg="white", bg="#1e1e1e",
            ).pack()

            self.camera_labels[cam] = tk.Label(cam_frame, bg="#1e1e1e")
            self.camera_labels[cam].pack()
            self.camera_labels[cam].bind(
                "<Button-1>",
                lambda event, camera=cam: self._on_camera_click(event, camera),
            )
            self.camera_labels[cam].bind(
                "<Button-3>",
                lambda event, camera=cam: self._on_camera_right_down(event, camera),
            )
            self.camera_labels[cam].bind(
                "<B3-Motion>",
                lambda event, camera=cam: self._on_camera_right_drag(event, camera),
            )
            self.camera_labels[cam].bind("<ButtonRelease-3>", self._on_right_release)

        top_camera_row = tk.Frame(middle_frame, bg="#2b2b2b")
        top_camera_row.pack(fill=tk.X, pady=(5, 4))
        main_camera_area = tk.Frame(middle_frame, bg="#2b2b2b")
        main_camera_area.pack(fill=tk.BOTH, expand=True)

        top_cameras = [cam for cam in ("RL", "RR") if cam in self.cameras]
        remaining_top_cameras = [
            cam for cam in self.cameras
            if cam not in top_cameras and cam != "FC"
        ]
        top_cameras.extend(remaining_top_cameras)

        for cam in top_cameras:
            create_camera_panel(top_camera_row, cam, side=tk.LEFT, expand=True)

        if "FC" in self.cameras:
            create_camera_panel(main_camera_area, "FC", side=tk.TOP, expand=False)
        
        # Right panel - Trajectory list
        right_frame = tk.Frame(content_frame, bg="#2b2b2b", width=460)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        
        tk.Label(
            right_frame, text="Trajectories:",
            font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
        ).pack(anchor=tk.W, pady=(0, 5))
        
        list_frame = tk.Frame(right_frame, bg="#2b2b2b")
        list_frame.pack(fill=tk.X, pady=(0, 6))
        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.traj_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scroll.set,
            font=("Courier", 10), bg="#1e1e1e", fg="white",
            selectbackground="#444444", selectforeground="white", height=20,
            width=56,
            justify=tk.LEFT,
        )
        self.traj_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.traj_listbox.yview)
        self.traj_listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.traj_listbox.bind("<Motion>", self._on_traj_list_motion)
        self.traj_listbox.bind("<Leave>", lambda _event: self._hide_traj_list_tooltip())

        tk.Label(
            right_frame, text="CoT:",
            font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
        ).pack(anchor=tk.W, pady=(10, 5))

        self.cot_text = tk.Text(
            right_frame,
            height=12,
            width=42,
            wrap=tk.WORD,
            font=("Arial", 9),
            bg="#1e1e1e",
            fg="#dddddd",
            insertbackground="white",
        )
        self.cot_text.pack(fill=tk.BOTH, expand=False)
        self.cot_text.configure(state=tk.DISABLED)

        cluster_frame = tk.Frame(right_frame, bg="#2b2b2b")
        cluster_frame.pack(fill=tk.X, pady=(10, 4))
        tk.Label(
            cluster_frame, text="Cluster Centers:",
            font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
        ).pack(anchor=tk.W, pady=(0, 5))

        cluster_select_frame = tk.Frame(cluster_frame, bg="#2b2b2b")
        cluster_select_frame.pack(fill=tk.X)
        cluster_button_frame = tk.Frame(cluster_frame, bg="#2b2b2b")
        cluster_button_frame.pack(fill=tk.X, pady=(5, 0))

        self.cluster_category_var = tk.StringVar(value="stop")
        self.cluster_category_combo = ttk.Combobox(
            cluster_select_frame,
            textvariable=self.cluster_category_var,
            values=CLUSTER_CATEGORY_ORDER,
            state="readonly",
            width=12,
        )
        self.cluster_category_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_category_combo.bind(
            "<<ComboboxSelected>>",
            self._on_cluster_category_selected,
        )

        self.cluster_choice_var = tk.StringVar()
        self.cluster_choice_combo = ttk.Combobox(
            cluster_select_frame,
            textvariable=self.cluster_choice_var,
            state="readonly",
            width=26,
        )
        self.cluster_choice_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_choice_combo.bind(
            "<<ComboboxSelected>>",
            self._on_cluster_choice_selected,
        )

        tk.Button(
            cluster_button_frame, text="-", command=lambda: self._cycle_selected_cluster_center(-1),
            bg="#7f8c8d", fg="white", padx=8, width=2,
        ).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(
            cluster_button_frame, text="+", command=lambda: self._cycle_selected_cluster_center(1),
            bg="#7f8c8d", fg="white", padx=8, width=2,
        ).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(
            cluster_button_frame, text="Confirm Save", command=self._save_selected_cluster_center_trajectory,
            bg="#ba6f1e", fg="white", padx=8,
        ).pack(side=tk.LEFT)
        self._refresh_cluster_choice_values()
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg="#2b2b2b")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = tk.Label(
            status_frame, text="",
            font=("Arial", 10), fg="#888888", bg="#2b2b2b",
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Buttons
        controls_frame = tk.Frame(main_frame, bg="#2b2b2b")
        controls_frame.pack(fill=tk.X, pady=(8, 0))
        btn_frame = tk.Frame(controls_frame, bg="#2b2b2b")
        btn_frame.pack(anchor=tk.E, pady=(0, 6))
        btn_frame_2 = tk.Frame(controls_frame, bg="#2b2b2b")
        btn_frame_2.pack(anchor=tk.E)

        control_font = ("Arial", 10, "bold")
        control_button_opts = {
            "font": control_font,
            "padx": 14,
            "pady": 6,
        }

        self.draw_line_var = tk.BooleanVar(value=self.draw_line_enabled)
        tk.Checkbutton(
            btn_frame, text="Draw Bezier", variable=self.draw_line_var,
            command=self._toggle_draw_line,
            bg="#2b2b2b", fg="white", selectcolor="#444444",
            activebackground="#2b2b2b", activeforeground="white",
            font=control_font,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="Add Final Stop", command=self._add_final_stop_point,
            bg="#b03a2e", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(
            btn_frame, text="Stop Time(s)", bg="#2b2b2b", fg="#dddddd",
            font=control_font,
        ).pack(side=tk.LEFT, padx=(4, 2))
        self.stop_duration_var = tk.DoubleVar(value=self.stop_duration_seconds)
        stop_spin = tk.Spinbox(
            btn_frame,
            from_=0.5,
            to=5.0,
            increment=0.5,
            width=4,
            textvariable=self.stop_duration_var,
            command=self._update_stop_duration_seconds,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white",
            font=control_font,
        )
        stop_spin.pack(side=tk.LEFT, padx=(0, 5))
        stop_spin.bind("<Return>", lambda _event: self._update_stop_duration_seconds())
        stop_spin.bind("<FocusOut>", lambda _event: self._update_stop_duration_seconds())

        tk.Button(
            btn_frame, text="Undo Control", command=self._undo_manual_point,
            bg="#7f8c8d", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame_2, text="Save Curve Traj", command=self._save_manual_bezier_trajectory,
            bg="#16a085", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Save Bezier Center", command=self._save_manual_bezier_as_cluster_center,
            bg="#d68910", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Delete Current Bezier Center", command=self._delete_current_bezier_cluster_center,
            bg="#a93226", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame_2, text="Repair GT", command=self._repair_gt_future,
            bg="#9b59b6", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Restore GT", command=self._restore_gt_future,
            bg="#566573", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame_2, text="Delete (Del)", command=self._delete_traj,
            bg="#c0392b", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame_2, text="Keep", command=self._keep_traj,
            bg="#27ae60", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame_2, text="Save (Ctrl+S)", command=self._save_results,
            bg="#2980b9", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        # Help
        help_frame = tk.Frame(main_frame, bg="#2b2b2b")
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            help_frame,
            text="Controls: ←/→ Samples | ↑/↓ Trajectories | +/- Cluster centers | Draw Bezier adds controls | Right-drag controls in BEV/FC edits the same curve | Add Final Stop uses the Bezier endpoint and Stop Time(s) | Repair/Restore GT toggles raw vs velocity-integrated GT | Ctrl+S Save | Q Quit",
            font=("Arial", 9), fg="#666666", bg="#2b2b2b",
        ).pack()
    
    def _set_projection_camera(self, cam):
        """Set camera for trajectory projection."""
        self.current_cam_for_projection = cam
        self._update_display()

    def _cluster_records_for_current_category(self) -> list[dict]:
        if self.cluster_category_var is None:
            return []
        return self.cluster_center_library.get(self.cluster_category_var.get(), [])

    def _refresh_cluster_choice_values(self):
        if self.cluster_choice_combo is None or self.cluster_choice_var is None:
            return
        values = [record["label"] for record in self._cluster_records_for_current_category()]
        self.cluster_choice_combo["values"] = values
        if values:
            if self.cluster_choice_var.get() not in values:
                self.cluster_choice_var.set(values[0])
        else:
            self.cluster_choice_var.set("")

    def _on_cluster_category_selected(self, _event=None):
        self._refresh_cluster_choice_values()
        self._preview_selected_cluster_center(show_warning=False)

    def _on_cluster_choice_selected(self, _event=None):
        self._preview_selected_cluster_center(show_warning=False)

    def _cycle_selected_cluster_center(self, direction: int):
        records = self._cluster_records_for_current_category()
        if not records or self.cluster_choice_var is None:
            self.cluster_preview_record = None
            self.cluster_preview_traj = None
            self._update_display()
            return

        labels = [record["label"] for record in records]
        current = self.cluster_choice_var.get()
        if current in labels:
            current_idx = labels.index(current)
        else:
            current_idx = 0
        next_idx = (current_idx + direction) % len(labels)
        self.cluster_choice_var.set(labels[next_idx])
        self._preview_selected_cluster_center(show_warning=False)

    def _selected_cluster_record(self) -> Optional[dict]:
        label = self.cluster_choice_var.get() if self.cluster_choice_var is not None else ""
        for record in self._cluster_records_for_current_category():
            if record["label"] == label:
                return record
        return None

    def _preview_selected_cluster_center(self, show_warning: bool = True):
        record = self._selected_cluster_record()
        if record is None:
            self.cluster_preview_record = None
            self.cluster_preview_traj = None
            self.cluster_preview_is_edited = False
            if show_warning:
                messagebox.showwarning("No Cluster", "No cluster center is available for this category.")
            self._update_display()
            return
        self.cluster_preview_record = record
        raw_traj = np.asarray(record["trajectory"], dtype=np.float32)
        smoothed_traj = _prepare_cluster_preview_trajectory(
            raw_traj,
            initial_speed_mps=self._estimate_t0_speed_mps(),
        )
        if smoothed_traj is None:
            self.cluster_preview_traj = None
            self.cluster_preview_is_edited = False
            if show_warning:
                messagebox.showwarning(
                    "Cluster Unavailable",
                    "This cluster center cannot be fit to the current t0 speed and acceleration limits.",
                )
            self._update_display()
            return
        self.cluster_preview_traj = smoothed_traj.astype(np.float32)
        self.cluster_preview_is_edited = False
        self._update_display()

    def _current_cluster_preview_label(self) -> str:
        if self.cluster_preview_traj is None or len(self.cluster_preview_traj) == 0:
            return "Cluster"
        cluster_id = "Cluster"
        count = None
        if self.cluster_preview_record is not None:
            cluster_id = f"C{int(self.cluster_preview_record.get('id', 0)):02d}"
            count = self.cluster_preview_record.get("count")
        final_forward = float(self.cluster_preview_traj[-1, 0])
        final_right = -float(self.cluster_preview_traj[-1, 1])
        suffix = " edited" if self.cluster_preview_is_edited else ""
        if count is None:
            return f"{cluster_id}{suffix} right={final_right:.1f} fwd={final_forward:.1f}"
        return f"{cluster_id}{suffix} n={int(count)} right={final_right:.1f} fwd={final_forward:.1f}"

    def _save_selected_cluster_center_trajectory(self):
        if self.gt_only:
            messagebox.showwarning(
                "GT Only",
                "GT-only mode does not load or append generated trajectory parquet files.",
            )
            return
        if self.cluster_preview_traj is None:
            self._preview_selected_cluster_center()
        if self.cluster_preview_traj is None:
            return

        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        smoothed_preview = _prepare_cluster_preview_trajectory(
            self.cluster_preview_traj,
            initial_speed_mps=self._estimate_t0_speed_mps(),
        )
        if smoothed_preview is None:
            messagebox.showwarning(
                "Save Failed",
                "The dragged cluster trajectory cannot be fit within the t0 speed and acceleration limits.",
            )
            return
        self.cluster_preview_traj = smoothed_preview.astype(np.float32)
        row = self._manual_trajectory_to_row(self.cluster_preview_traj, df)

        for column in df.columns:
            if column not in row:
                row[column] = None
        new_row = pd.DataFrame([{column: row[column] for column in df.columns}])
        df_appended = pd.concat([df, new_row], ignore_index=True)

        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df_appended.to_parquet(traj_file, index=False)

        self._load_sample(self.current_idx)
        self._update_display()

        cluster_label = (
            self._current_cluster_preview_label()
            if self.cluster_preview_traj is not None
            else ""
        )
        messagebox.showinfo(
            "Saved Cluster Center",
            (
                f"Appended {cluster_label} as sample_idx={row['sample_idx']} "
                f"for t0={int(t0_us)} to {traj_file}"
            ),
        )

    def _sync_sample_selectors(self):
        """Sync dataset/clip/t0 controls to the current sample."""
        if self.dataset_var is None or not (0 <= self.current_idx < len(self.samples)):
            return

        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        self.dataset_var.set(dataset_name)
        if self.clip_combo is not None:
            clips = self.clips_by_dataset.get(dataset_name, [])
            self.clip_combo.configure(values=clips)
        if self.clip_var is not None:
            self.clip_var.set(clip_stem)
        if self.t0_combo is not None:
            t0_values = [str(value) for value in self.t0_by_dataset_clip.get((dataset_name, clip_stem), [])]
            self.t0_combo.configure(values=t0_values)
        if self.t0_var is not None:
            self.t0_var.set(str(int(t0_us)))

    def _on_dataset_combo_selected(self, _event=None):
        dataset_name = self.dataset_var.get() if self.dataset_var is not None else ""
        clips = self.clips_by_dataset.get(dataset_name, [])
        if self.clip_combo is not None:
            self.clip_combo.configure(values=clips)
        if clips and self.clip_var is not None:
            self.clip_var.set(clips[0])
            self._on_clip_combo_selected()

    def _on_clip_combo_selected(self, _event=None):
        dataset_name = self.dataset_var.get() if self.dataset_var is not None else ""
        clip_stem = self.clip_var.get() if self.clip_var is not None else ""
        t0_values = [str(value) for value in self.t0_by_dataset_clip.get((dataset_name, clip_stem), [])]
        if self.t0_combo is not None:
            self.t0_combo.configure(values=t0_values)
        if t0_values and self.t0_var is not None:
            self.t0_var.set(t0_values[0])

    def _jump_to_selected_sample(self):
        dataset_name = self.dataset_var.get() if self.dataset_var is not None else ""
        clip_stem = self.clip_var.get() if self.clip_var is not None else ""
        t0_text = self.t0_var.get() if self.t0_var is not None else ""
        try:
            t0_us = int(t0_text) if t0_text else None
        except ValueError:
            t0_us = None
        target_idx = self._find_sample_index(dataset_name, clip_stem, t0_us)
        if target_idx is None:
            messagebox.showwarning(
                "Sample Not Found",
                f"No sample found for dataset={dataset_name}, clip={clip_stem}, t0={t0_text}",
            )
            return
        self._load_sample(target_idx)
        self._update_display()
    
    def _toggle_projection_camera(self):
        """Toggle to next camera for projection."""
        cams = ["FL", "FC", "FR", "RL", "RC", "RR"]
        current_idx = cams.index(self.current_cam_for_projection)
        next_idx = (current_idx + 1) % len(cams)
        self.current_cam_for_projection = cams[next_idx]
        self.cam_var.set(self.current_cam_for_projection)
        self._update_display()

    def _toggle_draw_line(self):
        self.draw_line_enabled = bool(self.draw_line_var.get())
        self._update_draw_cursor()

    def _update_stop_duration_seconds(self):
        try:
            value = float(self.stop_duration_var.get())
        except (TypeError, ValueError, tk.TclError):
            value = self.stop_duration_seconds
        self.stop_duration_seconds = float(np.clip(value, 0.5, 5.0))
        if self.stop_duration_var is not None:
            self.stop_duration_var.set(self.stop_duration_seconds)

    def _update_draw_cursor(self):
        cursor = "crosshair" if self.draw_line_enabled else ""
        self.traj_canvas.config(cursor=cursor)
        for label in self.camera_labels.values():
            label.config(cursor=cursor)

    def _manual_line_points_to_canvas(self) -> list[float]:
        points = []
        for point in self.manual_line_points:
            px, py = self._world_to_canvas(float(point["x"]), float(point["y"]))
            points.extend([px, py])
        return points

    def _camera_curve_control_indices(self) -> list[int]:
        """Return camera controls currently used to build the editable curve."""
        fc_indices = [
            idx for idx, point in enumerate(self.manual_camera_line_points)
            if point.get("camera") == "FC"
        ]
        if len(fc_indices) >= 2:
            return fc_indices
        return []

    def _active_manual_control_refs(self) -> list[dict]:
        """Return the control handles that define the current manual curve."""
        camera_indices = self._camera_curve_control_indices()
        if camera_indices:
            return [
                {"type": "camera_line", "index": index}
                for index in camera_indices
            ]
        return [
            {"type": "line", "index": index}
            for index in range(len(self.manual_line_points))
        ]

    def _manual_control_ref_to_ego(self, ref: dict):
        """Resolve a control handle to an ego-ground point."""
        point_type = ref.get("type")
        index = int(ref.get("index", -1))
        if point_type == "line":
            if not (0 <= index < len(self.manual_line_points)):
                return None
            point = self.manual_line_points[index]
            return (
                float(point["x"]),
                float(point["y"]),
                float(point.get("z", 0.0)),
            )

        if point_type == "camera_line":
            if not (0 <= index < len(self.manual_camera_line_points)):
                return None
            point = self.manual_camera_line_points[index]
            return self._camera_image_point_to_ego_ground(
                point.get("camera", ""),
                float(point["u"]),
                float(point["v"]),
            )
        return None

    def _ego_ground_to_source_image_point(self, cam_name: str, ego_point):
        """Project an ego-ground point to source image coordinates."""
        if self.calibration is None or cam_name not in self.calibration:
            return None

        calib = self.calibration[cam_name]
        point = np.asarray(ego_point, dtype=np.float32).reshape(1, 3)
        bev_point = ego_to_bev_points(point)
        u, v, z = calib.project_bev_to_image(bev_point)
        visible = calib.is_point_visible(u, v, z)
        if not bool(visible[0]):
            return None
        return float(u[0]), float(v[0])

    def _camera_image_point_to_ego_ground(self, cam_name: str, u: float, v: float):
        """Back-project an image point to the ego ground plane using calibration."""
        if self.calibration is None or cam_name not in self.calibration:
            return None

        calib = self.calibration[cam_name]
        camera_matrix = np.array(
            [
                [calib.fx, 0.0, calib.cx],
                [0.0, calib.fy, calib.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        image_point = np.array([[[float(u), float(v)]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(
            image_point,
            camera_matrix,
            calib.distortion_coeffs.astype(np.float64),
        )
        ray_cam = np.array(
            [undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0],
            dtype=np.float64,
        )

        transform = np.asarray(calib.T_bev_to_camera, dtype=np.float64)
        rotation = transform[:, :3]
        translation = transform[:, 3]
        try:
            camera_to_bev = np.linalg.inv(rotation)
        except np.linalg.LinAlgError:
            return None

        camera_center_bev = -camera_to_bev @ translation
        ray_bev = camera_to_bev @ ray_cam
        if abs(ray_bev[2]) < 1e-9:
            return None

        scale = -camera_center_bev[2] / ray_bev[2]
        if scale <= 0:
            return None

        point_bev = camera_center_bev + scale * ray_bev
        ego_x = float(point_bev[1])
        ego_y = float(-point_bev[0])
        return ego_x, ego_y, 0.0

    def _manual_camera_line_points_to_canvas(self) -> list[float]:
        points = []
        for point in self.manual_camera_line_points:
            ego_point = self._camera_image_point_to_ego_ground(
                point.get("camera", ""),
                float(point["u"]),
                float(point["v"]),
            )
            if ego_point is None:
                continue
            px, py = self._world_to_canvas(ego_point[0], ego_point[1])
            points.extend([px, py])
        return points

    def _add_manual_line_point(self, canvas_x: float, canvas_y: float):
        wx, wy = self._canvas_to_world(canvas_x, canvas_y)
        self.manual_line_points.append({
            "x": round(float(wx), 3),
            "y": round(float(wy), 3),
            "z": 0.0,
        })
        self.manual_point_actions.append(("line", len(self.manual_line_points) - 1))
        self.manual_line_points_dirty = True
        self._update_display()

    def _display_to_source_image_point(self, cam_name: str, display_x: float, display_y: float):
        meta = self.camera_display_meta.get(cam_name)
        if not meta:
            return None

        display_w = max(float(meta["display_width"]), 1.0)
        display_h = max(float(meta["display_height"]), 1.0)
        u = display_x * float(meta["source_width"]) / display_w
        v = display_y * float(meta["source_height"]) / display_h
        return u, v

    def _add_manual_camera_line_point(self, cam_name: str, display_x: float, display_y: float):
        source_point = self._display_to_source_image_point(cam_name, display_x, display_y)
        if source_point is None:
            return
        u, v = source_point

        self.manual_camera_line_points.append({
            "camera": cam_name,
            "u": round(float(u), 2),
            "v": round(float(v), 2),
        })
        self.manual_point_actions.append(("camera_line", len(self.manual_camera_line_points) - 1))
        self.manual_camera_line_points_dirty = True
        self._update_display()

    def _base_manual_bezier_trajectory(self) -> Optional[np.ndarray]:
        return self._build_manual_bezier_trajectory(include_stops=False)

    def _add_manual_stop_point_from_ego(self, ego_x: float, ego_y: float):
        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            messagebox.showwarning(
                "No Bezier Path",
                "Draw at least two Bezier control points before adding a stop marker.",
            )
            return

        fraction = _nearest_path_fraction(base_traj, (ego_x, ego_y))
        if fraction is None:
            return
        self._update_stop_duration_seconds()
        stop = {
            "fraction": round(float(fraction), 6),
            "duration_s": round(float(self.stop_duration_seconds), 2),
        }
        self.manual_stop_points.append(stop)
        self.manual_point_actions.append(("stop", len(self.manual_stop_points) - 1))
        self.manual_stop_points_dirty = True
        self._update_display()

    def _add_final_stop_point(self):
        """Add one stop marker at the final point of the current Bezier path."""
        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            messagebox.showwarning(
                "No Bezier Path",
                "Draw at least two Bezier control points before adding a final stop.",
            )
            return

        self._update_stop_duration_seconds()
        stop = {
            "fraction": 1.0,
            "duration_s": round(float(self.stop_duration_seconds), 2),
        }
        candidate = self._build_manual_bezier_trajectory(
            include_stops=True,
            stop_points=[stop],
        )
        if candidate is None:
            messagebox.showwarning(
                "Invalid Stop",
                "Cannot fit this final stop within the speed and acceleration limits. Try a shorter stop time or a longer path.",
            )
            return

        diagnostics = _trajectory_quality_diagnostics(candidate)
        if not bool(diagnostics.get("ok", False)):
            messagebox.showwarning(
                "Invalid Stop",
                (
                    "Final stop exceeds dynamics limits: "
                    f"acc={len(diagnostics.get('bad_accel_indices', []))}, "
                    f"jump={len(diagnostics.get('jump_indices', []))}."
                ),
            )
            return

        self.manual_stop_points = [stop]
        self.manual_point_actions.append(("stop", 0))
        self.manual_stop_points_dirty = True
        self._update_display()

    def _undo_manual_line_point(self) -> bool:
        while self.manual_point_actions:
            point_type, index = self.manual_point_actions.pop()
            if point_type == "line" and index == len(self.manual_line_points) - 1:
                self.manual_line_points.pop()
                self.manual_line_points_dirty = True
                self._update_display()
                return True
            if (
                point_type == "camera_line"
                and index == len(self.manual_camera_line_points) - 1
            ):
                self.manual_camera_line_points.pop()
                self.manual_camera_line_points_dirty = True
                self._update_display()
                return True
            if point_type == "stop" and index < len(self.manual_stop_points):
                self.manual_stop_points.pop(index)
                self.manual_stop_points_dirty = True
                self._update_display()
                return True

        if self.manual_camera_line_points:
            self.manual_camera_line_points.pop()
            self.manual_camera_line_points_dirty = True
            self._update_display()
            return True
        if self.manual_line_points:
            self.manual_line_points.pop()
            self.manual_line_points_dirty = True
            self._update_display()
            return True
        if self.manual_stop_points:
            self.manual_stop_points.pop()
            self.manual_stop_points_dirty = True
            self._update_display()
            return True
        return False

    def _undo_manual_point(self):
        self._undo_manual_line_point()

    def _save_manual_points(self):
        self._persist_current_manual_points()
        self._update_display()

        messagebox.showinfo(
            "Saved Controls",
            (
                f"Saved {len(self.manual_line_points)} BEV controls and "
                f"{len(self.manual_camera_line_points)} image controls and "
                f"{len(self.manual_stop_points)} stop markers to {self.manual_points_file}"
            ),
        )

    def _persist_current_manual_points(self):
        key = self._current_manual_points_key()
        if self.manual_line_points:
            self.manual_line_points_index[key] = [
                dict(point) for point in self.manual_line_points
            ]
        else:
            self.manual_line_points_index.pop(key, None)
        if self.manual_camera_line_points:
            self.manual_camera_line_points_index[key] = [
                dict(point) for point in self.manual_camera_line_points
            ]
        else:
            self.manual_camera_line_points_index.pop(key, None)
        if self.manual_stop_points:
            self.manual_stop_points_index[key] = [
                dict(point) for point in self.manual_stop_points
            ]
        else:
            self.manual_stop_points_index.pop(key, None)

        self._write_manual_points_index()
        self.manual_line_points_dirty = False
        self.manual_camera_line_points_dirty = False
        self.manual_stop_points_dirty = False

    def _estimate_t0_speed_mps(self) -> float:
        """Estimate current speed from the local history in the current sample."""
        if self.conv_data is None or "ego_history_xyz" not in self.conv_data:
            return 2.0

        hist = self.conv_data["ego_history_xyz"]
        if hasattr(hist, "detach"):
            hist = hist.detach().cpu().numpy()
        hist = np.asarray(hist, dtype=np.float64).reshape(-1, 3)
        if len(hist) < 2:
            return 2.0

        tail_delta = hist[-1, :2] - hist[-2, :2]
        distance = float(np.linalg.norm(tail_delta))
        if not np.isfinite(distance):
            return 2.0
        speed = distance / TRAJ_DT_SECONDS
        return float(np.clip(speed, 0.0, 12.0))

    def _manual_curve_ground_points(self) -> list[tuple[float, float, float]]:
        """Return the active manual controls as ego-ground points."""
        ground_points = []
        for ref in self._active_manual_control_refs():
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is not None:
                ground_points.append(ego_point)
        return ground_points

    def _build_manual_bezier_trajectory(
        self,
        include_stops: bool = True,
        stop_points: Optional[list[dict]] = None,
    ) -> Optional[np.ndarray]:
        """Build a smooth 64-step future trajectory from manual control points."""
        clicked_points = self._manual_curve_ground_points()
        if len(clicked_points) < 2:
            return None

        clicked = np.asarray(clicked_points, dtype=np.float64)
        clicked[:, 2] = 0.0

        first_xy = clicked[0, :2]
        first_distance = float(np.linalg.norm(first_xy))
        initial_speed = self._estimate_t0_speed_mps()
        anchor_distance = max(initial_speed * 0.45, first_distance * 0.35, 1.0)
        anchor_distance = min(anchor_distance, max(first_distance * 0.8, 1.0), 6.0)
        start_tangent = np.array([anchor_distance, 0.0, 0.0], dtype=np.float64)

        waypoints = np.vstack([np.zeros((1, 3), dtype=np.float64), clicked])
        dense = _sample_cubic_bezier_chain(
            waypoints,
            samples_per_segment=96,
            start_tangent=start_tangent,
        )
        active_stop_points = self.manual_stop_points if stop_points is None else stop_points
        if include_stops and active_stop_points:
            final_stop_points = [
                stop for stop in active_stop_points
                if float(stop.get("fraction", 0.0)) >= 0.999
            ]
            if len(final_stop_points) == len(active_stop_points):
                duration_s = max(float(stop.get("duration_s", 0.0)) for stop in final_stop_points)
                sampled = _resample_curve_with_final_stop(
                    dense,
                    num_steps=FUTURE_TRAJ_STEPS,
                    initial_speed_mps=initial_speed,
                    dt_seconds=TRAJ_DT_SECONDS,
                    duration_s=duration_s,
                )
                if sampled is None:
                    return None
                return sampled.astype(np.float32)
            sampled = _resample_curve_with_stops(
                dense,
                num_steps=FUTURE_TRAJ_STEPS,
                initial_speed_mps=initial_speed,
                dt_seconds=TRAJ_DT_SECONDS,
                stop_points=active_stop_points,
            )
        else:
            sampled = _resample_curve_by_distance(
                dense,
                num_steps=FUTURE_TRAJ_STEPS,
                initial_speed_mps=initial_speed,
                dt_seconds=TRAJ_DT_SECONDS,
            )
            sampled = _smooth_xy(sampled, passes=2)
        sampled[:, 2] = 0.0
        limited = _acceleration_limited_resample_path(sampled)
        if limited is None:
            return None
        sampled = limited
        return sampled.astype(np.float32)

    def _manual_trajectory_to_row(self, trajectory_xyz: np.ndarray, df: pd.DataFrame) -> dict:
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_xyz = np.asarray(trajectory_xyz, dtype=np.float64)
        num_steps = len(traj_xyz)

        prev = np.vstack([np.zeros((1, 3), dtype=np.float64), traj_xyz[:-1]])
        velocity = (traj_xyz - prev) / TRAJ_DT_SECONDS
        vx_list = velocity[:, 0].tolist()
        vy_list = velocity[:, 1].tolist()
        vz_list = velocity[:, 2].tolist()

        yaws = np.arctan2(velocity[:, 1], velocity[:, 0])
        if num_steps > 1:
            yaws = np.unwrap(yaws)
            stationary = np.linalg.norm(velocity[:, :2], axis=1) < 1e-3
            moving_indices = np.where(~stationary)[0]
            if len(moving_indices) == 0:
                yaws[:] = 0.0
            else:
                first_moving = int(moving_indices[0])
                yaws[:first_moving] = yaws[first_moving]
                for idx in range(first_moving + 1, num_steps):
                    if stationary[idx]:
                        yaws[idx] = yaws[idx - 1]
        qw = np.cos(yaws / 2.0)
        qz = np.sin(yaws / 2.0)

        if "t0_us" in df.columns:
            current_df = df[df["t0_us"].astype("int64") == int(t0_us)]
        else:
            current_df = df
        if "sample_idx" in current_df.columns and len(current_df) > 0:
            sample_idx = int(current_df["sample_idx"].max()) + 1
        else:
            sample_idx = len(current_df)

        return {
            "t0_us": int(t0_us),
            "sample_idx": sample_idx,
            "timestamp": [
                int(t0_us) + int((i + 1) * TRAJ_DT_SECONDS * 1_000_000)
                for i in range(num_steps)
            ],
            "qx": [0.0] * num_steps,
            "qy": [0.0] * num_steps,
            "qz": qz.tolist(),
            "qw": qw.tolist(),
            "x": traj_xyz[:, 0].tolist(),
            "y": traj_xyz[:, 1].tolist(),
            "z": traj_xyz[:, 2].tolist(),
            "vx": vx_list,
            "vy": vy_list,
            "vz": vz_list,
            "curvature": np.gradient(yaws).tolist(),
        }

    def _save_manual_bezier_trajectory(self):
        if self.gt_only:
            messagebox.showwarning(
                "GT Only",
                "GT-only mode does not load or append generated trajectory parquet files.",
            )
            return
        trajectory_xyz = self._build_manual_bezier_trajectory()
        if trajectory_xyz is None:
            messagebox.showwarning(
                "No Curve",
                "Use Draw Bezier to add at least two control points, and keep the curve within acceleration limits.",
            )
            return

        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        row = self._manual_trajectory_to_row(trajectory_xyz, df)

        for column in df.columns:
            if column not in row:
                row[column] = None
        new_row = pd.DataFrame([{column: row[column] for column in df.columns}])
        df_appended = pd.concat([df, new_row], ignore_index=True)

        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df_appended.to_parquet(traj_file, index=False)

        self._persist_current_manual_points()
        self._load_sample(self.current_idx)
        self._update_display()

        messagebox.showinfo(
            "Saved Curve Trajectory",
            (
                f"Appended manual Bezier trajectory sample_idx={row['sample_idx']} "
                f"for t0={int(t0_us)} to {traj_file}"
            ),
        )

    def _save_manual_bezier_as_cluster_center(self):
        trajectory_xyz = self._build_manual_bezier_trajectory()
        if trajectory_xyz is None:
            messagebox.showwarning(
                "No Valid Center",
                "Draw at least two Bezier control points and keep the curve within acceleration limits.",
            )
            return

        diagnostics = _trajectory_quality_diagnostics(trajectory_xyz)
        if not bool(diagnostics.get("ok", False)):
            messagebox.showwarning(
                "Invalid Center",
                (
                    "Bezier center exceeds limits: "
                    f"acc={len(diagnostics.get('bad_accel_indices', []))}, "
                    f"jump={len(diagnostics.get('jump_indices', []))}."
                ),
            )
            return

        category = self.cluster_category_var.get() if self.cluster_category_var is not None else "straight"
        if category not in CLUSTER_CATEGORY_FILES:
            category = "straight"

        records = list(self.cluster_center_library.get(category, []))
        next_id = (max((int(record["id"]) for record in records), default=-1) + 1)
        record = {
            "id": next_id,
            "label": "",
            "trajectory": np.asarray(trajectory_xyz, dtype=np.float32),
            "source": CLUSTER_CATEGORY_FILES[category],
            "count": 1,
            "category": category,
            "is_bezier_added": True,
        }
        final_right = -float(record["trajectory"][-1, 1])
        record["label"] = (
            f"B{next_id:02d} n=1 right={final_right:.1f} "
            f"fwd={float(record['trajectory'][-1, 0]):.1f}"
        )
        records.append(record)
        self.cluster_center_library[category] = records
        self.bezier_cluster_center_ids.setdefault(category, set()).add(int(next_id))
        self._write_cluster_category_file(category, records)
        self._write_bezier_cluster_center_ids()

        if self.cluster_category_var is not None:
            self.cluster_category_var.set(category)
        self._refresh_cluster_choice_values()
        if self.cluster_choice_var is not None:
            self.cluster_choice_var.set(record["label"])
        self.cluster_preview_record = record
        self.cluster_preview_traj = np.asarray(record["trajectory"], dtype=np.float32)
        self.cluster_preview_is_edited = False
        self._update_display()

        messagebox.showinfo(
            "Saved Bezier Center",
            f"Saved Bezier center as {category}/{record['label']} to {self._cluster_category_file(category)}",
        )

    def _delete_current_bezier_cluster_center(self):
        """Delete the currently displayed center only if it was added from a Bezier curve."""
        record = self.cluster_preview_record
        if record is None and self.cluster_preview_traj is None:
            record = self._selected_cluster_record()
        category = str(record.get("category", "")) if record is not None else ""
        if record is None or category not in CLUSTER_CATEGORY_FILES:
            messagebox.showwarning("No Center", "Display a Bezier-added center before deleting.")
            return

        center_id = int(record.get("id", -1))
        if not bool(record.get("is_bezier_added", False)):
            messagebox.showwarning(
                "Cannot Delete",
                "Only the currently displayed center created by Save Bezier Center can be deleted.",
            )
            return

        label = str(record.get("label", f"C{center_id:02d}"))
        if not messagebox.askyesno(
            "Delete Bezier Center",
            f"Delete the currently displayed {category}/{label} from {self._cluster_category_file(category)}?",
        ):
            return

        records = [
            item for item in self.cluster_center_library.get(category, [])
            if int(item.get("id", -1)) != center_id
        ]
        self.cluster_center_library[category] = records
        self.bezier_cluster_center_ids.setdefault(category, set()).discard(center_id)
        self._write_cluster_category_file(category, records)
        self._write_bezier_cluster_center_ids()

        self.cluster_preview_record = None
        self.cluster_preview_traj = None
        self.cluster_preview_is_edited = False
        self._refresh_cluster_choice_values()
        self._preview_selected_cluster_center(show_warning=False)
        self._update_display()

        messagebox.showinfo(
            "Deleted Bezier Center",
            f"Deleted {category}/{label}.",
        )

    def _remove_manual_point_action(self, point_type: str, removed_index: int):
        updated_actions = []
        for action_type, action_index in self.manual_point_actions:
            if action_type != point_type:
                updated_actions.append((action_type, action_index))
            elif action_index < removed_index:
                updated_actions.append((action_type, action_index))
            elif action_index > removed_index:
                updated_actions.append((action_type, action_index - 1))
        self.manual_point_actions = updated_actions

    def _selected_trajectory_matches_manual_curve(self) -> bool:
        """Return whether the selected trajectory is the current manual curve."""
        if not (0 <= self.current_traj_idx < len(self.trajectories)):
            return False

        manual_curve = self._build_manual_bezier_trajectory()
        if manual_curve is None or len(manual_curve) == 0:
            return False

        traj = self.trajectories[self.current_traj_idx]
        selected_curve = np.column_stack([traj["x"], traj["y"], traj["z"]])
        if len(selected_curve) == 0:
            return False

        count = min(len(selected_curve), len(manual_curve))
        deltas = np.linalg.norm(
            selected_curve[:count, :2] - manual_curve[:count, :2],
            axis=1,
        )
        return bool(float(np.nanmean(deltas)) <= 0.25)

    def _clear_current_manual_points(self):
        """Clear manual controls and stops for the current sample."""
        self.manual_line_points = []
        self.manual_camera_line_points = []
        self.manual_stop_points = []
        self.manual_point_actions = []
        self.manual_line_points_dirty = True
        self.manual_camera_line_points_dirty = True
        self.manual_stop_points_dirty = True
        self._persist_current_manual_points()

    def _nearest_manual_line_point_at_canvas(self, canvas_x: float, canvas_y: float):
        hit_index = None
        hit_distance = None
        for idx, point in enumerate(self.manual_line_points):
            px, py = self._world_to_canvas(float(point["x"]), float(point["y"]))
            distance = ((canvas_x - px) ** 2 + (canvas_y - py) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_index = idx
                hit_distance = distance
        return hit_index

    def _nearest_manual_camera_line_point_at_display(
        self,
        cam_name: str,
        display_x: float,
        display_y: float,
    ):
        meta = self.camera_display_meta.get(cam_name)
        if not meta:
            return None

        display_w = max(float(meta["display_width"]), 1.0)
        display_h = max(float(meta["display_height"]), 1.0)
        source_w = max(float(meta["source_width"]), 1.0)
        source_h = max(float(meta["source_height"]), 1.0)

        hit_index = None
        hit_distance = None
        for idx, point in enumerate(self.manual_camera_line_points):
            if point.get("camera") != cam_name:
                continue
            point_x = float(point["u"]) * display_w / source_w
            point_y = float(point["v"]) * display_h / source_h
            distance = ((display_x - point_x) ** 2 + (display_y - point_y) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_index = idx
                hit_distance = distance
        return hit_index

    def _nearest_active_manual_control_at_canvas(self, canvas_x: float, canvas_y: float):
        """Find the active manual control nearest to a BEV canvas point."""
        hit_ref = None
        hit_distance = None
        for order, ref in enumerate(self._active_manual_control_refs()):
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is None:
                continue
            px, py = self._world_to_canvas(float(ego_point[0]), float(ego_point[1]))
            distance = ((canvas_x - px) ** 2 + (canvas_y - py) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_ref = dict(ref)
                hit_ref["order"] = order
                hit_distance = distance
        return hit_ref

    def _nearest_cluster_endpoint_at_canvas(self, canvas_x: float, canvas_y: float) -> bool:
        """Return whether a canvas point hits the current cluster preview endpoint."""
        if self.cluster_preview_traj is None or len(self.cluster_preview_traj) == 0:
            return False
        end_x, end_y = self._world_to_canvas(
            float(self.cluster_preview_traj[-1, 0]),
            float(self.cluster_preview_traj[-1, 1]),
        )
        distance = ((canvas_x - end_x) ** 2 + (canvas_y - end_y) ** 2) ** 0.5
        return distance <= CLUSTER_ENDPOINT_HIT_RADIUS_PX

    def _nearest_active_manual_control_at_display(
        self,
        cam_name: str,
        display_x: float,
        display_y: float,
    ):
        """Find the active manual control nearest to a camera display point."""
        meta = self.camera_display_meta.get(cam_name)
        if not meta:
            return None

        display_w = max(float(meta["display_width"]), 1.0)
        display_h = max(float(meta["display_height"]), 1.0)
        source_w = max(float(meta["source_width"]), 1.0)
        source_h = max(float(meta["source_height"]), 1.0)

        hit_ref = None
        hit_distance = None
        for order, ref in enumerate(self._active_manual_control_refs()):
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is None:
                continue
            image_point = self._ego_ground_to_source_image_point(cam_name, ego_point)
            if image_point is None:
                continue
            point_x = image_point[0] * display_w / source_w
            point_y = image_point[1] * display_h / source_h
            distance = ((display_x - point_x) ** 2 + (display_y - point_y) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_ref = dict(ref)
                hit_ref["order"] = order
                hit_distance = distance
        return hit_ref

    def _set_manual_control_ref_from_ego(self, ref: dict, ego_point) -> bool:
        """Move a manual control handle to a new ego-ground point."""
        point_type = ref.get("type")
        index = int(ref.get("index", -1))
        if point_type == "line":
            if not (0 <= index < len(self.manual_line_points)):
                return False
            self.manual_line_points[index] = {
                "x": round(float(ego_point[0]), 3),
                "y": round(float(ego_point[1]), 3),
                "z": 0.0,
            }
            self.manual_line_points_dirty = True
            return True

        if point_type == "camera_line":
            if not (0 <= index < len(self.manual_camera_line_points)):
                return False
            cam_name = self.manual_camera_line_points[index].get("camera", "")
            image_point = self._ego_ground_to_source_image_point(cam_name, ego_point)
            if image_point is None:
                return False
            self.manual_camera_line_points[index] = {
                "camera": cam_name,
                "u": round(float(image_point[0]), 2),
                "v": round(float(image_point[1]), 2),
            }
            self.manual_camera_line_points_dirty = True
            return True
        return False

    def _convert_active_camera_controls_to_bev_controls(self, selected_order: int) -> int:
        """Convert FC image controls into BEV controls so BEV dragging is unconstrained."""
        refs = self._active_manual_control_refs()
        ground_points = []
        for ref in refs:
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is not None:
                ground_points.append(ego_point)

        if len(ground_points) < 2:
            return -1

        self.manual_line_points = [
            {
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "z": 0.0,
            }
            for point in ground_points
        ]
        self.manual_camera_line_points = []
        self.manual_point_actions = [
            ("line", index) for index in range(len(self.manual_line_points))
        ]
        self.manual_line_points_dirty = True
        self.manual_camera_line_points_dirty = True
        return int(np.clip(selected_order, 0, len(self.manual_line_points) - 1))

    def _on_camera_right_down(self, event, cam_name: str):
        hit_ref = self._nearest_active_manual_control_at_display(
            cam_name, event.x, event.y
        )
        if hit_ref is None:
            return
        self.drag_state = {
            "type": "manual_control",
            "camera": cam_name,
            "ref": hit_ref,
        }

    def _on_camera_right_drag(self, event, cam_name: str):
        if not self.drag_state or self.drag_state.get("type") != "manual_control":
            return
        if self.drag_state.get("camera") != cam_name:
            return
        ref = self.drag_state.get("ref", {})
        source_point = self._display_to_source_image_point(cam_name, event.x, event.y)
        if source_point is None:
            return

        if ref.get("type") == "camera_line":
            index = int(ref.get("index", -1))
            if not (0 <= index < len(self.manual_camera_line_points)):
                return
            source_cam = self.manual_camera_line_points[index].get("camera", "")
            if source_cam == cam_name:
                u, v = source_point
                meta = self.camera_display_meta.get(cam_name, {})
                source_w = max(float(meta.get("source_width", 1.0)), 1.0)
                source_h = max(float(meta.get("source_height", 1.0)), 1.0)
                self.manual_camera_line_points[index] = {
                    "camera": cam_name,
                    "u": round(float(np.clip(u, 0.0, source_w - 1.0)), 2),
                    "v": round(float(np.clip(v, 0.0, source_h - 1.0)), 2),
                }
                self.manual_camera_line_points_dirty = True
                self._update_display()
                return

        ego_point = self._camera_image_point_to_ego_ground(
            cam_name, source_point[0], source_point[1]
        )
        if ego_point is None:
            return
        if self._set_manual_control_ref_from_ego(ref, ego_point):
            self._update_display()

    def _on_canvas_right_down(self, event):
        hit_ref = self._nearest_active_manual_control_at_canvas(event.x, event.y)
        if hit_ref is None:
            return
        if hit_ref.get("type") == "camera_line":
            converted_index = self._convert_active_camera_controls_to_bev_controls(
                int(hit_ref.get("order", 0))
            )
            if converted_index < 0:
                return
            hit_ref = {"type": "line", "index": converted_index}
        self.drag_state = {
            "type": "manual_control",
            "ref": hit_ref,
        }

    def _on_canvas_right_drag(self, event):
        if not self.drag_state or self.drag_state.get("type") != "manual_control":
            return
        wx, wy = self._canvas_to_world(event.x, event.y)
        if self._set_manual_control_ref_from_ego(
            self.drag_state.get("ref", {}),
            (wx, wy, 0.0),
        ):
            self._update_display()

    def _on_canvas_left_drag(self, event):
        if not self.drag_state or self.drag_state.get("type") != "cluster_endpoint":
            return
        base_traj = self.drag_state.get("base_traj")
        if base_traj is None:
            return
        wx, wy = self._canvas_to_world(event.x, event.y)
        candidate = _cluster_drag_candidate(
            base_traj,
            (wx, wy),
            initial_speed_mps=self._estimate_t0_speed_mps(),
        )
        if candidate is None:
            return
        self.cluster_preview_traj = candidate.astype(np.float32)
        self.cluster_preview_is_edited = True
        self._update_display()

    def _on_left_release(self, _event):
        if self.drag_state and self.drag_state.get("type") == "cluster_endpoint":
            self.drag_state = None

    def _on_right_release(self, _event):
        self.drag_state = None

    def _on_camera_click(self, event, cam_name: str):
        if self.draw_line_enabled:
            self._add_manual_camera_line_point(cam_name, event.x, event.y)
            return
    
    def _on_canvas_click(self, event):
        """Handle canvas click to select trajectory."""
        if self.draw_line_enabled:
            self._add_manual_line_point(event.x, event.y)
            return

        if self._nearest_cluster_endpoint_at_canvas(event.x, event.y):
            record = self.cluster_preview_record
            if record is None:
                return
            self.drag_state = {
                "type": "cluster_endpoint",
                "base_traj": np.asarray(record["trajectory"], dtype=np.float64).copy(),
            }
            return

        for i, traj in enumerate(self.trajectories):
            if self._is_gt_trajectory(traj, i):
                continue

            x_coords = traj["x"]
            y_coords = traj["y"]
            if len(x_coords) > 0:
                cx = sum(x_coords) / len(x_coords)
                cy = sum(y_coords) / len(y_coords)
                canvas_x, canvas_y = self._world_to_canvas(cx, cy)
                dist = ((event.x - canvas_x)**2 + (event.y - canvas_y)**2)**0.5
                if dist < 50:
                    self.current_traj_idx = i
                    self._update_display()
                    return
    
    def _world_to_canvas(self, wx, wy, scale=None, offset=None):
        """Convert ego-local meters to BEV canvas pixels."""
        forward_scale = self.bev_forward_scale if scale is None else scale
        lateral_scale = self.bev_lateral_scale if scale is None else scale
        offset = self.bev_origin if offset is None else offset
        # Ego x is forward, ego y is left. Draw forward upward and left leftward.
        return offset[0] - wy * lateral_scale, offset[1] - wx * forward_scale
    
    def _canvas_to_world(self, cx, cy, scale=None, offset=None):
        """Convert BEV canvas pixels to ego-local meters."""
        forward_scale = self.bev_forward_scale if scale is None else scale
        lateral_scale = self.bev_lateral_scale if scale is None else scale
        offset = self.bev_origin if offset is None else offset
        return (offset[1] - cy) / forward_scale, (offset[0] - cx) / lateral_scale

    def _draw_bev_grid(self):
        """Draw a meter grid and ego axes on the BEV canvas."""
        width = self.bev_canvas_width
        height = self.bev_canvas_height
        grid_color = "#2f2f2f"
        major_color = "#3c3c3c"
        axis_color = "#6f6f6f"
        label_color = "#8a8a8a"
        label_font = ("Arial", 8)

        for forward_m in range(-10, 106, 5):
            _, y = self._world_to_canvas(forward_m, 0)
            if 0 <= y <= height:
                color = major_color if forward_m % 10 == 0 else grid_color
                self.traj_canvas.create_line(0, y, width, y, fill=color)
                if forward_m % 10 == 0:
                    self.traj_canvas.create_text(
                        10, y - 7, text=f"{forward_m}m",
                        fill=label_color, font=label_font, anchor=tk.W,
                    )

        for left_m in np.arange(-30.0, 30.01, 5.0):
            x, _ = self._world_to_canvas(0, left_m)
            if 0 <= x <= width:
                is_major = abs((left_m / 10.0) - round(left_m / 10.0)) < 1e-6
                color = major_color if is_major else grid_color
                self.traj_canvas.create_line(x, 0, x, height, fill=color)
                if is_major and abs(left_m) > 1e-6:
                    self.traj_canvas.create_text(
                        x, height - 12, text=f"{left_m:g}m",
                        fill=label_color, font=label_font,
                    )

        origin_x, origin_y = self._world_to_canvas(0, 0)
        self.traj_canvas.create_line(origin_x, 0, origin_x, height, fill=axis_color)
        self.traj_canvas.create_line(0, origin_y, width, origin_y, fill=axis_color)
        self.traj_canvas.create_text(
            origin_x + 6, origin_y + 10, text="0m",
            fill=label_color, font=label_font, anchor=tk.W,
        )

    def _draw_history_trajectory(self):
        """Draw the ego history loaded for the current sample."""
        if self.conv_data is None or "ego_history_xyz" not in self.conv_data:
            return

        hist = self.conv_data["ego_history_xyz"]
        if hasattr(hist, "detach"):
            hist = hist.detach().cpu().numpy()
        hist = np.asarray(hist).reshape(-1, 3)
        if len(hist) < 2:
            return

        points = []
        for x, y, _z in hist:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])

        self.traj_canvas.create_line(
            *points,
            fill="#b0b0b0",
            width=3,
            dash=(6, 4),
            smooth=True,
        )
        start_x, start_y = self._world_to_canvas(float(hist[0, 0]), float(hist[0, 1]))
        self.traj_canvas.create_oval(
            start_x - 3, start_y - 3, start_x + 3, start_y + 3,
            fill="#b0b0b0", outline="",
        )

    def _gt_future_key_for_mode(self, mode: Optional[str] = None) -> str:
        mode = mode or self.gt_future_mode
        if mode == "repaired":
            return "ego_future_xyz_repaired"
        return "ego_future_xyz_raw"

    def _get_source_gt_future_xyz(self, mode: Optional[str] = None):
        """Return source ground-truth future trajectory in ego-local coordinates."""
        if self.conv_data is None:
            return None

        key = self._gt_future_key_for_mode(mode)
        if key not in self.conv_data:
            key = "ego_future_xyz"
        if key not in self.conv_data:
            return None

        gt = self.conv_data[key]
        if hasattr(gt, "detach"):
            gt = gt.detach().cpu().numpy()
        gt = np.asarray(gt).reshape(-1, 3)
        if len(gt) < 2:
            return None
        return gt

    def _gt_trajectory_from_current_parquet(self) -> Optional[np.ndarray]:
        if self.gt_only:
            return None
        for traj in self.trajectories:
            source = str(traj.get("source", "")).lower()
            sample_idx = int(traj.get("sample_idx", -1))
            if source == "gt" or (not source and sample_idx == 0):
                return np.column_stack([traj["x"], traj["y"], traj["z"]]).astype(np.float32)
        return None

    def _get_gt_future_xyz(self, mode: Optional[str] = None):
        """Return active ground-truth future trajectory, preferring the editable parquet GT row."""
        if mode is None:
            if self.gt_edit_active and self.gt_edit_preview_xyz is not None:
                return np.asarray(self.gt_edit_preview_xyz, dtype=np.float32).reshape(-1, 3)
            parquet_gt = self._gt_trajectory_from_current_parquet()
            if parquet_gt is not None:
                return parquet_gt
        return self._get_source_gt_future_xyz(mode)

    def _get_gt_forward_acceleration(self, mode: Optional[str] = None):
        """Return source longitudinal GT acceleration for diagnostics when available."""
        if self.conv_data is None:
            return None
        if mode is None and self._gt_trajectory_from_current_parquet() is not None:
            return None

        mode = mode or self.gt_future_mode
        if mode == "repaired" and "ego_future_forward_acceleration_repaired" in self.conv_data:
            key = "ego_future_forward_acceleration_repaired"
        elif "ego_future_forward_acceleration_raw" in self.conv_data:
            key = "ego_future_forward_acceleration_raw"
        else:
            key = "ego_future_forward_acceleration"
        if key not in self.conv_data:
            return None

        accel = self.conv_data[key]
        if hasattr(accel, "detach"):
            accel = accel.detach().cpu().numpy()
        accel = np.asarray(accel, dtype=np.float64).reshape(-1)
        if len(accel) < 2 or not np.isfinite(accel).all():
            return None
        return accel

    def _get_gt_quality_diagnostics(self, mode: Optional[str] = None) -> Optional[dict[str, object]]:
        gt = self._get_gt_future_xyz(mode)
        if gt is None:
            return None
        return _trajectory_quality_diagnostics(
            gt,
            acceleration_mps2=self._get_gt_forward_acceleration(mode),
        )

    def _repair_gt_future(self):
        """Write the velocity-integrated GT future into the editable parquet GT row."""
        if self.conv_data is None:
            messagebox.showwarning("No GT", "Current sample has no loaded GT data.")
            return
        if "ego_future_xyz_repaired" not in self.conv_data:
            messagebox.showwarning("No Repair", "This sample does not include repaired GT data.")
            return
        if not bool(self.conv_data.get("gt_repair_available", True)):
            error = str(self.conv_data.get("gt_repair_error", "")).strip()
            message = "Velocity-based GT repair is unavailable for this sample."
            if error:
                message += f"\n\n{error}"
            messagebox.showwarning("No Repair", message)
            return

        repaired = self._get_source_gt_future_xyz("repaired")
        if repaired is None:
            messagebox.showwarning("No Repair", "Could not load repaired GT for this sample.")
            return
        if not self._write_gt_trajectory_to_parquet(repaired):
            return
        self.gt_future_mode = "raw"
        self._load_sample(self.current_idx)
        self._update_display()

    def _restore_gt_future(self):
        """Write the raw source GT future back into the editable parquet GT row."""
        raw = self._get_source_gt_future_xyz("raw")
        if raw is None:
            messagebox.showwarning("No GT", "Could not load raw GT for this sample.")
            return
        if not self._write_gt_trajectory_to_parquet(raw):
            return
        self.gt_future_mode = "raw"
        self._load_sample(self.current_idx)
        self._update_display()

    def _draw_gt_future_trajectory(self):
        """Draw ground-truth future trajectory on the BEV canvas."""
        gt = self._get_gt_future_xyz()
        if gt is None:
            return
        diagnostics = self._get_gt_quality_diagnostics() or _trajectory_quality_diagnostics(gt)
        has_problem = not bool(diagnostics.get("ok", True))

        points = []
        for x, y, _z in gt:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])

        self.traj_canvas.create_line(
            *points,
            fill="#ff6b6b" if has_problem else GT_COLOR_HEX,
            width=4,
            smooth=True,
        )
        for idx in diagnostics.get("bad_accel_indices", []):
            if 0 <= idx < len(gt):
                px, py = self._world_to_canvas(float(gt[idx, 0]), float(gt[idx, 1]))
                self.traj_canvas.create_oval(
                    px - 6, py - 6, px + 6, py + 6,
                    fill="#ff3333",
                    outline="white",
                    width=2,
                )
        for idx in diagnostics.get("jump_indices", []):
            if 0 <= idx < len(gt):
                px, py = self._world_to_canvas(float(gt[idx, 0]), float(gt[idx, 1]))
                self.traj_canvas.create_rectangle(
                    px - 6, py - 6, px + 6, py + 6,
                    fill="#d36bff",
                    outline="white",
                    width=2,
                )
        end_x, end_y = self._world_to_canvas(float(gt[-1, 0]), float(gt[-1, 1]))
        self.traj_canvas.create_oval(
            end_x - 5, end_y - 5, end_x + 5, end_y + 5,
            fill="#ff6b6b" if has_problem else GT_COLOR_HEX,
            outline="black",
            width=1,
        )
        gt_speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        for segment in _detect_stop_segments(gt_speed):
            marker_idx = min(int(segment["end"]), len(gt) - 1)
            px, py = self._world_to_canvas(float(gt[marker_idx, 0]), float(gt[marker_idx, 1]))
            radius = STOP_MARKER_RADIUS_PX
            self.traj_canvas.create_oval(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                fill="#ff2020",
                outline="white",
                width=1,
            )
            self.stop_marker_hitboxes.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius + 6),
                "text": (
                    "GT STOP\n"
                    f"Duration: {float(segment['duration_s']):.1f}s "
                    f"({int(segment['frames'])} frames)\n"
                    f"Frames: {int(segment['start'])}-{int(segment['end'])}\n"
                    f"Mean speed: {float(segment['mean_speed_mps']):.2f} m/s"
                ),
            })
        if has_problem:
            self.traj_canvas.create_text(
                end_x + 8,
                end_y + 10,
                text=(
                    f"GT acc {len(diagnostics.get('bad_accel_indices', []))} "
                    f"jump {len(diagnostics.get('jump_indices', []))}"
                ),
                fill="#ffb3b3",
                font=("Arial", 9, "bold"),
                anchor=tk.W,
            )

    def _draw_manual_bezier_preview(self):
        """Draw the generated manual Bezier trajectory preview on the BEV canvas."""
        trajectory_xyz = self._build_manual_bezier_trajectory()
        if trajectory_xyz is None or len(trajectory_xyz) < 2:
            return

        points = []
        for x, y, _z in trajectory_xyz:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])

        self.traj_canvas.create_line(
            *points,
            fill=MANUAL_TRAJ_COLOR_HEX,
            width=4,
            smooth=True,
        )
        end_x, end_y = self._world_to_canvas(
            float(trajectory_xyz[-1, 0]),
            float(trajectory_xyz[-1, 1]),
        )
        self.traj_canvas.create_oval(
            end_x - 5, end_y - 5, end_x + 5, end_y + 5,
            fill=MANUAL_TRAJ_COLOR_HEX,
            outline="black",
            width=1,
        )

    def _draw_cluster_center_preview(self):
        """Draw the selected cluster center preview on the BEV canvas."""
        if self.cluster_preview_traj is None or len(self.cluster_preview_traj) < 2:
            return

        points = []
        for x, y, _z in self.cluster_preview_traj:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])
        self.traj_canvas.create_line(
            *points,
            fill=CLUSTER_TRAJ_COLOR_HEX,
            width=5,
            smooth=True,
        )
        end_x, end_y = self._world_to_canvas(
            float(self.cluster_preview_traj[-1, 0]),
            float(self.cluster_preview_traj[-1, 1]),
        )
        self.traj_canvas.create_oval(
            end_x - 6, end_y - 6, end_x + 6, end_y + 6,
            fill=CLUSTER_TRAJ_COLOR_HEX,
            outline="black",
            width=2,
        )
        label = self._current_cluster_preview_label()
        self.traj_canvas.create_text(
            end_x + 8,
            end_y - 8,
            text=label,
            fill=CLUSTER_TRAJ_COLOR_HEX,
            font=("Arial", 9, "bold"),
            anchor=tk.W,
        )

    def _draw_manual_stop_markers(self):
        """Draw stop markers on top of the Bezier path."""
        if not self.manual_stop_points:
            return
        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            return

        for idx, stop in enumerate(self.manual_stop_points):
            point = _point_at_path_fraction(base_traj, float(stop.get("fraction", 0.0)))
            if point is None:
                continue
            px, py = self._world_to_canvas(float(point[0]), float(point[1]))
            radius = STOP_MARKER_RADIUS_PX
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="#ff5c5c", outline="white", width=2,
            )
            self.stop_marker_hitboxes.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius + 6),
                "text": (
                    "Manual STOP\n"
                    f"Duration: {float(stop.get('duration_s', 0.0)):.1f}s"
                ),
            })

    def _draw_manual_line(self):
        """Draw manually clicked BEV control points."""
        if not self.manual_line_points:
            return

        for idx, point in enumerate(self.manual_line_points):
            px, py = self._world_to_canvas(float(point["x"]), float(point["y"]))
            radius = 5
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="#00d4ff", outline="black", width=2,
            )
            self.traj_canvas.create_text(
                px + 8, py + 9, text=f"L{idx + 1}",
                fill="#00d4ff", font=("Arial", 9, "bold"), anchor=tk.W,
            )

    def _draw_manual_camera_line_on_bev(self):
        """Draw camera image control points back-projected onto the BEV ground plane."""
        points = self._manual_camera_line_points_to_canvas()
        if not points:
            return

        for idx in range(0, len(points), 2):
            px, py = points[idx], points[idx + 1]
            line_idx = idx // 2 + 1
            radius = 5
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="#00d4ff", outline="black", width=2,
            )
            self.traj_canvas.create_text(
                px + 8, py + 9, text=f"CL{line_idx}",
                fill="#00d4ff", font=("Arial", 8, "bold"), anchor=tk.W,
            )

    def _trajectory_draw_style(self, traj_index: int, is_selected: bool, is_kept: bool) -> dict:
        """Return shared BEV/camera drawing style for a generated trajectory."""
        is_unsmooth = not bool(
            self.trajectory_smoothness.get(traj_index, {}).get("ok", True)
        )
        if not is_kept:
            rgb = (92, 92, 92)
            if is_selected:
                rgb = _brighten_rgb(rgb, amount=0.25)
            return {
                "rgb": rgb,
                "hex": _rgb_to_hex(rgb),
                "bev_width": 3 if is_selected else 1,
                "camera_width": 3 if is_selected else 1,
                "dash": (5, 5),
                "draw_points": is_selected,
                "point_radius": 5 if is_selected else 3,
            }

        rgb = (235, 64, 64) if is_unsmooth else _trajectory_base_color(
            traj_index,
            len(self.trajectories),
        )
        if is_selected:
            rgb = _brighten_rgb(rgb, amount=0.32)
        return {
            "rgb": rgb,
            "hex": _rgb_to_hex(rgb),
            "bev_width": 5 if is_selected else 2,
            "camera_width": 5 if is_selected else 2,
            "dash": None,
            "draw_points": is_selected,
            "point_radius": 6 if is_selected else 4,
        }

    def _trajectory_speed_profile(self, traj: dict) -> np.ndarray:
        return _speed_profile_from_trajectory(
            traj.get("x", []),
            traj.get("y", []),
            traj.get("z", []),
            traj.get("vx"),
            traj.get("vy"),
            traj.get("vz"),
        )

    def _draw_generated_stop_markers(
        self,
        traj: dict,
        traj_index: int,
        is_selected: bool,
        is_kept: bool,
    ) -> None:
        """Draw detected stop actions for a generated trajectory on BEV."""
        speed = self._trajectory_speed_profile(traj)
        stop_segments = _detect_stop_segments(speed)
        if not stop_segments:
            return

        x_coords = np.asarray(traj.get("x", []), dtype=np.float64).reshape(-1)
        y_coords = np.asarray(traj.get("y", []), dtype=np.float64).reshape(-1)
        if len(x_coords) == 0 or len(y_coords) != len(x_coords):
            return

        for segment in stop_segments:
            marker_idx = min(int(segment["end"]), len(x_coords) - 1)
            px, py = self._world_to_canvas(float(x_coords[marker_idx]), float(y_coords[marker_idx]))
            radius = STOP_MARKER_RADIUS_PX + (1 if is_selected else 0)
            fill = "#ff2020" if is_kept else "#8a2020"
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill=fill,
                outline="white" if is_selected else "#1e1e1e",
                width=2 if is_selected else 1,
            )
            self.stop_marker_hitboxes.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius + 6),
                "text": (
                    f"T{traj_index} STOP\n"
                    f"Duration: {float(segment['duration_s']):.1f}s "
                    f"({int(segment['frames'])} frames)\n"
                    f"Frames: {int(segment['start'])}-{int(segment['end'])}\n"
                    f"Mean speed: {float(segment['mean_speed_mps']):.2f} m/s"
                ),
            })

    def _hide_stop_tooltip(self) -> None:
        if not hasattr(self, "traj_canvas"):
            return
        for item in self.stop_tooltip_items:
            self.traj_canvas.delete(item)
        self.stop_tooltip_items = []

    def _show_stop_tooltip(self, canvas_x: float, canvas_y: float, text: str) -> None:
        self._hide_stop_tooltip()
        padding = 7
        tooltip_x = min(float(canvas_x) + 14, self.bev_canvas_width - 190)
        tooltip_y = min(float(canvas_y) + 14, self.bev_canvas_height - 78)
        tooltip_x = max(8, tooltip_x)
        tooltip_y = max(8, tooltip_y)

        text_item = self.traj_canvas.create_text(
            tooltip_x + padding,
            tooltip_y + padding,
            text=text,
            fill="#f6f6f6",
            font=("Arial", 9),
            anchor=tk.NW,
            justify=tk.LEFT,
            tags=("stop_tooltip",),
        )
        bbox = self.traj_canvas.bbox(text_item)
        if bbox is None:
            return
        rect_item = self.traj_canvas.create_rectangle(
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
            fill="#111111",
            outline="#ff5c5c",
            width=1,
            tags=("stop_tooltip",),
        )
        self.traj_canvas.tag_raise(text_item, rect_item)
        self.stop_tooltip_items = [rect_item, text_item]

    def _on_traj_canvas_motion(self, event) -> None:
        for marker in reversed(self.stop_marker_hitboxes):
            dx = float(event.x) - float(marker["x"])
            dy = float(event.y) - float(marker["y"])
            if (dx * dx + dy * dy) ** 0.5 <= float(marker["radius"]):
                self._show_stop_tooltip(event.x, event.y, str(marker["text"]))
                return
        self._hide_stop_tooltip()

    def _refresh_trajectory_smoothness(self) -> None:
        self.trajectory_smoothness = {}
        _label, gt_speed, _stops, _color = self._gt_speed_profile_source()
        reference_speed = gt_speed if len(gt_speed) else None
        for idx, traj in enumerate(self.trajectories):
            speed = self._trajectory_speed_profile(traj)
            self.trajectory_smoothness[idx] = _speed_smoothness_diagnostics(
                speed,
                reference_speed=reference_speed,
            )

    def _hide_traj_list_tooltip(self) -> None:
        if self.traj_list_tooltip is not None:
            self.traj_list_tooltip.destroy()
            self.traj_list_tooltip = None

    def _show_traj_list_tooltip(self, event, text: str) -> None:
        self._hide_traj_list_tooltip()
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 10}")
        label = tk.Label(
            tooltip,
            text=text,
            bg="#111111",
            fg="#f6f6f6",
            bd=1,
            relief=tk.SOLID,
            padx=7,
            pady=5,
            font=("Arial", 9),
            justify=tk.LEFT,
        )
        label.pack()
        self.traj_list_tooltip = tooltip

    def _on_traj_list_motion(self, event) -> None:
        if not hasattr(self, "traj_listbox"):
            return
        row_idx = self.traj_listbox.nearest(event.y)
        bbox = self.traj_listbox.bbox(row_idx)
        if bbox is None or not (bbox[1] <= event.y <= bbox[1] + bbox[3]):
            self._hide_traj_list_tooltip()
            return
        diagnostics = self.trajectory_smoothness.get(row_idx, {})
        if event.x <= 24 and not bool(diagnostics.get("ok", True)):
            self._show_traj_list_tooltip(event, "建议删除原因：速度不够平滑")
            return
        self._hide_traj_list_tooltip()

    def _manual_preview_is_active_for_speed(self) -> bool:
        return (
            self.draw_line_enabled
            or self.manual_line_points_dirty
            or self.manual_camera_line_points_dirty
            or self.manual_stop_points_dirty
        )

    def _gt_speed_profile_source(self) -> tuple[str, np.ndarray, list[dict], str]:
        gt = self._get_gt_future_xyz()
        if gt is None:
            return "GT", np.zeros(0, dtype=np.float64), [], GT_COLOR_HEX
        speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        return "GT", speed, _detect_stop_segments(speed), GT_COLOR_HEX

    def _selected_speed_profile_source(self) -> tuple[str, np.ndarray, list[dict], str]:
        if self.speed_edit_active and self.speed_edit_speed is not None:
            speed = np.asarray(self.speed_edit_speed, dtype=np.float64)
            label = f"Editing T{self.speed_edit_traj_idx}"
            style = self._trajectory_draw_style(
                int(self.speed_edit_traj_idx),
                True,
                self.trajectory_states.get(int(self.speed_edit_traj_idx), True),
            )
            return label, speed, _detect_stop_segments(speed), style["hex"]

        manual_preview = None
        if self._manual_preview_is_active_for_speed():
            manual_preview = self._build_manual_bezier_trajectory()
        if manual_preview is not None and len(manual_preview) >= 2:
            speed = _speed_profile_from_trajectory(
                manual_preview[:, 0],
                manual_preview[:, 1],
                manual_preview[:, 2],
            )
            return "Manual preview", speed, _detect_stop_segments(speed), MANUAL_TRAJ_COLOR_HEX

        if self.cluster_preview_traj is not None and len(self.cluster_preview_traj) >= 2:
            cluster = np.asarray(self.cluster_preview_traj, dtype=np.float64)
            speed = _speed_profile_from_trajectory(cluster[:, 0], cluster[:, 1], cluster[:, 2])
            return "Cluster preview", speed, _detect_stop_segments(speed), "#f5b041"

        if 0 <= self.current_traj_idx < len(self.trajectories):
            traj = self.trajectories[self.current_traj_idx]
            style = self._trajectory_draw_style(
                self.current_traj_idx,
                True,
                self.trajectory_states.get(self.current_traj_idx, True),
            )
            speed = self._trajectory_speed_profile(traj)
            return f"T{self.current_traj_idx}", speed, _detect_stop_segments(speed), style["hex"]
        return "", np.zeros(0, dtype=np.float64), [], "#6f6f6f"

    def _speed_profile_source(self, source: str) -> tuple[str, np.ndarray, list[dict], str]:
        if source == "gt":
            return self._gt_speed_profile_source()
        return self._selected_speed_profile_source()

    def _selected_trajectory_points_xyz(self):
        if self._manual_preview_is_active_for_speed():
            manual_preview = self._build_manual_bezier_trajectory()
            if manual_preview is not None and len(manual_preview) >= 2:
                return manual_preview
        if self.cluster_preview_traj is not None and len(self.cluster_preview_traj) >= 2:
            return np.asarray(self.cluster_preview_traj, dtype=np.float64)
        if 0 <= self.current_traj_idx < len(self.trajectories):
            traj = self.trajectories[self.current_traj_idx]
            if self._is_gt_trajectory(traj, self.current_traj_idx):
                return None
            return np.column_stack([traj["x"], traj["y"], traj["z"]])
        return None

    def _trajectory_points_for_speed_source(self, source: str):
        if source == "gt":
            return self._get_gt_future_xyz()
        return self._selected_trajectory_points_xyz()

    def _speed_plot_geometry(self) -> dict[str, float]:
        margin_left = 44
        margin_right = 14
        margin_top = 22
        margin_bottom = 30
        plot_w = self.speed_canvas_width - margin_left - margin_right
        plot_h = self.speed_canvas_height - margin_top - margin_bottom
        return {
            "left": float(margin_left),
            "top": float(margin_top),
            "right": float(margin_left + plot_w),
            "bottom": float(margin_top + plot_h),
            "width": float(plot_w),
            "height": float(plot_h),
        }

    def _set_speed_hover_frame(self, source: str, frame_idx: Optional[int]) -> None:
        if frame_idx is None:
            next_idx = None
            next_source = None
        else:
            _label, speed, _stops, _color = self._speed_profile_source(source)
            if len(speed) == 0:
                next_idx = None
                next_source = None
            else:
                next_idx = int(np.clip(int(frame_idx), 0, len(speed) - 1))
                next_source = source
        if self.speed_hover_frame_idx == next_idx and self.speed_hover_source == next_source:
            return
        self.speed_hover_frame_idx = next_idx
        self.speed_hover_source = next_source
        self._draw_trajectories()
        self._draw_speed_profile()
        self._draw_gt_speed_profile()
        self._draw_camera_images()

    def _on_speed_canvas_motion(self, event, source: str) -> None:
        if source == "gt":
            rect = self.gt_speed_plot_rect or self._speed_plot_geometry()
        else:
            rect = self.speed_plot_rect or self._speed_plot_geometry()
        inside = (
            rect["left"] <= float(event.x) <= rect["right"]
            and rect["top"] <= float(event.y) <= rect["bottom"]
        )
        if not inside:
            self._set_speed_hover_frame(source, None)
            return

        _label, speed, _stops, _color = self._speed_profile_source(source)
        if len(speed) <= 1:
            self._set_speed_hover_frame(source, 0 if len(speed) == 1 else None)
            return
        fraction = (float(event.x) - rect["left"]) / max(rect["width"], 1.0)
        frame_idx = int(round(np.clip(fraction, 0.0, 1.0) * (len(speed) - 1)))
        self._set_speed_hover_frame(source, frame_idx)

    def _speed_canvas_frame_from_x(self, x: float) -> Optional[int]:
        rect = self.speed_plot_rect or self._speed_plot_geometry()
        _label, speed, _stops, _color = self._selected_speed_profile_source()
        if len(speed) == 0 or not (rect["left"] <= float(x) <= rect["right"]):
            return None
        if len(speed) == 1:
            return 0
        fraction = (float(x) - rect["left"]) / max(rect["width"], 1.0)
        return int(round(np.clip(fraction, 0.0, 1.0) * (len(speed) - 1)))

    def _speed_canvas_value_from_y(self, y: float) -> float:
        rect = self.speed_plot_rect or self._speed_plot_geometry()
        _label, speed, _stops, _color = self._selected_speed_profile_source()
        finite_speed = speed[np.isfinite(speed)]
        max_speed = float(np.nanmax(finite_speed)) if len(finite_speed) else 1.0
        y_max = max(1.0, np.ceil(max_speed * 1.15))
        fraction = (rect["bottom"] - float(y)) / max(rect["height"], 1.0)
        return float(np.clip(fraction, 0.0, 1.0) * y_max)

    def _on_speed_canvas_right_click(self, event) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        can_edit = (
            not self.gt_only
            and 0 <= self.current_traj_idx < len(self.trajectories)
            and not self._is_gt_trajectory(self.trajectories[self.current_traj_idx], self.current_traj_idx)
            and not self._manual_preview_is_active_for_speed()
        )
        if not self.speed_edit_active:
            menu.add_command(
                label="进行调整",
                command=self._start_speed_edit,
                state=tk.NORMAL if can_edit else tk.DISABLED,
            )
        else:
            menu.add_command(label="保存更改", command=self._save_speed_edit)
            menu.add_command(label="不保存，恢复原状", command=lambda: self._cancel_speed_edit(redraw=True))
        menu.tk_popup(event.x_root, event.y_root)

    def _pack_pred_speed_actions(self) -> None:
        if self.pred_speed_action_frame is not None:
            if hasattr(self, "gt_speed_header_frame") and self.gt_speed_header_frame is not None:
                self.pred_speed_action_frame.pack(
                    before=self.gt_speed_header_frame,
                    pady=(0, 6),
                )
            else:
                self.pred_speed_action_frame.pack(pady=(0, 6))

    def _hide_pred_speed_actions(self) -> None:
        if self.pred_speed_action_frame is not None:
            self.pred_speed_action_frame.pack_forget()

    def _pack_gt_speed_actions(self) -> None:
        if self.gt_speed_action_frame is not None:
            self.gt_speed_action_frame.pack(pady=(0, 6))
        if self.gt_stop_action_frame is not None:
            self.gt_stop_action_frame.pack_forget()

    def _pack_gt_stop_actions(self) -> None:
        if self.gt_speed_action_frame is not None:
            self.gt_speed_action_frame.pack_forget()
        if self.gt_stop_action_frame is not None:
            self.gt_stop_action_frame.pack(pady=(0, 6))

    def _hide_gt_action_frames(self) -> None:
        if self.gt_speed_action_frame is not None:
            self.gt_speed_action_frame.pack_forget()
        if self.gt_stop_action_frame is not None:
            self.gt_stop_action_frame.pack_forget()

    def _optimize_pred_speed_curve(self) -> None:
        if self.gt_only or not (0 <= self.current_traj_idx < len(self.trajectories)):
            messagebox.showwarning("No Trajectory", "Select a diversity trajectory before optimizing speed.")
            return
        if self._is_gt_trajectory(self.trajectories[self.current_traj_idx], self.current_traj_idx):
            messagebox.showwarning("Keep GT", "Select a rule/manual trajectory; GT is optimized from the GT panel.")
            return
        if self._manual_preview_is_active_for_speed():
            messagebox.showwarning("Manual Preview", "Finish manual Bezier editing before optimizing a saved trajectory.")
            return
        self._start_speed_edit()
        if not self.speed_edit_active or self.speed_edit_speed is None:
            return
        self.speed_edit_speed = _smooth_speed_profile(self.speed_edit_speed, passes=3)
        self.speed_edit_dirty = True
        if not self._apply_speed_edit_to_trajectory():
            self._cancel_speed_edit(redraw=False)
            messagebox.showwarning("Optimize Failed", "Could not optimize this trajectory speed curve.")
            return
        self._pack_pred_speed_actions()
        self._update_display()

    def _optimize_gt_speed_curve(self) -> None:
        gt = self._get_gt_future_xyz()
        if gt is None:
            messagebox.showwarning("No GT", "Current sample has no GT trajectory.")
            return
        self._cancel_gt_speed_edit(redraw=False)
        speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        smoothed = _smooth_speed_profile(speed, passes=3)
        sampled = _resample_xyz_by_speed_profile(gt, smoothed)
        if sampled is None:
            messagebox.showwarning("Optimize Failed", "Could not optimize GT speed curve.")
            return
        self.gt_edit_active = True
        self.gt_edit_mode = "optimize"
        self.gt_edit_original_xyz = np.asarray(gt, dtype=np.float32).copy()
        self.gt_edit_preview_xyz = sampled.astype(np.float32)
        self.gt_stop_frame_idx = None
        self._pack_gt_speed_actions()
        self._update_display()

    def _start_gt_stop_add(self) -> None:
        gt = self._get_gt_future_xyz()
        if gt is None:
            messagebox.showwarning("No GT", "Current sample has no GT trajectory.")
            return
        self._cancel_gt_speed_edit(redraw=False)
        self.gt_edit_active = True
        self.gt_edit_mode = "stop"
        self.gt_edit_original_xyz = np.asarray(gt, dtype=np.float32).copy()
        self.gt_edit_preview_xyz = np.asarray(gt, dtype=np.float32).copy()
        self.gt_stop_frame_idx = None
        self._pack_gt_stop_actions()
        self._update_display()

    def _on_gt_speed_canvas_click(self, event) -> None:
        if not self.gt_edit_active or self.gt_edit_mode != "stop":
            return
        rect = self.gt_speed_plot_rect or self._speed_plot_geometry()
        if not (
            rect["left"] <= float(event.x) <= rect["right"]
            and rect["top"] <= float(event.y) <= rect["bottom"]
        ):
            return
        gt = self.gt_edit_original_xyz
        if gt is None or len(gt) == 0:
            return
        if len(gt) == 1:
            frame_idx = 0
        else:
            fraction = (float(event.x) - rect["left"]) / max(rect["width"], 1.0)
            frame_idx = int(round(np.clip(fraction, 0.0, 1.0) * (len(gt) - 1)))
        preview = np.asarray(gt, dtype=np.float32).copy()
        stop_anchor_idx = max(0, frame_idx - 1)
        preview[frame_idx:] = preview[stop_anchor_idx]
        self.gt_edit_preview_xyz = preview
        self.gt_stop_frame_idx = frame_idx
        self._update_display()

    def _undo_gt_stop_add(self) -> None:
        if not self.gt_edit_active or self.gt_edit_original_xyz is None:
            return
        self.gt_edit_preview_xyz = np.asarray(self.gt_edit_original_xyz, dtype=np.float32).copy()
        self.gt_stop_frame_idx = None
        self._update_display()

    def _save_gt_speed_edit(self) -> None:
        if not self.gt_edit_active or self.gt_edit_preview_xyz is None:
            return
        if not self._write_gt_trajectory_to_parquet(
            np.asarray(self.gt_edit_preview_xyz, dtype=np.float32),
            speed_optimized=self.gt_edit_mode == "optimize",
        ):
            return
        self.gt_edit_active = False
        self.gt_edit_mode = None
        self.gt_edit_original_xyz = None
        self.gt_edit_preview_xyz = None
        self.gt_stop_frame_idx = None
        self._hide_gt_action_frames()
        self._load_sample(self.current_idx)
        self._update_display()

    def _cancel_gt_speed_edit(self, redraw: bool = True) -> None:
        self.gt_edit_active = False
        self.gt_edit_mode = None
        self.gt_edit_original_xyz = None
        self.gt_edit_preview_xyz = None
        self.gt_stop_frame_idx = None
        self._hide_gt_action_frames()
        if redraw and hasattr(self, "root"):
            self._update_display()

    def _on_speed_canvas_left_down(self, event) -> None:
        if not self.speed_edit_active:
            return
        self.speed_edit_last_frame = self._speed_canvas_frame_from_x(event.x)
        self._apply_speed_canvas_edit(event)

    def _on_speed_canvas_left_drag(self, event) -> None:
        if self.speed_edit_active:
            self._apply_speed_canvas_edit(event)

    def _on_speed_canvas_left_release(self, _event) -> None:
        self.speed_edit_last_frame = None

    def _apply_speed_canvas_edit(self, event) -> None:
        if not self.speed_edit_active or self.speed_edit_speed is None:
            return
        frame_idx = self._speed_canvas_frame_from_x(event.x)
        if frame_idx is None:
            return
        new_speed = self._speed_canvas_value_from_y(event.y)
        editable = np.asarray(self.speed_edit_speed, dtype=np.float64).copy()
        center = int(frame_idx)
        radius = min(SPEED_EDIT_LOCAL_RADIUS_FRAMES, max(len(editable) // 10, 2))
        start = max(0, center - radius)
        end = min(len(editable) - 1, center + radius)
        for idx in range(start, end + 1):
            distance = abs(idx - center) / max(float(radius), 1.0)
            weight = 0.5 * (1.0 + np.cos(np.pi * distance))
            editable[idx] = editable[idx] * (1.0 - weight) + new_speed * weight
        if self.speed_edit_last_frame is not None and abs(center - self.speed_edit_last_frame) > 1:
            bridge_start = int(min(self.speed_edit_last_frame, center))
            bridge_end = int(max(self.speed_edit_last_frame, center))
            for idx in range(bridge_start, bridge_end + 1):
                distance = abs(idx - center) / max(float(radius), 1.0)
                if distance > 1.0:
                    continue
                weight = 0.35 * 0.5 * (1.0 + np.cos(np.pi * distance))
                editable[idx] = editable[idx] * (1.0 - weight) + new_speed * weight
        self.speed_edit_last_frame = frame_idx
        editable = _enforce_speed_acceleration_limits(editable)
        self.speed_edit_speed = editable
        self.speed_edit_dirty = True
        self._apply_speed_edit_to_trajectory()
        self.speed_hover_source = "pred"
        self.speed_hover_frame_idx = frame_idx
        self._draw_trajectories()
        self._draw_speed_profile()
        self._draw_camera_images()

    def _start_speed_edit(self) -> None:
        if self.gt_only or not (0 <= self.current_traj_idx < len(self.trajectories)):
            return
        self._cancel_speed_edit(redraw=False)
        traj = self.trajectories[self.current_traj_idx]
        if self._is_gt_trajectory(traj, self.current_traj_idx):
            messagebox.showwarning("Keep GT", "Select a rule/manual trajectory; GT is optimized from the GT panel.")
            return
        self.speed_edit_active = True
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = int(self.current_traj_idx)
        self.speed_edit_original_traj = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in traj.items()
        }
        self.speed_edit_original_xyz = np.column_stack([traj["x"], traj["y"], traj["z"]]).copy()
        self.speed_edit_speed = self._trajectory_speed_profile(traj).copy()
        self.speed_edit_last_frame = None
        self._update_display()

    def _cancel_speed_edit(self, redraw: bool = True) -> None:
        if self.speed_edit_active and self.speed_edit_original_traj is not None:
            idx = self.speed_edit_traj_idx
            if idx is not None and 0 <= int(idx) < len(self.trajectories):
                self.trajectories[int(idx)] = {
                    key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                    for key, value in self.speed_edit_original_traj.items()
                }
        self.speed_edit_active = False
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = None
        self.speed_edit_original_traj = None
        self.speed_edit_original_xyz = None
        self.speed_edit_speed = None
        self.speed_edit_last_frame = None
        self._hide_pred_speed_actions()
        if redraw and hasattr(self, "root"):
            self._update_display()

    def _trajectory_components_from_xyz(self, traj_xyz: np.ndarray) -> dict:
        traj_xyz = np.asarray(traj_xyz, dtype=np.float64)
        num_steps = len(traj_xyz)
        prev = np.vstack([np.zeros((1, 3), dtype=np.float64), traj_xyz[:-1]])
        velocity = (traj_xyz - prev) / TRAJ_DT_SECONDS
        yaws = np.arctan2(velocity[:, 1], velocity[:, 0])
        if num_steps > 1:
            yaws = np.unwrap(yaws)
            stationary = np.linalg.norm(velocity[:, :2], axis=1) < 1e-3
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
            "x": traj_xyz[:, 0],
            "y": traj_xyz[:, 1],
            "z": traj_xyz[:, 2],
            "vx": velocity[:, 0],
            "vy": velocity[:, 1],
            "vz": velocity[:, 2],
            "qx": np.zeros(num_steps, dtype=np.float64),
            "qy": np.zeros(num_steps, dtype=np.float64),
            "qz": np.sin(yaws / 2.0),
            "qw": np.cos(yaws / 2.0),
            "curvature": np.gradient(yaws) if num_steps else np.zeros(0, dtype=np.float64),
        }

    def _apply_components_to_traj(self, traj: dict, components: dict) -> None:
        for key, values in components.items():
            if key in traj:
                traj[key] = np.asarray(values, dtype=np.float64)

    def _apply_speed_edit_to_trajectory(self) -> bool:
        if (
            self.speed_edit_original_xyz is None
            or self.speed_edit_speed is None
            or self.speed_edit_traj_idx is None
            or not (0 <= int(self.speed_edit_traj_idx) < len(self.trajectories))
        ):
            return False
        sampled = _resample_xyz_by_speed_profile(
            self.speed_edit_original_xyz,
            self.speed_edit_speed,
        )
        if sampled is None:
            return False
        limited = _acceleration_limited_resample_path(sampled)
        if limited is not None:
            sampled = limited
        traj = self.trajectories[int(self.speed_edit_traj_idx)]
        self._apply_components_to_traj(traj, self._trajectory_components_from_xyz(sampled))
        self.speed_edit_speed = self._trajectory_speed_profile(traj)
        self._refresh_trajectory_smoothness()
        return True

    def _save_speed_edit(self) -> None:
        if not self.speed_edit_active or self.speed_edit_traj_idx is None:
            return
        if not self.speed_edit_dirty:
            self._cancel_speed_edit(redraw=True)
            return
        edited_idx = int(self.speed_edit_traj_idx)
        edited_sample_idx = int(self.trajectories[edited_idx].get("sample_idx", edited_idx))
        if not self._write_selected_trajectory_to_parquet(edited_idx):
            return
        self.speed_edit_active = False
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = None
        self.speed_edit_original_traj = None
        self.speed_edit_original_xyz = None
        self.speed_edit_speed = None
        self.speed_edit_last_frame = None
        self._hide_pred_speed_actions()
        self._load_sample(self.current_idx)
        for idx, traj in enumerate(self.trajectories):
            if int(traj.get("sample_idx", idx)) == edited_sample_idx:
                self.current_traj_idx = idx
                break
        self._update_display()

    def _write_gt_trajectory_to_parquet(self, gt_xyz: np.ndarray, speed_optimized: bool = False) -> bool:
        if self.gt_only:
            messagebox.showwarning("GT Only", "GT-only mode cannot write generated trajectory parquet files.")
            return False
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        if not traj_file.exists():
            messagebox.showwarning("Save Failed", f"Trajectory parquet does not exist: {traj_file}")
            return False

        df = pd.read_parquet(traj_file)
        if "t0_us" in df.columns:
            current_mask = df["t0_us"].astype("int64") == int(t0_us)
        else:
            current_mask = pd.Series(True, index=df.index)

        if "source" in df.columns:
            gt_mask = df["source"].astype(str).str.lower() == "gt"
        elif "sample_idx" in df.columns:
            gt_mask = df["sample_idx"].astype("int64") == 0
        else:
            gt_mask = pd.Series(False, index=df.index)

        matched = df.index[current_mask & gt_mask].tolist()
        if matched:
            row_idx = matched[0]
        else:
            row_idx = df.index[current_mask].tolist()[0] if current_mask.any() else len(df)

        components = self._trajectory_components_from_xyz(gt_xyz)
        if GT_SPEED_OPTIMIZED_COLUMN not in df.columns:
            df[GT_SPEED_OPTIMIZED_COLUMN] = False
        if row_idx == len(df):
            row = self._manual_trajectory_to_row(gt_xyz, df)
            row["sample_idx"] = 0
            row["source"] = "gt"
            row[GT_SPEED_OPTIMIZED_COLUMN] = bool(speed_optimized)
            for column in df.columns:
                if column not in row:
                    row[column] = None
            df = pd.concat([df, pd.DataFrame([{column: row[column] for column in df.columns}])], ignore_index=True)
        else:
            for key, values in components.items():
                if key in df.columns:
                    df.at[row_idx, key] = np.asarray(values).tolist()
            if "source" in df.columns:
                df.at[row_idx, "source"] = "gt"
            if "sample_idx" in df.columns:
                df.at[row_idx, "sample_idx"] = 0
            df.at[row_idx, GT_SPEED_OPTIMIZED_COLUMN] = bool(speed_optimized)

        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(traj_file, index=False)
        return True
        messagebox.showinfo("Saved", "Saved adjusted speed and trajectory point density.")

    def _write_selected_trajectory_to_parquet(self, traj_idx: int) -> bool:
        if not (0 <= traj_idx < len(self.trajectories)):
            return False
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        traj = self.trajectories[traj_idx]
        if self._is_gt_trajectory(traj, traj_idx):
            messagebox.showwarning("Keep GT", "Use the GT speed panel to edit GT. Prediction speed edits only write rule/manual rows.")
            return False
        row_indices = df.index[df["t0_us"].astype("int64") == int(t0_us)].tolist() if "t0_us" in df.columns else list(df.index)
        if "sample_idx" in df.columns:
            sample_idx = int(traj.get("sample_idx", traj_idx))
            matched = df.index[
                (df["t0_us"].astype("int64") == int(t0_us))
                & (df["sample_idx"].astype("int64") == sample_idx)
            ].tolist() if "t0_us" in df.columns else df.index[df["sample_idx"].astype("int64") == sample_idx].tolist()
            if matched:
                row_idx = matched[0]
            elif traj_idx < len(row_indices):
                row_idx = row_indices[traj_idx]
            else:
                messagebox.showwarning("Save Failed", "Could not locate the selected trajectory row.")
                return False
        else:
            if traj_idx >= len(row_indices):
                messagebox.showwarning("Save Failed", "Could not locate the selected trajectory row.")
                return False
            row_idx = row_indices[traj_idx]

        for key in ("x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "curvature"):
            if key in df.columns and key in traj:
                df.at[row_idx, key] = np.asarray(traj[key]).tolist()

        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(traj_file, index=False)
        return True

    def _draw_speed_hover_marker_on_bev(self, source: str) -> None:
        if self.speed_hover_frame_idx is None or self.speed_hover_source != source:
            return
        points_xyz = self._trajectory_points_for_speed_source(source)
        if points_xyz is None or len(points_xyz) == 0:
            return
        frame_idx = int(np.clip(self.speed_hover_frame_idx, 0, len(points_xyz) - 1))
        point = points_xyz[frame_idx]
        px, py = self._world_to_canvas(float(point[0]), float(point[1]))
        radius = 8
        self.traj_canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            fill=HOVER_FRAME_COLOR_HEX,
            outline="black",
            width=2,
        )
        self.traj_canvas.create_text(
            px + radius + 6,
            py + radius,
            text=f"F{frame_idx}",
            fill=HOVER_FRAME_COLOR_HEX,
            font=("Arial", 9, "bold"),
            anchor=tk.W,
        )

    def _draw_speed_profile_on_canvas(self, canvas, source: str) -> None:
        """Draw a speed-over-frame chart for one trajectory source."""
        if canvas is None:
            return
        canvas.delete("all")
        width = self.speed_canvas_width
        height = self.speed_canvas_height
        rect = self._speed_plot_geometry()
        if source == "gt":
            self.gt_speed_plot_rect = rect
        else:
            self.speed_plot_rect = rect
        margin_left = rect["left"]
        margin_top = rect["top"]
        plot_w = rect["width"]
        plot_h = rect["height"]

        label, speed, stop_segments, color = self._speed_profile_source(source)
        axis_color = "#5a5a5a"
        grid_color = "#2c2c2c"
        text_color = "#b8b8b8"
        canvas.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_w,
            margin_top + plot_h,
            outline=axis_color,
            fill="#171717",
        )
        canvas.create_text(
            8,
            8,
            text=f"{label or 'No trajectory'} speed",
            fill=text_color,
            font=("Arial", 9, "bold"),
            anchor=tk.NW,
        )

        if len(speed) == 0:
            canvas.create_text(
                margin_left + plot_w / 2,
                margin_top + plot_h / 2,
                text="No speed data",
                fill="#777777",
                font=("Arial", 10),
            )
            return

        finite_speed = speed[np.isfinite(speed)]
        max_speed = float(np.nanmax(finite_speed)) if len(finite_speed) else 0.0
        y_max = max(1.0, np.ceil(max_speed * 1.15))
        frame_count = len(speed)

        for fraction in (0.25, 0.5, 0.75):
            y = margin_top + plot_h * (1.0 - fraction)
            canvas.create_line(
                margin_left, y, margin_left + plot_w, y,
                fill=grid_color,
            )
            canvas.create_text(
                margin_left - 6,
                y,
                text=f"{y_max * fraction:.1f}",
                fill=text_color,
                font=("Arial", 8),
                anchor=tk.E,
            )

        canvas.create_text(
            margin_left - 6,
            margin_top + plot_h,
            text="0",
            fill=text_color,
            font=("Arial", 8),
            anchor=tk.E,
        )
        canvas.create_text(
            margin_left + plot_w / 2,
            height - 8,
            text="frame",
            fill=text_color,
            font=("Arial", 8),
        )
        canvas.create_text(
            10,
            margin_top + plot_h / 2,
            text="m/s",
            fill=text_color,
            font=("Arial", 8),
        )

        def frame_to_x(idx: int) -> float:
            if frame_count <= 1:
                return margin_left
            return margin_left + (float(idx) / float(frame_count - 1)) * plot_w

        def speed_to_y(value: float) -> float:
            return margin_top + plot_h - (float(value) / y_max) * plot_h

        for segment in stop_segments:
            x0 = frame_to_x(int(segment["start"]))
            x1 = frame_to_x(int(segment["end"]))
            canvas.create_rectangle(
                x0,
                margin_top,
                x1,
                margin_top + plot_h,
                fill="#3a1717",
                outline="",
            )

        points = []
        for idx, value in enumerate(speed):
            if not np.isfinite(value):
                continue
            points.extend([frame_to_x(idx), speed_to_y(value)])
        if len(points) >= 4:
            canvas.create_line(
                *points,
                fill=color,
                width=3,
                smooth=True,
            )

        canvas.create_line(
            margin_left,
            speed_to_y(STOP_SPEED_THRESHOLD_MPS),
            margin_left + plot_w,
            speed_to_y(STOP_SPEED_THRESHOLD_MPS),
            fill="#ff5c5c",
            dash=(4, 3),
        )
        canvas.create_text(
            margin_left + plot_w - 2,
            speed_to_y(STOP_SPEED_THRESHOLD_MPS) - 8,
            text=f"stop < {STOP_SPEED_THRESHOLD_MPS:.1f}m/s",
            fill="#ff9b9b",
            font=("Arial", 8),
            anchor=tk.E,
        )

        canvas.create_text(
            margin_left,
            margin_top + plot_h + 12,
            text="0",
            fill=text_color,
            font=("Arial", 8),
            anchor=tk.N,
        )
        canvas.create_text(
            margin_left + plot_w,
            margin_top + plot_h + 12,
            text=str(frame_count - 1),
            fill=text_color,
            font=("Arial", 8),
            anchor=tk.N,
        )

        if self.speed_hover_frame_idx is not None and self.speed_hover_source == source:
            hover_idx = int(np.clip(self.speed_hover_frame_idx, 0, frame_count - 1))
            hover_x = frame_to_x(hover_idx)
            hover_y = speed_to_y(speed[hover_idx]) if np.isfinite(speed[hover_idx]) else margin_top + plot_h
            canvas.create_line(
                hover_x,
                margin_top,
                hover_x,
                margin_top + plot_h,
                fill=HOVER_FRAME_COLOR_HEX,
                width=2,
            )
            canvas.create_oval(
                hover_x - 5,
                hover_y - 5,
                hover_x + 5,
                hover_y + 5,
                fill=HOVER_FRAME_COLOR_HEX,
                outline="black",
                width=1,
            )
            canvas.create_text(
                min(hover_x + 8, margin_left + plot_w - 78),
                margin_top + 6,
                text=f"F{hover_idx} {speed[hover_idx]:.2f}m/s",
                fill=HOVER_FRAME_COLOR_HEX,
                font=("Arial", 8, "bold"),
                anchor=tk.NW,
            )

    def _draw_speed_profile(self) -> None:
        if not hasattr(self, "speed_canvas"):
            return
        self._draw_speed_profile_on_canvas(self.speed_canvas, "pred")

    def _draw_gt_speed_profile(self) -> None:
        if not hasattr(self, "gt_speed_canvas"):
            return
        self._draw_speed_profile_on_canvas(self.gt_speed_canvas, "gt")
    
    def _draw_trajectories(self):
        """Draw all trajectories on the canvas."""
        self.traj_canvas.delete("all")
        self.stop_marker_hitboxes = []
        self.stop_tooltip_items = []
        self._draw_bev_grid()
        self._draw_history_trajectory()
        self._draw_gt_future_trajectory()
        self._draw_manual_bezier_preview()

        if not self.trajectories:
            self._draw_cluster_center_preview()
            self._draw_manual_camera_line_on_bev()
            self._draw_manual_line()
            self._draw_manual_stop_markers()
            self._draw_speed_hover_marker_on_bev("pred")
            self._draw_speed_hover_marker_on_bev("gt")
            return
        
        for i, traj in enumerate(self.trajectories):
            x_coords = traj["x"]
            y_coords = traj["y"]
            
            if len(x_coords) < 2:
                continue
            
            is_selected = (i == self.current_traj_idx)
            is_kept = self.trajectory_states.get(i, True)
            style = self._trajectory_draw_style(i, is_selected, is_kept)
            color = style["hex"]
            
            points = []
            for x, y in zip(x_coords, y_coords):
                px, py = self._world_to_canvas(x, y)
                points.extend([px, py])
            
            if len(points) >= 4:
                self.traj_canvas.create_line(
                    *points,
                    fill=color,
                    width=style["bev_width"],
                    dash=style["dash"],
                    smooth=True,
                )
            
            # Draw start point
            if len(x_coords) > 0:
                px, py = self._world_to_canvas(x_coords[0], y_coords[0])
                r = style["point_radius"] if is_selected else 4
                self.traj_canvas.create_oval(
                    px - r, py - r, px + r, py + r,
                    fill=color, outline="white" if is_selected else "black",
                    width=2 if is_selected else 1,
                )
            
            # Draw trajectory ID
            if len(x_coords) > 5:
                px, py = self._world_to_canvas(x_coords[5], y_coords[5])
                self.traj_canvas.create_text(
                    px, py, text=f"T{i}", fill=color,
                    font=("Arial", 10, "bold"),
                )

            self._draw_generated_stop_markers(
                traj,
                i,
                is_selected=is_selected,
                is_kept=is_kept,
            )
        
        # Origin marker
        px, py = self._world_to_canvas(0, 0)
        self.traj_canvas.create_oval(
            px - 4, py - 4, px + 4, py + 4,
            fill="#2ecc71", outline="white",
        )
        self.traj_canvas.create_line(
            px, py, px, py - 24, fill="#2ecc71", width=3, arrow=tk.LAST,
        )
        self._draw_manual_camera_line_on_bev()
        self._draw_manual_line()
        self._draw_manual_stop_markers()
        self._draw_speed_hover_marker_on_bev("pred")
        self._draw_speed_hover_marker_on_bev("gt")
        self._draw_cluster_center_preview()
    
    def _draw_camera_images(self):
        """Draw camera images with trajectory projection."""
        if self.conv_data is None:
            for cam in self.cameras:
                self.camera_labels[cam].config(image="")
            return
        
        try:
            frames = self.conv_data["image_frames"]  # [6, T, C, H, W]
            
            for cam_idx, cam in enumerate(self.cameras):
                cam_frame_rgb = self._get_camera_base_image(frames, cam_idx, cam)
                if cam_frame_rgb is None:
                    continue
                
                # Project trajectories onto this camera
                if self.calibration and cam in self.calibration:
                    cam_frame_rgb = self._draw_trajectory_projection(cam_frame_rgb, cam)
                cam_frame_rgb = self._draw_manual_camera_points(cam_frame_rgb, cam)
                
                # Resize for display
                h, w = cam_frame_rgb.shape[:2]
                new_h = self._camera_display_height(cam)
                new_w = int(new_h * w / h)
                self.camera_display_meta[cam] = {
                    "source_width": w,
                    "source_height": h,
                    "display_width": new_w,
                    "display_height": new_h,
                }
                cam_frame_rgb = cv2.resize(cam_frame_rgb, (new_w, new_h))
                
                # Convert to PhotoImage
                image = Image.fromarray(cam_frame_rgb)
                self.camera_labels[cam].image = ImageTk.PhotoImage(image)
                self.camera_labels[cam].config(image=self.camera_labels[cam].image)
                
        except Exception as e:
            print(f"Warning: Could not display images: {e}")
            self.camera_display_meta = {}
            for cam in self.cameras:
                self.camera_labels[cam].config(image="")

    def _get_camera_base_image(self, frames, cam_idx: int, cam: str):
        """Return the cached RGB base image for a camera at calibration size."""
        if cam in self.camera_base_images:
            return self.camera_base_images[cam].copy()

        cam_frame = frames[cam_idx, 0]  # [C, H, W] or [H, W, C]
        cam_frame_rgb = load_image_from_frame(cam_frame)

        if self.calibration and cam in self.calibration:
            calib = self.calibration[cam]
            if (
                cam_frame_rgb.shape[1] != calib.image_width
                or cam_frame_rgb.shape[0] != calib.image_height
            ):
                cam_frame_rgb = cv2.resize(
                    cam_frame_rgb, (calib.image_width, calib.image_height)
                )

        self.camera_base_images[cam] = cam_frame_rgb
        return cam_frame_rgb.copy()

    def _camera_display_height(self, cam_name):
        """Return display height in pixels, giving FC the most screen space."""
        if len(self.cameras) <= 3:
            return 620 if cam_name == "FC" else 270
        return 460 if cam_name == "FC" else 190

    def _draw_manual_camera_points(self, img, cam_name):
        """Draw editable Bezier control handles before display resizing."""
        img = img.copy()
        height, width = img.shape[:2]

        line_points = []
        for ref in self._active_manual_control_refs():
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is None:
                continue
            image_point = self._ego_ground_to_source_image_point(cam_name, ego_point)
            if image_point is None:
                continue
            u, v = image_point
            if 0 <= u < width and 0 <= v < height:
                line_points.append((int(round(u)), int(round(v))))

        for idx, center in enumerate(line_points):
            cv2.circle(img, center, 7, (0, 212, 255), -1)
            cv2.circle(img, center, 7, (0, 0, 0), 2)
            cv2.putText(
                img,
                f"L{idx + 1}",
                (center[0] + 10, center[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 212, 255),
                2,
                cv2.LINE_AA,
            )

        return img

    def _draw_manual_stop_markers_on_image(self, img, cam_name):
        """Project stop markers onto a camera image."""
        if not self.manual_stop_points:
            return img
        if self.calibration is None or cam_name not in self.calibration:
            return img

        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            return img

        calib = self.calibration[cam_name]
        out = img.copy()
        height, width = out.shape[:2]

        for stop in self.manual_stop_points:
            ego_point = _point_at_path_fraction(base_traj, float(stop.get("fraction", 0.0)))
            if ego_point is None:
                continue

            bev_point = ego_to_bev_points(np.asarray(ego_point, dtype=np.float32).reshape(1, 3))
            u, v, z = calib.project_bev_to_image(bev_point)
            visible = calib.is_point_visible(u, v, z)
            if not bool(visible[0]):
                continue

            center = (int(round(u[0])), int(round(v[0])))
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                continue

            cv2.circle(out, center, 6, (255, 92, 92), -1)
            cv2.circle(out, center, 6, (255, 255, 255), 1)

        return out

    def _draw_generated_stop_markers_on_image(self, img, cam_name):
        """Project detected generated-trajectory stop markers onto FC."""
        if cam_name != "FC" or not self.trajectories:
            return img

        out = img.copy()
        height, width = out.shape[:2]
        for i, traj in enumerate(self.trajectories):
            speed = self._trajectory_speed_profile(traj)
            stop_segments = _detect_stop_segments(speed)
            if not stop_segments:
                continue
            points_xyz = np.column_stack([traj["x"], traj["y"], traj["z"]])
            if len(points_xyz) == 0:
                continue
            is_selected = i == self.current_traj_idx
            is_kept = self.trajectory_states.get(i, True)
            fill = (255, 32, 32) if is_kept else (138, 32, 32)
            for segment in stop_segments:
                marker_idx = min(int(segment["end"]), len(points_xyz) - 1)
                image_point = self._ego_ground_to_source_image_point(cam_name, points_xyz[marker_idx])
                if image_point is None:
                    continue
                u, v = image_point
                center = (int(round(u)), int(round(v)))
                if not (0 <= center[0] < width and 0 <= center[1] < height):
                    continue
                radius = 7 if is_selected else 6
                cv2.circle(out, center, radius, fill, -1)
                cv2.circle(out, center, radius, (255, 255, 255), 2 if is_selected else 1)
        return out

    def _draw_gt_stop_markers_on_image(self, img, cam_name):
        """Project detected GT stop markers onto FC."""
        if cam_name != "FC":
            return img
        gt = self._get_gt_future_xyz()
        if gt is None or len(gt) == 0:
            return img

        speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        stop_segments = _detect_stop_segments(speed)
        if not stop_segments:
            return img

        out = img.copy()
        height, width = out.shape[:2]
        for segment in stop_segments:
            marker_idx = min(int(segment["end"]), len(gt) - 1)
            image_point = self._ego_ground_to_source_image_point(cam_name, gt[marker_idx])
            if image_point is None:
                continue
            u, v = image_point
            center = (int(round(u)), int(round(v)))
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                continue
            cv2.circle(out, center, 6, (255, 32, 32), -1)
            cv2.circle(out, center, 6, (255, 255, 255), 1)
        return out

    def _draw_speed_hover_marker_on_image(self, img, cam_name, source: str):
        """Project the speed-window hover frame onto FC."""
        if (
            cam_name != "FC"
            or self.speed_hover_frame_idx is None
            or self.speed_hover_source != source
        ):
            return img

        points_xyz = self._trajectory_points_for_speed_source(source)
        if points_xyz is None or len(points_xyz) == 0:
            return img

        frame_idx = int(np.clip(self.speed_hover_frame_idx, 0, len(points_xyz) - 1))
        image_point = self._ego_ground_to_source_image_point(cam_name, points_xyz[frame_idx])
        if image_point is None:
            return img

        out = img.copy()
        height, width = out.shape[:2]
        u, v = image_point
        center = (int(round(u)), int(round(v)))
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            return out

        radius = 12
        cv2.circle(out, center, radius, HOVER_FRAME_COLOR_RGB, -1)
        cv2.circle(out, center, radius, (0, 0, 0), 2)
        cv2.putText(
            out,
            f"F{frame_idx}",
            (center[0] + radius + 5, center[1] + radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HOVER_FRAME_COLOR_RGB,
            2,
            cv2.LINE_AA,
        )
        return out
    
    def _draw_trajectory_projection(self, img, cam_name):
        """Draw trajectory projection onto camera image."""
        if self.calibration is None or cam_name not in self.calibration:
            return img
        
        calib = self.calibration[cam_name]

        gt = self._get_gt_future_xyz()
        if gt is not None:
            img = draw_trajectory_on_image(
                img,
                gt,
                calib,
                color=GT_COLOR_RGB,
                thickness=4,
                draw_points=True,
                point_radius=4,
                coordinate_frame="ego",
            )

        manual_preview = self._build_manual_bezier_trajectory()
        if manual_preview is not None:
            img = draw_trajectory_on_image(
                img,
                manual_preview,
                calib,
                color=MANUAL_TRAJ_COLOR_RGB,
                thickness=4,
                draw_points=True,
                point_radius=4,
                coordinate_frame="ego",
            )
            img = self._draw_manual_stop_markers_on_image(img, cam_name)

        if self.cluster_preview_traj is not None:
            img = draw_trajectory_on_image(
                img,
                self.cluster_preview_traj,
                calib,
                color=CLUSTER_TRAJ_COLOR_RGB,
                thickness=5,
                draw_points=True,
                point_radius=5,
                coordinate_frame="ego",
            )
        
        # Draw each trajectory
        for i, traj in enumerate(self.trajectories):
            if self._is_gt_trajectory(traj, i):
                continue

            is_selected = (i == self.current_traj_idx)
            is_kept = self.trajectory_states.get(i, True)
            style = self._trajectory_draw_style(i, is_selected, is_kept)
            
            # Get trajectory points in ego frame
            points_xyz = np.column_stack([
                traj["x"], traj["y"], traj["z"]
            ])
            
            img = draw_trajectory_on_image(
                img,
                points_xyz,
                calib,
                color=style["rgb"],
                thickness=style["camera_width"],
                draw_points=style["draw_points"],
                point_radius=style["point_radius"],
                coordinate_frame="ego",
            )

        img = self._draw_gt_stop_markers_on_image(img, cam_name)
        img = self._draw_generated_stop_markers_on_image(img, cam_name)
        img = self._draw_speed_hover_marker_on_image(img, cam_name, "pred")
        img = self._draw_speed_hover_marker_on_image(img, cam_name, "gt")
        
        return img
    
    def _get_default_extrinsics(self, cam_name):
        """Get default camera extrinsics based on typical vehicle camera setup.
        
        Ego frame: x=forward, y=left, z=up
        Camera frame: x=right, y=down, z=forward
        """
        # Camera mounting positions relative to ego center
        # [x_forward, y_left, z_up]
        cam_positions = {
            'FC': (0.0,  0.0,  1.5),   # Front Center
            'FL': (0.3,  0.5,  1.5),   # Front Left
            'FR': (0.3, -0.5,  1.5),   # Front Right
            'RC': (0.0,  0.0,  1.2),   # Rear Center
            'RL': (-0.3, 0.5,  1.5),   # Rear Left
            'RR': (-0.3,-0.5,  1.5),   # Rear Right
        }
        
        if cam_name not in cam_positions:
            return None
        
        tx, ty, tz = cam_positions[cam_name]
        
        # Rotation from ego to camera
        # ego_x=forward -> camera_z=forward
        # ego_y=left -> camera_x=-left
        # ego_z=up -> camera_y=-up
        # So camera = [[0,0,1], [0,-1,0], [1,0,0]] @ ego
        R_cam_ego = np.array([
            [ 0,  0, 1],
            [ 0, -1, 0],
            [ 1,  0, 0]
        ])
        
        return {
            'rotation': R_cam_ego.flatten().tolist(),
            'translation': [tx, ty, tz]
        }
    
    def _update_display(self):
        """Update the display."""
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        mode_suffix = (
            f" | GT-only stride={self.gt_stride_frames}"
            if self.gt_only
            else ""
        )
        
        self.title_label.config(
            text=f"Dataset: {dataset_name} | Clip: {clip_stem}{mode_suffix}"
        )
        self.nav_label.config(
            text=f"{self.current_idx + 1} / {len(self.samples)}"
        )
        self._sync_sample_selectors()
        
        kept = sum(1 for v in self.trajectory_states.values() if v)
        removed = len(self.trajectories) - kept
        points_suffix = (
            "*"
            if self.manual_line_points_dirty
            or self.manual_camera_line_points_dirty
            or self.manual_stop_points_dirty
            else ""
        )
        total_line_points = len(self.manual_line_points) + len(self.manual_camera_line_points)
        gt_diag = self._get_gt_quality_diagnostics()
        gt_status = "GT: n/a"
        if gt_diag is not None:
            bad_accel_count = len(gt_diag.get("bad_accel_indices", []))
            jump_count = len(gt_diag.get("jump_indices", []))
            gt_status = (
                f"GT({self.gt_future_mode}): acc[{gt_diag.get('min_accel', 0.0):.1f},"
                f"{gt_diag.get('max_accel', 0.0):.1f}] "
                f"jump={gt_diag.get('max_step_m', 0.0):.2f}m"
            )
            if bad_accel_count or jump_count:
                gt_status += f" WARN acc={bad_accel_count} jump={jump_count}"
            if self.gt_future_mode == "raw" and self.conv_data is not None:
                repaired_diag = self._get_gt_quality_diagnostics("repaired")
                if repaired_diag is not None:
                    repaired_acc = len(repaired_diag.get("bad_accel_indices", []))
                    repaired_jump = len(repaired_diag.get("jump_indices", []))
                    gt_status += f" | repaired acc={repaired_acc} jump={repaired_jump}"
        self.status_label.config(
            text=(
                f"Trajectories: {len(self.trajectories)} | Kept: {kept} | "
                f"Removed: {removed} | Bezier controls: {total_line_points}{points_suffix} | "
                f"Stops: {len(self.manual_stop_points)} | t0={int(t0_us)} | {gt_status}"
            )
        )
        
        # Update listbox
        self._refresh_trajectory_smoothness()
        self.traj_listbox.delete(0, tk.END)
        for i, traj in enumerate(self.trajectories):
            is_kept = self.trajectory_states.get(i, True)
            diagnostics = self.trajectory_smoothness.get(i, {})
            is_gt = self._is_gt_trajectory(traj, i)
            status = "GT" if is_gt else ("×" if not bool(diagnostics.get("ok", True)) else ("√" if is_kept else "×"))
            x_end = traj["x"][-1] if len(traj["x"]) > 0 else 0
            y_end = traj["y"][-1] if len(traj["y"]) > 0 else 0
            source = str(traj.get("source", "") or "traj")
            text = f"[{status}] T{i:<2} {source:<12} end=({x_end:6.1f}, {y_end:6.1f})"
            self.traj_listbox.insert(tk.END, text)
            if is_gt:
                self.traj_listbox.itemconfig(i, foreground=GT_COLOR_HEX)
            elif not bool(diagnostics.get("ok", True)):
                self.traj_listbox.itemconfig(i, foreground="#ff8a8a")
        
        if self.current_traj_idx < len(self.trajectories):
            self.traj_listbox.selection_clear(0, tk.END)
            self.traj_listbox.selection_set(self.current_traj_idx)
            self.traj_listbox.see(self.current_traj_idx)
        
        self._draw_trajectories()
        self._draw_speed_profile()
        self._draw_gt_speed_profile()
        self._draw_camera_images()
        self._update_cot_text()

    def _update_cot_text(self):
        """Display CoT for the selected trajectory when available."""
        cot = ""
        if 0 <= self.current_traj_idx < len(self.trajectories):
            cot = self.trajectories[self.current_traj_idx].get("cot", "")
        if not cot:
            cot = "No CoT saved for this trajectory."

        self.cot_text.configure(state=tk.NORMAL)
        self.cot_text.delete("1.0", tk.END)
        self.cot_text.insert(tk.END, cot)
        self.cot_text.configure(state=tk.DISABLED)
    
    def _on_list_select(self, event):
        selection = self.traj_listbox.curselection()
        if selection:
            if self.speed_edit_active and selection[0] != self.speed_edit_traj_idx:
                messagebox.showwarning(
                    "Speed Edit Active",
                    "Save or discard the current speed adjustment before selecting another trajectory.",
                )
                self.traj_listbox.selection_clear(0, tk.END)
                if self.speed_edit_traj_idx is not None:
                    self.traj_listbox.selection_set(int(self.speed_edit_traj_idx))
                return
            self.current_traj_idx = selection[0]
            self._update_display()
    
    def _prev_sample(self):
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before changing samples.",
            )
            return
        if self.current_idx > 0:
            self._load_sample(self.current_idx - 1)
            self._update_display()
    
    def _next_sample(self):
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before changing samples.",
            )
            return
        if self.current_idx < len(self.samples) - 1:
            self._load_sample(self.current_idx + 1)
            self._update_display()
    
    def _prev_traj(self):
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before selecting another trajectory.",
            )
            return
        for idx in range(self.current_traj_idx - 1, -1, -1):
            if not self._is_gt_trajectory(self.trajectories[idx], idx):
                self.current_traj_idx = idx
                break
        else:
            if self.current_traj_idx > 0:
                self.current_traj_idx -= 1
        self._update_display()
    
    def _next_traj(self):
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before selecting another trajectory.",
            )
            return
        for idx in range(self.current_traj_idx + 1, len(self.trajectories)):
            if not self._is_gt_trajectory(self.trajectories[idx], idx):
                self.current_traj_idx = idx
                break
        else:
            if self.current_traj_idx < len(self.trajectories) - 1:
                self.current_traj_idx += 1
        self._update_display()
    
    def _delete_traj(self):
        if 0 <= self.current_traj_idx < len(self.trajectories):
            traj = self.trajectories[self.current_traj_idx]
            source = str(traj.get("source", "")).lower()
            sample_idx = int(traj.get("sample_idx", self.current_traj_idx))
            if source == "gt" or (not source and sample_idx == 0):
                messagebox.showwarning("Keep GT", "The GT row stays in the parquet. Delete rule/manual trajectories only.")
                return
            delete_manual_controls = self._selected_trajectory_matches_manual_curve()
            self.trajectory_states[self.current_traj_idx] = False
            if delete_manual_controls:
                self._clear_current_manual_points()
            self._update_display()
    
    def _keep_traj(self):
        if 0 <= self.current_traj_idx < len(self.trajectories):
            self.trajectory_states[self.current_traj_idx] = True
            self._update_display()
    
    def _save_results(self):
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before writing the parquet.",
            )
            return
        if self.gt_only:
            messagebox.showwarning(
                "GT Only",
                "GT-only mode has no generated trajectory parquet to filter.",
            )
            return
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        
        df = pd.read_parquet(traj_file)
        if "t0_us" in df.columns:
            current_indices = df.index[df["t0_us"].astype("int64") == int(t0_us)].tolist()
            drop_indices = [
                row_idx
                for local_idx, row_idx in enumerate(current_indices)
                if not self.trajectory_states.get(local_idx, True)
            ]
            df_saved = df.drop(index=drop_indices).reset_index(drop=True)
        else:
            kept_mask = [self.trajectory_states.get(i, True) for i in range(len(df))]
            df_saved = df[kept_mask].reset_index(drop=True)
        
        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df_saved.to_parquet(traj_file, index=False)

        self._load_sample(self.current_idx)
        self._update_display()
        
        messagebox.showinfo(
            "Saved",
            f"Saved {len(df_saved)} trajectories to {traj_file}\n"
            f"(Removed {len(df) - len(df_saved)} trajectories)"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced GUI for viewing VLM-generated trajectories"
    )
    parser.add_argument(
        "--data_root", type=str, default="/home/tsingyu/train_data",
        help="Root directory for training data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory containing generated trajectories",
    )
    parser.add_argument(
        "--calibration_dir", type=str,
        default="/home/tsingyu/yzb/triplane_tokenization/cailibration",
        help="Directory containing calibration files",
    )
    parser.add_argument(
        "--cameras", type=str, default="RL,FC,RR",
        help="Comma-separated list of cameras to display",
    )
    parser.add_argument(
        "--start_index", type=int, default=None,
        help="1-based sample index to show at startup",
    )
    parser.add_argument(
        "--start_dataset", type=str, default="",
        help="Dataset name to show at startup, e.g. data_26_3_24_1_converted",
    )
    parser.add_argument(
        "--start_clip", type=str, default="",
        help="Clip stem to show at startup, e.g. 2026-03-24-12-06-59",
    )
    parser.add_argument(
        "--start_t0", type=int, default=None,
        help="Exact t0_us timestamp to show at startup",
    )
    parser.add_argument(
        "--no_restore_last", action="store_true",
        help="Do not restore the last viewed sample from the previous GUI session",
    )
    parser.add_argument(
        "--gt_only", action="store_true",
        help="Show GT samples directly from data_root and do not load generated VLA trajectories",
    )
    parser.add_argument(
        "--gt_stride_frames", type=int, default=3,
        help="Frame stride for GT-only t0 sampling",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",")]
    viewer = TrajectoryViewerEnhanced(
        data_root=args.data_root,
        output_dir=args.output_dir,
        calibration_dir=args.calibration_dir,
        cameras=cameras,
        start_index=args.start_index,
        start_dataset=args.start_dataset,
        start_clip=args.start_clip,
        start_t0=args.start_t0,
        restore_last=not args.no_restore_last,
        gt_only=args.gt_only,
        gt_stride_frames=args.gt_stride_frames,
    )
