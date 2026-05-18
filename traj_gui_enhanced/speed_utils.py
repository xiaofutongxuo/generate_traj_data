"""Speed-profile and dynamics helpers for the enhanced trajectory GUI."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import *
from .math_utils import *

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

def _history_speed_profile_from_xyz(
    history_xyz,
    dt_seconds: float = TRAJ_DT_SECONDS,
) -> np.ndarray:
    """Return one scalar speed value per history frame from adjacent history points."""
    points = np.asarray(history_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(points) == 1:
        return np.zeros(1, dtype=np.float64)

    transition_speed = (
        np.linalg.norm(np.diff(points, axis=0), axis=1)
        / max(float(dt_seconds), 1e-6)
    )
    return np.concatenate([[transition_speed[0]], transition_speed])

def _smooth_history_xyz_for_display(
    history_xyz,
    passes: int = 2,
) -> np.ndarray:
    """Return denoised history points for GUI display while preserving endpoints."""
    points = np.asarray(history_xyz, dtype=np.float64).reshape(-1, 3)
    smoothed = points.copy()
    if len(smoothed) < 5:
        return smoothed

    first = smoothed[0].copy()
    current = smoothed[-1].copy()
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    for _ in range(max(1, int(passes))):
        prev = smoothed.copy()
        for idx in range(2, len(smoothed) - 2):
            smoothed[idx] = (
                kernel[0] * prev[idx - 2]
                + kernel[1] * prev[idx - 1]
                + kernel[2] * prev[idx]
                + kernel[3] * prev[idx + 1]
                + kernel[4] * prev[idx + 2]
            )
        smoothed[0] = first
        smoothed[-1] = current
    return smoothed

def _smoothed_history_speed_profile_from_xyz(
    history_xyz,
    dt_seconds: float = TRAJ_DT_SECONDS,
    xyz_passes: int = 2,
    speed_passes: int = 1,
) -> np.ndarray:
    """Return a display-oriented history speed curve from denoised history points."""
    points = np.asarray(history_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    smoothed_points = _smooth_history_xyz_for_display(points, passes=xyz_passes)
    raw_speed = _history_speed_profile_from_xyz(smoothed_points, dt_seconds=dt_seconds)
    display_speed = _smooth_display_speed_profile(raw_speed, passes=speed_passes)
    if len(display_speed):
        display_speed[-1] = raw_speed[-1]
    return display_speed

def _smooth_display_speed_profile(
    speed: np.ndarray,
    passes: int = 2,
) -> np.ndarray:
    """Lightly smooth a speed curve for visualization without enforcing edit limits."""
    smoothed = np.asarray(speed, dtype=np.float64).reshape(-1).copy()
    if len(smoothed) < 5:
        return np.clip(smoothed, 0.0, np.inf)
    smoothed[~np.isfinite(smoothed)] = 0.0
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    for _ in range(max(1, int(passes))):
        padded = np.pad(smoothed, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    return np.clip(smoothed, 0.0, np.inf)

def _smoothed_gt_speed_profile_from_xyz(
    gt_xyz,
    dt_seconds: float = TRAJ_DT_SECONDS,
    passes: int = 2,
) -> np.ndarray:
    """Return a display-oriented GT speed curve that damps position-diff noise."""
    points = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    raw_speed = _speed_profile_from_trajectory(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        dt_seconds=dt_seconds,
    )
    return _smooth_display_speed_profile(raw_speed, passes=passes)

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

    from .cluster_utils import _trajectory_quality_diagnostics

    diagnostics = _trajectory_quality_diagnostics(sampled, dt_seconds)
    if not bool(diagnostics.get("ok", False)):
        return None
    return sampled

__all__ = [name for name in globals() if (name.startswith("_") and not name.startswith("__")) or name.isupper()]
