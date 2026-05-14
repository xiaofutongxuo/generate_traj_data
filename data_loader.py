# SPDX-License-Identifier: Apache-2.0
"""Data loader for training data compatible with Alpamayo 1.5 format.

This module loads data from the train_data directory which contains
converted video data in a format compatible with Alpamayo inference.
"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt

if os.environ.get("GENERATE_TRAJ_USE_TORCH", "1").lower() in {"0", "false", "no"}:
    torch = None
else:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

sys.path.insert(0, "/home/tsingyu/lxh/alpamayo_1.5/src")

CAMERA_FILE_MAP = {
    "FL": 0, "FC": 1, "FR": 2, "RL": 3, "RC": 4, "RR": 5,
    "FC_FAR": 6, "FL_FAR": 7, "FR_FAR": 8,
}

ALL_CAMERAS = ["FL", "FC", "FR", "RL", "RC", "RR"]
GT_ACCEL_MIN_MPS2 = -6.0
GT_ACCEL_MAX_MPS2 = 2.0
GT_MAX_STEP_SPEED_MPS = 15.0
DEFAULT_CONTINUITY_MAX_GAP_SECONDS = 0.3


def _resize_frames(frames, target_hw: tuple[int, int]):
    """Resize frames to target size using bilinear interpolation."""
    if torch is not None and torch.is_tensor(frames):
        from torchvision.transforms.functional import resize

        return resize(frames, target_hw, antialias=True)

    import cv2

    target_h, target_w = target_hw
    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for frame in frames
    ]
    return np.stack(resized, axis=0)


def _frames_to_model_array(frames_np: np.ndarray, target_hw: tuple[int, int]):
    """Return frames as [T, C, H, W], using torch when available."""
    if torch is not None:
        frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2)
        return _resize_frames(frames_t, target_hw)

    frames_np = _resize_frames(frames_np, target_hw)
    return np.transpose(frames_np, (0, 3, 1, 2))


def _array_from_numpy(array: np.ndarray):
    if torch is not None:
        return torch.from_numpy(array)
    return np.asarray(array)


def _stack_arrays(arrays: list):
    if torch is not None:
        return torch.stack(arrays, dim=0)
    return np.stack(arrays, axis=0)


def _unsqueeze_twice(array):
    if torch is not None and torch.is_tensor(array):
        return array.unsqueeze(0).unsqueeze(0)
    return np.expand_dims(np.expand_dims(array, axis=0), axis=0)


def _normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternions stored in scipy xyzw order."""
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return quat / norm


@lru_cache(maxsize=16)
def _load_dataset_egomotion_dataframe_cached(
    data_root: str,
    dataset_name: str,
) -> pd.DataFrame:
    """Load all egomotion rows for a dataset, sorted by global timestamp."""
    egomotion_dir = Path(data_root) / dataset_name / "data-egomotion"
    if not egomotion_dir.exists():
        raise FileNotFoundError(f"Egomotion data not found: {egomotion_dir}")

    frames = []
    for ego_file in sorted(egomotion_dir.glob("*.egomotion.parquet")):
        df = pd.read_parquet(ego_file)
        if "timestamp" not in df.columns or len(df) == 0:
            continue
        df = df.copy()
        df["_clip_stem"] = ego_file.name.replace(".egomotion.parquet", "")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No egomotion parquet files found in {egomotion_dir}")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values("timestamp")
    merged = merged.drop_duplicates(subset=["timestamp"], keep="first")
    return merged.reset_index(drop=True)


def _load_dataset_egomotion_dataframe(data_root: str | Path, dataset_name: str) -> pd.DataFrame:
    return _load_dataset_egomotion_dataframe_cached(str(Path(data_root)), str(dataset_name)).copy()


def _sorted_unique_timestamps(source_timestamps: np.ndarray) -> np.ndarray:
    source = np.asarray(source_timestamps, dtype=np.int64).reshape(-1)
    if len(source) == 0:
        return source
    return np.unique(np.sort(source))


def _coverage_mask_for_sorted_timestamps(
    source_timestamps_sorted: np.ndarray,
    target_timestamps: np.ndarray,
    max_gap_us: int,
) -> np.ndarray:
    source = np.asarray(source_timestamps_sorted, dtype=np.int64).reshape(-1)
    target = np.asarray(target_timestamps, dtype=np.int64).reshape(-1)
    if len(target) == 0:
        return np.zeros(0, dtype=bool)
    if len(source) == 0:
        return np.zeros(len(target), dtype=bool)

    insert_at = np.searchsorted(source, target, side="left")
    covered = np.zeros(len(target), dtype=bool)

    exact_idx = insert_at < len(source)
    if exact_idx.any():
        covered[exact_idx] = source[insert_at[exact_idx]] == target[exact_idx]

    between_idx = (~covered) & (insert_at > 0) & (insert_at < len(source))
    if between_idx.any():
        left = source[insert_at[between_idx] - 1]
        right = source[insert_at[between_idx]]
        covered[between_idx] = (right - left) <= int(max_gap_us)
    return covered


def _coverage_mask_for_timestamps(
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    max_gap_seconds: float = DEFAULT_CONTINUITY_MAX_GAP_SECONDS,
) -> np.ndarray:
    """Return whether each target can be read directly or interpolated across a small gap."""
    max_gap_us = int(round(float(max_gap_seconds) * 1_000_000))
    return _coverage_mask_for_sorted_timestamps(
        _sorted_unique_timestamps(source_timestamps),
        target_timestamps,
        max_gap_us,
    )


def filter_t0s_with_full_future(
    data_root: str | Path,
    dataset_name: str,
    t0_values: list[int],
    num_future_steps: int = 64,
    time_step: float = 0.1,
    max_gap_seconds: float = DEFAULT_CONTINUITY_MAX_GAP_SECONDS,
) -> list[int]:
    """Keep t0 values whose full future horizon is covered by continuous egomotion."""
    if not t0_values:
        return []
    try:
        df_ego = _load_dataset_egomotion_dataframe(data_root, dataset_name)
    except Exception:
        return []

    source_ts = _sorted_unique_timestamps(df_ego["timestamp"].to_numpy(dtype=np.int64))
    dt_us = int(round(float(time_step) * 1_000_000))
    max_gap_us = int(round(float(max_gap_seconds) * 1_000_000))
    t0_arr = np.asarray([int(t0) for t0 in t0_values], dtype=np.int64)
    offsets = np.arange(0, int(num_future_steps) + 1, dtype=np.int64) * dt_us
    target_ts = t0_arr[:, None] + offsets[None, :]
    coverage = _coverage_mask_for_sorted_timestamps(
        source_ts,
        target_ts.reshape(-1),
        max_gap_us,
    ).reshape(target_ts.shape)
    keep_mask = coverage.all(axis=1)
    return [int(t0) for t0, keep in zip(t0_arr, keep_mask) if bool(keep)]


def _interp_ego_at_timestamps(df, timestamps: np.ndarray):
    """Interpolate ego pose at given timestamps."""
    from scipy.interpolate import interp1d
    ts = df["timestamp"].values.astype(np.float64)
    xyz = df[["x", "y", "z"]].values.astype(np.float64)
    quat = df[["qx", "qy", "qz", "qw"]].values.astype(np.float64)

    interp_xyz = interp1d(ts, xyz, kind="linear", axis=0, fill_value="extrapolate")
    interp_quat = interp1d(ts, quat, kind="linear", axis=0, fill_value="extrapolate")

    return interp_xyz(timestamps), _normalize_quat_xyzw(interp_quat(timestamps))


def _interp_velocity_and_quat_at_timestamps(
    df,
    timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate body-frame velocity and xyzw quaternion at timestamps."""
    from scipy.interpolate import interp1d

    ts = df["timestamp"].values.astype(np.float64)
    velocity = df[["vx", "vy", "vz"]].values.astype(np.float64)
    quat = df[["qx", "qy", "qz", "qw"]].values.astype(np.float64)

    interp_velocity = interp1d(ts, velocity, kind="linear", axis=0, fill_value="extrapolate")
    interp_quat = interp1d(ts, quat, kind="linear", axis=0, fill_value="extrapolate")
    return interp_velocity(timestamps), _normalize_quat_xyzw(interp_quat(timestamps))


def _interp_forward_acceleration_at_timestamps(df, timestamps: np.ndarray) -> np.ndarray:
    """Interpolate longitudinal acceleration if present in the egomotion parquet."""
    if "ax" not in df.columns:
        return np.full(len(timestamps), np.nan, dtype=np.float32)

    from scipy.interpolate import interp1d

    ts = df["timestamp"].values.astype(np.float64)
    accel = df["ax"].values.astype(np.float64)
    interp_accel = interp1d(ts, accel, kind="linear", axis=0, fill_value="extrapolate")
    return interp_accel(timestamps).astype(np.float32)


def _clamp_speed_profile_by_acceleration(
    speed: np.ndarray,
    dt: np.ndarray,
    min_accel_mps2: float = GT_ACCEL_MIN_MPS2,
    max_accel_mps2: float = GT_ACCEL_MAX_MPS2,
) -> np.ndarray:
    """Clamp scalar speed transitions to the configured acceleration envelope."""
    repaired = np.asarray(speed, dtype=np.float64).copy()
    if len(repaired) == 0:
        return repaired
    repaired[~np.isfinite(repaired)] = 0.0
    repaired = np.maximum(repaired, 0.0)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)
    if len(dt) != max(0, len(repaired) - 1):
        raise ValueError("dt length must be speed length minus one")

    for idx in range(1, len(repaired)):
        step_dt = max(float(dt[idx - 1]), 1e-6)
        lower = max(0.0, repaired[idx - 1] + min_accel_mps2 * step_dt)
        upper = repaired[idx - 1] + max_accel_mps2 * step_dt
        repaired[idx] = float(np.clip(repaired[idx], lower, max(lower, upper)))
    return repaired


def _interp_polyline_by_distance(
    dense_points: np.ndarray,
    cumulative: np.ndarray,
    target_distances: np.ndarray,
) -> np.ndarray:
    sampled = np.empty((len(target_distances), dense_points.shape[1]), dtype=np.float64)
    for dim in range(dense_points.shape[1]):
        sampled[:, dim] = np.interp(target_distances, cumulative, dense_points[:, dim])
    return sampled


def _acceleration_limited_resample_local_path(
    xyz: np.ndarray,
    dt_seconds: float = 0.1,
    initial_speed_mps: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample local GT path so endpoint is preserved and scalar acceleration is clamped."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        raise ValueError("Need at least two xyz points for acceleration-limited GT repair")

    xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), points[:, :2]])
    seg_lengths = np.linalg.norm(np.diff(xy_from_origin, axis=0), axis=1)
    total_length = float(seg_lengths.sum())
    if total_length <= 1e-6:
        output = points.copy()
        output[:, :2] = 0.0
        if output.shape[1] > 2:
            output[:, 2] = 0.0
        return output.astype(np.float32), np.zeros(len(points), dtype=np.float32)

    desired_speed = seg_lengths / dt_seconds
    desired_speed = np.clip(desired_speed, 0.0, GT_MAX_STEP_SPEED_MPS)
    dt = np.full(len(desired_speed) - 1, dt_seconds, dtype=np.float64)
    finite_initial_speed = None
    if initial_speed_mps is not None and np.isfinite(initial_speed_mps):
        finite_initial_speed = float(np.clip(initial_speed_mps, 0.0, GT_MAX_STEP_SPEED_MPS))
    best_speed = None
    best_error = float("inf")
    lo = -GT_MAX_STEP_SPEED_MPS
    hi = GT_MAX_STEP_SPEED_MPS
    for _ in range(36):
        bias = (lo + hi) * 0.5
        candidate_speed = desired_speed + bias
        if finite_initial_speed is None:
            speed = _clamp_speed_profile_by_acceleration(
                candidate_speed,
                dt,
                min_accel_mps2=GT_ACCEL_MIN_MPS2,
                max_accel_mps2=GT_ACCEL_MAX_MPS2,
            )
        else:
            speed_with_initial = _clamp_speed_profile_by_acceleration(
                np.concatenate([[finite_initial_speed], candidate_speed]),
                np.full(len(candidate_speed), dt_seconds, dtype=np.float64),
                min_accel_mps2=GT_ACCEL_MIN_MPS2,
                max_accel_mps2=GT_ACCEL_MAX_MPS2,
            )
            speed = speed_with_initial[1:]
        speed = np.clip(speed, 0.0, GT_MAX_STEP_SPEED_MPS)
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
        raise ValueError("Could not preserve GT endpoint within acceleration limits")

    target_distances = np.cumsum(best_speed * dt_seconds)
    target_distances = np.maximum.accumulate(np.clip(target_distances, 0.0, total_length))
    if abs(float(target_distances[-1]) - total_length) <= max(0.25, total_length * 0.02):
        target_distances[-1] = total_length

    dense_points = np.vstack([
        np.zeros((1, points.shape[1]), dtype=np.float64),
        points.copy(),
    ])
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    output = _interp_polyline_by_distance(dense_points, cumulative, target_distances)
    if output.shape[1] > 2:
        output[:, 2] = 0.0

    accel = np.zeros(len(best_speed), dtype=np.float64)
    if finite_initial_speed is not None and len(best_speed) > 0:
        accel[0] = (best_speed[0] - finite_initial_speed) / dt_seconds
    if len(best_speed) > 1:
        accel[1:] = np.diff(best_speed) / dt_seconds
    accel = np.clip(accel, GT_ACCEL_MIN_MPS2, GT_ACCEL_MAX_MPS2)
    return output.astype(np.float32), accel.astype(np.float32)


def repair_future_xyz_by_velocity(
    df_ego,
    t0_us: int,
    future_ts: np.ndarray,
    t0_rot_inv: spt.Rotation,
    raw_future_xyz_local: Optional[np.ndarray] = None,
    initial_speed_mps: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild future local xyz after clamping acceleration.

    When raw future positions are available, the repaired path preserves the raw
    endpoint and redistributes motion along the path. This avoids velocity-only
    integration stretching GT far beyond the original future horizon.
    """
    if raw_future_xyz_local is not None:
        try:
            return _acceleration_limited_resample_local_path(
                raw_future_xyz_local,
                initial_speed_mps=initial_speed_mps,
            )
        except ValueError:
            pass

    required = {"timestamp", "vx", "vy", "vz", "qx", "qy", "qz", "qw"}
    if not required.issubset(df_ego.columns):
        raise ValueError("Egomotion parquet lacks velocity/quaternion columns needed for GT repair")

    integration_ts = np.concatenate([
        np.asarray([int(t0_us)], dtype=np.int64),
        np.asarray(future_ts, dtype=np.int64),
    ])
    if np.any(np.diff(integration_ts) <= 0):
        raise ValueError("Future timestamps must be strictly increasing for GT repair")

    body_velocity, quat_xyzw = _interp_velocity_and_quat_at_timestamps(df_ego, integration_ts)
    world_velocity = spt.Rotation.from_quat(quat_xyzw).apply(body_velocity)
    local_velocity = t0_rot_inv.apply(world_velocity)
    dt = np.diff(integration_ts.astype(np.float64)) * 1e-6

    planar_velocity = local_velocity[:, :2]
    raw_speed = np.linalg.norm(planar_velocity, axis=1)
    speed = raw_speed.copy()
    if initial_speed_mps is not None and np.isfinite(initial_speed_mps):
        speed[0] = float(np.clip(initial_speed_mps, 0.0, GT_MAX_STEP_SPEED_MPS))
    repaired_speed = _clamp_speed_profile_by_acceleration(speed, dt)

    direction = np.zeros_like(planar_velocity)
    moving = raw_speed > 1e-3
    direction[moving] = planar_velocity[moving] / raw_speed[moving, None]
    if moving.any():
        first = int(np.flatnonzero(moving)[0])
        direction[:first] = direction[first]
        for idx in range(first + 1, len(direction)):
            if not moving[idx]:
                direction[idx] = direction[idx - 1]
    else:
        direction[:, 0] = 1.0

    local_velocity_repaired = np.zeros_like(local_velocity)
    local_velocity_repaired[:, :2] = direction * repaired_speed[:, None]
    if local_velocity.shape[1] > 2:
        local_velocity_repaired[:, 2] = local_velocity[:, 2]

    repaired_accel = np.zeros(len(future_ts), dtype=np.float64)
    for idx in range(len(future_ts)):
        repaired_accel[idx] = (
            (repaired_speed[idx + 1] - repaired_speed[idx])
            / max(float(dt[idx]), 1e-6)
        )
    repaired_accel = np.clip(repaired_accel, GT_ACCEL_MIN_MPS2, GT_ACCEL_MAX_MPS2)

    repaired = np.zeros((len(future_ts), 3), dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    for idx in range(len(future_ts)):
        pos = pos + 0.5 * (
            local_velocity_repaired[idx] + local_velocity_repaired[idx + 1]
        ) * dt[idx]
        repaired[idx] = pos

    repaired[:, 2] = 0.0
    return repaired.astype(np.float32), repaired_accel.astype(np.float32)


def _estimate_t0_motion_heading(
    hist_xyz: np.ndarray,
    hist_quat: np.ndarray,
    heading_num_steps: int,
    min_heading_displacement_m: float,
) -> tuple[spt.Rotation, str, float]:
    """Estimate the t0 rotation using recent planar displacement."""
    t0_quat = hist_quat[-1].copy()
    start_idx = max(0, len(hist_xyz) - 1 - int(heading_num_steps))
    disp_xy = hist_xyz[-1, :2] - hist_xyz[start_idx, :2]
    planar_disp_m = float(np.linalg.norm(disp_xy))

    if planar_disp_m >= float(min_heading_displacement_m):
        yaw = float(np.arctan2(disp_xy[1], disp_xy[0]))
        return spt.Rotation.from_euler("z", yaw), "motion", planar_disp_m

    fallback_rot = spt.Rotation.from_quat(t0_quat)
    return fallback_rot, "quaternion_fallback", planar_disp_m


def get_dataset_names(data_root: str) -> list[str]:
    """Get list of dataset names from train_data directory.

    Args:
        data_root: Root directory containing converted datasets

    Returns:
        List of dataset names (folder names ending with '_converted')
    """
    root = Path(data_root)
    if not root.exists():
        raise ValueError(f"Data root does not exist: {data_root}")

    datasets = []
    for item in root.iterdir():
        if item.is_dir() and item.name.endswith("_converted"):
            datasets.append(item.name)
    return sorted(datasets)


def get_clip_stems_from_dataset(dataset_path: Path) -> list[str]:
    """Get list of clip stems from a dataset directory.

    Args:
        dataset_path: Path to a converted dataset directory

    Returns:
        List of unique clip stems (segment names without camera suffix)
    """
    mp4_dir = dataset_path / "mp4-converted"
    if not mp4_dir.exists():
        return []

    stems = set()
    for f in mp4_dir.iterdir():
        if f.suffix == ".mp4":
            stem = f.stem
            for cam in ALL_CAMERAS + ["FC_FAR"]:
                suffix = f"_fovs_{cam}"
                if stem.endswith(suffix):
                    stems.add(stem[:-len(suffix)])
                    break

    return sorted(stems)


def load_data(
    data_root: str,
    clip_stem: str,
    dataset_name: str,
    t0_us: Optional[int] = None,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    num_frames: int = 4,
    target_image_hw: tuple[int, int] = (1280, 1920),
    heading_num_steps: int = 5,
    min_heading_displacement_m: float = 0.2,
    cameras: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Load data for a specific clip and timestamp.

    Args:
        data_root: Root directory containing converted datasets
        clip_stem: Clip stem name (segment name)
        dataset_name: Dataset folder name
        t0_us: Timestamp for t0 (microseconds). If None, uses default offset.
        num_history_steps: Number of history steps
        num_future_steps: Number of future steps
        time_step: Time step in seconds
        num_frames: Number of image frames to load
        target_image_hw: Target image (height, width)
        heading_num_steps: Number of steps for heading estimation
        min_heading_displacement_m: Minimum displacement for motion heading
        cameras: List of cameras to load. Defaults to RL, FC, RR.

    Returns:
        Dictionary containing:
            - image_frames: Stacked image tensors [num_cams, num_frames, C, H, W]
            - camera_indices: Camera indices [num_cams]
            - ego_history_xyz: History positions [1, 1, num_history, 3]
            - ego_history_rot: History rotations [1, 1, num_history, 3, 3]
            - ego_future_xyz: Future positions [1, 1, num_future, 3]
            - ego_future_rot: Future rotations [1, 1, num_future, 3, 3]
            - relative_timestamps: Relative timestamps [num_cams, num_frames]
            - absolute_timestamps: Absolute timestamps [num_cams, num_frames]
            - t0_us: t0 timestamp in microseconds
            - clip_id: Clip identifier
            - dataset_name: Dataset name
            - has_future_gt: Whether future GT is available
    """
    if cameras is None:
        cameras = ["RL", "FC", "RR"]

    dataset_path = Path(data_root) / dataset_name

    egomotion_path = dataset_path / "data-egomotion"
    timestamps_path = dataset_path / "data-timestamps"
    video_path = dataset_path / "mp4-converted"

    if not egomotion_path.exists():
        raise FileNotFoundError(f"Egomotion data not found: {egomotion_path}")
    if not timestamps_path.exists():
        raise FileNotFoundError(f"Timestamp data not found: {timestamps_path}")

    ego_file = egomotion_path / f"{clip_stem}.egomotion.parquet"
    ts_file = timestamps_path / f"{clip_stem}.timestamps.parquet"

    if not ego_file.exists():
        raise FileNotFoundError(f"Egomotion file not found: {ego_file}")
    if not ts_file.exists():
        raise FileNotFoundError(f"Timestamp file not found: {ts_file}")

    df_ego_current = pd.read_parquet(ego_file)
    try:
        df_ego = _load_dataset_egomotion_dataframe(data_root, dataset_name)
    except Exception:
        df_ego = df_ego_current
    df_master = pd.read_parquet(ts_file)

    master_ts = df_master["timestamp"].values
    clip_start_us = int(master_ts[0])
    clip_end_us = int(master_ts[-1])

    dt_us = int(round(time_step * 1_000_000))

    if t0_us is None:
        t0_abs = clip_start_us + 5_100_000
    elif t0_us < clip_start_us:
        t0_abs = clip_start_us + t0_us
    else:
        t0_abs = t0_us
    t0_abs = int(np.clip(t0_abs, clip_start_us, clip_end_us))

    history_ts = np.array(
        [t0_abs - (num_history_steps - 1 - i) * dt_us for i in range(num_history_steps)],
        dtype=np.int64,
    )
    future_ts = np.array(
        [t0_abs + (i + 1) * dt_us for i in range(num_future_steps)],
        dtype=np.int64,
    )
    source_ts = df_ego["timestamp"].to_numpy(dtype=np.int64)
    current_valid_mask = _coverage_mask_for_timestamps(
        source_ts,
        np.array([t0_abs], dtype=np.int64),
    )
    history_valid_mask = _coverage_mask_for_timestamps(source_ts, history_ts)
    future_valid_mask = _coverage_mask_for_timestamps(source_ts, future_ts)

    hist_xyz, hist_quat = _interp_ego_at_timestamps(df_ego, history_ts)
    fut_xyz, fut_quat = _interp_ego_at_timestamps(df_ego, future_ts)

    t0_xyz = hist_xyz[-1].copy()
    t0_rot, heading_source, heading_disp_m = _estimate_t0_motion_heading(
        hist_xyz=hist_xyz,
        hist_quat=hist_quat,
        heading_num_steps=heading_num_steps,
        min_heading_displacement_m=min_heading_displacement_m,
    )
    t0_rot_inv = t0_rot.inv()

    ego_hist_xyz_local = t0_rot_inv.apply(hist_xyz - t0_xyz).astype(np.float32)
    ego_fut_xyz_local = t0_rot_inv.apply(fut_xyz - t0_xyz).astype(np.float32)
    gt_initial_speed_mps = None
    if len(ego_hist_xyz_local) >= 2 and time_step > 0:
        gt_initial_speed_mps = float(
            np.linalg.norm(ego_hist_xyz_local[-1, :2] - ego_hist_xyz_local[-2, :2])
            / float(time_step)
        )
    ego_hist_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(hist_quat)
    ).as_matrix().astype(np.float32)
    ego_fut_rot_local = (
        t0_rot_inv * spt.Rotation.from_quat(fut_quat)
    ).as_matrix().astype(np.float32)
    ego_fut_forward_acc = _interp_forward_acceleration_at_timestamps(df_ego, future_ts)
    ego_fut_forward_acc_repaired = np.clip(
        ego_fut_forward_acc,
        GT_ACCEL_MIN_MPS2,
        GT_ACCEL_MAX_MPS2,
    ).astype(np.float32)
    try:
        ego_fut_xyz_repaired_local, ego_fut_forward_acc_repaired = repair_future_xyz_by_velocity(
            df_ego,
            t0_abs,
            future_ts,
            t0_rot_inv,
            raw_future_xyz_local=ego_fut_xyz_local,
            initial_speed_mps=gt_initial_speed_mps,
        )
        gt_repair_available = True
        gt_repair_error = ""
    except Exception as exc:
        ego_fut_xyz_repaired_local = ego_fut_xyz_local.copy()
        gt_repair_available = False
        gt_repair_error = str(exc)

    target_img_ts = np.array(
        [t0_abs - (num_frames - 1 - i) * dt_us for i in range(num_frames)],
        dtype=np.int64,
    )

    image_frames_list = []
    timestamps_list = []
    cam_indices = []

    import cv2

    for cam_name in cameras:
        video_file = video_path / f"{clip_stem}_fovs_{cam_name}.mp4"
        cam_ts_file = timestamps_path / f"{clip_stem}_fovs_{cam_name}.timestamps.parquet"

        if not video_file.exists():
            continue
        if not cam_ts_file.exists():
            continue

        df_cam_ts = pd.read_parquet(cam_ts_file)
        cam_ts_arr = df_cam_ts["timestamp"].values
        frame_idxs = [int(np.abs(cam_ts_arr - t).argmin()) for t in target_img_ts]

        cap = cv2.VideoCapture(str(video_file))
        frames_np = []
        for idx in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_np.append(frame[:, :, ::-1])
        cap.release()

        if len(frames_np) != num_frames:
            continue
        frames_np = np.stack(frames_np, axis=0)

        frames_t = _frames_to_model_array(frames_np, target_image_hw)

        image_frames_list.append(frames_t)
        timestamps_list.append(_array_from_numpy(cam_ts_arr[frame_idxs]))
        cam_indices.append(CAMERA_FILE_MAP.get(cam_name, 0))

    if not image_frames_list:
        raise ValueError(f"No valid cameras found for clip {clip_stem}")

    if torch is not None:
        cam_idx_t = torch.tensor(cam_indices, dtype=torch.int64)
        image_frames = _stack_arrays(image_frames_list)
        abs_ts = _stack_arrays(timestamps_list)
        rel_ts = (abs_ts - abs_ts.min()).float() * 1e-6
    else:
        cam_idx_t = np.asarray(cam_indices, dtype=np.int64)
        image_frames = _stack_arrays(image_frames_list)
        abs_ts = _stack_arrays(timestamps_list).astype(np.int64)
        rel_ts = (abs_ts - abs_ts.min()).astype(np.float32) * 1e-6

    has_future_gt = bool(current_valid_mask.all() and future_valid_mask.all())

    return {
        "image_frames": image_frames,
        "camera_indices": cam_idx_t,
        "ego_history_xyz": _unsqueeze_twice(_array_from_numpy(ego_hist_xyz_local)),
        "ego_history_rot": _unsqueeze_twice(_array_from_numpy(ego_hist_rot_local)),
        "ego_history_valid_mask": _unsqueeze_twice(_array_from_numpy(history_valid_mask.astype(bool))),
        "ego_future_xyz": _unsqueeze_twice(_array_from_numpy(ego_fut_xyz_local)),
        "ego_future_xyz_raw": _unsqueeze_twice(_array_from_numpy(ego_fut_xyz_local.copy())),
        "ego_future_xyz_repaired": _unsqueeze_twice(_array_from_numpy(ego_fut_xyz_repaired_local)),
        "ego_future_valid_mask": _unsqueeze_twice(_array_from_numpy(future_valid_mask.astype(bool))),
        "ego_future_forward_acceleration": _unsqueeze_twice(_array_from_numpy(ego_fut_forward_acc)),
        "ego_future_forward_acceleration_raw": _unsqueeze_twice(_array_from_numpy(ego_fut_forward_acc.copy())),
        "ego_future_forward_acceleration_repaired": _unsqueeze_twice(_array_from_numpy(ego_fut_forward_acc_repaired)),
        "ego_future_rot": _unsqueeze_twice(_array_from_numpy(ego_fut_rot_local)),
        "relative_timestamps": rel_ts,
        "absolute_timestamps": abs_ts,
        "t0_us": t0_abs,
        "clip_id": clip_stem,
        "dataset_name": dataset_name,
        "has_future_gt": has_future_gt,
        "gt_repair_available": gt_repair_available,
        "gt_repair_error": gt_repair_error,
        "gt_repair_initial_speed_mps": gt_initial_speed_mps,
        "t0_heading_source": heading_source,
        "t0_heading_disp_m": heading_disp_m,
        "t0_heading_yaw_rad": float(t0_rot.as_euler("zyx")[0]),
    }


def to_device(data: dict, device: str) -> dict:
    """Move data tensors to device.

    Args:
        data: Data dictionary
        device: Target device

    Returns:
        Data dictionary with tensors moved to device
    """
    output = {}
    for key, value in data.items():
        if torch is not None and torch.is_tensor(value):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def get_t0_candidates(
    data_root: str,
    dataset_name: str,
    clip_stem: str,
    min_speed_mps: float = 2.0,
) -> list[tuple[int, float]]:
    """Get valid t0 candidates for a clip based on speed.

    Args:
        data_root: Root directory containing converted datasets
        dataset_name: Dataset folder name
        clip_stem: Clip stem name
        min_speed_mps: Minimum speed requirement in m/s

    Returns:
        List of (timestamp_us, speed_mps) tuples
    """
    dataset_path = Path(data_root) / dataset_name
    egomotion_path = dataset_path / "data-egomotion"

    ego_file = egomotion_path / f"{clip_stem}.egomotion.parquet"

    if not ego_file.exists():
        return []

    df_ego = pd.read_parquet(ego_file)

    spd = np.sqrt(df_ego["vx"]**2 + df_ego["vy"]**2 + df_ego["vz"]**2).values
    ts = df_ego["timestamp"].values

    lo, hi = 16, len(ts) - 65
    if lo >= hi:
        return []

    candidates = []
    for idx in range(lo, hi + 1, 30):
        if spd[idx] >= min_speed_mps:
            candidates.append((int(ts[idx]), float(spd[idx])))

    return candidates
