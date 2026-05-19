#!/usr/bin/env python3
"""Cluster GT trajectories by acceleration/curvature and visualize one match.

This is an experiment script for the copied k_means folder.  It intentionally
leaves the original xy-based K-Means pipeline untouched.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_TXT = SCRIPT_DIR / "future_trajectories_xy.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "acc_curvature_experiment"
DEFAULT_TRAIN_DATA_ROOT = Path("/home/ubuntu/Public/train_data")
DEFAULT_GENERATE_OUTPUT_ROOT = SCRIPT_DIR.parent / "output"
DEFAULT_SOURCE_DATA_ROOT = Path("/home/ubuntu/Public/train_data")
DEFAULT_DATASET_FILTER = "data_26_5_8_converted"
DT_SECONDS = 0.1


@dataclass(frozen=True)
class TrajectoryRow:
    traj_id: int
    dataset: str
    clip: str
    t0_us: int
    t0_index: int
    xy: np.ndarray


@dataclass(frozen=True)
class SmoothingConfig:
    speed_passes: int = 1
    acceleration_passes: int = 2
    curvature_passes: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_txt", type=Path, default=DEFAULT_DATA_TXT)
    parser.add_argument(
        "--match_data_txt",
        type=Path,
        default=None,
        help="Optional trajectory txt for GT samples to match against clusters fit from --data_txt.",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_data_root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--generate_output_root", type=Path, default=DEFAULT_GENERATE_OUTPUT_ROOT)
    parser.add_argument("--source_data_root", type=Path, default=DEFAULT_SOURCE_DATA_ROOT)
    parser.add_argument(
        "--dataset_filter",
        type=str,
        default=DEFAULT_DATASET_FILTER,
        help="Only cluster rows from this dataset. Use 'all' to disable filtering.",
    )
    parser.add_argument(
        "--match_dataset_filter",
        type=str,
        default="",
        help="Only use matched GT rows from this dataset. Empty means use --dataset_filter.",
    )
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n_clusters", type=int, default=1000)
    parser.add_argument(
        "--sample_row",
        type=int,
        default=0,
        help="Zero-based row in future_trajectories_xy.txt to visualize.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Number of nearest KMeans centers to visualize.",
    )
    parser.add_argument(
        "--speed_window",
        type=int,
        default=3,
        help="Average speed over [t0-speed_window, t0+speed_window] when parquet is available.",
    )
    parser.add_argument(
        "--feature_clip_percentile",
        type=float,
        default=99.0,
        help="Robustly clip acceleration/curvature features before standardization.",
    )
    parser.add_argument(
        "--smooth_passes",
        type=int,
        default=None,
        help=(
            "Legacy override: use the same smoothing passes for speed, "
            "raw acceleration, and curvature."
        ),
    )
    parser.add_argument("--speed_smooth_passes", type=int, default=None)
    parser.add_argument("--acceleration_smooth_passes", type=int, default=None)
    parser.add_argument("--curvature_smooth_passes", type=int, default=None)
    parser.add_argument(
        "--feature_source",
        choices=["xy_derived", "native_parquet"],
        default="native_parquet",
        help="Use xy-derived dynamics or native parquet scalar-speed acceleration plus curvature.",
    )
    parser.add_argument(
        "--selection_method",
        choices=["dynamic_feature", "rollout_xy_distance"],
        default="dynamic_feature",
        help=(
            "Choose top candidates by standardized acceleration/curvature distance, "
            "or by xy distance after rolling every center out from the current speed."
        ),
    )
    parser.add_argument(
        "--candidate_representation",
        choices=[
            "center_rollout",
            "medoid_xy",
            "gt_nearest_member_xy",
            "acc_curvature_nearest_member_rollout",
        ],
        default="acc_curvature_nearest_member_rollout",
        help=(
            "Draw candidates by integrating the KMeans center, or by using each "
            "selected cluster medoid's real xy trajectory, or by using each "
            "selected cluster member nearest to the current GT in xy, or by "
            "rolling out the selected cluster member nearest to the current GT "
            "in smoothed acceleration/curvature feature space."
        ),
    )
    parser.add_argument(
        "--curvature_min_speed_mps",
        type=float,
        default=0.0,
        help="For native_parquet features, set curvature to 0 where future speed is below this threshold; 0 keeps all native curvature.",
    )
    parser.add_argument(
        "--member_endpoint_weight",
        type=float,
        default=0.15,
        help=(
            "Small meters-to-score weight used when choosing a feature-nearest member "
            "inside each selected cluster: score=feature_distance+weight*endpoint_distance_m."
        ),
    )
    parser.add_argument(
        "--members_per_cluster",
        type=int,
        default=1,
        help="For member rollout visualization, draw this many best-scoring members from each selected cluster.",
    )
    parser.add_argument(
        "--endpoint_constraint_mode",
        choices=["off", "scan_clusters"],
        default="scan_clusters",
        help=(
            "When using member rollouts, require candidate endpoints to fall inside "
            "a GT-endpoint-yaw box. scan_clusters scans clusters by dynamic distance "
            "until top_k feasible clusters are found."
        ),
    )
    parser.add_argument("--endpoint_constraint_lateral_m", type=float, default=2.0)
    parser.add_argument("--endpoint_constraint_longitudinal_m", type=float, default=4.0)
    parser.add_argument(
        "--endpoint_constraint_short_longitudinal_m",
        type=float,
        default=5.0,
        help=(
            "Allowed negative longitudinal endpoint error in the GT-endpoint-yaw "
            "frame. With the default positive limit this gives (-5m, +4m)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def smoothing_pass_value(
    explicit_value: int | None,
    legacy_value: int | None,
    default_value: int,
) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    if legacy_value is not None:
        return int(legacy_value)
    return int(default_value)


def resolve_smoothing_config(args: argparse.Namespace) -> SmoothingConfig:
    return SmoothingConfig(
        speed_passes=smoothing_pass_value(args.speed_smooth_passes, args.smooth_passes, 1),
        acceleration_passes=smoothing_pass_value(args.acceleration_smooth_passes, args.smooth_passes, 2),
        curvature_passes=smoothing_pass_value(args.curvature_smooth_passes, args.smooth_passes, 1),
    )


def load_future_rows(path: Path, steps: int) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5 + steps * 2:
                raise ValueError(f"Bad row with {len(parts)} columns: {line[:120]}")
            rows.append(
                TrajectoryRow(
                    traj_id=int(parts[0]),
                    dataset=parts[1],
                    clip=parts[2],
                    t0_us=int(parts[3]),
                    t0_index=int(parts[4]),
                    xy=np.asarray(parts[5:], dtype=np.float64).reshape(steps, 2),
                )
            )
    if not rows:
        raise ValueError(f"No trajectories found in {path}")
    return rows


def filter_rows_by_dataset(rows: list[TrajectoryRow], dataset_filter: str | None) -> list[TrajectoryRow]:
    if dataset_filter is None or dataset_filter == "" or dataset_filter.lower() == "all":
        return rows
    filtered = [row for row in rows if row.dataset == dataset_filter]
    if not filtered:
        raise ValueError(f"No trajectories found for dataset_filter={dataset_filter!r}")
    return filtered


def select_sample_index(rows: list[TrajectoryRow], sample_row: int) -> int:
    row_by_traj_id = {row.traj_id: idx for idx, row in enumerate(rows)}
    if int(sample_row) in row_by_traj_id:
        return row_by_traj_id[int(sample_row)]
    if 0 <= int(sample_row) < len(rows):
        return int(sample_row)
    raise ValueError(
        "--sample_row must be either a traj_id in matched rows "
        f"or a filtered row index in [0, {len(rows) - 1}]"
    )


def dynamics_from_xy(xy: np.ndarray, dt: float = DT_SECONDS) -> tuple[np.ndarray, np.ndarray]:
    """Return scalar acceleration and curvature traces from ego-local xy."""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    points_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), points])
    delta = np.diff(points_from_origin, axis=0)
    step_distance = np.linalg.norm(delta, axis=1)
    speed = step_distance / dt

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

    acceleration = np.zeros_like(speed)
    if len(speed) > 1:
        acceleration[1:] = np.diff(speed) / dt

    curvature = np.zeros_like(speed)
    if len(speed) > 1:
        curvature[1:] = np.diff(yaw) / np.maximum(step_distance[1:], 1e-6)
    curvature[~np.isfinite(curvature)] = 0.0
    return acceleration, curvature


def robust_standardize(features: np.ndarray, clip_percentile: float) -> tuple[np.ndarray, dict]:
    clipped = np.asarray(features, dtype=np.float64).copy()
    clipped[~np.isfinite(clipped)] = 0.0
    clip_abs = np.nanpercentile(np.abs(clipped), float(clip_percentile), axis=0)
    clip_abs = np.maximum(clip_abs, 1e-6)
    clipped = np.clip(clipped, -clip_abs, clip_abs)
    mean = clipped.mean(axis=0)
    std = clipped.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    standardized = (clipped - mean) / std
    return standardized, {
        "clip_percentile": float(clip_percentile),
        "mean_abs_feature": float(np.mean(np.abs(features))),
        "mean_abs_clipped_feature": float(np.mean(np.abs(clipped))),
        "mean": mean,
        "std": std,
    }


def feature_stats_for_json(feature_stats: dict) -> dict:
    return {
        key: value for key, value in feature_stats.items()
        if key not in {"mean", "std"}
    }


def inverse_standardized_feature(feature: np.ndarray, feature_stats: dict) -> np.ndarray:
    return np.asarray(feature, dtype=np.float64) * feature_stats["std"] + feature_stats["mean"]


def smooth_trace(values: np.ndarray, passes: int) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if len(smoothed) < 5 or passes <= 0:
        smoothed[~np.isfinite(smoothed)] = 0.0
        return smoothed
    smoothed[~np.isfinite(smoothed)] = 0.0
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= kernel.sum()
    for _ in range(int(passes)):
        padded = np.pad(smoothed, (2, 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def build_acc_curvature_features(
    rows: list[TrajectoryRow],
    smooth_passes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accelerations = []
    curvatures = []
    raw_accelerations = []
    raw_curvatures = []
    for row in rows:
        acceleration, curvature = dynamics_from_xy(row.xy)
        raw_accelerations.append(acceleration)
        raw_curvatures.append(curvature)
        accelerations.append(smooth_trace(acceleration, smooth_passes))
        curvatures.append(smooth_trace(curvature, smooth_passes))
    acceleration_arr = np.stack(accelerations, axis=0)
    curvature_arr = np.stack(curvatures, axis=0)
    raw_acceleration_arr = np.stack(raw_accelerations, axis=0)
    raw_curvature_arr = np.stack(raw_curvatures, axis=0)
    features = np.concatenate([acceleration_arr, curvature_arr], axis=1)
    return features, acceleration_arr, curvature_arr, raw_acceleration_arr, raw_curvature_arr


def native_parquet_path(row: TrajectoryRow, train_data_root: Path) -> Path:
    return (
        train_data_root
        / row.dataset
        / "data-egomotion"
        / f"{row.clip}.egomotion.parquet"
    )


def native_t0_frame_index(df: pd.DataFrame, row: TrajectoryRow) -> int | None:
    if "timestamp" in df.columns:
        try:
            timestamps = df["timestamp"].to_numpy(dtype=np.int64)
        except (TypeError, ValueError):
            timestamps = np.asarray([], dtype=np.int64)

        if len(timestamps):
            target = int(row.t0_us)
            exact = np.flatnonzero(timestamps == target)
            timestamp_index: int | None = int(exact[0]) if len(exact) else None
            if timestamp_index is None:
                nearest = int(np.argmin(np.abs(timestamps - target)))
                tolerance_us = max(1, int(DT_SECONDS * 1_000_000 * 0.5))
                if abs(int(timestamps[nearest]) - target) <= tolerance_us:
                    timestamp_index = nearest

            if timestamp_index is not None:
                return int(timestamp_index)

    frame_index = int(row.t0_index)
    if 0 <= frame_index < len(df):
        return frame_index
    return None


def native_future_window_start(df: pd.DataFrame, row: TrajectoryRow, steps: int) -> int | None:
    def valid_start(start: int) -> bool:
        return 0 <= start and start + int(steps) <= len(df)

    frame_index = native_t0_frame_index(df, row)
    if frame_index is None:
        return None

    start = int(frame_index) + 1
    if valid_start(start):
        return start
    return None


def acceleration_from_smoothed_speed(
    speed: np.ndarray,
    smooth_passes: int | None = None,
    dt: float = DT_SECONDS,
    speed_smooth_passes: int | None = None,
    acceleration_smooth_passes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth scalar speed once, difference it, then smooth acceleration."""
    speed_passes = smoothing_pass_value(speed_smooth_passes, smooth_passes, 1)
    acceleration_passes = smoothing_pass_value(acceleration_smooth_passes, smooth_passes, 2)
    smoothed_speed = smooth_trace(speed, speed_passes)
    raw_acceleration = np.zeros_like(smoothed_speed)
    if len(smoothed_speed) > 1:
        raw_acceleration[1:] = np.diff(smoothed_speed) / dt
    acceleration = smooth_trace(raw_acceleration, acceleration_passes)
    return raw_acceleration, acceleration


def resolve_dynamics_smoothing_config(
    smooth_passes: int | None = None,
    speed_smooth_passes: int | None = None,
    acceleration_smooth_passes: int | None = None,
    curvature_smooth_passes: int | None = None,
) -> SmoothingConfig:
    return SmoothingConfig(
        speed_passes=smoothing_pass_value(speed_smooth_passes, smooth_passes, 1),
        acceleration_passes=smoothing_pass_value(
            acceleration_smooth_passes,
            smooth_passes,
            2,
        ),
        curvature_passes=smoothing_pass_value(curvature_smooth_passes, smooth_passes, 1),
    )


def build_native_acc_curvature_features(
    rows: list[TrajectoryRow],
    train_data_root: Path,
    steps: int,
    smooth_passes: int | None = None,
    curvature_min_speed_mps: float = 0.0,
    dt: float = DT_SECONDS,
    speed_smooth_passes: int | None = None,
    acceleration_smooth_passes: int | None = None,
    curvature_smooth_passes: int | None = None,
) -> tuple[
    list[TrajectoryRow],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int | float],
]:
    """Return position-derived acceleration plus native parquet curvature traces."""
    kept_rows: list[TrajectoryRow] = []
    accelerations = []
    curvatures = []
    raw_accelerations = []
    raw_curvatures = []
    speeds = []
    skipped_missing = 0
    skipped_bad = 0
    zeroed_curvature_points = 0
    smoothing = resolve_dynamics_smoothing_config(
        smooth_passes=smooth_passes,
        speed_smooth_passes=speed_smooth_passes,
        acceleration_smooth_passes=acceleration_smooth_passes,
        curvature_smooth_passes=curvature_smooth_passes,
    )

    for row in rows:
        if len(row.xy) != steps:
            skipped_bad += 1
            continue
        if not np.isfinite(row.xy).all():
            skipped_bad += 1
            continue

        parquet_path = native_parquet_path(row, train_data_root)
        if not parquet_path.exists():
            skipped_missing += 1
            continue
        try:
            df_native = pd.read_parquet(parquet_path, columns=["timestamp", "curvature"])
        except (FileNotFoundError, KeyError, ValueError, OSError):
            skipped_missing += 1
            continue
        start = native_future_window_start(df_native, row, steps)
        if start is None:
            skipped_bad += 1
            continue
        native_curvature = df_native["curvature"].iloc[start : start + int(steps)].to_numpy(dtype=np.float64)
        if len(native_curvature) != int(steps) or not np.isfinite(native_curvature).all():
            skipped_bad += 1
            continue

        speed, raw_acceleration, acceleration, _xy_raw_curvature, _xy_curvature = (
            smoothed_position_dynamics_from_xy(
                row.xy,
                dt=dt,
                speed_smooth_passes=smoothing.speed_passes,
                acceleration_smooth_passes=smoothing.acceleration_passes,
                curvature_smooth_passes=smoothing.curvature_passes,
            )
        )
        raw_curvature = native_curvature.copy()
        curvature = smooth_trace(raw_curvature, smoothing.curvature_passes)
        low_speed = speed < float(curvature_min_speed_mps)
        zeroed_curvature_points += int(low_speed.sum())
        raw_curvature[low_speed] = 0.0
        curvature[low_speed] = 0.0

        kept_rows.append(row)
        raw_accelerations.append(raw_acceleration)
        raw_curvatures.append(raw_curvature)
        accelerations.append(acceleration)
        curvatures.append(curvature)
        speeds.append(speed)

    if not kept_rows:
        raise ValueError(
            f"No position-derived feature rows found under {train_data_root}"
        )

    acceleration_arr = np.stack(accelerations, axis=0)
    curvature_arr = np.stack(curvatures, axis=0)
    raw_acceleration_arr = np.stack(raw_accelerations, axis=0)
    raw_curvature_arr = np.stack(raw_curvatures, axis=0)
    speed_arr = np.stack(speeds, axis=0)
    features = np.concatenate([acceleration_arr, curvature_arr], axis=1)
    stats = {
        "input_rows": len(rows),
        "kept_rows": len(kept_rows),
        "skipped_missing_parquet": skipped_missing,
        "skipped_bad_window": skipped_bad,
        "curvature_min_speed_mps": float(curvature_min_speed_mps),
        "zeroed_curvature_points": zeroed_curvature_points,
        "speed_smooth_passes": smoothing.speed_passes,
        "acceleration_smooth_passes": smoothing.acceleration_passes,
        "curvature_smooth_passes": smoothing.curvature_passes,
    }
    return (
        kept_rows,
        features,
        acceleration_arr,
        curvature_arr,
        raw_acceleration_arr,
        raw_curvature_arr,
        speed_arr,
        stats,
    )


def ego_xy_to_plot_xy(xy: np.ndarray) -> np.ndarray:
    """Plot ego-local x-forward/y-left as x-right/y-forward BEV axes."""
    out = np.empty_like(xy)
    out[..., 0] = -xy[..., 1]
    out[..., 1] = xy[..., 0]
    return out


def xy_to_xyz(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([xy[:, 0], xy[:, 1], np.zeros(len(xy), dtype=np.float64)])


def path_length_xy(xy: np.ndarray) -> float:
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    points_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), points])
    return float(np.linalg.norm(np.diff(points_from_origin, axis=0), axis=1).sum())


def speed_profile_from_xy(xy: np.ndarray, dt: float = DT_SECONDS) -> np.ndarray:
    """Return scalar speed per future step from ego-local xy positions."""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    points_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), points])
    step_distance = np.linalg.norm(np.diff(points_from_origin, axis=0), axis=1)
    speed = step_distance / float(dt)
    speed[~np.isfinite(speed)] = 0.0
    return speed


def speed_profile_from_xyz_positions(xyz: np.ndarray, dt: float = DT_SECONDS) -> np.ndarray:
    """Return scalar speed per frame from adjacent xyz positions."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(points) == 1:
        return np.zeros(1, dtype=np.float64)
    step_distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = np.concatenate([[step_distance[0]], step_distance]) / float(dt)
    speed[~np.isfinite(speed)] = 0.0
    return speed


def smoothed_position_dynamics_from_xy(
    xy: np.ndarray,
    dt: float = DT_SECONDS,
    speed_smooth_passes: int = 1,
    acceleration_smooth_passes: int = 2,
    curvature_smooth_passes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive speed, acceleration, and curvature from positions, smoothing in order."""
    raw_speed = speed_profile_from_xy(xy, dt)
    speed = smooth_trace(raw_speed, int(speed_smooth_passes))
    raw_acceleration = np.zeros_like(speed)
    if len(speed) > 1:
        raw_acceleration[1:] = np.diff(speed) / float(dt)
    acceleration = smooth_trace(raw_acceleration, int(acceleration_smooth_passes))

    _xy_acceleration, raw_curvature = dynamics_from_xy(xy, dt)
    curvature = smooth_trace(raw_curvature, int(curvature_smooth_passes))
    return speed, raw_acceleration, acceleration, raw_curvature, curvature


def gt_speed_profile_for_visualization(speed_profile: np.ndarray, smooth_passes: int) -> np.ndarray:
    """Return the GT speed profile shown in visualization, using the feature smoothing policy."""
    return smooth_trace(np.asarray(speed_profile, dtype=np.float64).reshape(-1), smooth_passes)


def standardize_with_stats(features: np.ndarray, feature_stats: dict) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - feature_stats["mean"]) / feature_stats["std"]


def trajectory_xy_distance(gt_xy: np.ndarray, candidate_xy: np.ndarray) -> float:
    """Return mean pointwise ego-local xy distance between two equal-step trajectories."""
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    candidate = np.asarray(candidate_xy, dtype=np.float64).reshape(-1, 2)
    steps = min(len(gt), len(candidate))
    if steps == 0:
        return float("inf")
    per_step = np.linalg.norm(gt[:steps] - candidate[:steps], axis=1)
    return float(np.mean(per_step))


def endpoint_yaw_unit_vectors(gt_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        raise ValueError("gt_xy must contain at least one point")
    if len(points) == 1:
        forward = points[0]
    else:
        deltas = np.diff(points, axis=0)
        distances = np.linalg.norm(deltas, axis=1)
        moving = np.flatnonzero(distances > 1e-6)
        forward = deltas[int(moving[-1])] if len(moving) else points[-1]
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-6:
        forward_unit = np.array([1.0, 0.0], dtype=np.float64)
    else:
        forward_unit = forward / norm
    left_unit = np.array([-forward_unit[1], forward_unit[0]], dtype=np.float64)
    return forward_unit, left_unit


def endpoint_longitudinal_bounds(
    max_longitudinal_m: float,
    max_short_longitudinal_m: float | None = None,
) -> tuple[float, float]:
    forward_limit = float(max_longitudinal_m)
    short_limit = forward_limit if max_short_longitudinal_m is None else float(max_short_longitudinal_m)
    return -abs(short_limit), forward_limit


def endpoint_constraint_metrics(
    gt_xy: np.ndarray,
    candidate_endpoint_xy: np.ndarray,
    max_lateral_m: float,
    max_longitudinal_m: float,
    max_short_longitudinal_m: float | None = None,
) -> dict[str, float | bool]:
    gt_points = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if len(gt_points) == 0:
        raise ValueError("gt_xy must contain at least one point")
    forward_unit, left_unit = endpoint_yaw_unit_vectors(gt_points)
    delta = np.asarray(candidate_endpoint_xy, dtype=np.float64).reshape(2) - gt_points[-1]
    longitudinal = float(np.dot(delta, forward_unit))
    lateral = float(np.dot(delta, left_unit))
    min_longitudinal_m, max_forward_longitudinal_m = endpoint_longitudinal_bounds(
        max_longitudinal_m,
        max_short_longitudinal_m,
    )
    return {
        "endpoint_longitudinal_error_m": longitudinal,
        "endpoint_lateral_error_m": lateral,
        "endpoint_abs_longitudinal_error_m": abs(longitudinal),
        "endpoint_abs_lateral_error_m": abs(lateral),
        "endpoint_distance_m": float(np.linalg.norm(delta)),
        "constraint_satisfied": (
            abs(lateral) <= float(max_lateral_m)
            and min_longitudinal_m <= longitudinal <= max_forward_longitudinal_m
        ),
    }


def candidate_match_label(
    rank: int,
    cluster_id: int,
    distance_label: str,
    distance: float,
    average_distance_m: float | None = None,
    endpoint_distance_m: float | None = None,
) -> str:
    label = f"Top{rank} cluster={cluster_id} {distance_label}={distance:.2f}"
    if average_distance_m is not None:
        label += f" avg={average_distance_m:.1f}m"
    if endpoint_distance_m is not None:
        label += f" end={endpoint_distance_m:.1f}m"
    return label


def select_top_rollouts_by_xy_distance(
    gt_xy: np.ndarray,
    rollouts: list[np.ndarray],
    dynamic_distances: list[float] | np.ndarray,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """Select rollout positions by xy distance, using dynamic distance as a tie-break."""
    xy_distances = [trajectory_xy_distance(gt_xy, rollout) for rollout in rollouts]
    dyn = np.asarray(dynamic_distances, dtype=np.float64).reshape(-1)
    order = sorted(
        range(len(rollouts)),
        key=lambda idx: (
            xy_distances[idx],
            float(dyn[idx]) if idx < len(dyn) and np.isfinite(dyn[idx]) else float("inf"),
        ),
    )
    chosen = order[: max(1, min(int(top_k), len(order)))]
    return chosen, [float(xy_distances[idx]) for idx in chosen]


def candidate_trajectories_for_representation(
    representation: str,
    center_rollouts: list[np.ndarray],
    representative_rows: list[TrajectoryRow],
    member_rollouts: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    if representation in {"medoid_xy", "gt_nearest_member_xy"}:
        return [np.asarray(row.xy, dtype=np.float64) for row in representative_rows]
    if representation == "acc_curvature_nearest_member_rollout":
        if member_rollouts is None:
            raise ValueError("member_rollouts is required for acc_curvature_nearest_member_rollout")
        return [np.asarray(rollout, dtype=np.float64) for rollout in member_rollouts]
    return [np.asarray(rollout, dtype=np.float64) for rollout in center_rollouts]


def rank_color_bgr(rank: int, total: int) -> tuple[int, int, int]:
    """Return a distinct BGR color for a one-based candidate rank."""
    if total <= 2:
        palette = [(237, 128, 47), (87, 87, 235)]
        return palette[(rank - 1) % len(palette)]
    hue = ((rank - 1) / max(1, total)) % 1.0
    rgb_float = plt.cm.hsv(hue)[:3]
    rgb = tuple(int(round(channel * 255.0)) for channel in rgb_float)
    return (rgb[2], rgb[1], rgb[0])


def rank_color_hex(rank: int, total: int) -> str:
    bgr = rank_color_bgr(rank, total)
    return f"#{bgr[2]:02x}{bgr[1]:02x}{bgr[0]:02x}"


def integrate_acc_curvature(
    acceleration: np.ndarray,
    curvature: np.ndarray,
    initial_speed_mps: float,
    dt: float = DT_SECONDS,
    max_speed_mps: float = 15.0,
) -> np.ndarray:
    """Integrate an acceleration/curvature profile into ego-local xy."""
    acceleration = np.asarray(acceleration, dtype=np.float64).reshape(-1)
    curvature = np.asarray(curvature, dtype=np.float64).reshape(-1)
    steps = min(len(acceleration), len(curvature))
    xy = np.zeros((steps, 2), dtype=np.float64)
    speed = np.zeros(steps, dtype=np.float64)
    yaw = np.zeros(steps, dtype=np.float64)
    pos = np.zeros(2, dtype=np.float64)
    current_speed = float(np.clip(initial_speed_mps, 0.0, max_speed_mps))
    current_yaw = 0.0

    for idx in range(steps):
        if idx > 0:
            current_speed = float(
                np.clip(current_speed + float(acceleration[idx]) * dt, 0.0, max_speed_mps)
            )
        ds = current_speed * dt
        if idx > 0:
            current_yaw += float(curvature[idx]) * ds
        pos = pos + np.array(
            [np.cos(current_yaw) * ds, np.sin(current_yaw) * ds],
            dtype=np.float64,
        )
        speed[idx] = current_speed
        yaw[idx] = current_yaw
        xy[idx] = pos
    return xy


def integrate_acc_curvature_endpoints_batch(
    acceleration: np.ndarray,
    curvature: np.ndarray,
    initial_speed_mps: float,
    dt: float = DT_SECONDS,
    max_speed_mps: float = 15.0,
) -> np.ndarray:
    """Vectorized endpoint-only rollout for many acceleration/curvature profiles."""
    acceleration = np.asarray(acceleration, dtype=np.float64)
    curvature = np.asarray(curvature, dtype=np.float64)
    if acceleration.ndim == 1:
        acceleration = acceleration[None, :]
    if curvature.ndim == 1:
        curvature = curvature[None, :]
    steps = min(acceleration.shape[1], curvature.shape[1])
    n = acceleration.shape[0]
    pos = np.zeros((n, 2), dtype=np.float64)
    speed = np.full(n, float(np.clip(initial_speed_mps, 0.0, max_speed_mps)), dtype=np.float64)
    yaw = np.zeros(n, dtype=np.float64)
    for idx in range(steps):
        if idx > 0:
            speed = np.clip(speed + acceleration[:, idx] * dt, 0.0, max_speed_mps)
        ds = speed * dt
        if idx > 0:
            yaw = yaw + curvature[:, idx] * ds
        pos[:, 0] += np.cos(yaw) * ds
        pos[:, 1] += np.sin(yaw) * ds
    return pos


def parquet_candidates(row: TrajectoryRow, train_data_root: Path, generate_output_root: Path) -> list[Path]:
    return [
        train_data_root / row.dataset / "data-egomotion" / f"{row.clip}.egomotion.parquet",
        train_data_root / row.dataset / f"{row.clip}.egomotion.parquet",
        generate_output_root / row.dataset / f"{row.clip}.egomotion.parquet",
    ]


def fallback_t0_speed_from_xy(row: TrajectoryRow, steps: int = 5) -> float:
    points_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), row.xy])
    step_distance = np.linalg.norm(np.diff(points_from_origin, axis=0), axis=1)
    if len(step_distance) == 0:
        return 0.0
    return float(np.nanmean(step_distance[: max(1, min(steps, len(step_distance)))]) / DT_SECONDS)


def speed_from_array_parquet_row(df: pd.DataFrame, row: TrajectoryRow, speed_window: int) -> float | None:
    """Read future-array position rows used by generate_traj_data/output."""
    if not {"x", "y"}.issubset(df.columns):
        return None
    selected = df
    if "t0_us" in df.columns:
        selected = df[df["t0_us"].astype("int64") == int(row.t0_us)]
    if selected.empty and "sample_idx" in df.columns:
        selected = df[df["sample_idx"].astype("int64") == int(row.t0_index)]
    if selected.empty:
        return None

    first = selected.iloc[0]
    if not isinstance(first["x"], (list, tuple, np.ndarray)):
        return None
    try:
        x = np.asarray(first["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(first["y"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(x) == 0 or len(y) != len(x):
        return None
    speed = smooth_trace(speed_profile_from_xy(np.column_stack([x, y])), passes=1)
    n = max(1, min(len(speed), int(speed_window) + 1))
    if not np.isfinite(speed[:n]).any():
        return None
    return float(np.nanmean(speed[:n]))


def speed_profile_from_array_parquet_row(df: pd.DataFrame, row: TrajectoryRow, steps: int) -> np.ndarray | None:
    """Read a future-array position speed profile row from generate_traj_data/output."""
    if not {"x", "y"}.issubset(df.columns):
        return None
    selected = df
    if "t0_us" in df.columns:
        selected = df[df["t0_us"].astype("int64") == int(row.t0_us)]
    if selected.empty and "sample_idx" in df.columns:
        selected = df[df["sample_idx"].astype("int64") == int(row.t0_index)]
    if selected.empty:
        return None

    first = selected.iloc[0]
    if not isinstance(first["x"], (list, tuple, np.ndarray)):
        return None
    try:
        x = np.asarray(first["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(first["y"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(x) < steps or len(y) != len(x):
        return None
    speed = smooth_trace(speed_profile_from_xy(np.column_stack([x[:steps], y[:steps]])), passes=1)
    if len(speed) != steps or not np.isfinite(speed).all():
        return None
    return speed


def has_array_velocity_columns(df: pd.DataFrame) -> bool:
    if not {"vx", "vy", "vz"}.issubset(df.columns) or df.empty:
        return False
    first = df.iloc[0]["vx"]
    return isinstance(first, (list, tuple, np.ndarray))


def t0_average_speed(
    row: TrajectoryRow,
    train_data_root: Path,
    generate_output_root: Path,
    speed_window: int,
    cache: dict[Path, object],
) -> float:
    parquet_path = next(
        (candidate for candidate in parquet_candidates(row, train_data_root, generate_output_root) if candidate.exists()),
        None,
    )
    if parquet_path is None:
        return fallback_t0_speed_from_xy(row)

    cached = cache.get(parquet_path)
    df: pd.DataFrame | None = None
    if isinstance(cached, pd.DataFrame):
        array_speed = speed_from_array_parquet_row(cached, row, speed_window)
        if array_speed is not None:
            return array_speed
        if has_array_velocity_columns(cached):
            return fallback_t0_speed_from_xy(row)
        df = cached
    elif isinstance(cached, np.ndarray):
        speed = cached
        center = int(row.t0_index)
    else:
        df = pd.read_parquet(parquet_path)
        array_speed = speed_from_array_parquet_row(df, row, speed_window)
        if array_speed is not None:
            cache[parquet_path] = df
            return array_speed
        if has_array_velocity_columns(df):
            cache[parquet_path] = df
            return fallback_t0_speed_from_xy(row)

    if df is not None:
        if not {"x", "y", "z"}.issubset(df.columns):
            return fallback_t0_speed_from_xy(row)
        speed = smooth_trace(
            speed_profile_from_xyz_positions(df[["x", "y", "z"]].to_numpy(dtype=np.float64)),
            passes=1,
        )
        center = native_t0_frame_index(df, row)
        cache[parquet_path] = df
        if center is None:
            return fallback_t0_speed_from_xy(row)
    start = max(0, center - int(speed_window))
    end = min(len(speed), center + int(speed_window) + 1)
    if start >= end:
        return fallback_t0_speed_from_xy(row)
    values = speed[start:end]
    if len(values) == 0 or not np.isfinite(values).any():
        return fallback_t0_speed_from_xy(row)
    return float(np.nanmean(values))


def gt_future_speed_profile(
    row: TrajectoryRow,
    train_data_root: Path,
    generate_output_root: Path,
    steps: int,
    cache: dict[Path, object],
) -> np.ndarray:
    """Read the GT future scalar-speed profile, falling back to xy-derived speed."""
    fallback = speed_profile_from_xy(row.xy)
    parquet_path = next(
        (candidate for candidate in parquet_candidates(row, train_data_root, generate_output_root) if candidate.exists()),
        None,
    )
    if parquet_path is None:
        return fallback

    cached = cache.get(parquet_path)
    df: pd.DataFrame | None = None
    if isinstance(cached, pd.DataFrame):
        array_speed = speed_profile_from_array_parquet_row(cached, row, steps)
        if array_speed is not None:
            return array_speed
        if has_array_velocity_columns(cached):
            return fallback
        df = cached
    elif isinstance(cached, np.ndarray):
        speed = cached
        start = int(row.t0_index) + 1
    else:
        df = pd.read_parquet(parquet_path)
        array_speed = speed_profile_from_array_parquet_row(df, row, steps)
        if array_speed is not None:
            cache[parquet_path] = df
            return array_speed
        if has_array_velocity_columns(df):
            cache[parquet_path] = df
            return fallback

    if df is not None:
        if not {"x", "y", "z"}.issubset(df.columns):
            return fallback
        speed = smooth_trace(
            speed_profile_from_xyz_positions(df[["x", "y", "z"]].to_numpy(dtype=np.float64)),
            passes=1,
        )
        start = native_future_window_start(df, row, steps)
        cache[parquet_path] = df
        if start is None:
            return fallback

    end = start + int(steps)
    if start < 0 or end > len(speed):
        return fallback
    profile = np.asarray(speed[start:end], dtype=np.float64)
    if len(profile) != steps or not np.isfinite(profile).all():
        return fallback
    return profile


def choose_medoids(
    standardized_features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    medoid_indices = np.full(n_clusters, -1, dtype=np.int64)
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        if len(members) == 0:
            continue
        distances = np.linalg.norm(
            standardized_features[members] - centers[cluster_id],
            axis=1,
        )
        medoid_indices[cluster_id] = int(members[int(np.argmin(distances))])
    return medoid_indices


def choose_gt_nearest_member_indices(
    rows: list[TrajectoryRow],
    labels: np.ndarray,
    chosen_clusters: list[int],
    gt_xy: np.ndarray,
) -> list[int]:
    """Return the member in each chosen cluster with the smallest xy distance to GT."""
    nearest_indices: list[int] = []
    for cluster_id in chosen_clusters:
        members = np.flatnonzero(labels == int(cluster_id))
        if len(members) == 0:
            raise ValueError(f"Selected cluster {cluster_id} has no members")
        distances = [
            trajectory_xy_distance(gt_xy, rows[int(member_idx)].xy)
            for member_idx in members
        ]
        nearest_indices.append(int(members[int(np.argmin(distances))]))
    return nearest_indices


def choose_feature_nearest_member_indices(
    features: np.ndarray,
    labels: np.ndarray,
    chosen_clusters: list[int],
    gt_feature: np.ndarray,
) -> list[int]:
    """Return the member in each chosen cluster nearest to GT in feature space."""
    features = np.asarray(features, dtype=np.float64)
    gt_feature = np.asarray(gt_feature, dtype=np.float64).reshape(-1)
    nearest_indices: list[int] = []
    for cluster_id in chosen_clusters:
        members = np.flatnonzero(labels == int(cluster_id))
        if len(members) == 0:
            raise ValueError(f"Selected cluster {cluster_id} has no members")
        distances = np.linalg.norm(features[members] - gt_feature, axis=1)
        nearest_indices.append(int(members[int(np.argmin(distances))]))
    return nearest_indices


def choose_feature_nearest_member_indices_with_endpoint(
    features: np.ndarray,
    labels: np.ndarray,
    chosen_clusters: list[int],
    gt_feature: np.ndarray,
    rows: list[TrajectoryRow],
    gt_xy: np.ndarray,
    endpoint_weight: float,
    endpoint_xy_by_row: np.ndarray | None = None,
    members_per_cluster: int = 1,
) -> tuple[list[int], list[dict[str, float | int]]]:
    """Choose feature-nearest members with a small GT endpoint penalty."""
    features = np.asarray(features, dtype=np.float64)
    gt_feature = np.asarray(gt_feature, dtype=np.float64).reshape(-1)
    gt_points = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    if len(gt_points) == 0:
        raise ValueError("gt_xy must contain at least one point")
    gt_endpoint = gt_points[-1]
    endpoint_xy = None
    if endpoint_xy_by_row is not None:
        endpoint_xy = np.asarray(endpoint_xy_by_row, dtype=np.float64).reshape(len(rows), 2)
    weight = float(endpoint_weight)

    nearest_indices: list[int] = []
    details: list[dict[str, float | int]] = []
    for cluster_id in chosen_clusters:
        members = np.flatnonzero(labels == int(cluster_id))
        if len(members) == 0:
            raise ValueError(f"Selected cluster {cluster_id} has no members")
        feature_distances = np.linalg.norm(features[members] - gt_feature, axis=1)
        endpoint_distances = np.asarray(
            [
                np.linalg.norm(
                    (
                        endpoint_xy[int(member_idx)]
                        if endpoint_xy is not None
                        else np.asarray(rows[int(member_idx)].xy, dtype=np.float64).reshape(-1, 2)[-1]
                    )
                    - gt_endpoint
                )
                for member_idx in members
            ],
            dtype=np.float64,
        )
        scores = feature_distances + weight * endpoint_distances
        n_members = max(1, min(int(members_per_cluster), len(members)))
        for member_rank, position in enumerate(np.argsort(scores)[:n_members], start=1):
            position = int(position)
            member_index = int(members[position])
            nearest_indices.append(member_index)
            details.append(
                {
                    "cluster": int(cluster_id),
                    "member_index": member_index,
                    "member_rank_in_cluster": int(member_rank),
                    "feature_distance": float(feature_distances[position]),
                    "endpoint_distance_m": float(endpoint_distances[position]),
                    "endpoint_weight": weight,
                    "selection_score": float(scores[position]),
                }
            )
    return nearest_indices, details


def choose_endpoint_constrained_feature_members(
    features: np.ndarray,
    labels: np.ndarray,
    chosen_clusters: list[int],
    gt_feature: np.ndarray,
    rows: list[TrajectoryRow],
    gt_xy: np.ndarray,
    endpoint_xy_by_row: np.ndarray,
    max_lateral_m: float,
    max_longitudinal_m: float,
    max_short_longitudinal_m: float | None = None,
    members_per_cluster: int = 1,
) -> tuple[list[int], list[dict[str, float | int | bool]]]:
    """Choose feature-nearest members that satisfy the GT-endpoint-yaw box."""
    features = np.asarray(features, dtype=np.float64)
    gt_feature = np.asarray(gt_feature, dtype=np.float64).reshape(-1)
    endpoint_xy = np.asarray(endpoint_xy_by_row, dtype=np.float64).reshape(len(rows), 2)

    selected: list[int] = []
    details: list[dict[str, float | int | bool]] = []
    for cluster_id in chosen_clusters:
        members = np.flatnonzero(labels == int(cluster_id))
        if len(members) == 0:
            raise ValueError(f"Selected cluster {cluster_id} has no members")
        feature_distances = np.linalg.norm(features[members] - gt_feature, axis=1)
        feasible: list[tuple[float, int, dict[str, float | bool]]] = []
        for pos, member_idx in enumerate(members):
            metrics = endpoint_constraint_metrics(
                gt_xy,
                endpoint_xy[int(member_idx)],
                max_lateral_m=max_lateral_m,
                max_longitudinal_m=max_longitudinal_m,
                max_short_longitudinal_m=max_short_longitudinal_m,
            )
            if bool(metrics["constraint_satisfied"]):
                feasible.append((float(feature_distances[pos]), int(pos), metrics))
        feasible.sort(key=lambda item: item[0])
        n_members = max(1, min(int(members_per_cluster), len(feasible)))
        for member_rank, (feature_distance, pos, metrics) in enumerate(feasible[:n_members], start=1):
            member_index = int(members[pos])
            selected.append(member_index)
            details.append(
                {
                    "cluster": int(cluster_id),
                    "member_index": member_index,
                    "member_rank_in_cluster": int(member_rank),
                    "feature_distance": feature_distance,
                    "endpoint_constraint_satisfied": True,
                    **metrics,
                }
            )
    return selected, details


def build_cluster_member_indices(labels: np.ndarray, n_clusters: int) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    return [np.flatnonzero(labels == cluster_id) for cluster_id in range(int(n_clusters))]


def select_endpoint_first_top_cluster_members_fast(
    features: np.ndarray,
    cluster_member_indices: list[np.ndarray],
    valid_clusters: np.ndarray,
    dynamic_distances: np.ndarray,
    gt_feature: np.ndarray,
    gt_xy: np.ndarray,
    member_endpoint_xy: np.ndarray,
    max_lateral_m: float,
    max_longitudinal_m: float,
    max_short_longitudinal_m: float | None = None,
    top_clusters: int = 2,
    members_per_cluster: int = 2,
) -> tuple[list[int], list[dict[str, float | int | bool]], list[int], list[float]]:
    """Endpoint-feasible top-cluster member selection using vectorized per-cluster math."""
    features = np.asarray(features, dtype=np.float64)
    gt_feature = np.asarray(gt_feature, dtype=np.float64).reshape(-1)
    endpoint_xy = np.asarray(member_endpoint_xy, dtype=np.float64)
    gt_points = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    gt_endpoint = gt_points[-1]
    forward_unit, left_unit = endpoint_yaw_unit_vectors(gt_points)
    min_longitudinal_m, max_forward_longitudinal_m = endpoint_longitudinal_bounds(
        max_longitudinal_m,
        max_short_longitudinal_m,
    )

    selected_indices: list[int] = []
    selected_details: list[dict[str, float | int | bool]] = []
    selected_clusters: list[int] = []
    selected_distances: list[float] = []
    ordered_positions = [int(pos) for pos in np.argsort(np.asarray(dynamic_distances, dtype=np.float64))]

    for pos in ordered_positions:
        cluster_id = int(valid_clusters[pos])
        if cluster_id < 0 or cluster_id >= len(cluster_member_indices):
            continue
        members = np.asarray(cluster_member_indices[cluster_id], dtype=np.int64)
        if len(members) == 0:
            continue
        delta = endpoint_xy[members] - gt_endpoint[None, :]
        longitudinal = delta @ forward_unit
        lateral = delta @ left_unit
        feasible_mask = (
            (np.abs(lateral) <= float(max_lateral_m))
            & (longitudinal >= min_longitudinal_m)
            & (longitudinal <= max_forward_longitudinal_m)
        )
        feasible_members = members[feasible_mask]
        if len(feasible_members) == 0:
            continue

        feature_distances = np.linalg.norm(features[feasible_members] - gt_feature[None, :], axis=1)
        order = np.argsort(feature_distances)[: max(1, min(int(members_per_cluster), len(feasible_members)))]
        selected_clusters.append(cluster_id)
        selected_distances.append(float(dynamic_distances[pos]))
        feasible_positions = np.flatnonzero(feasible_mask)
        for rank, local_pos in enumerate(order, start=1):
            member_index = int(feasible_members[int(local_pos)])
            original_pos = int(feasible_positions[int(local_pos)])
            endpoint_delta = delta[original_pos]
            endpoint_distance = float(np.linalg.norm(endpoint_delta))
            selected_indices.append(member_index)
            selected_details.append(
                {
                    "cluster": cluster_id,
                    "member_index": member_index,
                    "member_rank_in_cluster": int(rank),
                    "feature_distance": float(feature_distances[int(local_pos)]),
                    "endpoint_constraint_satisfied": True,
                    "endpoint_longitudinal_error_m": float(longitudinal[original_pos]),
                    "endpoint_lateral_error_m": float(lateral[original_pos]),
                    "endpoint_abs_longitudinal_error_m": float(abs(longitudinal[original_pos])),
                    "endpoint_abs_lateral_error_m": float(abs(lateral[original_pos])),
                    "endpoint_distance_m": endpoint_distance,
                    "constraint_satisfied": True,
                }
            )
        if len(selected_clusters) >= int(top_clusters):
            break

    return selected_indices, selected_details, selected_clusters, selected_distances


def select_top_clusters_then_endpoint_members_fast(
    features: np.ndarray,
    cluster_member_indices: list[np.ndarray],
    valid_clusters: np.ndarray,
    dynamic_distances: np.ndarray,
    gt_feature: np.ndarray,
    gt_xy: np.ndarray,
    member_endpoint_xy: np.ndarray,
    max_lateral_m: float,
    max_longitudinal_m: float,
    max_short_longitudinal_m: float | None = None,
    endpoint_weight: float = 0.15,
    top_clusters: int = 2,
    members_per_cluster: int = 2,
) -> tuple[list[int], list[dict[str, float | int | bool]], list[int], list[float]]:
    """Pick dynamic top clusters first, then endpoint-feasible feature-nearest members."""
    features = np.asarray(features, dtype=np.float64)
    gt_feature = np.asarray(gt_feature, dtype=np.float64).reshape(-1)
    endpoint_xy = np.asarray(member_endpoint_xy, dtype=np.float64)
    gt_points = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    gt_endpoint = gt_points[-1]
    forward_unit, left_unit = endpoint_yaw_unit_vectors(gt_points)
    min_longitudinal_m, max_forward_longitudinal_m = endpoint_longitudinal_bounds(
        max_longitudinal_m,
        max_short_longitudinal_m,
    )

    ordered_positions = [int(pos) for pos in np.argsort(np.asarray(dynamic_distances, dtype=np.float64))]
    top_positions = ordered_positions[: max(1, min(int(top_clusters), len(ordered_positions)))]
    selected_clusters = [int(valid_clusters[pos]) for pos in top_positions]
    selected_distances = [float(dynamic_distances[pos]) for pos in top_positions]

    selected_indices: list[int] = []
    selected_details: list[dict[str, float | int | bool]] = []
    for cluster_id in selected_clusters:
        if cluster_id < 0 or cluster_id >= len(cluster_member_indices):
            continue
        members = np.asarray(cluster_member_indices[cluster_id], dtype=np.int64)
        if len(members) == 0:
            continue

        delta = endpoint_xy[members] - gt_endpoint[None, :]
        longitudinal = delta @ forward_unit
        lateral = delta @ left_unit
        feasible_mask = (
            (np.abs(lateral) <= float(max_lateral_m))
            & (longitudinal >= min_longitudinal_m)
            & (longitudinal <= max_forward_longitudinal_m)
        )
        feasible_members = members[feasible_mask]
        if len(feasible_members) == 0:
            continue

        feasible_positions = np.flatnonzero(feasible_mask)
        feature_distances = np.linalg.norm(features[feasible_members] - gt_feature[None, :], axis=1)
        endpoint_distances = np.linalg.norm(delta[feasible_positions], axis=1)
        scores = feature_distances + float(endpoint_weight) * endpoint_distances
        order = np.argsort(scores)[: max(1, min(int(members_per_cluster), len(feasible_members)))]
        for rank, local_pos in enumerate(order, start=1):
            member_index = int(feasible_members[int(local_pos)])
            original_pos = int(feasible_positions[int(local_pos)])
            endpoint_delta = delta[original_pos]
            selected_indices.append(member_index)
            selected_details.append(
                {
                    "cluster": cluster_id,
                    "member_index": member_index,
                    "member_rank_in_cluster": int(rank),
                    "feature_distance": float(feature_distances[int(local_pos)]),
                    "endpoint_weight": float(endpoint_weight),
                    "selection_score": float(scores[int(local_pos)]),
                    "endpoint_constraint_satisfied": True,
                    "endpoint_longitudinal_error_m": float(longitudinal[original_pos]),
                    "endpoint_lateral_error_m": float(lateral[original_pos]),
                    "endpoint_abs_longitudinal_error_m": float(abs(longitudinal[original_pos])),
                    "endpoint_abs_lateral_error_m": float(abs(lateral[original_pos])),
                    "endpoint_distance_m": float(np.linalg.norm(endpoint_delta)),
                    "constraint_satisfied": True,
                }
            )

    return selected_indices, selected_details, selected_clusters, selected_distances


def select_endpoint_first_top_cluster_members(
    features: np.ndarray,
    labels: np.ndarray,
    valid_clusters: np.ndarray,
    dynamic_distances: np.ndarray,
    gt_feature: np.ndarray,
    rows: list[TrajectoryRow],
    gt_xy: np.ndarray,
    member_endpoint_xy: np.ndarray,
    fill_member_rollout_endpoints,
    max_lateral_m: float,
    max_longitudinal_m: float,
    max_short_longitudinal_m: float | None = None,
    top_clusters: int = 2,
    members_per_cluster: int = 2,
) -> tuple[list[int], list[dict[str, float | int | bool]], list[int], list[float]]:
    """Filter by endpoint feasibility first, then keep top clusters by dynamic distance."""
    selected_indices: list[int] = []
    selected_details: list[dict[str, float | int | bool]] = []
    selected_clusters: list[int] = []
    selected_distances: list[float] = []
    ordered_positions = [int(pos) for pos in np.argsort(np.asarray(dynamic_distances, dtype=np.float64))]

    for pos in ordered_positions:
        cluster_id = int(valid_clusters[pos])
        fill_member_rollout_endpoints(cluster_id)
        cluster_indices, cluster_details = choose_endpoint_constrained_feature_members(
            features,
            labels,
            [cluster_id],
            gt_feature,
            rows,
            gt_xy,
            member_endpoint_xy,
            max_lateral_m=max_lateral_m,
            max_longitudinal_m=max_longitudinal_m,
            max_short_longitudinal_m=max_short_longitudinal_m,
            members_per_cluster=members_per_cluster,
        )
        if not cluster_indices:
            continue
        selected_clusters.append(cluster_id)
        selected_distances.append(float(dynamic_distances[pos]))
        selected_indices.extend(cluster_indices)
        selected_details.extend(cluster_details)
        if len(selected_clusters) >= int(top_clusters):
            break

    return selected_indices, selected_details, selected_clusters, selected_distances


def draw_match(
    output_path: Path,
    gt_row: TrajectoryRow,
    gt_speed: float,
    center_trajs: list[np.ndarray],
    center_distances: list[float],
    center_clusters: list[int],
    distance_label: str = "dyn_dist",
    average_distances: list[float] | None = None,
    endpoint_distances: list[float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    gt_plot = ego_xy_to_plot_xy(gt_row.xy)
    ax.plot(gt_plot[:, 0], gt_plot[:, 1], color="#ffd84d", linewidth=3.0, label=f"GT speed={gt_speed:.2f}m/s")
    ax.scatter(gt_plot[0, 0], gt_plot[0, 1], color="#ffd84d", edgecolor="black", s=45, zorder=4)
    ax.scatter(gt_plot[-1, 0], gt_plot[-1, 1], color="#ffd84d", marker="x", s=65, zorder=4)

    for rank, (traj_xy, distance, cluster_id) in enumerate(zip(center_trajs, center_distances, center_clusters), start=1):
        plot_xy = ego_xy_to_plot_xy(traj_xy)
        color = rank_color_hex(rank, len(center_trajs))
        avg_distance = (
            None
            if average_distances is None or rank - 1 >= len(average_distances)
            else average_distances[rank - 1]
        )
        endpoint_distance = (
            None
            if endpoint_distances is None or rank - 1 >= len(endpoint_distances)
            else endpoint_distances[rank - 1]
        )
        ax.plot(
            plot_xy[:, 0],
            plot_xy[:, 1],
            color=color,
            linewidth=2.2 if len(center_trajs) <= 2 else 1.4,
            linestyle="--",
            alpha=1.0 if len(center_trajs) <= 2 else 0.72,
            label=candidate_match_label(
                rank,
                cluster_id,
                distance_label,
                distance,
                average_distance_m=avg_distance,
                endpoint_distance_m=endpoint_distance,
            ),
        )
        ax.scatter(plot_xy[0, 0], plot_xy[0, 1], color=color, edgecolor="black", s=28, zorder=4)
        ax.scatter(plot_xy[-1, 0], plot_xy[-1, 1], color=color, marker="x", s=42, zorder=4)

    ax.scatter([0.0], [0.0], color="black", s=30, label="ego t0", zorder=5)
    title_metric = "XY Rollout Distance" if distance_label == "xy_dist" else "Dynamic Distance"
    ax.set_title(f"Acceleration+Curvature KMeans Top{len(center_trajs)} by {title_metric}\nGT traj_id={gt_row.traj_id}, rollout speed={gt_speed:.2f}m/s")
    ax.set_xlabel("right (m)")
    ax.set_ylabel("forward (m)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.axis("equal")
    if len(center_trajs) <= 8:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def draw_speed_profiles(
    output_path: Path,
    gt_speed_profile: np.ndarray,
    candidate_trajs: list[np.ndarray],
    center_clusters: list[int],
) -> None:
    """Draw GT and selected candidate scalar-speed curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_speed_profile = np.asarray(gt_speed_profile, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))

    frames = np.arange(len(gt_speed_profile), dtype=np.int64)
    ax.plot(
        frames,
        gt_speed_profile,
        color="#ffd84d",
        linewidth=2.8,
        label="GT speed",
    )
    for rank, (traj_xy, cluster_id) in enumerate(zip(candidate_trajs, center_clusters), start=1):
        speed = speed_profile_from_xy(traj_xy)
        candidate_frames = np.arange(len(speed), dtype=np.int64)
        color = rank_color_hex(rank, len(candidate_trajs))
        ax.plot(
            candidate_frames,
            speed,
            color=color,
            linewidth=2.2 if len(candidate_trajs) <= 2 else 1.5,
            linestyle="--",
            alpha=1.0 if len(candidate_trajs) <= 2 else 0.78,
            label=f"Top{rank} cluster={cluster_id}",
        )

    ax.set_title("GT and Top Candidate Speed Profiles")
    ax.set_xlabel("future frame index")
    ax.set_ylabel("speed (m/s)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_fc_frame(
    source_data_root: Path,
    row: TrajectoryRow,
) -> tuple[np.ndarray, int, int] | None:
    video_path = (
        source_data_root
        / row.dataset
        / "mp4-converted"
        / f"{row.clip}_fovs_FC.mp4"
    )
    timestamps_path = (
        source_data_root
        / row.dataset
        / "data-timestamps"
        / f"{row.clip}_fovs_FC.timestamps.parquet"
    )
    if not video_path.exists() or not timestamps_path.exists():
        return None

    df_ts = pd.read_parquet(timestamps_path)
    if "timestamp" not in df_ts.columns:
        return None
    timestamps = df_ts["timestamp"].to_numpy(dtype=np.int64)
    nearest = int(np.abs(timestamps - int(row.t0_us)).argmin())
    if "frame_index" in df_ts.columns:
        frame_idx = int(df_ts.iloc[nearest]["frame_index"])
    else:
        frame_idx = nearest

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        return None
    return frame_bgr, frame_idx, int(timestamps[nearest])


def project_ego_to_fc_pixels(
    xyz: np.ndarray,
    image_shape: tuple[int, int],
    fx: float = 1000.0,
    fy: float = 1000.0,
    camera_height_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate FC projection using the GUI fallback ego-to-camera transform."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    height, width = image_shape
    cx = width * 0.5
    cy = height * 0.5
    x_cam = -points[:, 1]
    y_cam = -(points[:, 2] - camera_height_m)
    z_cam = points[:, 0]
    valid = z_cam > 0.2
    u = np.full(len(points), np.nan, dtype=np.float64)
    v = np.full(len(points), np.nan, dtype=np.float64)
    u[valid] = fx * (x_cam[valid] / z_cam[valid]) + cx
    v[valid] = fy * (y_cam[valid] / z_cam[valid]) + cy
    in_bounds = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.column_stack([u, v]), in_bounds


def draw_projected_polyline(
    image_bgr: np.ndarray,
    xyz: np.ndarray,
    color_bgr: tuple[int, int, int],
    label: str,
    thickness: int = 4,
) -> tuple[np.ndarray, int]:
    out = image_bgr
    pixels, visible = project_ego_to_fc_pixels(xyz, out.shape[:2])
    visible_indices = np.flatnonzero(visible)
    if len(visible_indices) >= 2:
        for a, b in zip(visible_indices[:-1], visible_indices[1:]):
            pt1 = tuple(np.round(pixels[a]).astype(int))
            pt2 = tuple(np.round(pixels[b]).astype(int))
            cv2.line(out, pt1, pt2, color_bgr, thickness, cv2.LINE_AA)
    for idx in visible_indices[:: max(1, len(visible_indices) // 16)]:
        center = tuple(np.round(pixels[idx]).astype(int))
        cv2.circle(out, center, max(3, thickness), color_bgr, -1, cv2.LINE_AA)
        cv2.circle(out, center, max(3, thickness), (0, 0, 0), 1, cv2.LINE_AA)
    if len(visible_indices):
        pt = tuple(np.round(pixels[visible_indices[-1]]).astype(int))
        cv2.putText(
            out,
            label,
            (pt[0] + 8, max(24, pt[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color_bgr,
            2,
            cv2.LINE_AA,
        )
    return out, int(len(visible_indices))


def draw_fc_projection(
    output_path: Path,
    source_data_root: Path,
    gt_row: TrajectoryRow,
    gt_speed: float,
    center_trajs: list[np.ndarray],
    center_clusters: list[int],
) -> dict[str, object] | None:
    loaded = load_fc_frame(source_data_root, gt_row)
    if loaded is None:
        return None
    frame_bgr, frame_idx, frame_ts = loaded
    out = frame_bgr.copy()
    counts: dict[str, int] = {}
    out, counts["gt_visible_points"] = draw_projected_polyline(
        out,
        xy_to_xyz(gt_row.xy),
        (65, 216, 255),
        "GT",
        thickness=5,
    )
    for rank, (traj_xy, cluster_id) in enumerate(zip(center_trajs, center_clusters), start=1):
        out, counts[f"top{rank}_visible_points"] = draw_projected_polyline(
            out,
            xy_to_xyz(traj_xy),
            rank_color_bgr(rank, len(center_trajs)),
            f"T{rank} C{cluster_id}",
            thickness=4 if len(center_trajs) <= 2 else 2,
        )
    cv2.putText(
        out,
        f"FC approximate projection | traj_id={gt_row.traj_id} | t0 speed={gt_speed:.2f}m/s | frame={frame_idx}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"FC approximate projection | traj_id={gt_row.traj_id} | t0 speed={gt_speed:.2f}m/s | frame={frame_idx}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)
    return {
        "image": str(output_path),
        "frame_index": frame_idx,
        "frame_timestamp": frame_ts,
        "projection": "approximate_default_fc_intrinsics_extrinsics",
        **counts,
    }


def main() -> None:
    args = parse_args()
    smoothing = resolve_smoothing_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded_rows = filter_rows_by_dataset(
        load_future_rows(args.data_txt, args.steps),
        args.dataset_filter,
    )
    match_data_txt = args.match_data_txt or args.data_txt
    match_dataset_filter = args.match_dataset_filter or args.dataset_filter
    loaded_match_rows = filter_rows_by_dataset(
        load_future_rows(match_data_txt, args.steps),
        match_dataset_filter,
    )
    if args.feature_source == "native_parquet":
        (
            rows,
            raw_features,
            acceleration,
            curvature,
            raw_acceleration,
            raw_curvature,
            speed,
            filter_stats,
        ) = build_native_acc_curvature_features(
            loaded_rows,
            args.train_data_root,
            args.steps,
            curvature_min_speed_mps=args.curvature_min_speed_mps,
            speed_smooth_passes=smoothing.speed_passes,
            acceleration_smooth_passes=smoothing.acceleration_passes,
            curvature_smooth_passes=smoothing.curvature_passes,
        )
        (
            match_rows,
            match_raw_features,
            match_acceleration,
            match_curvature,
            _match_raw_acceleration,
            _match_raw_curvature,
            match_speed,
            match_filter_stats,
        ) = build_native_acc_curvature_features(
            loaded_match_rows,
            args.train_data_root,
            args.steps,
            curvature_min_speed_mps=args.curvature_min_speed_mps,
            speed_smooth_passes=smoothing.speed_passes,
            acceleration_smooth_passes=smoothing.acceleration_passes,
            curvature_smooth_passes=smoothing.curvature_passes,
        )
    else:
        rows = loaded_rows
        raw_features, acceleration, curvature, raw_acceleration, raw_curvature = build_acc_curvature_features(
            rows,
            smoothing.speed_passes,
        )
        speed = None
        match_rows = loaded_match_rows
        match_raw_features, match_acceleration, match_curvature, _match_raw_acceleration, _match_raw_curvature = build_acc_curvature_features(
            match_rows,
            smoothing.speed_passes,
        )
        match_speed = None
        match_filter_stats = {
            "input_rows": len(loaded_match_rows),
            "kept_rows": len(match_rows),
            "skipped_missing_parquet": 0,
            "skipped_bad_window": 0,
        }
        filter_stats = {
            "input_rows": len(loaded_rows),
            "kept_rows": len(rows),
            "skipped_missing_parquet": 0,
            "skipped_bad_window": 0,
        }
    features, feature_stats = robust_standardize(raw_features, args.feature_clip_percentile)
    match_features = standardize_with_stats(match_raw_features, feature_stats)
    sample_index = select_sample_index(match_rows, int(args.sample_row))

    n_clusters = min(int(args.n_clusters), len(rows))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        n_init="auto",
        max_iter=300,
    )
    labels = kmeans.fit_predict(features)
    medoid_indices = choose_medoids(features, labels, kmeans.cluster_centers_, n_clusters)
    valid_clusters = np.flatnonzero(medoid_indices >= 0)
    speed_cache: dict[Path, object] = {}
    gt_row = match_rows[sample_index]
    gt_speed = t0_average_speed(
        gt_row,
        args.train_data_root,
        args.generate_output_root,
        args.speed_window,
        speed_cache,
    )
    gt_path_length = path_length_xy(gt_row.xy)
    gt_feature = match_features[sample_index]
    dynamic_distances = np.linalg.norm(
        kmeans.cluster_centers_[valid_clusters] - gt_feature,
        axis=1,
    )
    top_k = max(1, min(int(args.top_k), len(valid_clusters)))
    all_center_features = [
        inverse_standardized_feature(kmeans.cluster_centers_[int(cluster_id)], feature_stats)
        for cluster_id in valid_clusters
    ]
    all_center_acc = [center_feature[:args.steps] for center_feature in all_center_features]
    all_center_curv = [center_feature[args.steps:] for center_feature in all_center_features]
    all_rollouts = [
        integrate_acc_curvature(center_acc, center_curv, initial_speed_mps=gt_speed)
        for center_acc, center_curv in zip(all_center_acc, all_center_curv)
    ]
    member_endpoint_xy = np.asarray(
        [
            np.asarray(row.xy, dtype=np.float64).reshape(-1, 2)[-1]
            for row in rows
        ],
        dtype=np.float64,
    )

    def fill_member_rollout_endpoints(cluster_id: int) -> None:
        for row_idx in np.flatnonzero(labels == int(cluster_id)):
            rollout = integrate_acc_curvature(
                acceleration[int(row_idx)],
                curvature[int(row_idx)],
                initial_speed_mps=gt_speed,
            )
            member_endpoint_xy[int(row_idx)] = rollout[-1]

    if args.selection_method == "rollout_xy_distance":
        cluster_order = sorted(
            range(len(valid_clusters)),
            key=lambda pos: (
                trajectory_xy_distance(gt_row.xy, all_rollouts[int(pos)]),
                float(dynamic_distances[int(pos)]),
            ),
        )
        distance_label = "xy_dist"
    else:
        cluster_order = [int(pos) for pos in np.argsort(dynamic_distances)]
        distance_label = "dyn_dist"

    member_selection_details: list[dict[str, float | int | bool]] = []
    chosen_indices: list[int] = []
    if (
        args.candidate_representation == "acc_curvature_nearest_member_rollout"
        and args.endpoint_constraint_mode == "scan_clusters"
    ):
        chosen_positions = []
        chosen_clusters = []
        for pos in cluster_order:
            cluster_id = int(valid_clusters[int(pos)])
            fill_member_rollout_endpoints(cluster_id)
            cluster_indices, cluster_details = choose_endpoint_constrained_feature_members(
                features,
                labels,
                [cluster_id],
                gt_feature,
                rows,
                gt_row.xy,
                member_endpoint_xy,
                max_lateral_m=args.endpoint_constraint_lateral_m,
                max_longitudinal_m=args.endpoint_constraint_longitudinal_m,
                max_short_longitudinal_m=args.endpoint_constraint_short_longitudinal_m,
                members_per_cluster=args.members_per_cluster,
            )
            if not cluster_indices:
                continue
            chosen_positions.append(int(pos))
            chosen_clusters.append(cluster_id)
            chosen_indices.extend(cluster_indices)
            member_selection_details.extend(cluster_details)
            if len(chosen_clusters) >= top_k:
                break
        if not chosen_indices:
            raise ValueError(
                "No candidate members satisfied endpoint constraints "
                f"(lat<={args.endpoint_constraint_lateral_m}, "
                f"long<={args.endpoint_constraint_longitudinal_m})"
            )
        chosen_distances = [float(dynamic_distances[pos]) for pos in chosen_positions]
    else:
        if args.selection_method == "rollout_xy_distance":
            chosen_positions, chosen_distances = select_top_rollouts_by_xy_distance(
                gt_row.xy,
                all_rollouts,
                dynamic_distances,
                top_k=top_k,
            )
        else:
            chosen_positions = cluster_order[:top_k]
            chosen_distances = [float(dynamic_distances[pos]) for pos in chosen_positions]

        chosen_clusters = valid_clusters[chosen_positions].astype(int).tolist()
        if args.candidate_representation == "gt_nearest_member_xy":
            chosen_indices = choose_gt_nearest_member_indices(
                rows,
                labels,
                chosen_clusters,
                gt_row.xy,
            )
        elif args.candidate_representation == "acc_curvature_nearest_member_rollout":
            for cluster_id in chosen_clusters:
                fill_member_rollout_endpoints(int(cluster_id))
            chosen_indices, member_selection_details = choose_feature_nearest_member_indices_with_endpoint(
                features,
                labels,
                chosen_clusters,
                gt_feature,
                rows,
                gt_row.xy,
                args.member_endpoint_weight,
                endpoint_xy_by_row=member_endpoint_xy,
                members_per_cluster=args.members_per_cluster,
            )
        else:
            chosen_indices = [int(medoid_indices[cluster_id]) for cluster_id in chosen_clusters]

    cluster_position_by_id = {
        int(cluster_id): int(position)
        for cluster_id, position in zip(chosen_clusters, chosen_positions)
    }
    chosen_medoid_indices = [int(medoid_indices[cluster_id]) for cluster_id in chosen_clusters]
    candidate_clusters = (
        [int(item["cluster"]) for item in member_selection_details]
        if member_selection_details
        else chosen_clusters
    )
    candidate_positions = [cluster_position_by_id[int(cluster_id)] for cluster_id in candidate_clusters]
    candidate_distances = [float(dynamic_distances[int(position)]) for position in candidate_positions]
    candidate_medoid_indices = [int(medoid_indices[int(cluster_id)]) for cluster_id in candidate_clusters]
    chosen_representative_rows = [rows[row_idx] for row_idx in chosen_indices]
    chosen_center_features = [all_center_features[pos] for pos in candidate_positions]
    chosen_center_acc = [center_feature[:args.steps] for center_feature in chosen_center_features]
    chosen_center_curv = [center_feature[args.steps:] for center_feature in chosen_center_features]
    chosen_rollouts = [all_rollouts[pos] for pos in candidate_positions]
    chosen_member_rollouts = [
        integrate_acc_curvature(
            acceleration[row_idx],
            curvature[row_idx],
            initial_speed_mps=gt_speed,
        )
        for row_idx in chosen_indices
    ]
    chosen_candidate_trajs = candidate_trajectories_for_representation(
        args.candidate_representation,
        chosen_rollouts,
        chosen_representative_rows,
        chosen_member_rollouts,
    )
    chosen_average_distances = [
        trajectory_xy_distance(gt_row.xy, candidate_xy)
        for candidate_xy in chosen_candidate_trajs
    ]
    rollout_description = (
        "integrate_feature_nearest_member_smoothed_acceleration_curvature_with_gt_t0_average_speed"
        if args.candidate_representation == "acc_curvature_nearest_member_rollout"
        else "integrate_selected_center_acceleration_curvature_with_gt_t0_average_speed"
    )

    output_suffix = "" if args.selection_method == "dynamic_feature" else "_xy_distance"
    if args.candidate_representation == "medoid_xy":
        output_suffix += "_medoid"
    elif args.candidate_representation == "gt_nearest_member_xy":
        output_suffix += "_gt_nearest"
    elif args.candidate_representation == "acc_curvature_nearest_member_rollout":
        output_suffix += "_feature_member_rollout"
    top_tag = f"top{top_k}"
    if int(args.members_per_cluster) > 1:
        top_tag += f"_m{int(args.members_per_cluster)}"
    image_path = args.output_dir / f"sample_{gt_row.traj_id:05d}{output_suffix}_{top_tag}.png"
    draw_match(
        image_path,
        gt_row,
        gt_speed,
        chosen_candidate_trajs,
        candidate_distances,
        candidate_clusters,
        distance_label=distance_label,
        average_distances=chosen_average_distances,
        endpoint_distances=[
            float(item["endpoint_distance_m"])
            for item in member_selection_details
        ] if member_selection_details else None,
    )
    if match_speed is not None:
        gt_speed_profile = np.asarray(match_speed[sample_index], dtype=np.float64)
    else:
        gt_speed_profile = gt_future_speed_profile(
            gt_row,
            args.train_data_root,
            args.generate_output_root,
            args.steps,
            speed_cache,
        )
    if len(gt_speed_profile) != args.steps or not np.isfinite(gt_speed_profile).all():
        gt_speed_profile = speed_profile_from_xy(gt_row.xy)
    gt_speed_profile = gt_speed_profile_for_visualization(
        gt_speed_profile,
        smoothing.speed_passes,
    )
    speed_profile_path = args.output_dir / f"sample_{gt_row.traj_id:05d}{output_suffix}_{top_tag}_speed_profile.png"
    draw_speed_profiles(
        speed_profile_path,
        gt_speed_profile,
        chosen_candidate_trajs,
        candidate_clusters,
    )
    fc_image_path = args.output_dir / f"sample_{gt_row.traj_id:05d}{output_suffix}_{top_tag}_fc_projection.png"
    fc_projection = draw_fc_projection(
        fc_image_path,
        args.source_data_root,
        gt_row,
        gt_speed,
        chosen_candidate_trajs,
        candidate_clusters,
    )

    summary = {
        "data_txt": str(args.data_txt),
        "match_data_txt": str(match_data_txt),
        "n_rows": len(rows),
        "n_match_rows": len(match_rows),
        "n_clusters": n_clusters,
        "feature": "scalar_acceleration_plus_curvature_only",
        "feature_source": args.feature_source,
        "dataset_filter": args.dataset_filter,
        "filter_stats": filter_stats,
        "match_filter_stats": match_filter_stats,
        "smoothing": {
            "applied_before_clustering": True,
            "legacy_smooth_passes": args.smooth_passes,
            "speed_smooth_passes": int(smoothing.speed_passes),
            "acceleration_smooth_passes": int(smoothing.acceleration_passes),
            "curvature_smooth_passes": int(smoothing.curvature_passes),
        },
        "feature_shape": list(features.shape),
        "feature_stats": feature_stats_for_json(feature_stats),
        "inertia": float(kmeans.inertia_),
        "selection": args.selection_method,
        "candidate_representation": args.candidate_representation,
        "endpoint_constraint": {
            "mode": args.endpoint_constraint_mode,
            "frame": "gt_endpoint_yaw",
            "max_lateral_m": float(args.endpoint_constraint_lateral_m),
            "max_longitudinal_m": float(args.endpoint_constraint_longitudinal_m),
            "max_short_longitudinal_m": float(args.endpoint_constraint_short_longitudinal_m),
        },
        "member_endpoint_weight": float(args.member_endpoint_weight),
        "members_per_cluster": int(args.members_per_cluster),
        "n_candidates": len(chosen_candidate_trajs),
        "rollout": rollout_description,
        "top_k": top_k,
        "sample_row": int(args.sample_row),
        "sample_index_after_filter": int(sample_index),
        "match_dataset_filter": match_dataset_filter,
        "gt": {
            "traj_id": gt_row.traj_id,
            "dataset": gt_row.dataset,
            "clip": gt_row.clip,
            "t0_index": gt_row.t0_index,
            "t0_avg_speed_mps": gt_speed,
            "path_length_m": gt_path_length,
            "speed_profile_visualization": "smoothed_scalar_speed",
        },
        "top_matches": [
            {
                "rank": rank,
                "cluster": int(cluster_id),
                "representative_row": int(row_idx),
                "representative_kind": args.candidate_representation,
                "member_rank_in_cluster": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else int(member_selection_details[rank - 1]["member_rank_in_cluster"])
                ),
                "medoid_row": int(medoid_row_idx),
                "traj_id": rows[row_idx].traj_id,
                "dataset": rows[row_idx].dataset,
                "clip": rows[row_idx].clip,
                "t0_index": rows[row_idx].t0_index,
                "dynamic_distance": float(dynamic_distances[int(valid_position)]),
                "xy_distance": trajectory_xy_distance(gt_row.xy, rollout_xy),
                "avg_step_distance_m": float(chosen_average_distances[rank - 1]),
                "selection_distance": float(distance),
                "member_feature_distance": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else float(member_selection_details[rank - 1]["feature_distance"])
                ),
                "member_endpoint_distance_m": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else float(member_selection_details[rank - 1]["endpoint_distance_m"])
                ),
                "member_endpoint_lateral_error_m": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else float(member_selection_details[rank - 1].get("endpoint_lateral_error_m", 0.0))
                ),
                "member_endpoint_longitudinal_error_m": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else float(member_selection_details[rank - 1].get("endpoint_longitudinal_error_m", 0.0))
                ),
                "member_endpoint_constraint_satisfied": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else bool(member_selection_details[rank - 1].get("endpoint_constraint_satisfied", False))
                ),
                "member_selection_score": (
                    None
                    if rank - 1 >= len(member_selection_details)
                    else (
                        None
                        if "selection_score" not in member_selection_details[rank - 1]
                        else float(member_selection_details[rank - 1]["selection_score"])
                    )
                ),
                "rollout_initial_speed_mps": float(gt_speed),
                "medoid_path_length_m": path_length_xy(rows[medoid_row_idx].xy),
                "representative_path_length_m": path_length_xy(rows[row_idx].xy),
                "center_rollout_path_length_m": path_length_xy(chosen_rollouts[rank - 1]),
                "member_rollout_path_length_m": path_length_xy(chosen_member_rollouts[rank - 1]),
                "candidate_path_length_m": path_length_xy(rollout_xy),
                "rollout_path_length_m": path_length_xy(rollout_xy),
                "center_acceleration_mean_mps2": float(np.mean(chosen_center_acc[rank - 1])),
                "center_curvature_mean_abs": float(np.mean(np.abs(chosen_center_curv[rank - 1]))),
            }
            for rank, (cluster_id, row_idx, medoid_row_idx, distance, rollout_xy, valid_position) in enumerate(
                zip(
                    candidate_clusters,
                    chosen_indices,
                    candidate_medoid_indices,
                    candidate_distances,
                    chosen_candidate_trajs,
                    candidate_positions,
                ),
                start=1,
            )
        ],
        "image": str(image_path),
        "speed_profile_image": str(speed_profile_path),
        "fc_projection": fc_projection,
        "raw_acceleration_mean_abs": float(np.mean(np.abs(raw_acceleration))),
        "raw_curvature_mean_abs": float(np.mean(np.abs(raw_curvature))),
        "smoothed_acceleration_mean_abs": float(np.mean(np.abs(acceleration))),
        "smoothed_curvature_mean_abs": float(np.mean(np.abs(curvature))),
        "native_speed_mean_mps": None if speed is None else float(np.mean(speed)),
        "match_native_speed_mean_mps": None if match_speed is None else float(np.mean(match_speed)),
    }
    summary_path = args.output_dir / f"sample_{gt_row.traj_id:05d}{output_suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Loaded {len(rows)} trajectories")
    print(f"Clustered by scalar acceleration+curvature into {n_clusters} clusters")
    print(f"Feature source: {args.feature_source}")
    print(f"Selection method: {args.selection_method}")
    print(f"Candidate representation: {args.candidate_representation}")
    print(f"Filter stats: {filter_stats}")
    print(f"GT row {gt_row.traj_id} t0_avg_speed={gt_speed:.3f} m/s")
    for item in summary["top_matches"]:
        print(
            f"Top{item['rank']}: cluster={item['cluster']} "
            f"traj_id={item['traj_id']} "
            f"selection_distance={item['selection_distance']:.3f} "
            f"dynamic_distance={item['dynamic_distance']:.3f} "
            f"xy_distance={item['xy_distance']:.3f} "
            f"candidate_length={item['candidate_path_length_m']:.3f} m"
        )
    print(f"Wrote image: {image_path}")
    print(f"Wrote speed profile: {speed_profile_path}")
    if fc_projection is None:
        print("FC projection not written: FC video/timestamps were not found")
    else:
        print(f"Wrote FC projection: {fc_projection['image']}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
