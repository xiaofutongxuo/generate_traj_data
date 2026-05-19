#!/usr/bin/env python3
"""Generate target GT plus acceleration/curvature cluster candidates."""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import cluster_acc_curvature_top2 as acc
from kmeans_cluster import estimate_t0_rotation, export_future_xy_txt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_ROOT = Path("/home/ubuntu/Public/train_data")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output"
DEFAULT_TRAIN_DATA_TXT = SCRIPT_DIR / "future_trajectories_5_8_xy.txt"
FALLBACK_TRAIN_DATA_TXT = SCRIPT_DIR / "future_trajectories_xy.txt"
DEFAULT_TARGET_FUTURE_TXT = DEFAULT_OUTPUT_DIR / "target_future_trajectories_xy.txt"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "acc_curvature_diverse_gt_summary.csv"
DEFAULT_TRAIN_DATASET = "data_26_5_8_converted"
EGOMOTION_COLUMNS = [
    "timestamp",
    "qx",
    "qy",
    "qz",
    "qw",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "ax",
    "ay",
    "az",
    "curvature",
]
SUMMARY_FIELDNAMES = [
    "traj_id",
    "dataset",
    "clip",
    "t0_us",
    "t0_index",
    "gt_speed_mps",
    "num_candidates",
    "candidate_sample_idx",
    "cluster",
    "member_index",
    "member_rank_in_cluster",
    "feature_distance",
    "dynamic_cluster_distance",
    "endpoint_distance_m",
    "endpoint_lateral_error_m",
    "endpoint_longitudinal_error_m",
    "avg_step_distance_m",
]
DEFAULT_DATASETS = ",".join(
    [
        "data_26_5_8_converted",
        "data_26_3_24_1_converted",
        "data_26_3_24_2_converted",
        "data_26_3_24_3_converted",
        "data_26_3_25_1_converted",
        "data_26_3_25_2_converted",
    ]
)
DT_SECONDS = acc.DT_SECONDS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--datasets", type=str, default=DEFAULT_DATASETS)
    parser.add_argument("--train_data_txt", type=Path, default=DEFAULT_TRAIN_DATA_TXT)
    parser.add_argument("--train_dataset_filter", type=str, default=DEFAULT_TRAIN_DATASET)
    parser.add_argument("--target_future_txt", type=Path, default=DEFAULT_TARGET_FUTURE_TXT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument("--t0_stride", type=int, default=1)
    parser.add_argument("--min_speed_mps", type=float, default=0.0)
    parser.add_argument("--min_forward_acc_mps2", type=float, default=-6.0)
    parser.add_argument("--max_forward_acc_mps2", type=float, default=2.0)
    parser.add_argument("--max_step_speed_mps", type=float, default=15.0)
    parser.add_argument("--max_step_acc_mps2", type=float, default=0.0)
    parser.add_argument("--allow_backward", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--n_clusters", type=int, default=500)
    parser.add_argument("--top_clusters", type=int, default=2)
    parser.add_argument("--members_per_cluster", type=int, default=2)
    parser.add_argument("--member_endpoint_weight", type=float, default=0.15)
    parser.add_argument("--speed_window", type=int, default=3)
    parser.add_argument("--feature_clip_percentile", type=float, default=99.0)
    parser.add_argument("--speed_smooth_passes", type=int, default=1)
    parser.add_argument("--acceleration_smooth_passes", type=int, default=2)
    parser.add_argument("--curvature_smooth_passes", type=int, default=1)
    parser.add_argument("--curvature_min_speed_mps", type=float, default=0.0)
    parser.add_argument(
        "--endpoint_speed_bin_mps",
        type=float,
        default=0.05,
        help=(
            "Quantize t0 speed for endpoint-feasibility rollout cache. "
            "Use 0 to disable and recompute endpoints with exact speed."
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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_train_data_txt(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_TRAIN_DATA_TXT and FALLBACK_TRAIN_DATA_TXT.exists():
        return FALLBACK_TRAIN_DATA_TXT
    raise FileNotFoundError(f"Training trajectory txt not found: {path}")


def load_target_future_rows(path: Path, steps: int) -> list[acc.TrajectoryRow]:
    rows: list[acc.TrajectoryRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            values = np.asarray(parts[5:], dtype=np.float64)
            if len(values) % 2 != 0 or len(values) < int(steps) * 2:
                raise ValueError(f"Bad row with {len(parts)} columns: {line[:120]}")
            xy = values.reshape(-1, 2)[: int(steps)]
            rows.append(
                acc.TrajectoryRow(
                    traj_id=int(parts[0]),
                    dataset=parts[1],
                    clip=parts[2],
                    t0_us=int(parts[3]),
                    t0_index=int(parts[4]),
                    xy=xy,
                )
            )
    if not rows:
        raise ValueError(f"No trajectories found in {path}")
    return rows


def export_target_future_rows(args: argparse.Namespace) -> Path:
    if args.skip_export:
        if not args.target_future_txt.exists():
            raise FileNotFoundError(f"Missing --target_future_txt: {args.target_future_txt}")
        return args.target_future_txt

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        if not args.target_future_txt.exists():
            raise FileNotFoundError(f"Missing --target_future_txt: {args.target_future_txt}")
        return args.target_future_txt

    with tempfile.TemporaryDirectory(prefix="acc_curv_target_export_") as tmp_name:
        tmp_root = Path(tmp_name)
        for dataset in datasets:
            source_dir = args.train_data_root / dataset
            if not source_dir.exists():
                raise FileNotFoundError(f"Target dataset not found: {source_dir}")
            (tmp_root / dataset).symlink_to(source_dir, target_is_directory=True)
        stats = export_future_xy_txt(
            tmp_root,
            args.target_future_txt,
            args.steps,
            args.history_steps,
            args.t0_stride,
            args.min_speed_mps,
            args.min_forward_acc_mps2,
            args.max_forward_acc_mps2,
            args.max_step_speed_mps,
            args.max_step_acc_mps2,
            args.allow_backward,
        )
    print(
        f"Exported {stats.kept} target GT rows to {args.target_future_txt} "
        f"from datasets={','.join(datasets)} stride={args.t0_stride}"
    )
    print(
        "Target export filters: "
        f"skipped_slow={stats.skipped_slow}, "
        f"skipped_backward={stats.skipped_backward}, "
        f"skipped_acc={stats.skipped_acc}, "
        f"skipped_jump={stats.skipped_jump}, "
        f"skipped_short_or_bad={stats.skipped_short_or_bad}"
    )
    return args.target_future_txt


def build_cluster_model(args: argparse.Namespace) -> dict[str, object]:
    train_txt = resolve_train_data_txt(args.train_data_txt)
    loaded_rows = acc.load_future_rows(train_txt, args.steps)
    rows = acc.filter_rows_by_dataset(loaded_rows, args.train_dataset_filter)
    (
        rows,
        raw_features,
        acceleration,
        curvature,
        _raw_acceleration,
        _raw_curvature,
        _speed,
        filter_stats,
    ) = acc.build_native_acc_curvature_features(
        rows,
        args.train_data_root,
        args.steps,
        curvature_min_speed_mps=args.curvature_min_speed_mps,
        speed_smooth_passes=args.speed_smooth_passes,
        acceleration_smooth_passes=args.acceleration_smooth_passes,
        curvature_smooth_passes=args.curvature_smooth_passes,
    )
    features, feature_stats = acc.robust_standardize(raw_features, args.feature_clip_percentile)
    n_clusters = min(int(args.n_clusters), len(rows))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        n_init="auto",
        max_iter=300,
    )
    labels = kmeans.fit_predict(features)
    medoid_indices = acc.choose_medoids(features, labels, kmeans.cluster_centers_, n_clusters)
    return {
        "train_txt": train_txt,
        "rows": rows,
        "features": features,
        "acceleration": acceleration,
        "curvature": curvature,
        "kmeans": kmeans,
        "labels": labels,
        "medoid_indices": medoid_indices,
        "valid_clusters": np.flatnonzero(medoid_indices >= 0),
        "feature_stats": feature_stats,
        "filter_stats": filter_stats,
    }


def load_target_features(
    args: argparse.Namespace,
    target_future_txt: Path,
    feature_stats: dict,
) -> tuple[list[acc.TrajectoryRow], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    loaded_rows = load_target_future_rows(target_future_txt, args.steps)
    if args.limit and args.limit > 0:
        loaded_rows = loaded_rows[: int(args.limit)]
    (
        rows,
        raw_features,
        acceleration,
        curvature,
        _raw_acceleration,
        _raw_curvature,
        speed,
        filter_stats,
    ) = acc.build_native_acc_curvature_features(
        loaded_rows,
        args.train_data_root,
        args.steps,
        curvature_min_speed_mps=args.curvature_min_speed_mps,
        speed_smooth_passes=args.speed_smooth_passes,
        acceleration_smooth_passes=args.acceleration_smooth_passes,
        curvature_smooth_passes=args.curvature_smooth_passes,
    )
    features = acc.standardize_with_stats(raw_features, feature_stats)
    return rows, features, acceleration, curvature, speed, filter_stats


def derive_vector_kinematics_from_xyz(
    xyz: np.ndarray,
    speed_smooth_passes: int = 1,
    acceleration_smooth_passes: int = 2,
    curvature_smooth_passes: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive velocity, acceleration, and curvature from positions with smoothing."""
    points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    prev = np.vstack([points[:1], points[:-1]])
    raw_velocity = (points - prev) / DT_SECONDS
    if len(raw_velocity) > 1:
        raw_velocity[0] = raw_velocity[1]
    velocity = np.column_stack([
        acc.smooth_trace(raw_velocity[:, dim], speed_smooth_passes)
        for dim in range(3)
    ])

    raw_acceleration = np.vstack([
        np.zeros((1, 3), dtype=np.float64),
        np.diff(velocity, axis=0) / DT_SECONDS,
    ])
    acceleration = np.column_stack([
        acc.smooth_trace(raw_acceleration[:, dim], acceleration_smooth_passes)
        for dim in range(3)
    ])

    yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
    speed = np.linalg.norm(velocity[:, :2], axis=1)
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
    step_distance = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), points[:, :2]]), axis=0), axis=1)
    curvature = np.zeros(len(points), dtype=np.float64)
    if len(points) > 1:
        curvature[1:] = np.diff(yaw) / np.maximum(step_distance[1:], 1e-6)
    curvature[~np.isfinite(curvature)] = 0.0
    curvature = acc.smooth_trace(curvature, curvature_smooth_passes)
    return velocity, acceleration, curvature


def dataset_clip_stems_by_time(train_data_root: Path, dataset: str) -> list[str]:
    dataset_path = train_data_root / dataset
    ego_dir = dataset_path / "data-egomotion"
    ts_dir = dataset_path / "data-timestamps"
    stems = set()
    if ego_dir.exists():
        stems.update(path.name.replace(".egomotion.parquet", "") for path in ego_dir.glob("*.egomotion.parquet"))
    if ts_dir.exists():
        stems.update(path.name.replace(".timestamps.parquet", "") for path in ts_dir.glob("*.timestamps.parquet"))

    def sort_key(stem: str) -> tuple[int, str]:
        ts_file = ts_dir / f"{stem}.timestamps.parquet"
        if ts_file.exists():
            try:
                df_ts = pd.read_parquet(ts_file, columns=["timestamp"])
                if len(df_ts):
                    return int(df_ts["timestamp"].iloc[0]), stem
            except Exception:
                pass
        return 2**63 - 1, stem

    return sorted(stems, key=sort_key)


def load_history_context_df(
    row: acc.TrajectoryRow,
    train_data_root: Path,
    history_steps: int,
    cache: dict[tuple[str, str], pd.DataFrame],
    clip_stems_cache: dict[str, list[str]] | None = None,
) -> pd.DataFrame | None:
    if clip_stems_cache is None:
        stems = dataset_clip_stems_by_time(train_data_root, row.dataset)
    else:
        if row.dataset not in clip_stems_cache:
            clip_stems_cache[row.dataset] = dataset_clip_stems_by_time(train_data_root, row.dataset)
        stems = clip_stems_cache[row.dataset]
    ego_dir = train_data_root / row.dataset / "data-egomotion"
    if row.clip not in stems:
        stems = sorted(set(stems) | {row.clip})
    current_pos = stems.index(row.clip)
    frames: list[pd.DataFrame] = []
    for stem in reversed(stems[: current_pos + 1]):
        key = (row.dataset, stem)
        if key not in cache:
            path = ego_dir / f"{stem}.egomotion.parquet"
            if not path.exists():
                continue
            cache[key] = pd.read_parquet(path)
        frames.insert(0, cache[key])
        combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        t0_idx = acc.native_t0_frame_index(combined, row)
        if t0_idx is not None and int(t0_idx) - int(history_steps) + 1 >= 0:
            return combined
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def trajectory_to_parquet_row(
    row: acc.TrajectoryRow,
    xy: np.ndarray,
    sample_idx: int,
    source: str,
    history_egomotion: dict[str, list[float] | list[int]] | None = None,
) -> dict:
    xyz = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
    velocity, acceleration, curvature = derive_vector_kinematics_from_xyz(xyz)
    yaw = np.arctan2(velocity[:, 1], velocity[:, 0])
    moving = np.linalg.norm(velocity[:, :2], axis=1) > 1e-3
    if moving.any():
        first = int(np.flatnonzero(moving)[0])
        yaw[:first] = yaw[first]
        for idx in range(first + 1, len(yaw)):
            if not moving[idx]:
                yaw[idx] = yaw[idx - 1]
        yaw = np.unwrap(yaw)
    else:
        yaw[:] = 0.0
    out = {
        "t0_us": int(row.t0_us),
        "sample_idx": int(sample_idx),
        "source": source,
        "timestamp": [int(row.t0_us) + int((i + 1) * DT_SECONDS * 1_000_000) for i in range(len(xy))],
        "qx": [0.0] * len(xy),
        "qy": [0.0] * len(xy),
        "qz": np.sin(yaw / 2.0).tolist(),
        "qw": np.cos(yaw / 2.0).tolist(),
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "vx": velocity[:, 0].tolist(),
        "vy": velocity[:, 1].tolist(),
        "vz": velocity[:, 2].tolist(),
        "ax": acceleration[:, 0].tolist(),
        "ay": acceleration[:, 1].tolist(),
        "az": acceleration[:, 2].tolist(),
        "curvature": curvature.tolist(),
    }
    if history_egomotion is not None:
        for name in EGOMOTION_COLUMNS:
            if name in history_egomotion:
                out[f"history_{name}"] = history_egomotion[name]
    return out


def load_history_egomotion(
    row: acc.TrajectoryRow,
    train_data_root: Path,
    history_steps: int,
    cache: dict[tuple[str, str], pd.DataFrame],
    clip_stems_cache: dict[str, list[str]] | None = None,
) -> dict[str, list[float] | list[int]] | None:
    df = load_history_context_df(
        row,
        train_data_root,
        history_steps,
        cache,
        clip_stems_cache=clip_stems_cache,
    )
    if df is None:
        return None
    t0_idx = acc.native_t0_frame_index(df, row)
    if t0_idx is None:
        return None
    start = int(t0_idx) - int(history_steps) + 1
    if start < 0:
        return None
    hist = df.iloc[start : int(t0_idx) + 1]
    if len(hist) != int(history_steps):
        return None
    required = set(EGOMOTION_COLUMNS)
    if not required.issubset(hist.columns):
        return None
    xyz = hist[["x", "y", "z"]].to_numpy(dtype=np.float64)
    quat = hist[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    t0_rot_inv = estimate_t0_rotation(xyz, quat).inv()
    local_xyz = t0_rot_inv.apply(xyz - xyz[-1])
    local_velocity, local_acceleration, local_curvature = derive_vector_kinematics_from_xyz(local_xyz)
    out: dict[str, list[float] | list[int]] = {
        "timestamp": hist["timestamp"].to_numpy(dtype=np.int64).tolist(),
        "qx": hist["qx"].to_numpy(dtype=np.float64).tolist(),
        "qy": hist["qy"].to_numpy(dtype=np.float64).tolist(),
        "qz": hist["qz"].to_numpy(dtype=np.float64).tolist(),
        "qw": hist["qw"].to_numpy(dtype=np.float64).tolist(),
        "x": local_xyz[:, 0].tolist(),
        "y": local_xyz[:, 1].tolist(),
        "z": local_xyz[:, 2].tolist(),
        "vx": local_velocity[:, 0].tolist(),
        "vy": local_velocity[:, 1].tolist(),
        "vz": local_velocity[:, 2].tolist(),
        "ax": local_acceleration[:, 0].tolist(),
        "ay": local_acceleration[:, 1].tolist(),
        "az": local_acceleration[:, 2].tolist(),
        "curvature": local_curvature.tolist(),
    }
    return out


def endpoint_cache_speed(initial_speed_mps: float, endpoint_speed_bin_mps: float) -> float:
    speed = float(initial_speed_mps)
    if not np.isfinite(speed):
        speed = 0.0
    speed = max(0.0, speed)
    bin_mps = float(endpoint_speed_bin_mps)
    if not np.isfinite(bin_mps) or bin_mps <= 0.0:
        return speed
    bucket = int(np.floor(speed / bin_mps + 0.5))
    return float(bucket * bin_mps)


def ego_rollout_curvature_from_native(curvature: np.ndarray) -> np.ndarray:
    """Convert parquet native curvature sign to ego-local xy rollout curvature."""
    return -np.asarray(curvature, dtype=np.float64)


def cached_member_endpoint_xy(
    acceleration: np.ndarray,
    curvature: np.ndarray,
    initial_speed_mps: float,
    endpoint_speed_bin_mps: float,
    cache: dict[float, np.ndarray],
) -> tuple[np.ndarray, float]:
    rollout_speed = endpoint_cache_speed(initial_speed_mps, endpoint_speed_bin_mps)
    if rollout_speed not in cache:
        cache[rollout_speed] = acc.integrate_acc_curvature_endpoints_batch(
            acceleration,
            ego_rollout_curvature_from_native(curvature),
            initial_speed_mps=rollout_speed,
        )
    return cache[rollout_speed], rollout_speed


def output_row_for_clip(row: acc.TrajectoryRow, values: dict) -> dict:
    out = dict(values)
    out["_dataset"] = row.dataset
    out["_clip"] = row.clip
    return out


def candidate_rollouts_from_members(
    member_indices: Iterable[int],
    acceleration: np.ndarray,
    curvature: np.ndarray,
    initial_speed_mps: float,
) -> list[np.ndarray]:
    rollout_curvature = ego_rollout_curvature_from_native(curvature)
    return [
        acc.integrate_acc_curvature(
            acceleration[int(member_idx)],
            rollout_curvature[int(member_idx)],
            initial_speed_mps=initial_speed_mps,
        )
        for member_idx in member_indices
    ]


def select_candidate_members(
    args: argparse.Namespace,
    gt_feature: np.ndarray,
    gt_xy: np.ndarray,
    train_features: np.ndarray,
    train_acceleration: np.ndarray,
    train_curvature: np.ndarray,
    cluster_centers: np.ndarray,
    valid_clusters: np.ndarray,
    cluster_member_indices: list[np.ndarray],
    gt_speed: float,
    endpoint_cache: dict[float, np.ndarray],
) -> tuple[list[int], list[dict], np.ndarray, np.ndarray]:
    dynamic_distances = np.linalg.norm(
        cluster_centers[valid_clusters] - gt_feature,
        axis=1,
    )
    member_endpoint_xy, _endpoint_rollout_speed = cached_member_endpoint_xy(
        train_acceleration,
        train_curvature,
        initial_speed_mps=gt_speed,
        endpoint_speed_bin_mps=args.endpoint_speed_bin_mps,
        cache=endpoint_cache,
    )
    return acc.select_top_clusters_then_endpoint_members_fast(
        features=train_features,
        cluster_member_indices=cluster_member_indices,
        valid_clusters=valid_clusters,
        dynamic_distances=dynamic_distances,
        gt_feature=gt_feature,
        gt_xy=gt_xy,
        member_endpoint_xy=member_endpoint_xy,
        max_lateral_m=args.endpoint_constraint_lateral_m,
        max_longitudinal_m=args.endpoint_constraint_longitudinal_m,
        max_short_longitudinal_m=args.endpoint_constraint_short_longitudinal_m,
        endpoint_weight=args.member_endpoint_weight,
        top_clusters=args.top_clusters,
        members_per_cluster=args.members_per_cluster,
    )


def gt_output_row(
    row: acc.TrajectoryRow,
    history_egomotion: dict[str, list[float] | list[int]] | None,
) -> dict:
    return output_row_for_clip(
        row,
        trajectory_to_parquet_row(
            row,
            row.xy,
            0,
            "gt",
            history_egomotion=history_egomotion,
        ),
    )


def candidate_output_row(
    row: acc.TrajectoryRow,
    candidate_xy: np.ndarray,
    sample_idx: int,
    history_egomotion: dict[str, list[float] | list[int]] | None,
) -> dict:
    return output_row_for_clip(
        row,
        trajectory_to_parquet_row(
            row,
            candidate_xy,
            sample_idx,
            "acc_curvature_cluster",
            history_egomotion=history_egomotion,
        ),
    )


def candidate_summary_row(
    row: acc.TrajectoryRow,
    gt_speed: float,
    num_candidates: int,
    sample_idx: int,
    member_idx: int,
    detail: dict,
    dynamic_cluster_distance: float,
    candidate_xy: np.ndarray,
) -> dict:
    return {
        "traj_id": row.traj_id,
        "dataset": row.dataset,
        "clip": row.clip,
        "t0_us": row.t0_us,
        "t0_index": row.t0_index,
        "gt_speed_mps": gt_speed,
        "num_candidates": num_candidates,
        "candidate_sample_idx": sample_idx,
        "cluster": int(detail["cluster"]),
        "member_index": int(member_idx),
        "member_rank_in_cluster": int(detail["member_rank_in_cluster"]),
        "feature_distance": float(detail["feature_distance"]),
        "dynamic_cluster_distance": float(dynamic_cluster_distance),
        "endpoint_distance_m": float(detail["endpoint_distance_m"]),
        "endpoint_lateral_error_m": float(detail["endpoint_lateral_error_m"]),
        "endpoint_longitudinal_error_m": float(detail["endpoint_longitudinal_error_m"]),
        "avg_step_distance_m": acc.trajectory_xy_distance(row.xy, candidate_xy),
    }


def no_candidate_summary_row(row: acc.TrajectoryRow, gt_speed: float) -> dict:
    out = {name: "" for name in SUMMARY_FIELDNAMES}
    out.update(
        {
            "traj_id": row.traj_id,
            "dataset": row.dataset,
            "clip": row.clip,
            "t0_us": row.t0_us,
            "t0_index": row.t0_index,
            "gt_speed_mps": gt_speed,
            "num_candidates": 0,
        }
    )
    return out


def should_expand_candidates(
    row: acc.TrajectoryRow,
    history_egomotion: dict[str, list[float] | list[int]] | None,
    history_steps: int,
) -> bool:
    if int(row.t0_index) >= int(history_steps) - 1:
        return True
    return history_egomotion is not None


def log_progress(row_idx: int, total_rows: int, start_time: float, progress_interval: int) -> None:
    if progress_interval <= 0 or row_idx % int(progress_interval) != 0:
        return
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    rate = row_idx / elapsed
    remaining = (total_rows - row_idx) / max(rate, 1e-6)
    print(
        f"Processed {row_idx}/{total_rows} target GT rows "
        f"({rate:.2f} rows/s, eta={remaining / 60.0:.1f} min)"
    )


def write_outputs(
    output_dir: Path,
    parquet_rows: Iterable[dict],
    summary_rows: list[dict],
    summary_csv: Path,
) -> None:
    by_clip: dict[tuple[str, str], list[dict]] = {}
    for item in parquet_rows:
        row = dict(item)
        dataset = row.pop("_dataset")
        clip = row.pop("_clip")
        by_clip.setdefault((dataset, clip), []).append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for (dataset, clip), rows in sorted(by_clip.items()):
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).sort_values(["t0_us", "sample_idx"])
        df.to_parquet(dataset_dir / f"{clip}.egomotion.parquet", index=False)
        total_rows += len(df)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(by_clip)} parquet files / {total_rows} rows to {output_dir}")
    print(f"Wrote summary to {summary_csv}")


def generate(args: argparse.Namespace) -> None:
    target_future_txt = export_target_future_rows(args)
    model = build_cluster_model(args)
    train_rows: list[acc.TrajectoryRow] = model["rows"]  # type: ignore[assignment]
    train_features: np.ndarray = model["features"]  # type: ignore[assignment]
    train_acceleration: np.ndarray = model["acceleration"]  # type: ignore[assignment]
    train_curvature: np.ndarray = model["curvature"]  # type: ignore[assignment]
    labels: np.ndarray = model["labels"]  # type: ignore[assignment]
    valid_clusters: np.ndarray = model["valid_clusters"]  # type: ignore[assignment]
    kmeans: KMeans = model["kmeans"]  # type: ignore[assignment]
    feature_stats: dict = model["feature_stats"]  # type: ignore[assignment]
    cluster_member_indices = acc.build_cluster_member_indices(labels, len(kmeans.cluster_centers_))

    target_rows, target_features, _target_acc, _target_curv, _target_speed, target_filter_stats = (
        load_target_features(args, target_future_txt, feature_stats)
    )
    speed_cache: dict[Path, object] = {}
    history_cache: dict[tuple[str, str], pd.DataFrame] = {}
    clip_stems_cache: dict[str, list[str]] = {}
    endpoint_cache: dict[float, np.ndarray] = {}
    parquet_rows: list[dict] = []
    summary_rows: list[dict] = []
    start_time = time.perf_counter()

    for row_idx, row in enumerate(target_rows, start=1):
        gt_feature = target_features[row_idx - 1]
        gt_speed = acc.t0_average_speed(
            row,
            args.train_data_root,
            args.output_dir,
            args.speed_window,
            speed_cache,
        )
        history_egomotion = load_history_egomotion(
            row,
            args.train_data_root,
            args.history_steps,
            history_cache,
            clip_stems_cache=clip_stems_cache,
        )
        parquet_rows.append(gt_output_row(row, history_egomotion))
        if not should_expand_candidates(row, history_egomotion, args.history_steps):
            summary_rows.append(no_candidate_summary_row(row, gt_speed))
            log_progress(
                row_idx,
                len(target_rows),
                start_time,
                int(args.progress_interval),
            )
            continue

        chosen_indices, details, chosen_clusters, chosen_cluster_distances = select_candidate_members(
            args,
            gt_feature,
            row.xy,
            train_features,
            train_acceleration,
            train_curvature,
            kmeans.cluster_centers_,
            valid_clusters,
            cluster_member_indices,
            gt_speed,
            endpoint_cache,
        )
        cluster_distance_by_id = {
            int(cluster_id): float(distance)
            for cluster_id, distance in zip(chosen_clusters, chosen_cluster_distances)
        }
        candidate_rollouts = candidate_rollouts_from_members(
            chosen_indices,
            train_acceleration,
            train_curvature,
            gt_speed,
        )

        for sample_idx, (member_idx, detail, candidate_xy) in enumerate(
            zip(chosen_indices, details, candidate_rollouts),
            start=1,
        ):
            parquet_rows.append(
                candidate_output_row(
                    row,
                    candidate_xy,
                    sample_idx,
                    history_egomotion,
                )
            )
            summary_rows.append(
                candidate_summary_row(
                    row,
                    gt_speed,
                    len(chosen_indices),
                    sample_idx,
                    int(member_idx),
                    detail,
                    cluster_distance_by_id[int(detail["cluster"])],
                    candidate_xy,
                )
            )
        if not chosen_indices:
            summary_rows.append(no_candidate_summary_row(row, gt_speed))

        log_progress(
            row_idx,
            len(target_rows),
            start_time,
            int(args.progress_interval),
        )

    write_outputs(args.output_dir, parquet_rows, summary_rows, args.summary_csv)
    print(
        "Cluster config: "
        f"train_rows={len(train_rows)}, clusters={len(valid_clusters)}, "
        f"top_clusters={args.top_clusters}, members_per_cluster={args.members_per_cluster}"
    )
    print(f"Target feature filter stats: {target_filter_stats}")


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
