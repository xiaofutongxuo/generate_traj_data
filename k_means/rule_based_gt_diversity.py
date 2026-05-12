#!/usr/bin/env python3
"""Generate rule-based diverse trajectories around each GT future path.

The workflow is intentionally lightweight:
1. Assign each GT row to a motion category using existing split txt files when
   available, otherwise fall back to curvature/endpoint rules.
2. Pick nearby cluster centers from the same category.
3. Apply a small grid of speed, acceleration-ramp, and curvature scaling to
   each candidate center.
4. Keep only trajectories close to GT by endpoint, mean point distance, and
   residual shape distance.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kmeans_cluster import export_future_xy_txt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FUTURE_TXT = SCRIPT_DIR / "future_trajectories_xy.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "rule_based_gt_diversity"
DEFAULT_TRAIN_DATA_ROOT = Path("/home/ubuntu/Public/train_data")
DT_SECONDS = 0.1
STEPS = 64
GT_ACCEL_MIN_MPS2 = -6.0
GT_ACCEL_MAX_MPS2 = 2.0
GT_MAX_STEP_SPEED_MPS = 15.0
STRAIGHT_SPEED_MPS = 5.0
STRAIGHT_MIN_FORWARD_M = 12.0
STRAIGHT_MAX_FINAL_LATERAL_M = 2.0
STRAIGHT_MAX_LATERAL_M = 2.5
STRAIGHT_MAX_SLOPE = 0.06

CATEGORY_FILES = {
    "stop": "stop.txt",
    "straight": "straight.txt",
    "left": "left.txt",
    "right": "right.txt",
    "s_curve": "s_curve.txt",
}

SPLIT_FILES = {
    "stop": "stop_trajectories_xy.txt",
    "straight": "straight_trajectories_xy.txt",
    "turn": "turn_trajectories_xy.txt",
    "s_curve": "s_curve_trajectories_xy.txt",
}


@dataclass(frozen=True)
class TrajectoryRow:
    traj_id: int
    dataset: str
    clip: str
    t0_us: int
    t0_index: int
    xy: np.ndarray
    compare_xy: np.ndarray
    initial_speed_mps: float | None = None


@dataclass(frozen=True)
class CenterLibrary:
    ids: np.ndarray
    counts: np.ndarray
    centers: np.ndarray
    center_residuals: np.ndarray
    variants: np.ndarray
    variant_residuals: np.ndarray
    variant_sign_sequences: list[tuple[int, ...]]
    variant_center_ids: np.ndarray
    speed_scales: np.ndarray
    accel_ramps: np.ndarray
    curvature_scales: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future_txt", type=Path, default=DEFAULT_FUTURE_TXT)
    parser.add_argument("--kmeans_dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--train_data_root",
        type=Path,
        default=DEFAULT_TRAIN_DATA_ROOT,
        help="Train-data root used when exporting target GT rows from data-egomotion.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated target datasets to export, e.g. data_26_3_24_1_converted. "
        "When set, future_txt is generated from these datasets instead of using the default 5-8-1 txt.",
    )
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument("--t0_stride", type=int, default=3)
    parser.add_argument("--min_speed_mps", type=float, default=0.0)
    parser.add_argument("--min_forward_acc_mps2", type=float, default=-6.0)
    parser.add_argument("--max_forward_acc_mps2", type=float, default=2.0)
    parser.add_argument("--max_step_speed_mps", type=float, default=15.0)
    parser.add_argument("--max_step_acc_mps2", type=float, default=0.0)
    parser.add_argument("--allow_backward", action="store_true")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--compare_steps",
        type=int,
        default=74,
        help="Export this many target GT steps for temporal-shift matching; output remains --steps.",
    )
    parser.add_argument(
        "--temporal_match_min",
        type=int,
        default=56,
        help="Minimum GT horizon sampled for temporal-shift matching.",
    )
    parser.add_argument(
        "--temporal_match_max",
        type=int,
        default=72,
        help="Maximum GT horizon sampled for temporal-shift matching.",
    )
    parser.add_argument(
        "--temporal_match_samples",
        type=int,
        default=9,
        help="Number of uniformly sampled integer horizons from [min, max] per GT row.",
    )
    parser.add_argument("--min_diversity_mean_dist", type=float, default=0.5)
    parser.add_argument(
        "--min_pairwise_mean_dist",
        type=float,
        default=0.5,
        help="Minimum mean Euclidean distance between any two kept rule trajectories for the same GT.",
    )
    parser.add_argument(
        "--curvature_sign_threshold",
        type=float,
        default=0.01,
        help="Curvature magnitude threshold used only for sign-transition extraction.",
    )
    parser.add_argument(
        "--curvature_sign_min_run_steps",
        type=int,
        default=3,
        help="Minimum consecutive points needed to accept a curvature sign segment.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional number of GT rows to process.")
    parser.add_argument("--include_gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preselect_straight", type=int, default=4)
    parser.add_argument("--preselect_stop", type=int, default=6)
    parser.add_argument("--preselect_turn", type=int, default=10)
    parser.add_argument("--preselect_s_curve", type=int, default=12)
    parser.add_argument("--max_straight", type=int, default=2)
    parser.add_argument("--max_stop", type=int, default=3)
    parser.add_argument("--max_turn", type=int, default=6)
    parser.add_argument("--max_s_curve", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary_csv", type=Path, default=None)
    return parser.parse_args()


def load_future_rows(path: Path, steps: int) -> list[TrajectoryRow]:
    rows: list[TrajectoryRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            values = np.asarray(parts[5:], dtype=np.float64)
            total_steps = len(values) // 2
            if total_steps < steps or len(values) % 2 != 0:
                raise ValueError(f"Bad row with {len(parts)} columns: {line[:120]}")
            compare_xy = values.reshape(total_steps, 2)
            xy = compare_xy[:steps]
            rows.append(
                TrajectoryRow(
                    traj_id=int(parts[0]),
                    dataset=parts[1],
                    clip=parts[2],
                    t0_us=int(parts[3]),
                    t0_index=int(parts[4]),
                    xy=xy,
                    compare_xy=compare_xy,
                )
            )
    return rows


def fallback_initial_speed_mps(xy: np.ndarray) -> float:
    """Use the first future step as a last-resort t0 speed estimate."""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return 0.0
    return float(np.clip(np.linalg.norm(points[0]) / DT_SECONDS, 0.0, GT_MAX_STEP_SPEED_MPS))


def estimate_t0_speed_from_train_data(
    train_data_root: Path,
    rows: list[TrajectoryRow],
) -> list[TrajectoryRow]:
    """Attach history-tail t0 speed to each row, falling back to GT first-step speed."""
    if not rows:
        return rows

    by_clip: dict[tuple[str, str], list[tuple[int, TrajectoryRow]]] = {}
    for idx, row in enumerate(rows):
        by_clip.setdefault((row.dataset, row.clip), []).append((idx, row))

    speeds: list[float] = [fallback_initial_speed_mps(row.xy) for row in rows]
    loaded = 0
    missing = 0
    for (dataset, clip), indexed_rows in by_clip.items():
        parquet_file = train_data_root / dataset / "data-egomotion" / f"{clip}.egomotion.parquet"
        if not parquet_file.exists():
            missing += len(indexed_rows)
            continue
        try:
            df = pd.read_parquet(parquet_file, columns=["timestamp", "x", "y", "z"])
        except Exception:
            missing += len(indexed_rows)
            continue

        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        timestamps = df["timestamp"].to_numpy(dtype=np.float64)
        if len(xyz) < 2:
            missing += len(indexed_rows)
            continue

        for idx, row in indexed_rows:
            t0_idx = int(row.t0_index)
            if 1 <= t0_idx < len(xyz):
                dt = (timestamps[t0_idx] - timestamps[t0_idx - 1]) * 1e-6
                if np.isfinite(dt) and dt > 1e-6:
                    speed = np.linalg.norm(xyz[t0_idx, :2] - xyz[t0_idx - 1, :2]) / dt
                else:
                    speed = np.nan
            else:
                speed = np.nan
            if np.isfinite(speed):
                speeds[idx] = float(np.clip(speed, 0.0, GT_MAX_STEP_SPEED_MPS))
                loaded += 1
            else:
                missing += 1

    if missing:
        print(
            f"Warning: used first future-step speed for {missing}/{len(rows)} rows "
            f"because t0 speed could not be read from {train_data_root}"
        )
    if loaded:
        print(f"Loaded history-tail t0 speed for {loaded}/{len(rows)} rows from {train_data_root}")

    return [
        TrajectoryRow(
            traj_id=row.traj_id,
            dataset=row.dataset,
            clip=row.clip,
            t0_us=row.t0_us,
            t0_index=row.t0_index,
            xy=row.xy,
            compare_xy=row.compare_xy,
            initial_speed_mps=speeds[idx],
        )
        for idx, row in enumerate(rows)
    ]


def export_target_future_rows(args: argparse.Namespace) -> Path:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        return args.future_txt

    out_txt = args.output_dir / "target_future_trajectories_xy.txt"
    with tempfile.TemporaryDirectory(prefix="rule_gt_export_") as tmp_name:
        tmp_root = Path(tmp_name)
        for dataset in datasets:
            source_dir = args.train_data_root / dataset
            if not source_dir.exists():
                raise FileNotFoundError(f"Target dataset not found: {source_dir}")
            (tmp_root / dataset).symlink_to(source_dir, target_is_directory=True)
        stats = export_future_xy_txt(
            tmp_root,
            out_txt,
            max(args.steps, args.compare_steps),
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
        f"Exported {stats.kept} target GT rows to {out_txt} "
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
    return out_txt


def load_split_categories(kmeans_dir: Path) -> tuple[dict[int, str], dict[tuple[str, str, int, int], str]]:
    id_mapping: dict[int, str] = {}
    key_mapping: dict[tuple[str, str, int, int], str] = {}
    for split_name, filename in SPLIT_FILES.items():
        path = kmeans_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                traj_id = int(parts[0])
                dataset = parts[4]
                clip = parts[5]
                t0_us = int(parts[6])
                t0_index = int(parts[7])
                if split_name == "turn":
                    direction = parts[2]
                    category = "right" if direction == "right" else "left"
                elif split_name == "s_curve":
                    category = "s_curve"
                else:
                    category = split_name
                id_mapping[traj_id] = category
                key_mapping[(dataset, clip, t0_us, t0_index)] = category
    return id_mapping, key_mapping


def trajectory_dynamics(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin_xy = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    delta = np.diff(origin_xy, axis=0)
    step_distance = np.linalg.norm(delta, axis=1)
    speed = step_distance / DT_SECONDS
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
    return speed, yaw, curvature


def resample_by_time(xy: np.ndarray, steps: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == steps:
        return xy
    src = np.linspace(0.0, 1.0, len(xy), dtype=np.float64)
    dst = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    out = np.empty((steps, 2), dtype=np.float64)
    out[:, 0] = np.interp(dst, src, xy[:, 0])
    out[:, 1] = np.interp(dst, src, xy[:, 1])
    return out


def _clamp_speed_profile_by_acceleration(speed: np.ndarray, initial_speed_mps: float | None = None) -> np.ndarray:
    values = np.asarray(speed, dtype=np.float64).copy()
    if len(values) == 0:
        return values
    values[~np.isfinite(values)] = 0.0
    values = np.clip(values, 0.0, GT_MAX_STEP_SPEED_MPS)
    if initial_speed_mps is not None and np.isfinite(initial_speed_mps):
        prev = float(np.clip(initial_speed_mps, 0.0, GT_MAX_STEP_SPEED_MPS))
    else:
        prev = float(values[0])
    for idx in range(len(values)):
        lower = max(0.0, prev + GT_ACCEL_MIN_MPS2 * DT_SECONDS)
        upper = prev + GT_ACCEL_MAX_MPS2 * DT_SECONDS
        values[idx] = float(np.clip(values[idx], lower, max(lower, upper)))
        prev = values[idx]
    return values


def _interp_polyline_by_distance(points: np.ndarray, cumulative: np.ndarray, distances: np.ndarray) -> np.ndarray:
    out = np.empty((len(distances), 2), dtype=np.float64)
    out[:, 0] = np.interp(distances, cumulative, points[:, 0])
    out[:, 1] = np.interp(distances, cumulative, points[:, 1])
    return out


def smooth_xy_from_t0_speed(xy: np.ndarray, initial_speed_mps: float | None) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2 or initial_speed_mps is None or not np.isfinite(initial_speed_mps):
        return points.copy()

    polyline = np.vstack([np.zeros((1, 2), dtype=np.float64), points])
    seg_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    total_length = float(seg_lengths.sum())
    if total_length <= 1e-6:
        return np.zeros_like(points)

    desired_speed = np.clip(seg_lengths / DT_SECONDS, 0.0, GT_MAX_STEP_SPEED_MPS)
    initial_speed = float(np.clip(initial_speed_mps, 0.0, GT_MAX_STEP_SPEED_MPS))
    best_speed = None
    best_error = float("inf")
    lo = -GT_MAX_STEP_SPEED_MPS
    hi = GT_MAX_STEP_SPEED_MPS
    for _ in range(36):
        bias = (lo + hi) * 0.5
        speed = _clamp_speed_profile_by_acceleration(desired_speed + bias, initial_speed)
        distance = float(speed.sum() * DT_SECONDS)
        error = abs(distance - total_length)
        if error < best_error:
            best_error = error
            best_speed = speed
        if distance < total_length:
            lo = bias
        else:
            hi = bias

    if best_speed is None or best_error > max(0.25, total_length * 0.02):
        return points.copy()

    target_distances = np.maximum.accumulate(np.clip(np.cumsum(best_speed * DT_SECONDS), 0.0, total_length))
    if abs(float(target_distances[-1]) - total_length) <= max(0.25, total_length * 0.02):
        target_distances[-1] = total_length
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return _interp_polyline_by_distance(polyline, cumulative, target_distances)


def smooth_variants_from_t0_speed(variants: np.ndarray, initial_speed_mps: float | None) -> np.ndarray:
    if initial_speed_mps is None or not np.isfinite(initial_speed_mps) or len(variants) == 0:
        return variants
    return np.stack(
        [smooth_xy_from_t0_speed(variant, initial_speed_mps) for variant in variants],
        axis=0,
    )


def temporal_reference_paths(compare_xy: np.ndarray, horizons: Iterable[int], steps: int) -> list[tuple[int, np.ndarray]]:
    refs: list[tuple[int, np.ndarray]] = []
    seen: set[int] = set()
    for horizon in horizons:
        horizon = int(horizon)
        if horizon in seen or horizon <= 1 or horizon > len(compare_xy):
            continue
        seen.add(horizon)
        refs.append((horizon, resample_by_time(compare_xy[:horizon], steps)))
    if not refs:
        refs.append((min(steps, len(compare_xy)), resample_by_time(compare_xy[:steps], steps)))
    return refs


def sampled_temporal_horizons(row: TrajectoryRow, args: argparse.Namespace) -> list[int]:
    lo = max(2, int(args.temporal_match_min))
    hi = min(int(args.temporal_match_max), len(row.compare_xy), int(args.compare_steps))
    if hi < lo:
        return [min(args.steps, len(row.compare_xy))]

    choices = np.arange(lo, hi + 1, dtype=np.int64)
    sample_count = min(max(1, int(args.temporal_match_samples)), len(choices))
    seed = (
        int(args.seed)
        + int(row.traj_id) * 1_000_003
        + int(row.t0_index) * 97
        + (int(row.t0_us) & 0xFFFF)
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    sampled = rng.choice(choices, size=sample_count, replace=False)
    return sorted(int(v) for v in sampled)


def max_run(mask: np.ndarray) -> int:
    best = 0
    run = 0
    for value in mask:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def curvature_sign_sequence(
    xy: np.ndarray,
    threshold: float = 0.01,
    min_run_steps: int = 3,
) -> tuple[int, ...]:
    _, _, curvature = trajectory_dynamics(xy)
    raw = np.zeros(len(curvature), dtype=np.int8)
    raw[curvature >= threshold] = 1
    raw[curvature <= -threshold] = -1

    seq: list[int] = []
    run_sign = 0
    run_len = 0
    for value in raw:
        sign = int(value)
        if sign == 0:
            if run_sign and run_len >= min_run_steps:
                if not seq or seq[-1] != run_sign:
                    seq.append(run_sign)
            run_sign = 0
            run_len = 0
            continue
        if sign == run_sign:
            run_len += 1
        else:
            if run_sign and run_len >= min_run_steps:
                if not seq or seq[-1] != run_sign:
                    seq.append(run_sign)
            run_sign = sign
            run_len = 1
    if run_sign and run_len >= min_run_steps:
        if not seq or seq[-1] != run_sign:
            seq.append(run_sign)
    return tuple(seq)


def curvature_trend_ok(
    category: str,
    candidate_xy: np.ndarray,
    gt_refs: list[tuple[int, np.ndarray]],
    threshold: float = 0.01,
    min_run_steps: int = 3,
) -> bool:
    return curvature_trend_sequence_ok(
        category,
        curvature_sign_sequence(candidate_xy, threshold, min_run_steps),
        [curvature_sign_sequence(ref_xy, threshold, min_run_steps) for _, ref_xy in gt_refs],
    )


def curvature_trend_sequence_ok(
    category: str,
    cand_seq: tuple[int, ...],
    gt_sequences: list[tuple[int, ...]],
) -> bool:
    def first_sign_change(seq: tuple[int, ...]) -> tuple[int, int] | None:
        for prev, curr in zip(seq, seq[1:]):
            if prev != curr:
                return int(prev), int(curr)
        return None

    non_empty_gt = [seq for seq in gt_sequences if len(seq) > 0]
    if category == "s_curve":
        cand_change = first_sign_change(cand_seq)
        gt_changes = {first_sign_change(seq) for seq in non_empty_gt}
        gt_changes.discard(None)
        if gt_changes:
            return cand_change in gt_changes
        return cand_change is not None

    if non_empty_gt:
        return cand_seq in non_empty_gt

    if category == "straight":
        return len(cand_seq) == 0
    if category == "left":
        return bool(cand_seq) and all(sign >= 0 for sign in cand_seq)
    if category == "right":
        return bool(cand_seq) and all(sign <= 0 for sign in cand_seq)
    return True


def fallback_category(xy: np.ndarray) -> str:
    speed, _, curvature = trajectory_dynamics(xy)
    final_forward = float(xy[-1, 0])
    final_right = -float(xy[-1, 1])
    if final_forward < 2.0 or float(np.nanmean(speed)) < 0.35:
        return "stop"
    abs_curv_sum = float(np.abs(curvature).sum())
    final_lateral = abs(float(xy[-1, 1]))
    max_lateral = float(np.max(np.abs(xy[:, 1])))
    slope = final_lateral / max(final_forward, 1e-6)
    initial_speed = float(speed[0]) if len(speed) else 0.0
    gentle_high_speed_straight = (
        initial_speed >= STRAIGHT_SPEED_MPS
        and final_forward >= STRAIGHT_MIN_FORWARD_M
        and final_lateral <= STRAIGHT_MAX_FINAL_LATERAL_M
        and max_lateral <= STRAIGHT_MAX_LATERAL_M
        and slope <= STRAIGHT_MAX_SLOPE
    )
    pos_mask = curvature >= 0.01
    neg_mask = curvature <= -0.01
    significant = curvature[pos_mask | neg_mask]
    has_sign_change = (
        max_run(pos_mask) >= 3
        and max_run(neg_mask) >= 3
        and len(significant) > 1
        and bool(np.any(np.sign(significant[:-1]) * np.sign(significant[1:]) < 0))
    )
    if has_sign_change and abs_curv_sum > 0.15:
        return "s_curve"
    if abs_curv_sum <= 0.15 or gentle_high_speed_straight:
        return "straight"
    return "right" if final_right >= 0.0 else "left"


def load_centers(path: Path, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids: list[int] = []
    counts: list[int] = []
    centers: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "CENTER":
                continue
            xy = np.asarray(parts[3:], dtype=np.float64).reshape(steps, 2)
            ids.append(int(parts[1]))
            counts.append(int(parts[2]))
            centers.append(xy)
    if not centers:
        raise ValueError(f"No centers found in {path}")
    order = np.argsort(np.asarray(ids, dtype=np.int64))
    return (
        np.asarray(ids, dtype=np.int64)[order],
        np.asarray(counts, dtype=np.int64)[order],
        np.stack(centers, axis=0)[order],
    )


def parameter_grid(category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if category == "straight":
        return (
            np.asarray([0.94, 1.0, 1.06], dtype=np.float64),
            np.asarray([-0.10, 0.0, 0.10], dtype=np.float64),
            np.asarray([0.85, 1.0, 1.15], dtype=np.float64),
        )
    if category == "stop":
        return (
            np.asarray([0.70, 0.85, 1.0, 1.15], dtype=np.float64),
            np.asarray([-0.20, 0.0, 0.20], dtype=np.float64),
            np.asarray([0.80, 1.0, 1.20], dtype=np.float64),
        )
    if category in {"left", "right"}:
        return (
            np.asarray([0.80, 0.90, 1.0, 1.10, 1.20], dtype=np.float64),
            np.asarray([-0.25, -0.10, 0.0, 0.10, 0.25], dtype=np.float64),
            np.asarray([0.60, 0.75, 0.90, 1.0, 1.10, 1.25, 1.45, 1.65], dtype=np.float64),
        )
    return (
        np.asarray([0.80, 0.90, 1.0, 1.10, 1.20], dtype=np.float64),
        np.asarray([-0.25, -0.10, 0.0, 0.10, 0.25], dtype=np.float64),
        np.asarray([0.60, 0.75, 0.90, 1.0, 1.10, 1.25, 1.45], dtype=np.float64),
    )


def transform_center(
    center_xy: np.ndarray,
    speed_scale: float,
    accel_ramp: float,
    curvature_scale: float,
) -> np.ndarray:
    origin_xy = np.vstack([np.zeros((1, 2), dtype=np.float64), center_xy])
    delta = np.diff(origin_xy, axis=0)
    step_distance = np.linalg.norm(delta, axis=1)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    moving = step_distance > 1e-4
    if moving.any():
        first = int(np.flatnonzero(moving)[0])
        yaw[:first] = yaw[first]
        for idx in range(first + 1, len(yaw)):
            if not moving[idx]:
                yaw[idx] = yaw[idx - 1]
        yaw = np.unwrap(yaw)
    else:
        yaw[:] = 0.0

    tau = np.linspace(-0.5, 0.5, len(step_distance), dtype=np.float64)
    speed_factor = np.maximum(0.05, speed_scale * (1.0 + accel_ramp * tau))
    scaled_distance = step_distance * speed_factor
    first_yaw = float(yaw[0])
    scaled_yaw = first_yaw + curvature_scale * (yaw - first_yaw)
    delta_scaled = np.column_stack(
        [scaled_distance * np.cos(scaled_yaw), scaled_distance * np.sin(scaled_yaw)]
    )
    return np.cumsum(delta_scaled, axis=0)


def build_library(
    kmeans_dir: Path,
    category: str,
    steps: int,
    curvature_sign_threshold: float,
    curvature_sign_min_run_steps: int,
) -> CenterLibrary:
    center_ids, counts, centers = load_centers(kmeans_dir / CATEGORY_FILES[category], steps)
    speed_scales, accel_ramps, curvature_scales = parameter_grid(category)

    variants: list[np.ndarray] = []
    variant_sign_sequences: list[tuple[int, ...]] = []
    variant_center_ids: list[int] = []
    variant_speed_scales: list[float] = []
    variant_accel_ramps: list[float] = []
    variant_curvature_scales: list[float] = []
    for center_id, center in zip(center_ids, centers):
        for speed_scale in speed_scales:
            for accel_ramp in accel_ramps:
                for curvature_scale in curvature_scales:
                    variant = transform_center(center, speed_scale, accel_ramp, curvature_scale)
                    variants.append(variant)
                    variant_sign_sequences.append(
                        curvature_sign_sequence(
                            variant,
                            curvature_sign_threshold,
                            curvature_sign_min_run_steps,
                        )
                    )
                    variant_center_ids.append(int(center_id))
                    variant_speed_scales.append(float(speed_scale))
                    variant_accel_ramps.append(float(accel_ramp))
                    variant_curvature_scales.append(float(curvature_scale))

    return CenterLibrary(
        ids=center_ids,
        counts=counts,
        centers=centers,
        center_residuals=np.stack([residual_shape(center) for center in centers], axis=0),
        variants=np.stack(variants, axis=0),
        variant_residuals=np.stack([residual_shape(variant) for variant in variants], axis=0),
        variant_sign_sequences=variant_sign_sequences,
        variant_center_ids=np.asarray(variant_center_ids, dtype=np.int64),
        speed_scales=np.asarray(variant_speed_scales, dtype=np.float64),
        accel_ramps=np.asarray(variant_accel_ramps, dtype=np.float64),
        curvature_scales=np.asarray(variant_curvature_scales, dtype=np.float64),
    )


def residual_shape(xy: np.ndarray) -> np.ndarray:
    t = np.linspace(1.0 / len(xy), 1.0, len(xy), dtype=np.float64)
    return xy - t[:, None] * xy[-1]


def local_tangent_normal(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step tangent and left-normal in the reference path's local frame."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    prev_xy = np.vstack([np.zeros((1, 2), dtype=np.float64), xy[:-1]])
    next_xy = np.vstack([xy[1:], xy[-1:]])
    tangent = next_xy - prev_xy
    norm = np.linalg.norm(tangent, axis=1)

    valid = norm > 1e-4
    if valid.any():
        first = int(np.flatnonzero(valid)[0])
        tangent[:first] = tangent[first]
        norm[:first] = norm[first]
        for idx in range(first + 1, len(tangent)):
            if norm[idx] <= 1e-4:
                tangent[idx] = tangent[idx - 1]
                norm[idx] = norm[idx - 1]
    else:
        tangent[:, 0] = 1.0
        tangent[:, 1] = 0.0
        norm[:] = 1.0

    tangent = tangent / np.maximum(norm[:, None], 1e-6)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return tangent, normal


def heading_error_many(candidates: np.ndarray, gt: np.ndarray) -> np.ndarray:
    gt_delta = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float64), gt]), axis=0)
    cand_delta = np.diff(
        np.concatenate([np.zeros((len(candidates), 1, 2), dtype=np.float64), candidates], axis=1),
        axis=1,
    )
    gt_norm = np.linalg.norm(gt_delta, axis=1)
    cand_norm = np.linalg.norm(cand_delta, axis=2)
    moving = (gt_norm[None, :] > 1e-3) & (cand_norm > 1e-3)
    dot = np.einsum("vnd,nd->vn", cand_delta, gt_delta)
    denom = np.maximum(cand_norm * gt_norm[None, :], 1e-6)
    cos_sim = np.clip(dot / denom, -1.0, 1.0)
    summed = np.sum((1.0 - cos_sim) * moving, axis=1)
    counts = np.maximum(np.sum(moving, axis=1), 1)
    return summed / counts


def category_settings(category: str, path_length: float) -> dict[str, float | int]:
    if category == "straight":
        return {
            "preselect": 4,
            "max_keep": 2,
            "endpoint_abs": 2.0,
            "endpoint_rel": 0.06,
            "mean_abs": 0.80,
            "mean_rel": 0.035,
            "shape_abs": 0.65,
            "shape_rel": 0.030,
            "heading": 0.12,
            "endpoint_long": 4.0,
            "endpoint_lat": 0.45,
            "lateral_mean": 0.28,
            "lateral_max": 0.50,
            "lateral_shape": 0.24,
            "min_gt_mean": 0.65,
            "min_gt_endpoint": 0.90,
            "min_gt_shape": 0.32,
            "div_mean": 0.70,
            "div_endpoint": 0.75,
            "div_shape": 0.40,
        }
    if category == "stop":
        return {
            "preselect": 6,
            "max_keep": 3,
            "endpoint_abs": 1.4,
            "endpoint_rel": 0.25,
            "mean_abs": 0.65,
            "mean_rel": 0.20,
            "shape_abs": 0.60,
            "shape_rel": 0.15,
            "heading": 0.50,
            "endpoint_long": 1.50,
            "endpoint_lat": 0.35,
            "lateral_mean": 0.20,
            "lateral_max": 0.40,
            "lateral_shape": 0.18,
            "min_gt_mean": 0.25,
            "min_gt_endpoint": 0.30,
            "min_gt_shape": 0.14,
            "div_mean": 0.32,
            "div_endpoint": 0.35,
            "div_shape": 0.22,
        }
    if category == "s_curve":
        return {
            "preselect": 12,
            "max_keep": 6,
            "endpoint_abs": 3.0,
            "endpoint_rel": 0.13,
            "mean_abs": 1.60,
            "mean_rel": 0.070,
            "shape_abs": 1.35,
            "shape_rel": 0.060,
            "heading": 0.35,
            "endpoint_long": 4.50,
            "endpoint_lat": 0.50,
            "lateral_mean": 0.35,
            "lateral_max": 0.60,
            "lateral_shape": 0.35,
            "min_gt_mean": 0.85,
            "min_gt_endpoint": 1.20,
            "min_gt_shape": 0.60,
            "div_mean": 0.95,
            "div_endpoint": 1.05,
            "div_shape": 0.65,
        }
    return {
        "preselect": 10,
        "max_keep": 6,
        "endpoint_abs": 2.6,
        "endpoint_rel": 0.11,
        "mean_abs": 1.60,
        "mean_rel": 0.060,
        "shape_abs": 1.25,
        "shape_rel": 0.050,
        "heading": 0.30,
        "endpoint_long": 4.50,
        "endpoint_lat": 0.50,
        "lateral_mean": 0.35,
        "lateral_max": 0.60,
        "lateral_shape": 0.35,
        "min_gt_mean": 0.75,
        "min_gt_endpoint": 1.10,
        "min_gt_shape": 0.55,
        "div_mean": 0.85,
        "div_endpoint": 0.95,
        "div_shape": 0.58,
    }


def apply_arg_overrides(settings: dict[str, float | int], args: argparse.Namespace, category: str) -> None:
    if category == "straight":
        settings["preselect"] = args.preselect_straight
        settings["max_keep"] = args.max_straight
    elif category == "stop":
        settings["preselect"] = args.preselect_stop
        settings["max_keep"] = args.max_stop
    elif category == "s_curve":
        settings["preselect"] = args.preselect_s_curve
        settings["max_keep"] = args.max_s_curve
    else:
        settings["preselect"] = args.preselect_turn
        settings["max_keep"] = args.max_turn


def select_center_ids(gt_xy: np.ndarray, library: CenterLibrary, preselect: int) -> set[int]:
    if int(preselect) <= 0:
        return set()
    center_endpoint = np.linalg.norm(library.centers[:, -1] - gt_xy[-1], axis=1)
    center_mean = np.linalg.norm(library.centers - gt_xy[None, :, :], axis=2).mean(axis=1)
    center_shape = np.linalg.norm(
        library.center_residuals - residual_shape(gt_xy)[None, :, :],
        axis=2,
    ).mean(axis=1)
    score = 1.2 * center_endpoint + 0.8 * center_mean + 0.7 * center_shape
    count = min(max(1, int(preselect)), len(score))
    return set(int(library.ids[idx]) for idx in np.argsort(score)[:count])


def optimized_candidates(
    row: TrajectoryRow,
    category: str,
    library: CenterLibrary,
    args: argparse.Namespace,
) -> list[dict]:
    gt_xy = row.xy
    step_distance = np.linalg.norm(
        np.diff(np.vstack([np.zeros((1, 2), dtype=np.float64), gt_xy]), axis=0),
        axis=1,
    )
    path_length = float(step_distance.sum())
    settings = category_settings(category, path_length)
    apply_arg_overrides(settings, args, category)
    if int(settings["max_keep"]) <= 0 or int(settings["preselect"]) <= 0:
        return []
    temporal_horizons = sampled_temporal_horizons(row, args)
    gt_refs = temporal_reference_paths(row.compare_xy, temporal_horizons, args.steps)

    selected_center_ids = select_center_ids(gt_xy, library, int(settings["preselect"]))
    variant_mask = np.asarray(
        [int(center_id) in selected_center_ids for center_id in library.variant_center_ids],
        dtype=bool,
    )
    variants = library.variants[variant_mask]
    if len(variants) == 0:
        return []
    variants = smooth_variants_from_t0_speed(variants, row.initial_speed_mps)
    variant_residuals = np.stack([residual_shape(variant) for variant in variants], axis=0)
    variant_sign_sequences = [
        curvature_sign_sequence(
            variant,
            float(args.curvature_sign_threshold),
            int(args.curvature_sign_min_run_steps),
        )
        for variant in variants
    ]

    candidate_center_ids = library.variant_center_ids[variant_mask]
    speed_scales = library.speed_scales[variant_mask]
    accel_ramps = library.accel_ramps[variant_mask]
    curvature_scales = library.curvature_scales[variant_mask]

    best_score = np.full(len(variants), np.inf, dtype=np.float64)
    best_horizon = np.zeros(len(variants), dtype=np.int64)
    endpoint_dist = np.full(len(variants), np.inf, dtype=np.float64)
    point_mean_dist = np.full(len(variants), np.inf, dtype=np.float64)
    shape_dist = np.full(len(variants), np.inf, dtype=np.float64)
    heading_dist = np.full(len(variants), np.inf, dtype=np.float64)
    endpoint_long = np.full(len(variants), np.inf, dtype=np.float64)
    endpoint_lat = np.full(len(variants), np.inf, dtype=np.float64)
    lateral_mean = np.full(len(variants), np.inf, dtype=np.float64)
    lateral_max = np.full(len(variants), np.inf, dtype=np.float64)
    lateral_shape = np.full(len(variants), np.inf, dtype=np.float64)

    for horizon, ref_xy in gt_refs:
        ref_residual = residual_shape(ref_xy)
        ref_tangent, ref_normal = local_tangent_normal(ref_xy)
        ref_delta = variants - ref_xy[None, :, :]
        ref_longitudinal = np.abs(np.einsum("vnd,nd->vn", ref_delta, ref_tangent))
        ref_lateral = np.abs(np.einsum("vnd,nd->vn", ref_delta, ref_normal))
        ref_shape_delta = variant_residuals - ref_residual[None, :, :]
        ref_lateral_shape_values = np.abs(
            np.einsum("vnd,nd->vn", ref_shape_delta, ref_normal)
        )
        ref_endpoint_long = ref_longitudinal[:, -1]
        ref_endpoint_lat = ref_lateral[:, -1]
        ref_lateral_mean = ref_lateral.mean(axis=1)
        ref_lateral_max = ref_lateral.max(axis=1)
        ref_point_mean = np.linalg.norm(variants - ref_xy[None, :, :], axis=2).mean(axis=1)
        ref_shape = np.linalg.norm(variant_residuals - ref_residual[None, :, :], axis=2).mean(axis=1)
        ref_lateral_shape = ref_lateral_shape_values.mean(axis=1)
        ref_heading = heading_error_many(variants, ref_xy)
        ref_endpoint_dist = np.sqrt((0.35 * ref_endpoint_long) ** 2 + ref_endpoint_lat ** 2)
        ref_score = (
            0.55 * ref_endpoint_long
            + 2.4 * ref_endpoint_lat
            + 1.8 * ref_lateral_mean
            + 0.8 * ref_point_mean
            + 1.0 * ref_shape
            + 1.2 * ref_heading * max(path_length, 1.0) / max(len(gt_xy), 1)
        )
        better = ref_score < best_score
        best_score[better] = ref_score[better]
        best_horizon[better] = int(horizon)
        endpoint_dist[better] = ref_endpoint_dist[better]
        point_mean_dist[better] = ref_point_mean[better]
        shape_dist[better] = ref_shape[better]
        heading_dist[better] = ref_heading[better]
        endpoint_long[better] = ref_endpoint_long[better]
        endpoint_lat[better] = ref_endpoint_lat[better]
        lateral_mean[better] = ref_lateral_mean[better]
        lateral_max[better] = ref_lateral_max[better]
        lateral_shape[better] = ref_lateral_shape[better]

    gt_sign_sequences = [
        curvature_sign_sequence(
            ref_xy,
            float(args.curvature_sign_threshold),
            int(args.curvature_sign_min_run_steps),
        )
        for _, ref_xy in gt_refs
    ]
    trend_mask = np.asarray(
        [
            curvature_trend_sequence_ok(category, candidate_seq, gt_sign_sequences)
            for candidate_seq in variant_sign_sequences
        ],
        dtype=bool,
    )
    score = (
        best_score
        + 2.0 * np.maximum(0.0, lateral_max - float(settings["lateral_max"]))
    )

    endpoint_limit = max(float(settings["endpoint_abs"]), float(settings["endpoint_rel"]) * max(path_length, 1.0))
    mean_limit = max(float(settings["mean_abs"]), float(settings["mean_rel"]) * max(path_length, 1.0))
    shape_limit = max(float(settings["shape_abs"]), float(settings["shape_rel"]) * max(path_length, 1.0))
    heading_limit = float(settings["heading"])
    endpoint_lat_limit = np.full(len(variants), float(settings["endpoint_lat"]), dtype=np.float64)
    lateral_mean_limit = np.full(len(variants), float(settings["lateral_mean"]), dtype=np.float64)
    lateral_max_limit = np.full(len(variants), float(settings["lateral_max"]), dtype=np.float64)
    lateral_shape_limit = np.full(len(variants), float(settings["lateral_shape"]), dtype=np.float64)
    turn_inside_offset = np.zeros(len(variants), dtype=np.float64)
    turn_inside_ok = np.ones(len(variants), dtype=bool)
    if category in {"left", "right"}:
        turn_sign = 1.0 if category == "left" else -1.0
        turn_inside_offset = turn_sign * (variants[:, -1, 1] - gt_xy[-1, 1])
        turn_inside_mask = turn_inside_offset > 0.05
        endpoint_lat_limit[turn_inside_mask] = min(0.75, float(settings["endpoint_lat"]) * 1.50)
        lateral_mean_limit[turn_inside_mask] = min(0.48, float(settings["lateral_mean"]) * 1.35)
        lateral_max_limit[turn_inside_mask] = min(0.82, float(settings["lateral_max"]) * 1.35)
        lateral_shape_limit[turn_inside_mask] = min(0.48, float(settings["lateral_shape"]) * 1.35)
        turn_inside_ok = (turn_inside_offset >= -0.75) & (
            ~turn_inside_mask | (turn_inside_offset <= 0.75)
        )
    keep_mask = (
        (endpoint_dist <= endpoint_limit)
        & (endpoint_long <= float(settings["endpoint_long"]))
        & (endpoint_lat <= endpoint_lat_limit)
        & (lateral_mean <= lateral_mean_limit)
        & (lateral_max <= lateral_max_limit)
        & (lateral_shape <= lateral_shape_limit)
        & (point_mean_dist <= mean_limit)
        & (point_mean_dist > float(args.min_diversity_mean_dist))
        & (shape_dist <= shape_limit)
        & (heading_dist <= heading_limit)
        & trend_mask
        & turn_inside_ok
    )
    overlaps_gt = (
        (point_mean_dist < float(settings["min_gt_mean"]))
        | (
            (endpoint_dist < float(settings["min_gt_endpoint"]))
            & (shape_dist < float(settings["min_gt_shape"]))
        )
    )
    keep_mask &= ~overlaps_gt

    kept: list[dict] = []
    max_keep = int(settings["max_keep"])

    def try_keep_candidate(idx: int) -> bool:
        idx = int(idx)
        if not bool(keep_mask[idx]):
            return False
        candidate = variants[idx]
        for existing in kept:
            existing_xy = existing["xy"]
            mean_sep = float(np.linalg.norm(candidate - existing_xy, axis=1).mean())
            endpoint_sep = float(np.linalg.norm(candidate[-1] - existing_xy[-1]))
            shape_sep = float(
                np.linalg.norm(residual_shape(candidate) - residual_shape(existing_xy), axis=1).mean()
            )
            if (
                mean_sep <= float(args.min_pairwise_mean_dist)
                or mean_sep < float(settings["div_mean"])
                or (
                    endpoint_sep < float(settings["div_endpoint"])
                    and shape_sep < float(settings["div_shape"])
                )
            ):
                return False
        kept.append(
            {
                "xy": candidate,
                "center_id": int(candidate_center_ids[idx]),
                "speed_scale": float(speed_scales[idx]),
                "accel_ramp": float(accel_ramps[idx]),
                "curvature_scale": float(curvature_scales[idx]),
                "endpoint_dist": float(endpoint_dist[idx]),
                "endpoint_long": float(endpoint_long[idx]),
                "endpoint_lat": float(endpoint_lat[idx]),
                "mean_dist": float(point_mean_dist[idx]),
                "shape_dist": float(shape_dist[idx]),
                "lateral_mean": float(lateral_mean[idx]),
                "lateral_max": float(lateral_max[idx]),
                "lateral_shape": float(lateral_shape[idx]),
                "heading_dist": float(heading_dist[idx]),
                "best_horizon": int(best_horizon[idx]),
                "trend_sequence": "/".join(str(v) for v in variant_sign_sequences[idx]),
                "score": float(score[idx]),
            }
        )
        return True

    order = np.argsort(np.where(keep_mask, score, np.inf))
    valid_order = [int(idx) for idx in order if bool(keep_mask[int(idx)])]
    if category in {"left", "right"}:
        # Prefer a mild majority of candidates that end slightly inside the GT turn:
        # larger y for left turns, smaller y for right turns. Existing endpoint and
        # lateral thresholds still cap how aggressive these candidates can be.
        target_inside = int(np.ceil(max_keep * 0.60))
        target_outside = max_keep - target_inside

        inside_kept = 0
        turn_score = score - 1.4 * np.clip(turn_inside_offset, 0.0, 0.75)
        inside_order = [
            int(idx)
            for idx in np.argsort(np.where(keep_mask & (turn_inside_offset > 0.05), turn_score, np.inf))
            if bool(keep_mask[int(idx)]) and float(turn_inside_offset[int(idx)]) > 0.05
        ]
        for idx in inside_order:
            if inside_kept >= target_inside:
                break
            if try_keep_candidate(idx):
                inside_kept += 1

        outside_kept = 0
        for idx in valid_order:
            if len(kept) >= max_keep or outside_kept >= target_outside:
                break
            if float(turn_inside_offset[idx]) <= 0.05 and try_keep_candidate(idx):
                outside_kept += 1

        for idx in valid_order:
            if len(kept) >= max_keep:
                break
            try_keep_candidate(idx)
    else:
        for idx in valid_order:
            if len(kept) >= max_keep:
                break
            try_keep_candidate(idx)
    return kept


def trajectory_to_parquet_row(
    row: TrajectoryRow,
    xy: np.ndarray,
    sample_idx: int,
    source: str,
) -> dict:
    xyz = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
    prev = np.vstack([np.zeros((1, 3), dtype=np.float64), xyz[:-1]])
    velocity = (xyz - prev) / DT_SECONDS
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
    qw = np.cos(yaw / 2.0)
    qz = np.sin(yaw / 2.0)
    return {
        "t0_us": int(row.t0_us),
        "sample_idx": int(sample_idx),
        "source": source,
        "timestamp": [int(row.t0_us) + int((i + 1) * DT_SECONDS * 1_000_000) for i in range(len(xy))],
        "qx": [0.0] * len(xy),
        "qy": [0.0] * len(xy),
        "qz": qz.tolist(),
        "qw": qw.tolist(),
        "x": xyz[:, 0].tolist(),
        "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "vx": velocity[:, 0].tolist(),
        "vy": velocity[:, 1].tolist(),
        "vz": velocity[:, 2].tolist(),
        "curvature": np.gradient(yaw).tolist(),
    }


def write_outputs(
    output_dir: Path,
    parquet_rows: Iterable[dict],
    summary_rows: list[dict],
    summary_csv: Path,
) -> None:
    by_clip: dict[tuple[str, str], list[dict]] = {}
    for row in parquet_rows:
        by_clip.setdefault((row.pop("_dataset"), row.pop("_clip")), []).append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for (dataset, clip), rows in sorted(by_clip.items()):
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows).sort_values(["t0_us", "sample_idx"])
        out_file = dataset_dir / f"{clip}.egomotion.parquet"
        df.to_parquet(out_file, index=False)
        total_rows += len(df)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "traj_id",
                "dataset",
                "clip",
                "t0_us",
                "t0_index",
                "category",
                "num_kept",
                "center_id",
                "sample_idx",
                "speed_scale",
                "accel_ramp",
                "curvature_scale",
                "endpoint_dist",
                "endpoint_long",
                "endpoint_lat",
                "mean_dist",
                "shape_dist",
                "lateral_mean",
                "lateral_max",
                "lateral_shape",
                "heading_dist",
                "best_horizon",
                "trend_sequence",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(by_clip)} parquet files / {total_rows} rows to {output_dir}")
    print(f"Wrote summary to {summary_csv}")


def main() -> None:
    args = parse_args()
    future_txt = export_target_future_rows(args)
    rows = load_future_rows(future_txt, args.steps)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    rows = estimate_t0_speed_from_train_data(args.train_data_root, rows)

    split_categories_by_id, split_categories_by_key = load_split_categories(args.kmeans_dir)
    use_split_id_mapping = not bool(args.datasets.strip())
    libraries = {
        category: build_library(
            args.kmeans_dir,
            category,
            args.steps,
            float(args.curvature_sign_threshold),
            int(args.curvature_sign_min_run_steps),
        )
        for category in CATEGORY_FILES
    }

    parquet_rows: list[dict] = []
    summary_rows: list[dict] = []
    category_counts: dict[str, int] = {category: 0 for category in CATEGORY_FILES}
    kept_counts: dict[str, int] = {category: 0 for category in CATEGORY_FILES}

    for row_idx, row in enumerate(rows, start=1):
        row_key = (row.dataset, row.clip, int(row.t0_us), int(row.t0_index))
        category = split_categories_by_key.get(row_key)
        if category is None and use_split_id_mapping:
            category = split_categories_by_id.get(row.traj_id)
        if category is None:
            category = fallback_category(row.xy)
        category_counts[category] = category_counts.get(category, 0) + 1
        candidates = optimized_candidates(row, category, libraries[category], args)
        kept_counts[category] = kept_counts.get(category, 0) + len(candidates)

        sample_idx = 0
        if args.include_gt:
            gt_row = trajectory_to_parquet_row(row, row.xy, sample_idx, "gt")
            gt_row["_dataset"] = row.dataset
            gt_row["_clip"] = row.clip
            parquet_rows.append(gt_row)
            sample_idx += 1

        if candidates:
            for candidate in candidates:
                pred_row = trajectory_to_parquet_row(row, candidate["xy"], sample_idx, "rule_cluster")
                pred_row["_dataset"] = row.dataset
                pred_row["_clip"] = row.clip
                parquet_rows.append(pred_row)
                summary_rows.append(
                    {
                        "traj_id": row.traj_id,
                        "dataset": row.dataset,
                        "clip": row.clip,
                        "t0_us": row.t0_us,
                        "t0_index": row.t0_index,
                        "category": category,
                        "num_kept": len(candidates),
                        "center_id": candidate["center_id"],
                        "sample_idx": sample_idx,
                        "speed_scale": candidate["speed_scale"],
                        "accel_ramp": candidate["accel_ramp"],
                        "curvature_scale": candidate["curvature_scale"],
                        "endpoint_dist": candidate["endpoint_dist"],
                        "endpoint_long": candidate["endpoint_long"],
                        "endpoint_lat": candidate["endpoint_lat"],
                        "mean_dist": candidate["mean_dist"],
                        "shape_dist": candidate["shape_dist"],
                        "lateral_mean": candidate["lateral_mean"],
                        "lateral_max": candidate["lateral_max"],
                        "lateral_shape": candidate["lateral_shape"],
                        "heading_dist": candidate["heading_dist"],
                        "best_horizon": candidate["best_horizon"],
                        "trend_sequence": candidate["trend_sequence"],
                        "score": candidate["score"],
                    }
                )
                sample_idx += 1
        else:
            summary_rows.append(
                {
                    "traj_id": row.traj_id,
                    "dataset": row.dataset,
                    "clip": row.clip,
                    "t0_us": row.t0_us,
                    "t0_index": row.t0_index,
                    "category": category,
                    "num_kept": 0,
                    "center_id": "",
                    "sample_idx": "",
                    "speed_scale": "",
                    "accel_ramp": "",
                    "curvature_scale": "",
                    "endpoint_dist": "",
                    "endpoint_long": "",
                    "endpoint_lat": "",
                    "mean_dist": "",
                    "shape_dist": "",
                    "lateral_mean": "",
                    "lateral_max": "",
                    "lateral_shape": "",
                    "heading_dist": "",
                    "best_horizon": "",
                    "trend_sequence": "",
                    "score": "",
                }
            )

        if row_idx % 1000 == 0:
            print(f"Processed {row_idx}/{len(rows)} GT rows")

    summary_csv = args.summary_csv or (args.output_dir / "rule_based_gt_diversity_summary.csv")
    write_outputs(args.output_dir, parquet_rows, summary_rows, summary_csv)
    print("GT category counts:", category_counts)
    print("Kept rule trajectory counts:", kept_counts)


if __name__ == "__main__":
    main()
