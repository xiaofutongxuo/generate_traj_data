#!/usr/bin/env python3
"""Export GT future xy trajectories from train_data and cluster with K-Means."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
from sklearn.cluster import KMeans


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_ROOT = (
    SCRIPT_DIR.parent.parent
    / "triplane_tokenization"
    / "data_cache"
    / "alpamayo_extracted"
)
DEFAULT_DATA_TXT = SCRIPT_DIR / "future_trajectories_xy.txt"
DEFAULT_RESULT_TXT = SCRIPT_DIR / "kmeans_results.txt"
DEFAULT_STOP_RESULT_TXT = SCRIPT_DIR / "kmeans_stop_results.txt"
DEFAULT_STOP_SPLIT_TXT = SCRIPT_DIR / "stop_trajectories_xy.txt"
DEFAULT_STRAIGHT_SPLIT_TXT = SCRIPT_DIR / "straight_trajectories_xy.txt"
DEFAULT_TURN_SPLIT_TXT = SCRIPT_DIR / "turn_trajectories_xy.txt"
DEFAULT_S_CURVE_SPLIT_TXT = SCRIPT_DIR / "s_curve_trajectories_xy.txt"
DEFAULT_STOP_CENTER_TXT = SCRIPT_DIR / "stop.txt"
DEFAULT_STRAIGHT_CENTER_TXT = SCRIPT_DIR / "straight.txt"
DEFAULT_LEFT_CENTER_TXT = SCRIPT_DIR / "left.txt"
DEFAULT_RIGHT_CENTER_TXT = SCRIPT_DIR / "right.txt"
DEFAULT_S_CURVE_CENTER_TXT = SCRIPT_DIR / "s_curve.txt"


@dataclass
class TrajectoryRow:
    traj_id: int
    dataset: str
    clip: str
    t0_us: int
    sample_idx: int
    xy: np.ndarray


@dataclass
class ExportStats:
    kept: int = 0
    skipped_short_or_bad: int = 0
    skipped_slow: int = 0
    skipped_backward: int = 0
    skipped_acc: int = 0
    skipped_jump: int = 0
    max_speed_mps: float = 0.0
    max_acc_mps2: float = 0.0
    min_forward_acc_mps2: float = float("inf")
    max_forward_acc_mps2: float = float("-inf")
    max_raw_position_step_m: float = 0.0
    max_repaired_position_step_m: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GT future xy trajectories from train_data and run K-Means."
    )
    parser.add_argument("--train_data_root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--data_txt", type=Path, default=DEFAULT_DATA_TXT)
    parser.add_argument("--result_txt", type=Path, default=DEFAULT_RESULT_TXT)
    parser.add_argument("--stop_result_txt", type=Path, default=DEFAULT_STOP_RESULT_TXT)
    parser.add_argument("--stop_split_txt", type=Path, default=DEFAULT_STOP_SPLIT_TXT)
    parser.add_argument("--straight_split_txt", type=Path, default=DEFAULT_STRAIGHT_SPLIT_TXT)
    parser.add_argument("--turn_split_txt", type=Path, default=DEFAULT_TURN_SPLIT_TXT)
    parser.add_argument("--s_curve_split_txt", type=Path, default=DEFAULT_S_CURVE_SPLIT_TXT)
    parser.add_argument("--stop_center_txt", type=Path, default=DEFAULT_STOP_CENTER_TXT)
    parser.add_argument("--straight_center_txt", type=Path, default=DEFAULT_STRAIGHT_CENTER_TXT)
    parser.add_argument("--left_center_txt", type=Path, default=DEFAULT_LEFT_CENTER_TXT)
    parser.add_argument("--right_center_txt", type=Path, default=DEFAULT_RIGHT_CENTER_TXT)
    parser.add_argument("--s_curve_center_txt", type=Path, default=DEFAULT_S_CURVE_CENTER_TXT)
    parser.add_argument("--n_clusters", type=int, default=100)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--feature_mode",
        choices=["endpoint_weighted", "final_lateral", "final_xy"],
        default="endpoint_weighted",
        help="K-Means feature. final_lateral uses final right; final_xy uses final right and forward.",
    )
    parser.add_argument(
        "--endpoint_weight",
        type=float,
        default=2.0,
        help="For endpoint_weighted mode, append final xy as an extra weighted feature; <=0 disables.",
    )
    parser.add_argument(
        "--all_lateral_weight",
        type=float,
        default=0.0,
        help="Append all 64 plot-x/right coordinates multiplied by this weight; <=0 disables.",
    )
    parser.add_argument(
        "--split_by_curvature",
        action="store_true",
        help="Split data into straight/turn by abs(sum(curvature)) before clustering.",
    )
    parser.add_argument("--curvature_threshold", type=float, default=0.15)
    parser.add_argument(
        "--straight_speed_mps",
        type=float,
        default=5.0,
        help="Also classify gentle long-horizon motion as straight when t0 speed is at least this value.",
    )
    parser.add_argument(
        "--straight_min_forward_m",
        type=float,
        default=12.0,
        help="Minimum final forward distance for geometry/speed straight classification.",
    )
    parser.add_argument(
        "--straight_max_final_lateral_m",
        type=float,
        default=2.0,
        help="Maximum absolute final lateral offset for geometry/speed straight classification.",
    )
    parser.add_argument(
        "--straight_max_lateral_m",
        type=float,
        default=2.5,
        help="Maximum absolute lateral offset at any future step for geometry/speed straight classification.",
    )
    parser.add_argument(
        "--straight_max_slope",
        type=float,
        default=0.06,
        help="Maximum abs(final_lateral) / final_forward for geometry/speed straight classification.",
    )
    parser.add_argument("--straight_clusters", type=int, default=25)
    parser.add_argument("--turn_clusters", type=int, default=25)
    parser.add_argument("--s_curve_clusters", type=int, default=0)
    parser.add_argument(
        "--s_curve_curvature_threshold",
        type=float,
        default=0.003,
        help="Classify S-curves when future curvature has a sign change beyond this abs threshold; <=0 disables.",
    )
    parser.add_argument(
        "--s_curve_min_run_steps",
        type=int,
        default=1,
        help="Require both positive and negative significant curvature to persist for at least this many consecutive future steps.",
    )
    parser.add_argument(
        "--split_stop_and_curvature",
        action="store_true",
        help="Cluster near-stop trajectories separately, then split the rest by curvature.",
    )
    parser.add_argument("--stop_speed_threshold_mps", type=float, default=0.5)
    parser.add_argument(
        "--stop_duration_s",
        type=float,
        default=0.0,
        help="Require this many continuous seconds below stop_speed_threshold_mps; <=0 uses any future speed.",
    )
    parser.add_argument("--stop_clusters", type=int, default=10)
    parser.add_argument(
        "--stop_lateral_weight",
        type=float,
        default=1.2,
        help="For stop-only K-Means, append all plot-x/right coordinates with this weight.",
    )
    parser.add_argument(
        "--stop_endpoint_lateral_weight",
        type=float,
        default=8.0,
        help="For stop-only K-Means, append final plot-x/right coordinate with this weight.",
    )
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument("--t0_stride", type=int, default=30)
    parser.add_argument("--min_speed_mps", type=float, default=2.0)
    parser.add_argument(
        "--min_forward_acc_mps2",
        type=float,
        default=-6.0,
        help="Drop GT windows whose original local forward acceleration goes below this.",
    )
    parser.add_argument(
        "--max_forward_acc_mps2",
        type=float,
        default=2.0,
        help="Drop GT windows whose original local forward acceleration goes above this.",
    )
    parser.add_argument(
        "--max_step_speed_mps",
        type=float,
        default=15.0,
        help="Drop repaired windows with any step faster than this; <=0 disables.",
    )
    parser.add_argument(
        "--max_step_acc_mps2",
        type=float,
        default=0.0,
        help="Optional repaired-position second-difference acceleration sanity check; <=0 disables.",
    )
    parser.add_argument(
        "--allow_backward",
        action="store_true",
        help="Keep windows whose repaired future endpoint is behind ego; useful for all-speed clustering.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_export", action="store_true")
    return parser.parse_args()


def iter_egomotion_files(train_data_root: Path) -> list[Path]:
    files: list[Path] = []
    for dataset_dir in sorted(train_data_root.glob("*_converted")):
        if not dataset_dir.is_dir():
            continue
        files.extend(sorted((dataset_dir / "data-egomotion").glob("*.egomotion.parquet")))
    return files


def estimate_t0_rotation(history_xyz: np.ndarray, history_quat_xyzw: np.ndarray) -> spt.Rotation:
    """Estimate ego heading from recent motion, falling back to quaternion if stopped."""
    disp_xy = history_xyz[-1, :2] - history_xyz[max(0, len(history_xyz) - 6), :2]
    if float(np.linalg.norm(disp_xy)) >= 0.2:
        return spt.Rotation.from_euler("z", float(np.arctan2(disp_xy[1], disp_xy[0])))

    quat = history_quat_xyzw[-1].astype(np.float64)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-8)
    return spt.Rotation.from_quat(quat)


def integrate_local_velocity(
    timestamps_us: np.ndarray,
    velocity_xyz: np.ndarray,
    quat_xyzw: np.ndarray,
    t0_idx: int,
    steps: int,
    t0_rot_inv: spt.Rotation,
) -> np.ndarray:
    """Rebuild future local xy by integrating body velocity in the t0 frame."""
    body_velocity = velocity_xyz[t0_idx : t0_idx + steps + 1]
    future_rot = spt.Rotation.from_quat(quat_xyzw[t0_idx : t0_idx + steps + 1])
    world_velocity = future_rot.apply(body_velocity)
    vel_local = t0_rot_inv.apply(world_velocity)[:, :2]
    ts = timestamps_us[t0_idx : t0_idx + steps + 1].astype(np.float64)
    dt = np.diff(ts) * 1e-6
    if len(dt) != steps or not np.isfinite(dt).all() or np.any(dt <= 0):
        raise ValueError("Bad timestamps for velocity integration")

    xy = np.zeros((steps, 2), dtype=np.float64)
    pos = np.zeros(2, dtype=np.float64)
    for i in range(steps):
        pos = pos + 0.5 * (vel_local[i] + vel_local[i + 1]) * dt[i]
        xy[i] = pos
    return xy


def export_future_xy_txt(
    train_data_root: Path,
    data_txt: Path,
    steps: int,
    history_steps: int,
    t0_stride: int,
    min_speed_mps: float,
    min_forward_acc_mps2: float,
    max_forward_acc_mps2: float,
    max_step_speed_mps: float,
    max_step_acc_mps2: float,
    allow_backward: bool,
) -> ExportStats:
    data_txt.parent.mkdir(parents=True, exist_ok=True)
    stats = ExportStats()
    traj_id = 0

    with data_txt.open("w", encoding="utf-8") as f:
        f.write("# gt_future_trajectories_xy_v1\n")
        f.write(f"# source_train_data_root={train_data_root}\n")
        f.write(f"# steps={steps}\n")
        f.write(f"# history_steps={history_steps}\n")
        f.write(f"# t0_stride={t0_stride}\n")
        f.write(f"# min_speed_mps={min_speed_mps}\n")
        f.write(f"# min_forward_acc_mps2={min_forward_acc_mps2}\n")
        f.write(f"# max_forward_acc_mps2={max_forward_acc_mps2}\n")
        f.write(f"# max_step_speed_mps={max_step_speed_mps}\n")
        f.write(f"# max_step_acc_mps2={max_step_acc_mps2}\n")
        f.write("# ego-local xy: x=forward, y=left\n")
        f.write("# GT future is cut from train_data/data-egomotion, not model output\n")
        f.write("# future xy is repaired by integrating GT velocity to remove raw position jumps\n")
        f.write("# original local forward acceleration is filtered before export\n")
        f.write("# columns: traj_id dataset clip t0_us t0_index x0 y0 ... x63 y63\n")

        for parquet_file in iter_egomotion_files(train_data_root):
            dataset = parquet_file.parents[1].name
            clip = parquet_file.name.replace(".egomotion.parquet", "")
            df = pd.read_parquet(parquet_file)
            required = {"timestamp", "x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw"}
            if not required.issubset(df.columns) or len(df) < history_steps + steps + 1:
                stats.skipped_short_or_bad += 1
                continue

            xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
            quat_xyzw = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
            velocity_xyz = df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64)
            acceleration_xyz = df[["ax", "ay", "az"]].to_numpy(dtype=np.float64)
            speed = np.linalg.norm(velocity_xyz, axis=1)
            timestamps = df["timestamp"].to_numpy(dtype=np.int64)

            first_t0 = history_steps - 1
            last_t0 = len(df) - steps - 1
            for t0_idx in range(first_t0, last_t0 + 1, max(1, t0_stride)):
                if speed[t0_idx] < min_speed_mps:
                    stats.skipped_slow += 1
                    continue

                history_xyz = xyz[t0_idx - history_steps + 1 : t0_idx + 1]
                future_xyz = xyz[t0_idx + 1 : t0_idx + 1 + steps]
                if len(future_xyz) != steps:
                    stats.skipped_short_or_bad += 1
                    continue

                t0_rot_inv = estimate_t0_rotation(
                    history_xyz,
                    quat_xyzw[t0_idx - history_steps + 1 : t0_idx + 1],
                ).inv()
                raw_xy = t0_rot_inv.apply(future_xyz - xyz[t0_idx])[:, :2]
                # vx/vy and ax/ay in these parquet files are vehicle-dynamics
                # channels, so ax is already longitudinal acceleration.
                forward_acc = acceleration_xyz[t0_idx + 1 : t0_idx + 1 + steps, 0]
                if not np.isfinite(forward_acc).all():
                    stats.skipped_short_or_bad += 1
                    continue
                if (
                    float(forward_acc.min()) < min_forward_acc_mps2
                    or float(forward_acc.max()) > max_forward_acc_mps2
                ):
                    stats.skipped_acc += 1
                    continue

                try:
                    xy = integrate_local_velocity(
                        timestamps,
                        velocity_xyz,
                        quat_xyzw,
                        t0_idx,
                        steps,
                        t0_rot_inv,
                    )
                except ValueError:
                    stats.skipped_short_or_bad += 1
                    continue
                if not np.isfinite(xy).all():
                    stats.skipped_short_or_bad += 1
                    continue
                if not allow_backward and xy[-1, 0] < 0.0:
                    stats.skipped_backward += 1
                    continue

                xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
                velocity = np.diff(xy_from_origin, axis=0) / 0.1
                step_speed = np.linalg.norm(velocity, axis=1)
                step_acc = np.linalg.norm(np.diff(velocity, axis=0) / 0.1, axis=1)
                max_speed = float(step_speed.max(initial=0.0))
                max_acc = float(step_acc.max(initial=0.0))
                raw_step = float(
                    np.linalg.norm(
                        np.diff(np.vstack([np.zeros((1, 2), dtype=np.float64), raw_xy]), axis=0),
                        axis=1,
                    ).max(initial=0.0)
                )
                repaired_step = float(
                    np.linalg.norm(np.diff(xy_from_origin, axis=0), axis=1).max(initial=0.0)
                )

                too_fast = max_step_speed_mps > 0 and max_speed > max_step_speed_mps
                too_accel = max_step_acc_mps2 > 0 and max_acc > max_step_acc_mps2
                if too_fast or too_accel:
                    stats.skipped_jump += 1
                    continue

                values = " ".join(f"{v:.6f}" for v in xy.reshape(-1))
                f.write(
                    f"{traj_id} {dataset} {clip} {int(timestamps[t0_idx])} "
                    f"{int(t0_idx)} {values}\n"
                )
                traj_id += 1
                stats.kept += 1
                stats.max_speed_mps = max(stats.max_speed_mps, max_speed)
                stats.max_acc_mps2 = max(stats.max_acc_mps2, max_acc)
                stats.min_forward_acc_mps2 = min(
                    stats.min_forward_acc_mps2,
                    float(forward_acc.min()),
                )
                stats.max_forward_acc_mps2 = max(
                    stats.max_forward_acc_mps2,
                    float(forward_acc.max()),
                )
                stats.max_raw_position_step_m = max(stats.max_raw_position_step_m, raw_step)
                stats.max_repaired_position_step_m = max(
                    stats.max_repaired_position_step_m,
                    repaired_step,
                )

    return stats


def build_feature(
    xy: np.ndarray,
    endpoint_weight: float,
    feature_mode: str,
    all_lateral_weight: float,
) -> np.ndarray:
    if feature_mode == "final_lateral":
        parts = [np.asarray([-xy[-1, 1]], dtype=np.float64)]
    elif feature_mode == "final_xy":
        parts = [np.asarray([-xy[-1, 1], xy[-1, 0]], dtype=np.float64)]
    else:
        parts = [xy.reshape(-1)]
        if endpoint_weight > 0:
            parts.append(xy[-1] * endpoint_weight)

    if all_lateral_weight > 0:
        parts.append((-xy[:, 1]) * all_lateral_weight)
    return np.concatenate(parts)


def build_stop_feature(
    base_feature: np.ndarray,
    xy: np.ndarray,
    stop_lateral_weight: float,
    stop_endpoint_lateral_weight: float,
) -> np.ndarray:
    """Add stronger plot-x/right features for stop-only clustering."""
    parts = [np.asarray(base_feature, dtype=np.float64)]
    if stop_lateral_weight > 0:
        parts.append((-xy[:, 1]) * stop_lateral_weight)
    if stop_endpoint_lateral_weight > 0:
        parts.append(np.asarray([-xy[-1, 1] * stop_endpoint_lateral_weight], dtype=np.float64))
    return np.concatenate(parts)


def load_future_xy_txt(
    data_txt: Path,
    steps: int,
    endpoint_weight: float,
    feature_mode: str,
    all_lateral_weight: float,
) -> tuple[list[TrajectoryRow], np.ndarray]:
    rows: list[TrajectoryRow] = []
    features: list[np.ndarray] = []

    with data_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 5 + steps * 2:
                raise ValueError(f"Bad row with {len(parts)} columns: {line[:120]}")

            traj_id = int(parts[0])
            dataset = parts[1]
            clip = parts[2]
            t0_us = int(parts[3])
            sample_idx = int(parts[4])
            xy = np.asarray(parts[5:], dtype=np.float64).reshape(steps, 2)
            rows.append(TrajectoryRow(traj_id, dataset, clip, t0_us, sample_idx, xy))
            features.append(
                build_feature(xy, endpoint_weight, feature_mode, all_lateral_weight)
            )

    if not rows:
        raise ValueError(f"No trajectories found in {data_txt}")

    return rows, np.vstack(features)


def load_curvature_scores(
    rows: list[TrajectoryRow],
    train_data_root: Path,
    steps: int,
) -> np.ndarray:
    """Return sum(abs(curvature)) over each row's future window."""
    cache: dict[tuple[str, str], np.ndarray] = {}
    scores = []
    for row in rows:
        key = (row.dataset, row.clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / row.dataset
                / "data-egomotion"
                / f"{row.clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["curvature"])
            cache[key] = df["curvature"].to_numpy(dtype=np.float64)
        curv = cache[key][row.sample_idx + 1 : row.sample_idx + 1 + steps]
        if len(curv) != steps or not np.isfinite(curv).all():
            scores.append(np.nan)
        else:
            scores.append(float(np.abs(curv).sum()))
    return np.asarray(scores, dtype=np.float64)


def load_curvature_traces(
    rows: list[TrajectoryRow],
    train_data_root: Path,
    steps: int,
) -> np.ndarray:
    """Return curvature traces over each row's future window."""
    cache: dict[tuple[str, str], np.ndarray] = {}
    traces = []
    for row in rows:
        key = (row.dataset, row.clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / row.dataset
                / "data-egomotion"
                / f"{row.clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["curvature"])
            cache[key] = df["curvature"].to_numpy(dtype=np.float64)
        curv = cache[key][row.sample_idx + 1 : row.sample_idx + 1 + steps]
        if len(curv) != steps or not np.isfinite(curv).all():
            traces.append(np.full(steps, np.nan, dtype=np.float64))
        else:
            traces.append(curv)
    return np.asarray(traces, dtype=np.float64)


def has_s_curve_sign_change(
    curvature_traces: np.ndarray,
    curvature_threshold: float,
    min_run_steps: int,
) -> np.ndarray:
    """Return rows whose significant future curvature changes sign."""
    if curvature_threshold <= 0:
        return np.zeros(curvature_traces.shape[0], dtype=bool)
    min_run_steps = max(1, int(min_run_steps))

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

    s_mask = np.zeros(curvature_traces.shape[0], dtype=bool)
    for idx, curv in enumerate(curvature_traces):
        if not np.isfinite(curv).all():
            continue
        pos_mask = curv >= curvature_threshold
        neg_mask = curv <= -curvature_threshold
        if max_run(pos_mask) < min_run_steps or max_run(neg_mask) < min_run_steps:
            continue
        significant = curv[pos_mask | neg_mask]
        signs = np.sign(significant)
        if np.any(signs[:-1] * signs[1:] < 0):
            s_mask[idx] = True
    return s_mask


def load_min_future_speeds(
    rows: list[TrajectoryRow],
    train_data_root: Path,
    steps: int,
) -> np.ndarray:
    """Return min absolute speed over each row's future window."""
    cache: dict[tuple[str, str], np.ndarray] = {}
    speeds = []
    for row in rows:
        key = (row.dataset, row.clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / row.dataset
                / "data-egomotion"
                / f"{row.clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["vx", "vy", "vz"])
            cache[key] = np.linalg.norm(
                df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64),
                axis=1,
            )
        future_speed = cache[key][row.sample_idx + 1 : row.sample_idx + 1 + steps]
        if len(future_speed) != steps or not np.isfinite(future_speed).all():
            speeds.append(np.nan)
        else:
            speeds.append(float(future_speed.min()))
    return np.asarray(speeds, dtype=np.float64)


def load_t0_speeds(
    rows: list[TrajectoryRow],
    train_data_root: Path,
) -> np.ndarray:
    """Return absolute speed at each row's t0 index."""
    cache: dict[tuple[str, str], np.ndarray] = {}
    speeds = []
    for row in rows:
        key = (row.dataset, row.clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / row.dataset
                / "data-egomotion"
                / f"{row.clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["vx", "vy", "vz"])
            cache[key] = np.linalg.norm(
                df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64),
                axis=1,
            )
        speed = cache[key]
        if row.sample_idx >= len(speed) or not np.isfinite(speed[row.sample_idx]):
            speeds.append(np.nan)
        else:
            speeds.append(float(speed[row.sample_idx]))
    return np.asarray(speeds, dtype=np.float64)


def load_future_speed_traces(
    rows: list[TrajectoryRow],
    train_data_root: Path,
    steps: int,
) -> np.ndarray:
    """Return absolute speed traces over each row's future window."""
    cache: dict[tuple[str, str], np.ndarray] = {}
    traces = []
    for row in rows:
        key = (row.dataset, row.clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / row.dataset
                / "data-egomotion"
                / f"{row.clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["vx", "vy", "vz"])
            cache[key] = np.linalg.norm(
                df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64),
                axis=1,
            )
        future_speed = cache[key][row.sample_idx + 1 : row.sample_idx + 1 + steps]
        if len(future_speed) != steps or not np.isfinite(future_speed).all():
            traces.append(np.full(steps, np.nan, dtype=np.float64))
        else:
            traces.append(future_speed)
    return np.asarray(traces, dtype=np.float64)


def has_continuous_low_speed(
    speed_traces: np.ndarray,
    speed_threshold_mps: float,
    duration_s: float,
    dt_s: float = 0.1,
) -> np.ndarray:
    """Return rows with a continuous low-speed run of duration_s or longer."""
    required = max(1, int(np.ceil(duration_s / dt_s)))
    stop_mask = np.zeros(speed_traces.shape[0], dtype=bool)
    for idx, speeds in enumerate(speed_traces):
        if not np.isfinite(speeds).all():
            continue
        run = 0
        for speed in speeds:
            if float(speed) <= speed_threshold_mps:
                run += 1
                if run >= required:
                    stop_mask[idx] = True
                    break
            else:
                run = 0
    return stop_mask


def straight_geometry_mask(
    rows: list[TrajectoryRow],
    t0_speeds: np.ndarray,
    min_speed_mps: float,
    min_forward_m: float,
    max_final_lateral_m: float,
    max_lateral_m: float,
    max_slope: float,
) -> np.ndarray:
    """Return high-speed, low-slope trajectories that should be treated as straight."""
    xy = np.stack([row.xy for row in rows], axis=0)
    final_forward = xy[:, -1, 0]
    final_lateral = np.abs(xy[:, -1, 1])
    max_lateral = np.max(np.abs(xy[:, :, 1]), axis=1)
    slope = final_lateral / np.maximum(final_forward, 1e-6)
    return (
        np.isfinite(t0_speeds)
        & (t0_speeds >= float(min_speed_mps))
        & (final_forward >= float(min_forward_m))
        & (final_lateral <= float(max_final_lateral_m))
        & (max_lateral <= float(max_lateral_m))
        & (slope <= float(max_slope))
    )


def write_results(
    result_txt: Path,
    rows: list[TrajectoryRow],
    labels: np.ndarray,
    distances: np.ndarray,
    centers: np.ndarray,
    inertia: float,
    feature_mode: str,
    endpoint_weight: float,
    all_lateral_weight: float,
    export_stats: ExportStats | None = None,
    extra_headers: dict[str, object] | None = None,
) -> None:
    result_txt.parent.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(labels, minlength=len(centers))
    max_speed_mps, max_acc_mps2 = compute_motion_stats(rows)

    with result_txt.open("w", encoding="utf-8") as f:
        f.write("# kmeans_results_v1\n")
        f.write(f"# n_trajectories={len(rows)}\n")
        f.write(f"# n_clusters={len(centers)}\n")
        f.write(f"# feature_mode={feature_mode}\n")
        f.write(f"# endpoint_weight={endpoint_weight:.6f}\n")
        f.write(f"# all_lateral_weight={all_lateral_weight:.6f}\n")
        if extra_headers:
            for key, value in extra_headers.items():
                f.write(f"# {key}={value}\n")
        f.write(f"# inertia={inertia:.6f}\n")
        f.write(f"# filtered_repaired_max_step_speed_mps={max_speed_mps:.6f}\n")
        f.write(f"# filtered_repaired_max_second_diff_acc_mps2={max_acc_mps2:.6f}\n")
        if export_stats is not None:
            f.write(f"# filtered_original_forward_acc_min_mps2={export_stats.min_forward_acc_mps2:.6f}\n")
            f.write(f"# filtered_original_forward_acc_max_mps2={export_stats.max_forward_acc_mps2:.6f}\n")
            f.write(f"# max_raw_position_step_m={export_stats.max_raw_position_step_m:.6f}\n")
            f.write(f"# max_repaired_position_step_m={export_stats.max_repaired_position_step_m:.6f}\n")
            f.write(f"# skipped_slow={export_stats.skipped_slow}\n")
            f.write(f"# skipped_acc={export_stats.skipped_acc}\n")
            f.write(f"# skipped_jump={export_stats.skipped_jump}\n")
            f.write(f"# skipped_backward={export_stats.skipped_backward}\n")
            f.write(f"# skipped_short_or_bad={export_stats.skipped_short_or_bad}\n")
        f.write("# assignments: traj_id cluster distance dataset clip t0_us t0_index\n")
        for row, label, dist in zip(rows, labels, distances):
            f.write(
                f"{row.traj_id} {int(label)} {float(dist):.6f} "
                f"{row.dataset} {row.clip} {row.t0_us} {row.sample_idx}\n"
            )

        f.write("# centers: cluster count x0 y0 ... x63 y63\n")
        for cluster_id, center in enumerate(centers):
            members = [row.xy for row, label in zip(rows, labels) if int(label) == cluster_id]
            if members:
                center_xy = np.mean(np.stack(members, axis=0), axis=0).reshape(-1)
            else:
                center_xy = np.zeros(rows[0].xy.size, dtype=np.float64)
            values = " ".join(f"{v:.6f}" for v in center_xy)
            f.write(f"CENTER {cluster_id} {int(counts[cluster_id])} {values}\n")
            speed, accel, curvature = center_dynamics_from_xy(center_xy.reshape(-1, 2))
            speed_values = " ".join(f"{v:.6f}" for v in speed)
            accel_values = " ".join(f"{v:.6f}" for v in accel)
            curvature_values = " ".join(f"{v:.6f}" for v in curvature)
            f.write(f"CENTER_DYNAMICS {cluster_id} speed {speed_values}\n")
            f.write(f"CENTER_DYNAMICS {cluster_id} acceleration {accel_values}\n")
            f.write(f"CENTER_DYNAMICS {cluster_id} curvature {curvature_values}\n")


def mean_centers_from_labels(
    rows: list[TrajectoryRow],
    labels: np.ndarray,
    cluster_ids: np.ndarray | list[int],
) -> list[tuple[int, int, np.ndarray]]:
    out: list[tuple[int, int, np.ndarray]] = []
    for cluster_id in sorted(int(v) for v in cluster_ids):
        members = [row.xy for row, label in zip(rows, labels) if int(label) == cluster_id]
        if not members:
            continue
        center_xy = np.mean(np.stack(members, axis=0), axis=0)
        out.append((cluster_id, len(members), center_xy))
    return out


def write_category_center_txt(
    path: Path,
    category: str,
    centers: list[tuple[int, int, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# cluster_centers_xy_v1\n")
        f.write(f"# category={category}\n")
        f.write("# ego-local xy: x=forward, y=left; plot right=-y\n")
        f.write("# columns: CENTER center_id count x0 y0 ... x63 y63\n")
        for center_id, count, center_xy in centers:
            values = " ".join(f"{v:.6f}" for v in center_xy.reshape(-1))
            f.write(f"CENTER {center_id} {count} {values}\n")


def write_category_center_txts(
    stop_center_path: Path,
    straight_center_path: Path,
    left_center_path: Path,
    right_center_path: Path,
    s_curve_center_path: Path,
    stop_rows: list[TrajectoryRow],
    stop_labels: np.ndarray,
    main_rows: list[TrajectoryRow],
    main_labels: np.ndarray,
    straight_cluster_count: int,
    turn_cluster_count: int,
) -> None:
    stop_centers = mean_centers_from_labels(stop_rows, stop_labels, np.unique(stop_labels))
    straight_centers = mean_centers_from_labels(
        main_rows,
        main_labels,
        np.arange(straight_cluster_count),
    )
    turn_centers = mean_centers_from_labels(
        main_rows,
        main_labels,
        np.arange(straight_cluster_count, straight_cluster_count + turn_cluster_count),
    )
    s_curve_centers = mean_centers_from_labels(
        main_rows,
        main_labels,
        np.unique(main_labels[main_labels >= straight_cluster_count + turn_cluster_count]),
    )

    left_centers: list[tuple[int, int, np.ndarray]] = []
    right_centers: list[tuple[int, int, np.ndarray]] = []
    for center in turn_centers:
        _, _, center_xy = center
        final_right = -float(center_xy[-1, 1])
        if final_right >= 0.0:
            right_centers.append(center)
        else:
            left_centers.append(center)

    write_category_center_txt(stop_center_path, "stop", stop_centers)
    write_category_center_txt(straight_center_path, "straight", straight_centers)
    write_category_center_txt(left_center_path, "left", left_centers)
    write_category_center_txt(right_center_path, "right", right_centers)
    write_category_center_txt(s_curve_center_path, "s_curve", s_curve_centers)


def compute_motion_stats(rows: list[TrajectoryRow]) -> tuple[float, float]:
    max_speed_mps = 0.0
    max_acc_mps2 = 0.0
    for row in rows:
        xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), row.xy])
        velocity = np.diff(xy_from_origin, axis=0) / 0.1
        speed = np.linalg.norm(velocity, axis=1)
        acc = np.linalg.norm(np.diff(velocity, axis=0) / 0.1, axis=1)
        max_speed_mps = max(max_speed_mps, float(speed.max(initial=0.0)))
        max_acc_mps2 = max(max_acc_mps2, float(acc.max(initial=0.0)))
    return max_speed_mps, max_acc_mps2


def center_dynamics_from_xy(xy: np.ndarray, dt: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate unicycle dynamics from an ego-local xy trajectory."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy_from_origin = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    delta = np.diff(xy_from_origin, axis=0)
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
        dyaw = np.diff(yaw)
        ds = np.maximum(step_distance[1:], 1e-6)
        curvature[1:] = dyaw / ds
        curvature[~np.isfinite(curvature)] = 0.0
    return speed, acceleration, curvature


def _write_split_rows(
    path: Path,
    rows: list[TrajectoryRow],
    labels: np.ndarray,
    split_name: str,
    direction_mode: str = "none",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# split_trajectories_xy_v1\n")
        f.write("# ego-local xy: x=forward, y=left; plot right=-y\n")
        f.write("# columns: traj_id split direction cluster dataset clip t0_us t0_index x0 y0 ... x63 y63\n")
        for row, label in zip(rows, labels):
            final_right = -float(row.xy[-1, 1])
            if direction_mode == "turn":
                direction = "right" if final_right >= 0.0 else "left"
            else:
                direction = "none"
            values = " ".join(f"{v:.6f}" for v in row.xy.reshape(-1))
            f.write(
                f"{row.traj_id} {split_name} {direction} {int(label)} "
                f"{row.dataset} {row.clip} {row.t0_us} {row.sample_idx} {values}\n"
            )


def write_split_txts(
    stop_path: Path,
    straight_path: Path,
    turn_path: Path,
    s_curve_path: Path,
    stop_rows: list[TrajectoryRow],
    stop_labels: np.ndarray,
    main_rows: list[TrajectoryRow],
    main_labels: np.ndarray,
    straight_cluster_count: int,
    turn_cluster_count: int,
) -> None:
    _write_split_rows(stop_path, stop_rows, stop_labels, "stop")
    straight_mask = main_labels < straight_cluster_count
    turn_mask = (main_labels >= straight_cluster_count) & (
        main_labels < straight_cluster_count + turn_cluster_count
    )
    s_curve_mask = main_labels >= straight_cluster_count + turn_cluster_count
    _write_split_rows(
        straight_path,
        [row for row, keep in zip(main_rows, straight_mask) if bool(keep)],
        main_labels[straight_mask],
        "straight",
    )
    _write_split_rows(
        turn_path,
        [row for row, keep in zip(main_rows, turn_mask) if bool(keep)],
        main_labels[turn_mask],
        "turn",
        direction_mode="turn",
    )
    _write_split_rows(
        s_curve_path,
        [row for row, keep in zip(main_rows, s_curve_mask) if bool(keep)],
        main_labels[s_curve_mask],
        "s_curve",
        direction_mode="turn",
    )


def run_split_kmeans(
    rows: list[TrajectoryRow],
    features: np.ndarray,
    curvature_scores: np.ndarray,
    curvature_traces: np.ndarray | None,
    t0_speeds: np.ndarray,
    curvature_threshold: float,
    straight_speed_mps: float,
    straight_min_forward_m: float,
    straight_max_final_lateral_m: float,
    straight_max_lateral_m: float,
    straight_max_slope: float,
    straight_clusters: int,
    turn_clusters: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, object]]:
    labels = np.full(features.shape[0], -1, dtype=np.int64)
    distances = np.zeros(features.shape[0], dtype=np.float64)
    total_clusters = straight_clusters + turn_clusters
    centers = np.zeros((total_clusters, features.shape[1]), dtype=np.float64)
    inertia = 0.0

    valid_curvature = np.isfinite(curvature_scores)
    low_curvature_mask = valid_curvature & (curvature_scores <= curvature_threshold)
    non_straight_mask = valid_curvature & (curvature_scores > curvature_threshold)
    # This legacy two-way split has no S-curve bucket, so sign-changing rows
    # remain in turn unless they satisfy the explicit gentle-straight geometry.
    s_curve_mask = np.zeros(len(rows), dtype=bool)
    geometry_straight_mask = (
        non_straight_mask
        & ~s_curve_mask
        & straight_geometry_mask(
            rows,
            t0_speeds,
            straight_speed_mps,
            straight_min_forward_m,
            straight_max_final_lateral_m,
            straight_max_lateral_m,
            straight_max_slope,
        )
    )
    straight_mask = low_curvature_mask | geometry_straight_mask
    turn_mask = non_straight_mask & ~s_curve_mask & ~geometry_straight_mask
    split_defs = [
        ("straight", straight_mask, 0, straight_clusters),
        ("turn", turn_mask, straight_clusters, turn_clusters),
    ]
    extra_headers: dict[str, object] = {
        "split_by_curvature": "true",
        "curvature_metric": "sum_abs_future_curvature_plus_speed_slope_straight",
        "curvature_threshold": f"{curvature_threshold:.6f}",
        "geometry_straight_trajectories": int(geometry_straight_mask.sum()),
        "straight_speed_mps": f"{straight_speed_mps:.6f}",
        "straight_min_forward_m": f"{straight_min_forward_m:.6f}",
        "straight_max_final_lateral_m": f"{straight_max_final_lateral_m:.6f}",
        "straight_max_lateral_m": f"{straight_max_lateral_m:.6f}",
        "straight_max_slope": f"{straight_max_slope:.6f}",
    }

    for group_name, mask, offset, requested_clusters in split_defs:
        indices = np.flatnonzero(mask)
        n_group = len(indices)
        n_clusters = min(requested_clusters, n_group)
        extra_headers[f"{group_name}_trajectories"] = n_group
        extra_headers[f"{group_name}_clusters"] = n_clusters
        if n_group == 0:
            continue

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init="auto",
            max_iter=300,
        )
        local_labels = kmeans.fit_predict(features[indices])
        labels[indices] = local_labels + offset
        distances[indices] = np.linalg.norm(
            features[indices] - kmeans.cluster_centers_[local_labels],
            axis=1,
        )
        centers[offset : offset + n_clusters] = kmeans.cluster_centers_
        inertia += float(kmeans.inertia_)

    if np.any(labels < 0):
        raise ValueError("Some rows were not assigned to a curvature split")

    return labels, distances, centers, inertia, extra_headers


def run_kmeans_subset(
    rows: list[TrajectoryRow],
    features: np.ndarray,
    indices: np.ndarray,
    n_clusters: int,
    seed: int,
) -> tuple[list[TrajectoryRow], np.ndarray, np.ndarray, np.ndarray, float]:
    subset_rows = [rows[int(i)] for i in indices]
    subset_features = features[indices]
    actual_clusters = min(n_clusters, len(subset_rows))
    if actual_clusters <= 0:
        raise ValueError("Cannot cluster an empty subset")

    kmeans = KMeans(
        n_clusters=actual_clusters,
        random_state=seed,
        n_init="auto",
        max_iter=300,
    )
    labels = kmeans.fit_predict(subset_features)
    distances = np.linalg.norm(subset_features - kmeans.cluster_centers_[labels], axis=1)
    return subset_rows, labels, distances, kmeans.cluster_centers_, float(kmeans.inertia_)


def run_stop_and_curvature_split(
    rows: list[TrajectoryRow],
    features: np.ndarray,
    min_future_speeds: np.ndarray,
    future_speed_traces: np.ndarray | None,
    curvature_scores: np.ndarray,
    curvature_traces: np.ndarray | None,
    t0_speeds: np.ndarray,
    stop_speed_threshold_mps: float,
    stop_duration_s: float,
    stop_clusters: int,
    stop_lateral_weight: float,
    stop_endpoint_lateral_weight: float,
    straight_clusters: int,
    turn_clusters: int,
    s_curve_clusters: int,
    curvature_threshold: float,
    straight_speed_mps: float,
    straight_min_forward_m: float,
    straight_max_final_lateral_m: float,
    straight_max_lateral_m: float,
    straight_max_slope: float,
    s_curve_curvature_threshold: float,
    s_curve_min_run_steps: int,
    seed: int,
) -> tuple[
    list[TrajectoryRow],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    dict[str, object],
    list[TrajectoryRow],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    dict[str, object],
]:
    if stop_duration_s > 0:
        if future_speed_traces is None:
            raise ValueError("future_speed_traces is required when stop_duration_s > 0")
        stop_mask = has_continuous_low_speed(
            future_speed_traces,
            stop_speed_threshold_mps,
            stop_duration_s,
        )
    else:
        stop_mask = np.isfinite(min_future_speeds) & (min_future_speeds < stop_speed_threshold_mps)
    stop_indices = np.flatnonzero(stop_mask)
    remaining_indices = np.flatnonzero(~stop_mask & np.isfinite(curvature_scores))
    if len(stop_indices) == 0:
        raise ValueError("Cannot cluster stop split because no stop trajectories were found")
    stop_feature_rows = [
            build_stop_feature(
                features[int(i)],
                rows[int(i)].xy,
                stop_lateral_weight,
                stop_endpoint_lateral_weight,
            )
            for i in stop_indices
    ]
    stop_features = np.zeros((len(rows), len(stop_feature_rows[0])), dtype=np.float64)
    stop_features[stop_indices] = np.vstack(stop_feature_rows)

    stop_rows, stop_labels, stop_distances, stop_centers, stop_inertia = run_kmeans_subset(
        rows,
        stop_features,
        stop_indices,
        stop_clusters,
        seed,
    )

    remaining_rows = [rows[int(i)] for i in remaining_indices]
    remaining_features = features[remaining_indices]
    remaining_scores = curvature_scores[remaining_indices]
    remaining_t0_speeds = t0_speeds[remaining_indices]
    low_curvature_mask = np.isfinite(remaining_scores) & (remaining_scores <= curvature_threshold)
    non_straight_mask = np.isfinite(remaining_scores) & (remaining_scores > curvature_threshold)

    if s_curve_clusters > 0:
        if curvature_traces is None:
            raise ValueError("curvature_traces is required when s_curve_clusters > 0")
        remaining_curvature_traces = curvature_traces[remaining_indices]
        s_curve_mask = non_straight_mask & has_s_curve_sign_change(
            remaining_curvature_traces,
            s_curve_curvature_threshold,
            s_curve_min_run_steps,
        )
    else:
        s_curve_mask = np.zeros(len(remaining_indices), dtype=bool)
    geometry_straight_mask = (
        non_straight_mask
        & ~s_curve_mask
        & straight_geometry_mask(
            remaining_rows,
            remaining_t0_speeds,
            straight_speed_mps,
            straight_min_forward_m,
            straight_max_final_lateral_m,
            straight_max_lateral_m,
            straight_max_slope,
        )
    )
    straight_mask = low_curvature_mask | geometry_straight_mask
    turn_mask = non_straight_mask & ~s_curve_mask & ~geometry_straight_mask

    split_defs = [
        ("straight", straight_mask, 0, straight_clusters),
        ("turn", turn_mask, straight_clusters, turn_clusters),
        ("s_curve", s_curve_mask, straight_clusters + turn_clusters, s_curve_clusters),
    ]
    total_clusters = straight_clusters + turn_clusters + max(0, s_curve_clusters)
    rem_labels = np.full(len(remaining_rows), -1, dtype=np.int64)
    rem_distances = np.zeros(len(remaining_rows), dtype=np.float64)
    rem_centers = np.zeros((total_clusters, features.shape[1]), dtype=np.float64)
    rem_inertia = 0.0
    rem_headers: dict[str, object] = {
        "split_by_curvature": "true",
        "curvature_metric": "sum_abs_future_curvature_plus_speed_slope_straight",
        "curvature_threshold": f"{curvature_threshold:.6f}",
        "geometry_straight_trajectories": int(geometry_straight_mask.sum()),
        "straight_speed_mps": f"{straight_speed_mps:.6f}",
        "straight_min_forward_m": f"{straight_min_forward_m:.6f}",
        "straight_max_final_lateral_m": f"{straight_max_final_lateral_m:.6f}",
        "straight_max_lateral_m": f"{straight_max_lateral_m:.6f}",
        "straight_max_slope": f"{straight_max_slope:.6f}",
        "s_curve_curvature_threshold": f"{s_curve_curvature_threshold:.6f}",
        "s_curve_min_run_steps": s_curve_min_run_steps,
    }

    for group_name, mask, offset, requested_clusters in split_defs:
        indices = np.flatnonzero(mask)
        n_group = len(indices)
        n_clusters = min(requested_clusters, n_group)
        rem_headers[f"{group_name}_trajectories"] = n_group
        rem_headers[f"{group_name}_clusters"] = n_clusters
        if n_group == 0 or requested_clusters <= 0:
            continue
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init="auto",
            max_iter=300,
        )
        local_labels = kmeans.fit_predict(remaining_features[indices])
        rem_labels[indices] = local_labels + offset
        rem_distances[indices] = np.linalg.norm(
            remaining_features[indices] - kmeans.cluster_centers_[local_labels],
            axis=1,
        )
        rem_centers[offset : offset + n_clusters] = kmeans.cluster_centers_
        rem_inertia += float(kmeans.inertia_)

    if np.any(rem_labels < 0):
        raise ValueError("Some rows were not assigned to a stop/curvature split")

    main_headers: dict[str, object] = {
        "split_stop_and_curvature": "true",
        "excluded_stop_trajectories": len(stop_indices),
        "stop_speed_threshold_mps": f"{stop_speed_threshold_mps:.6f}",
        "stop_duration_s": f"{stop_duration_s:.6f}",
        **rem_headers,
    }
    stop_headers: dict[str, object] = {
        "stop_only": "true",
        "stop_speed_threshold_mps": f"{stop_speed_threshold_mps:.6f}",
        "stop_duration_s": f"{stop_duration_s:.6f}",
        "stop_trajectories": len(stop_indices),
        "stop_clusters": min(stop_clusters, len(stop_rows)),
        "stop_feature_lateral_weight": f"{stop_lateral_weight:.6f}",
        "stop_feature_endpoint_lateral_weight": f"{stop_endpoint_lateral_weight:.6f}",
    }

    return (
        remaining_rows,
        rem_labels,
        rem_distances,
        rem_centers,
        rem_inertia,
        main_headers,
        stop_rows,
        stop_labels,
        stop_distances,
        stop_centers,
        stop_inertia,
        stop_headers,
    )


def main() -> None:
    args = parse_args()
    export_stats = None

    if not args.skip_export:
        export_stats = export_future_xy_txt(
            args.train_data_root,
            args.data_txt,
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
        print(f"Exported {export_stats.kept} trajectories to {args.data_txt}")
        print(
            "Filtered export stats: "
            f"skipped_slow={export_stats.skipped_slow}, "
            f"skipped_backward={export_stats.skipped_backward}, "
            f"skipped_acc={export_stats.skipped_acc}, "
            f"skipped_jump={export_stats.skipped_jump}, "
            f"skipped_short_or_bad={export_stats.skipped_short_or_bad}"
        )
        print(
            "Filtered repaired motion diagnostic: "
            f"speed={export_stats.max_speed_mps:.6f} m/s, "
            f"second_diff_acc={export_stats.max_acc_mps2:.6f} m/s^2"
        )
        print(
            "Filtered original forward acceleration: "
            f"min={export_stats.min_forward_acc_mps2:.6f} m/s^2, "
            f"max={export_stats.max_forward_acc_mps2:.6f} m/s^2"
        )
        print(
            "Position jump repair: "
            f"max_raw_step={export_stats.max_raw_position_step_m:.6f} m, "
            f"max_repaired_step={export_stats.max_repaired_position_step_m:.6f} m"
        )

    rows, features = load_future_xy_txt(
        args.data_txt,
        args.steps,
        args.endpoint_weight,
        args.feature_mode,
        args.all_lateral_weight,
    )
    extra_headers: dict[str, object] | None = None
    if args.split_stop_and_curvature:
        curvature_scores = load_curvature_scores(rows, args.train_data_root, args.steps)
        min_future_speeds = load_min_future_speeds(rows, args.train_data_root, args.steps)
        t0_speeds = load_t0_speeds(rows, args.train_data_root)
        future_speed_traces = (
            load_future_speed_traces(rows, args.train_data_root, args.steps)
            if args.stop_duration_s > 0
            else None
        )
        curvature_traces = (
            load_curvature_traces(rows, args.train_data_root, args.steps)
            if args.s_curve_clusters > 0
            else None
        )
        (
            rows,
            labels,
            distances,
            centers,
            inertia,
            extra_headers,
            stop_rows,
            stop_labels,
            stop_distances,
            stop_centers,
            stop_inertia,
            stop_headers,
        ) = run_stop_and_curvature_split(
            rows,
            features,
            min_future_speeds,
            future_speed_traces,
            curvature_scores,
            curvature_traces,
            t0_speeds,
            args.stop_speed_threshold_mps,
            args.stop_duration_s,
            args.stop_clusters,
            args.stop_lateral_weight,
            args.stop_endpoint_lateral_weight,
            args.straight_clusters,
            args.turn_clusters,
            args.s_curve_clusters,
            args.curvature_threshold,
            args.straight_speed_mps,
            args.straight_min_forward_m,
            args.straight_max_final_lateral_m,
            args.straight_max_lateral_m,
            args.straight_max_slope,
            args.s_curve_curvature_threshold,
            args.s_curve_min_run_steps,
            args.seed,
        )
        print(
            "Stop split: "
            f"stop={stop_headers['stop_trajectories']} "
            f"remaining={len(rows)} "
            f"speed_threshold={args.stop_speed_threshold_mps} "
            f"duration_s={args.stop_duration_s}"
        )
        print(
            "Remaining curvature split: "
            f"straight={extra_headers['straight_trajectories']} "
            f"turn={extra_headers['turn_trajectories']} "
            f"s_curve={extra_headers.get('s_curve_trajectories', 0)} "
            f"geometry_straight={extra_headers.get('geometry_straight_trajectories', 0)} "
            f"threshold={args.curvature_threshold}"
        )
        write_results(
            args.stop_result_txt,
            stop_rows,
            stop_labels,
            stop_distances,
            stop_centers,
            stop_inertia,
            args.feature_mode,
            args.endpoint_weight,
            args.all_lateral_weight,
            export_stats,
            stop_headers,
        )
        print(f"Wrote stop K-Means results to {args.stop_result_txt}")
        write_split_txts(
            args.stop_split_txt,
            args.straight_split_txt,
            args.turn_split_txt,
            args.s_curve_split_txt,
            stop_rows,
            stop_labels,
            rows,
            labels,
            args.straight_clusters,
            args.turn_clusters,
        )
        write_category_center_txts(
            args.stop_center_txt,
            args.straight_center_txt,
            args.left_center_txt,
            args.right_center_txt,
            args.s_curve_center_txt,
            stop_rows,
            stop_labels,
            rows,
            labels,
            args.straight_clusters,
            args.turn_clusters,
        )
        print(
            "Wrote split trajectory and category center txt files: "
            f"{args.stop_split_txt}, {args.straight_split_txt}, "
            f"{args.turn_split_txt}, {args.s_curve_split_txt}, "
            f"{args.stop_center_txt}, {args.straight_center_txt}, "
            f"{args.left_center_txt}, {args.right_center_txt}, "
            f"{args.s_curve_center_txt}"
        )
    elif args.split_by_curvature:
        curvature_scores = load_curvature_scores(rows, args.train_data_root, args.steps)
        t0_speeds = load_t0_speeds(rows, args.train_data_root)
        labels, distances, centers, inertia, extra_headers = run_split_kmeans(
            rows,
            features,
            curvature_scores,
            None,
            t0_speeds,
            args.curvature_threshold,
            args.straight_speed_mps,
            args.straight_min_forward_m,
            args.straight_max_final_lateral_m,
            args.straight_max_lateral_m,
            args.straight_max_slope,
            args.straight_clusters,
            args.turn_clusters,
            args.seed,
        )
        print(
            "Curvature split: "
            f"straight={extra_headers['straight_trajectories']} "
            f"turn={extra_headers['turn_trajectories']} "
            f"geometry_straight={extra_headers.get('geometry_straight_trajectories', 0)} "
            f"threshold={args.curvature_threshold}"
        )
    else:
        n_clusters = min(args.n_clusters, len(rows))
        if n_clusters < args.n_clusters:
            print(f"Only {len(rows)} trajectories found; using n_clusters={n_clusters}")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=args.seed,
            n_init="auto",
            max_iter=300,
        )
        labels = kmeans.fit_predict(features)
        distances = np.linalg.norm(features - kmeans.cluster_centers_[labels], axis=1)
        centers = kmeans.cluster_centers_
        inertia = float(kmeans.inertia_)

    write_results(
        args.result_txt,
        rows,
        labels,
        distances,
        centers,
        inertia,
        args.feature_mode,
        args.endpoint_weight,
        args.all_lateral_weight,
        export_stats,
        extra_headers,
    )
    print(f"Wrote K-Means results to {args.result_txt}")


if __name__ == "__main__":
    main()
