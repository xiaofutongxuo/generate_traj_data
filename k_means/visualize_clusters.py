#!/usr/bin/env python3
"""Visualize K-Means trajectory clusters from txt outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_TXT = SCRIPT_DIR / "future_trajectories_xy.txt"
DEFAULT_RESULT_TXT = SCRIPT_DIR / "kmeans_results.txt"
DEFAULT_IMAGE = SCRIPT_DIR / "kmeans_clusters_xy.png"
DEFAULT_TRAIN_DATA_ROOT = (
    SCRIPT_DIR.parent.parent
    / "triplane_tokenization"
    / "data_cache"
    / "alpamayo_extracted"
)


def ego_xy_to_plot_xy(xy: np.ndarray) -> np.ndarray:
    """Plot ego-local x-forward/y-left as x-right/y-forward BEV axes."""
    out = np.empty_like(xy)
    out[..., 0] = -xy[..., 1]
    out[..., 1] = xy[..., 0]
    return out


def trajectory_curvature_abs_sum(xy: np.ndarray) -> float:
    """Approximate sum(abs(curvature)) from an ego-local xy trajectory."""
    origin_xy = np.vstack([np.zeros((1, 2), dtype=np.float64), xy])
    delta = np.diff(origin_xy, axis=0)
    step_distance = np.linalg.norm(delta, axis=1)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    moving = step_distance > 1e-3
    if moving.any():
        first = int(np.flatnonzero(moving)[0])
        yaw[:first] = yaw[first]
        for idx in range(first + 1, len(yaw)):
            if not moving[idx]:
                yaw[idx] = yaw[idx - 1]
        yaw = np.unwrap(yaw)
    else:
        yaw[:] = 0.0

    curvature = np.zeros_like(step_distance)
    if len(step_distance) > 1:
        curvature[1:] = np.diff(yaw) / np.maximum(step_distance[1:], 1e-6)
        curvature[~np.isfinite(curvature)] = 0.0
    return float(np.abs(curvature).sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw clustered xy trajectories.")
    parser.add_argument("--data_txt", type=Path, default=DEFAULT_DATA_TXT)
    parser.add_argument("--result_txt", type=Path, default=DEFAULT_RESULT_TXT)
    parser.add_argument(
        "--center_txt",
        type=Path,
        default=None,
        help="Draw centers from a category center txt such as left.txt/right.txt.",
    )
    parser.add_argument(
        "--split_txt",
        type=Path,
        default=None,
        help="Optional split trajectory txt to draw samples behind category centers.",
    )
    parser.add_argument(
        "--direction",
        choices=["left", "right", "none"],
        default=None,
        help="Optional direction filter for split trajectory txt.",
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--train_data_root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_curvature_labels", action="store_true")
    parser.add_argument(
        "--speed_labels",
        action="store_true",
        help="Annotate each cluster at the timestep where a member reaches its minimum future speed.",
    )
    return parser.parse_args()


def load_data(data_txt: Path, steps: int) -> tuple[np.ndarray, list[int], list[tuple[str, str, int]]]:
    trajs: list[np.ndarray] = []
    ids: list[int] = []
    metas: list[tuple[str, str, int]] = []
    with data_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            ids.append(int(parts[0]))
            metas.append((parts[1], parts[2], int(parts[4])))
            trajs.append(np.asarray(parts[5:], dtype=np.float64).reshape(steps, 2))
    return np.stack(trajs, axis=0), ids, metas


def load_results(result_txt: Path, steps: int) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    labels_by_id: dict[int, int] = {}
    centers: list[np.ndarray] = []
    counts: list[int] = []

    with result_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "CENTER":
                counts.append(int(parts[2]))
                centers.append(np.asarray(parts[3:], dtype=np.float64).reshape(steps, 2))
            elif parts[0] == "CENTER_DYNAMICS":
                continue
            else:
                labels_by_id[int(parts[0])] = int(parts[1])

    return labels_by_id, np.stack(centers, axis=0), np.asarray(counts, dtype=np.int64)


def load_center_file(center_txt: Path, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_ids: list[int] = []
    centers: list[np.ndarray] = []
    counts: list[int] = []

    with center_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "CENTER":
                continue
            xy = np.asarray(parts[3:], dtype=np.float64).reshape(-1, 2)
            if len(xy) != steps:
                raise ValueError(f"{center_txt} contains {len(xy)} steps, expected {steps}")
            center_ids.append(int(parts[1]))
            counts.append(int(parts[2]))
            centers.append(xy)

    if not centers:
        raise ValueError(f"No CENTER rows found in {center_txt}")

    order = np.argsort(np.asarray(center_ids, dtype=np.int64))
    return (
        np.asarray(center_ids, dtype=np.int64)[order],
        np.stack(centers, axis=0)[order],
        np.asarray(counts, dtype=np.int64)[order],
    )


def load_split_samples(
    split_txt: Path,
    center_ids: set[int],
    steps: int,
    direction: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    trajs: list[np.ndarray] = []
    labels: list[int] = []

    with split_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            row_direction = parts[2]
            cluster_id = int(parts[3])
            if cluster_id not in center_ids:
                continue
            if direction is not None and row_direction != direction:
                continue
            xy = np.asarray(parts[8:], dtype=np.float64).reshape(-1, 2)
            if len(xy) != steps:
                continue
            labels.append(cluster_id)
            trajs.append(xy)

    if not trajs:
        return (
            np.zeros((0, steps, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )
    return np.stack(trajs, axis=0), np.asarray(labels, dtype=np.int64)


def load_curvature_scores(
    train_data_root: Path,
    metas: list[tuple[str, str, int]],
    steps: int,
) -> np.ndarray:
    cache: dict[tuple[str, str], np.ndarray] = {}
    scores = []
    for dataset, clip, t0_idx in metas:
        key = (dataset, clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / dataset
                / "data-egomotion"
                / f"{clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["curvature"])
            cache[key] = df["curvature"].to_numpy(dtype=np.float64)
        curv = cache[key][t0_idx + 1 : t0_idx + 1 + steps]
        scores.append(float(np.abs(curv).sum()) if len(curv) == steps else np.nan)
    return np.asarray(scores, dtype=np.float64)


def load_future_speeds(
    train_data_root: Path,
    metas: list[tuple[str, str, int]],
    steps: int,
) -> np.ndarray:
    cache: dict[tuple[str, str], np.ndarray] = {}
    speeds = []
    for dataset, clip, t0_idx in metas:
        key = (dataset, clip)
        if key not in cache:
            parquet_file = (
                train_data_root
                / dataset
                / "data-egomotion"
                / f"{clip}.egomotion.parquet"
            )
            df = pd.read_parquet(parquet_file, columns=["vx", "vy", "vz"])
            cache[key] = np.linalg.norm(
                df[["vx", "vy", "vz"]].to_numpy(dtype=np.float64),
                axis=1,
            )
        future_speed = cache[key][t0_idx + 1 : t0_idx + 1 + steps]
        if len(future_speed) == steps:
            speeds.append(future_speed)
        else:
            speeds.append(np.full(steps, np.nan, dtype=np.float64))
    return np.asarray(speeds, dtype=np.float64)


def main() -> None:
    args = parse_args()
    metas = []
    center_ids = None
    if args.center_txt is not None:
        center_ids, centers, counts = load_center_file(args.center_txt, args.steps)
        if args.split_txt is not None:
            trajs, labels = load_split_samples(
                args.split_txt,
                set(int(v) for v in center_ids),
                args.steps,
                args.direction,
            )
        else:
            trajs = np.zeros((0, args.steps, 2), dtype=np.float64)
            labels = np.zeros((0,), dtype=np.int64)
    else:
        trajs, traj_ids, metas = load_data(args.data_txt, args.steps)
        labels_by_id, centers, counts = load_results(args.result_txt, args.steps)
        keep = np.asarray([traj_id in labels_by_id for traj_id in traj_ids], dtype=bool)
        trajs = trajs[keep]
        traj_ids = [traj_id for traj_id, keep_row in zip(traj_ids, keep) if keep_row]
        metas = [meta for meta, keep_row in zip(metas, keep) if keep_row]
        labels = np.asarray([labels_by_id[i] for i in traj_ids], dtype=np.int64)
    trajs_plot = ego_xy_to_plot_xy(trajs)
    centers_plot = ego_xy_to_plot_xy(centers)
    center_curvature = np.asarray(
        [trajectory_curvature_abs_sum(center) for center in centers],
        dtype=np.float64,
    )
    curvature_scores = None
    cluster_curvature = None
    if args.center_txt is None and not args.no_curvature_labels and not args.speed_labels:
        curvature_scores = load_curvature_scores(args.train_data_root, metas, args.steps)
        cluster_curvature = np.full(len(centers), np.nan, dtype=np.float64)
        for cluster_id in range(len(centers)):
            member_scores = curvature_scores[labels == cluster_id]
            member_scores = member_scores[np.isfinite(member_scores)]
            if len(member_scores):
                cluster_curvature[cluster_id] = float(member_scores.mean())
    future_speeds = None
    cluster_min_speed = None
    cluster_min_speed_index = None
    if args.center_txt is None and args.speed_labels:
        future_speeds = load_future_speeds(args.train_data_root, metas, args.steps)
        cluster_min_speed = np.full(len(centers), np.nan, dtype=np.float64)
        cluster_min_speed_index = np.zeros(len(centers), dtype=np.int64)
        for cluster_id in range(len(centers)):
            member_speeds = future_speeds[labels == cluster_id]
            if len(member_speeds) == 0 or not np.isfinite(member_speeds).any():
                continue
            flat_idx = int(np.nanargmin(member_speeds))
            _, step_idx = np.unravel_index(flat_idx, member_speeds.shape)
            cluster_min_speed[cluster_id] = float(member_speeds.reshape(-1)[flat_idx])
            cluster_min_speed_index[cluster_id] = int(step_idx)

    rng = np.random.default_rng(args.seed)
    if len(trajs) > args.max_samples:
        sample_idx = rng.choice(len(trajs), size=args.max_samples, replace=False)
    else:
        sample_idx = np.arange(len(trajs))

    cmap = plt.get_cmap("tab20", max(20, len(centers)))
    fig, ax = plt.subplots(figsize=(12, 10), dpi=160)

    if center_ids is None:
        color_ids = np.arange(len(centers), dtype=np.int64)
    else:
        color_ids = center_ids
    color_by_cluster = {
        int(cluster_id): cmap(color_idx % cmap.N)
        for color_idx, cluster_id in enumerate(color_ids)
    }

    for idx in sample_idx:
        color = color_by_cluster.get(int(labels[idx]), cmap(int(labels[idx]) % cmap.N))
        ax.plot(
            trajs_plot[idx, :, 0],
            trajs_plot[idx, :, 1],
            color=color,
            alpha=0.08,
            linewidth=0.7,
        )

    order = np.argsort(counts)
    for center_idx in order:
        center = centers_plot[center_idx]
        cluster_id = int(color_ids[center_idx])
        color = color_by_cluster.get(cluster_id, cmap(cluster_id % cmap.N))
        ax.plot(center[:, 0], center[:, 1], color=color, linewidth=2.4, alpha=0.95)
        ax.scatter(center[-1, 0], center[-1, 1], s=8 + 0.25 * counts[center_idx], color=color, alpha=0.9)
        ax.annotate(
            f"{cluster_id}",
            xy=(center[-1, 0], center[-1, 1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            color="black",
            alpha=0.8,
        )
        if np.isfinite(center_curvature[center_idx]):
            ax.annotate(
                f"{center_curvature[center_idx]:.2f}",
                xy=(center[-1, 0], center[-1, 1]),
                xytext=(3, -8),
                textcoords="offset points",
                fontsize=6,
                color="black",
                alpha=0.8,
            )
        if (
            cluster_min_speed is not None
            and cluster_min_speed_index is not None
            and np.isfinite(cluster_min_speed[center_idx])
        ):
            step_idx = int(cluster_min_speed_index[center_idx])
            point = center[step_idx]
            ax.scatter(point[0], point[1], s=18, marker="x", color="black", linewidths=0.9)
            ax.annotate(
                f"{cluster_min_speed[center_idx]:.2f}m/s",
                xy=(point[0], point[1]),
                xytext=(3, -8),
                textcoords="offset points",
                fontsize=6,
                color="black",
                alpha=0.85,
            )

    ax.scatter([0], [0], s=45, color="black", marker="+", linewidths=1.8, label="ego")
    title = args.title or f"K-Means future trajectories, k={len(centers)}, n={len(trajs)}"
    title += ", endpoint labels: center sum|curvature|"
    if cluster_min_speed is not None:
        title += ", x marks cluster min speed"
    ax.set_title(title)
    ax.set_xlabel("right (m)")
    ax.set_ylabel("forward (m)")
    ax.axis("equal")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper right")

    args.image.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.image)
    plt.close(fig)
    print(f"Wrote visualization to {args.image}")


if __name__ == "__main__":
    main()
