#!/usr/bin/env python3
"""Cluster 5-8 trajectories into 500 acceleration/curvature centers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import cluster_acc_curvature_top2 as source


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_TXT = SCRIPT_DIR / "future_trajectories_5_8_xy.txt"
FALLBACK_DATA_TXT = SCRIPT_DIR / "future_trajectories_xy.txt"
DEFAULT_TRAIN_DATA_ROOT = Path("/home/ubuntu/Public/train_data")
DEFAULT_OUTPUT_TXT = SCRIPT_DIR / "acc_curvature_500_centers.txt"
DEFAULT_SUMMARY_JSON = SCRIPT_DIR / "acc_curvature_500_summary.json"
DEFAULT_DATASET = "data_26_5_8_converted"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_txt", type=Path, default=DEFAULT_DATA_TXT)
    parser.add_argument("--train_data_root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--dataset_filter", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output_txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--n_clusters", type=int, default=500)
    parser.add_argument("--speed_smooth_passes", type=int, default=1)
    parser.add_argument("--acceleration_smooth_passes", type=int, default=2)
    parser.add_argument("--curvature_smooth_passes", type=int, default=1)
    parser.add_argument("--feature_clip_percentile", type=float, default=99.0)
    parser.add_argument("--curvature_min_speed_mps", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_data_txt(path: Path) -> Path:
    if path.exists():
        return path
    if path == DEFAULT_DATA_TXT and FALLBACK_DATA_TXT.exists():
        return FALLBACK_DATA_TXT
    raise FileNotFoundError(f"Trajectory txt not found: {path}")


def _jsonable_stats(feature_stats: dict) -> dict:
    out = source.feature_stats_for_json(feature_stats)
    out["mean"] = np.asarray(feature_stats["mean"], dtype=np.float64).tolist()
    out["std"] = np.asarray(feature_stats["std"], dtype=np.float64).tolist()
    return out


def write_center_file(
    output_txt: Path,
    rows: list[source.TrajectoryRow],
    labels: np.ndarray,
    centers: np.ndarray,
    medoid_indices: np.ndarray,
    steps: int,
    feature_stats: dict,
    metadata: dict[str, object],
) -> None:
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    counts = np.bincount(labels, minlength=len(centers))
    native_centers = source.inverse_standardized_feature(centers, feature_stats)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("# acc_curvature_kmeans_centers_v1\n")
        for key, value in metadata.items():
            f.write(f"# {key}={value}\n")
        f.write("# feature: acceleration[0:steps] curvature[0:steps]\n")
        f.write("# smoothing: existing native_parquet speed/acceleration/curvature policy\n")
        f.write("# CENTER center_id count medoid_traj_id acc0 ... acc63 curv0 ... curv63\n")
        f.write("# CENTER_MEDOID_XY center_id x0 y0 ... x63 y63\n")
        for center_id, center in enumerate(native_centers):
            medoid_row_idx = int(medoid_indices[center_id])
            medoid_traj_id = -1 if medoid_row_idx < 0 else int(rows[medoid_row_idx].traj_id)
            values = " ".join(f"{v:.6f}" for v in center.reshape(-1))
            f.write(f"CENTER {center_id} {int(counts[center_id])} {medoid_traj_id} {values}\n")
            if medoid_row_idx >= 0:
                xy_values = " ".join(f"{v:.6f}" for v in rows[medoid_row_idx].xy.reshape(-1))
                f.write(f"CENTER_MEDOID_XY {center_id} {xy_values}\n")


def run_clustering(args: argparse.Namespace) -> dict:
    data_txt = resolve_data_txt(args.data_txt)
    loaded_rows = source.load_future_rows(data_txt, args.steps)
    rows = source.filter_rows_by_dataset(loaded_rows, args.dataset_filter)
    (
        rows,
        raw_features,
        acceleration,
        curvature,
        raw_acceleration,
        raw_curvature,
        speed,
        filter_stats,
    ) = source.build_native_acc_curvature_features(
        rows,
        args.train_data_root,
        args.steps,
        curvature_min_speed_mps=args.curvature_min_speed_mps,
        speed_smooth_passes=args.speed_smooth_passes,
        acceleration_smooth_passes=args.acceleration_smooth_passes,
        curvature_smooth_passes=args.curvature_smooth_passes,
    )
    features, feature_stats = source.robust_standardize(
        raw_features,
        args.feature_clip_percentile,
    )
    n_clusters = min(int(args.n_clusters), len(rows))
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        n_init="auto",
        max_iter=300,
    )
    labels = kmeans.fit_predict(features)
    medoid_indices = source.choose_medoids(
        features,
        labels,
        kmeans.cluster_centers_,
        n_clusters,
    )
    metadata = {
        "data_txt": data_txt,
        "train_data_root": args.train_data_root,
        "dataset_filter": args.dataset_filter,
        "steps": int(args.steps),
        "n_rows": len(rows),
        "n_clusters": n_clusters,
        "speed_smooth_passes": int(args.speed_smooth_passes),
        "acceleration_smooth_passes": int(args.acceleration_smooth_passes),
        "curvature_smooth_passes": int(args.curvature_smooth_passes),
        "curvature_min_speed_mps": float(args.curvature_min_speed_mps),
        "feature_clip_percentile": float(args.feature_clip_percentile),
        "seed": int(args.seed),
        "inertia": f"{float(kmeans.inertia_):.6f}",
    }
    write_center_file(
        args.output_txt,
        rows,
        labels,
        kmeans.cluster_centers_,
        medoid_indices,
        args.steps,
        feature_stats,
        metadata,
    )
    summary = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in metadata.items()},
        "filter_stats": filter_stats,
        "feature_stats": _jsonable_stats(feature_stats),
        "raw_acceleration_mean_abs": float(np.mean(np.abs(raw_acceleration))),
        "raw_curvature_mean_abs": float(np.mean(np.abs(raw_curvature))),
        "smoothed_acceleration_mean_abs": float(np.mean(np.abs(acceleration))),
        "smoothed_curvature_mean_abs": float(np.mean(np.abs(curvature))),
        "native_speed_mean_mps": float(np.mean(speed)),
        "output_txt": str(args.output_txt),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_clustering(args)
    print(
        f"Clustered {summary['n_rows']} rows into {summary['n_clusters']} centers; "
        f"wrote {summary['output_txt']}"
    )
    print(f"Wrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
