#!/usr/bin/env python3
"""Cluster generated trajectories with K-Medoids.

This script is intentionally standalone and lives under output/ so it does not
touch the generation or GUI code.

The default v2 workflow clusters trajectories in a per-trajectory aligned
Cartesian frame: translate each trajectory to its start point, rotate its
initial motion direction to +x, then cluster weighted [x, y, vx, vy] curves.
This makes maneuvers with the same ego-relative intent comparable even when
their original world/clip headings differ.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "triplane_tokenization"
    / "data_cache"
    / "alpamayo_extracted"
)


@dataclass
class TrajectoryRecord:
    dataset: str
    clip: str
    file_path: Path
    t0_us: int
    sample_idx: int
    xy: np.ndarray
    aligned_xy: np.ndarray | None = None
    frenet: np.ndarray | None = None
    feature: np.ndarray | None = None
    behavior_label: str = ""
    cluster: int = -1
    is_medoid: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster generated trajectories using Frechet K-Medoids."
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result_dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["aligned_xy", "frenet"],
        default="aligned_xy",
        help="Feature space. aligned_xy is the v2 default; frenet keeps the first-version workflow.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of clusters.")
    parser.add_argument("--dt", type=float, default=0.1, help="Trajectory time step.")
    parser.add_argument("--frechet_points", type=int, default=16)
    parser.add_argument("--max_iter", type=int, default=50)
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument(
        "--weights",
        type=float,
        nargs=4,
        default=[0.7, 2.4, 0.5, 1.2],
        help="Feature weights. For aligned_xy: x y vx vy. For frenet: s l ds dl.",
    )
    parser.add_argument(
        "--output_prefix",
        default="cluster_v2",
        help="Prefix for result files, so v2 outputs do not overwrite the first-version images.",
    )
    parser.add_argument(
        "--exact_pairwise_max",
        type=int,
        default=1200,
        help="Use exact all-pairs Frechet only when N is at most this value.",
    )
    parser.add_argument(
        "--candidate_cap",
        type=int,
        default=160,
        help="Max candidate medoids evaluated per cluster in scalable mode.",
    )
    parser.add_argument(
        "--eval_cap",
        type=int,
        default=500,
        help="Max cluster members used to score candidate medoids in scalable mode.",
    )
    parser.add_argument(
        "--use_filtered",
        action="store_true",
        help="Use filtered parquet files instead of raw generated files when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick experiments; 0 means use all records.",
    )
    return parser.parse_args()


def list_trajectory_files(output_dir: Path, use_filtered: bool) -> list[Path]:
    files: list[Path] = []
    for dataset_dir in sorted(output_dir.glob("*_converted")):
        if not dataset_dir.is_dir():
            continue
        search_dir = dataset_dir / "filtered" if use_filtered and (dataset_dir / "filtered").exists() else dataset_dir
        files.extend(sorted(search_dir.glob("*.egomotion.parquet")))
    return files


def load_records(output_dir: Path, use_filtered: bool, limit: int = 0) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for file_path in list_trajectory_files(output_dir, use_filtered):
        dataset = file_path.parent.name
        if dataset == "filtered":
            dataset = file_path.parent.parent.name
        clip = file_path.name.replace(".egomotion.parquet", "")
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            x = np.asarray(row["x"], dtype=np.float64)
            y = np.asarray(row["y"], dtype=np.float64)
            if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 4:
                continue
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            records.append(
                TrajectoryRecord(
                    dataset=dataset,
                    clip=clip,
                    file_path=file_path,
                    t0_us=int(row["t0_us"]),
                    sample_idx=int(row.get("sample_idx", len(records))),
                    xy=np.column_stack([x, y]),
                )
            )
            if limit and len(records) >= limit:
                return records
    return records


def interpolate_pose(df_ego: pd.DataFrame, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ts = df_ego["timestamp"].to_numpy(dtype=np.float64)
    xyz = df_ego[["x", "y", "z"]].to_numpy(dtype=np.float64)
    quat = df_ego[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
    xyz_interp = interp1d(ts, xyz, kind="linear", axis=0, fill_value="extrapolate")
    quat_interp = interp1d(ts, quat, kind="linear", axis=0, fill_value="extrapolate")
    quat_out = quat_interp(timestamps)
    quat_norm = np.linalg.norm(quat_out, axis=1, keepdims=True)
    quat_out = quat_out / np.maximum(quat_norm, 1e-8)
    return xyz_interp(timestamps), quat_out


def estimate_heading(hist_xyz: np.ndarray, hist_quat: np.ndarray) -> Rotation:
    disp = hist_xyz[-1, :2] - hist_xyz[max(0, len(hist_xyz) - 6), :2]
    if float(np.linalg.norm(disp)) > 0.2:
        yaw = math.atan2(float(disp[1]), float(disp[0]))
        return Rotation.from_euler("z", yaw)
    return Rotation.from_quat(hist_quat[-1])


def load_history_local(
    data_root: Path,
    dataset: str,
    clip: str,
    t0_us: int,
    history_steps: int,
    dt: float,
) -> np.ndarray | None:
    ego_file = data_root / dataset / "data-egomotion" / f"{clip}.egomotion.parquet"
    if not ego_file.exists():
        return None
    df_ego = pd.read_parquet(ego_file)
    if df_ego.empty:
        return None
    dt_us = int(round(dt * 1_000_000))
    hist_ts = np.asarray(
        [t0_us - (history_steps - 1 - i) * dt_us for i in range(history_steps)],
        dtype=np.int64,
    )
    hist_xyz, hist_quat = interpolate_pose(df_ego, hist_ts)
    t0_xyz = hist_xyz[-1].copy()
    t0_rot_inv = estimate_heading(hist_xyz, hist_quat).inv()
    return t0_rot_inv.apply(hist_xyz - t0_xyz)[:, :2]


def build_history_reference(history_xy: np.ndarray, future_x_max: float) -> np.ndarray:
    """Create a local reference line from history plus forward extension."""
    history_xy = np.asarray(history_xy, dtype=np.float64)
    history_xy = history_xy[np.isfinite(history_xy).all(axis=1)]
    if len(history_xy) < 2:
        x_grid = np.linspace(-10.0, max(20.0, future_x_max + 5.0), 120)
        return np.column_stack([x_grid, np.zeros_like(x_grid)])

    x = history_xy[:, 0]
    y = history_xy[:, 1]
    unique_x = np.unique(np.round(x, 3))
    degree = int(min(2, max(1, len(unique_x) - 1)))
    try:
        coef = np.polyfit(x, y, deg=degree)
        x_min = min(float(np.nanmin(x)), -1.0)
        x_max = max(float(future_x_max) + 5.0, 20.0)
        x_grid = np.linspace(x_min, x_max, 160)
        y_grid = np.polyval(coef, x_grid)
    except Exception:
        direction = history_xy[-1] - history_xy[-2]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.asarray([1.0, 0.0])
        else:
            direction = direction / norm
        backward = history_xy[0]
        forward = history_xy[-1] + direction * max(float(future_x_max) + 5.0, 20.0)
        return np.vstack([backward, history_xy[-1], forward])

    ref = np.column_stack([x_grid, y_grid])
    t0 = np.asarray([[0.0, 0.0]])
    ref = np.vstack([ref, t0])
    order = np.argsort(ref[:, 0])
    return ref[order]


def project_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Project local xy points to signed Frenet-like [s, l]."""
    ref = np.asarray(reference, dtype=np.float64)
    seg_start = ref[:-1]
    seg_vec = ref[1:] - ref[:-1]
    seg_len = np.linalg.norm(seg_vec, axis=1)
    valid = seg_len > 1e-6
    seg_start = seg_start[valid]
    seg_vec = seg_vec[valid]
    seg_len = seg_len[valid]
    if len(seg_len) == 0:
        return np.column_stack([points[:, 0], points[:, 1]])

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    def _project_one(point: np.ndarray) -> tuple[float, float]:
        rel = point - seg_start
        t = np.sum(rel * seg_vec, axis=1) / (seg_len**2)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_start + seg_vec * t[:, None]
        d2 = np.sum((proj - point) ** 2, axis=1)
        idx = int(np.argmin(d2))
        tangent = seg_vec[idx] / seg_len[idx]
        normal_left = np.asarray([-tangent[1], tangent[0]])
        signed_l = float(np.dot(point - proj[idx], normal_left))
        s = float(cum[idx] + t[idx] * seg_len[idx])
        return s, signed_l

    projected = np.asarray([_project_one(p) for p in points], dtype=np.float64)
    t0_s, _ = _project_one(np.asarray([0.0, 0.0]))
    projected[:, 0] -= t0_s
    return projected


def make_features(records: list[TrajectoryRecord], data_root: Path, history_steps: int, dt: float) -> None:
    history_cache: dict[tuple[str, str, int], np.ndarray] = {}
    for rec in records:
        key = (rec.dataset, rec.clip, rec.t0_us)
        if key not in history_cache:
            hist = load_history_local(data_root, rec.dataset, rec.clip, rec.t0_us, history_steps, dt)
            if hist is None:
                hist = np.asarray([[-10.0, 0.0], [0.0, 0.0]])
            history_cache[key] = hist
        reference = build_history_reference(history_cache[key], float(np.nanmax(rec.xy[:, 0])))
        sl = project_to_reference(rec.xy, reference)
        ds = np.gradient(sl[:, 0], dt)
        dl = np.gradient(sl[:, 1], dt)
        rec.frenet = np.column_stack([sl[:, 0], sl[:, 1], ds, dl])


def estimate_initial_forward(xy: np.ndarray) -> np.ndarray:
    """Estimate the initial trajectory direction used for per-record alignment."""
    xy = np.asarray(xy, dtype=np.float64)
    origin = xy[0]
    max_probe = min(len(xy), 8)
    for i in range(1, max_probe):
        disp = xy[i] - origin
        norm = float(np.linalg.norm(disp))
        if norm > 0.25:
            return disp / norm

    disp = xy[-1] - origin
    norm = float(np.linalg.norm(disp))
    if norm > 1e-6:
        return disp / norm
    return np.asarray([1.0, 0.0], dtype=np.float64)


def align_trajectory_to_initial_frame(xy: np.ndarray) -> np.ndarray:
    """Translate to the first point and rotate initial motion to +x.

    Output convention is x forward and y left. A right turn should therefore
    move toward negative y after alignment, independent of the original heading.
    """
    xy = np.asarray(xy, dtype=np.float64)
    origin = xy[0]
    forward = estimate_initial_forward(xy)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    delta = xy - origin
    return np.column_stack([delta @ forward, delta @ left])


def make_aligned_xy_features(records: list[TrajectoryRecord], dt: float) -> None:
    for rec in records:
        aligned = align_trajectory_to_initial_frame(rec.xy)
        vx = np.gradient(aligned[:, 0], dt)
        vy = np.gradient(aligned[:, 1], dt)
        rec.aligned_xy = aligned
        rec.feature = np.column_stack([aligned[:, 0], aligned[:, 1], vx, vy])


def robust_standardize(features: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1])
    med = np.nanmedian(flat, axis=0)
    q25 = np.nanpercentile(flat, 25, axis=0)
    q75 = np.nanpercentile(flat, 75, axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    weighted = (features - med) / iqr
    return weighted * np.asarray(list(weights), dtype=np.float64)


def downsample_curve(curve: np.ndarray, n_points: int) -> np.ndarray:
    if len(curve) <= n_points:
        return curve
    idx = np.linspace(0, len(curve) - 1, n_points).round().astype(int)
    return curve[idx]


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    ca = np.full((n, m), np.inf, dtype=np.float64)
    d00 = np.linalg.norm(a[0] - b[0])
    ca[0, 0] = d00
    for i in range(n):
        for j in range(m):
            d = np.linalg.norm(a[i] - b[j])
            if i == 0 and j == 0:
                continue
            best_prev = np.inf
            if i > 0:
                best_prev = min(best_prev, ca[i - 1, j])
            if j > 0:
                best_prev = min(best_prev, ca[i, j - 1])
            if i > 0 and j > 0:
                best_prev = min(best_prev, ca[i - 1, j - 1])
            ca[i, j] = max(best_prev, d)
    return float(ca[-1, -1])


def pairwise_frechet(curves: list[np.ndarray]) -> np.ndarray:
    n = len(curves)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if i and i % 50 == 0:
            print(f"Computed distances for {i}/{n} trajectories...")
        for j in range(i + 1, n):
            d = discrete_frechet(curves[i], curves[j])
            dist[i, j] = dist[j, i] = d
    return dist


def kmedoids(distance: np.ndarray, k: int, max_iter: int) -> tuple[np.ndarray, np.ndarray]:
    n = distance.shape[0]
    k = min(max(1, k), n)
    medoids = [int(np.argmin(distance.sum(axis=1)))]
    while len(medoids) < k:
        nearest = np.min(distance[:, medoids], axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))
    medoids_arr = np.asarray(medoids, dtype=int)

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        labels = np.argmin(distance[:, medoids_arr], axis=1)
        new_medoids = medoids_arr.copy()
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(distance[:, medoids_arr], axis=1)))
                new_medoids[c] = farthest
                continue
            intra = distance[np.ix_(members, members)].sum(axis=1)
            new_medoids[c] = int(members[np.argmin(intra)])
        if np.array_equal(new_medoids, medoids_arr):
            break
        medoids_arr = new_medoids
    labels = np.argmin(distance[:, medoids_arr], axis=1)
    return labels, medoids_arr


def squared_euclidean_to_medoids(flat: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    med = flat[medoids]
    return ((flat[:, None, :] - med[None, :, :]) ** 2).sum(axis=2)


def init_medoids_from_vectors(flat: np.ndarray, k: int) -> np.ndarray:
    n = flat.shape[0]
    k = min(max(1, k), n)
    center = flat.mean(axis=0)
    medoids = [int(np.argmin(((flat - center) ** 2).sum(axis=1)))]
    min_dist = ((flat - flat[medoids[0]]) ** 2).sum(axis=1)
    while len(medoids) < k:
        min_dist[medoids] = -1.0
        next_idx = int(np.argmax(min_dist))
        medoids.append(next_idx)
        min_dist = np.minimum(min_dist, ((flat - flat[next_idx]) ** 2).sum(axis=1))
    return np.asarray(medoids, dtype=int)


def assign_to_frechet_medoids(curves: list[np.ndarray], medoids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(curves)
    k = len(medoids)
    dist_to_medoids = np.zeros((n, k), dtype=np.float64)
    for m_pos, m_idx in enumerate(medoids):
        medoid_curve = curves[int(m_idx)]
        for i, curve in enumerate(curves):
            dist_to_medoids[i, m_pos] = 0.0 if i == int(m_idx) else discrete_frechet(curve, medoid_curve)
    labels = np.argmin(dist_to_medoids, axis=1)
    return labels, dist_to_medoids


def choose_cluster_medoid_approx(
    curves: list[np.ndarray],
    flat: np.ndarray,
    members: np.ndarray,
    current_medoid: int,
    candidate_cap: int,
    eval_cap: int,
    rng: np.random.Generator,
) -> int:
    if len(members) <= 2:
        return int(members[0])

    cluster_center = flat[members].mean(axis=0)
    vec_dist = ((flat[members] - cluster_center) ** 2).sum(axis=1)
    ordered_members = members[np.argsort(vec_dist)]
    candidates = ordered_members[: min(candidate_cap, len(ordered_members))]
    if current_medoid in members and current_medoid not in candidates:
        candidates = np.concatenate([[current_medoid], candidates])

    if len(members) <= eval_cap:
        eval_members = members
    else:
        near_count = eval_cap // 2
        rand_count = eval_cap - near_count
        near = ordered_members[:near_count]
        remaining = np.setdiff1d(members, near, assume_unique=False)
        rand = rng.choice(remaining, size=min(rand_count, len(remaining)), replace=False)
        eval_members = np.unique(np.concatenate([near, rand, candidates]))

    best_idx = int(candidates[0])
    best_cost = np.inf
    for cand in candidates:
        cand_curve = curves[int(cand)]
        cost = 0.0
        for member in eval_members:
            if int(member) == int(cand):
                continue
            cost += discrete_frechet(cand_curve, curves[int(member)])
        if cost < best_cost:
            best_cost = cost
            best_idx = int(cand)
    return best_idx


def scalable_frechet_kmedoids(
    curves: list[np.ndarray],
    flat: np.ndarray,
    k: int,
    max_iter: int,
    candidate_cap: int,
    eval_cap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    medoids = init_medoids_from_vectors(flat, k)
    labels = np.full(len(curves), -1, dtype=int)
    dist_to_medoids = np.empty((len(curves), len(medoids)), dtype=np.float64)

    for iteration in range(max_iter):
        print(f"Scalable K-Medoids iteration {iteration + 1}/{max_iter}...")
        labels, dist_to_medoids = assign_to_frechet_medoids(curves, medoids)
        new_medoids = medoids.copy()
        for c in range(len(medoids)):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                farthest = int(np.argmax(np.min(dist_to_medoids, axis=1)))
                new_medoids[c] = farthest
                continue
            new_medoids[c] = choose_cluster_medoid_approx(
                curves=curves,
                flat=flat,
                members=members,
                current_medoid=int(medoids[c]),
                candidate_cap=candidate_cap,
                eval_cap=eval_cap,
                rng=rng,
            )
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids

    labels, dist_to_medoids = assign_to_frechet_medoids(curves, medoids)
    return labels, medoids, dist_to_medoids


def classical_mds(distance: np.ndarray) -> np.ndarray:
    n = distance.shape[0]
    d2 = distance**2
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order[:2]], 0.0)
    vecs = vecs[:, order[:2]]
    return vecs * np.sqrt(vals)


def pca_embedding(flat: np.ndarray) -> np.ndarray:
    centered = flat - flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def cluster_shape_stats(record: TrajectoryRecord) -> dict[str, float]:
    xy = record.aligned_xy if record.aligned_xy is not None else record.xy
    tail_n = max(2, min(5, len(xy)))
    tail = xy[-1] - xy[-tail_n]
    angle = math.atan2(float(tail[1]), float(tail[0])) if np.linalg.norm(tail) > 1e-6 else 0.0
    return {
        "end_x": float(xy[-1, 0]),
        "end_y": float(xy[-1, 1]),
        "max_abs_y": float(np.max(np.abs(xy[:, 1]))),
        "signed_lateral_area": float(np.trapezoid(xy[:, 1], xy[:, 0])),
        "final_heading_rad": float(angle),
    }


def infer_behavior_labels(records: list[TrajectoryRecord], medoids: np.ndarray) -> dict[int, str]:
    """Attach approximate maneuver labels to arbitrary K-Medoids cluster ids."""
    available = {int(records[int(i)].cluster): int(i) for i in medoids}
    labels: dict[int, str] = {}
    if not available:
        return labels

    stats = {cluster_id: cluster_shape_stats(records[idx]) for cluster_id, idx in available.items()}
    straight_cluster = min(
        stats,
        key=lambda c: abs(stats[c]["end_y"]) + 0.35 * stats[c]["max_abs_y"] + 8.0 * abs(stats[c]["final_heading_rad"]),
    )
    labels[straight_cluster] = "straight"

    remaining = [c for c in stats if c != straight_cluster]
    left = [c for c in remaining if stats[c]["end_y"] >= 0.0]
    right = [c for c in remaining if stats[c]["end_y"] < 0.0]

    def assign_side(side_clusters: list[int], turn_label: str, detour_label: str) -> None:
        if not side_clusters:
            return
        ordered = sorted(
            side_clusters,
            key=lambda c: abs(stats[c]["final_heading_rad"]) + 0.02 * abs(stats[c]["end_y"]),
            reverse=True,
        )
        labels[ordered[0]] = turn_label
        for c in ordered[1:]:
            labels[c] = detour_label

    assign_side(left, "left_turn", "left_detour")
    assign_side(right, "right_turn", "right_detour")

    desired = ["straight", "left_turn", "right_turn", "left_detour", "right_detour"]
    used = set(labels.values())
    fallback = [name for name in desired if name not in used]
    for c in sorted(stats):
        if c not in labels:
            labels[c] = fallback.pop(0) if fallback else "other"
    return labels


def save_assignments(
    records: list[TrajectoryRecord],
    medoids: np.ndarray,
    distance: np.ndarray | None,
    result_dir: Path,
    output_prefix: str,
    dist_to_medoids: np.ndarray | None = None,
) -> None:
    medoid_set = set(int(i) for i in medoids)
    label_map = infer_behavior_labels(records, medoids)
    for idx, rec in enumerate(records):
        rec.is_medoid = idx in medoid_set
        rec.behavior_label = label_map.get(rec.cluster, "")

    rows = []
    for idx, rec in enumerate(records):
        aligned = rec.aligned_xy if rec.aligned_xy is not None else rec.xy
        fre_end_s = float(rec.frenet[-1, 0]) if rec.frenet is not None else float("nan")
        fre_end_l = float(rec.frenet[-1, 1]) if rec.frenet is not None else float("nan")
        rows.append(
            {
                "record_idx": idx,
                "cluster": rec.cluster,
                "behavior_label": rec.behavior_label,
                "is_medoid": rec.is_medoid,
                "dataset": rec.dataset,
                "clip": rec.clip,
                "t0_us": rec.t0_us,
                "sample_idx": rec.sample_idx,
                "file_path": str(rec.file_path),
                "distance_to_medoid": (
                    float(dist_to_medoids[idx, rec.cluster])
                    if dist_to_medoids is not None and rec.cluster >= 0
                    else 0.0
                ),
                "end_x": float(rec.xy[-1, 0]),
                "end_y": float(rec.xy[-1, 1]),
                "aligned_end_x": float(aligned[-1, 0]),
                "aligned_end_y": float(aligned[-1, 1]),
                "end_s": fre_end_s,
                "end_l": fre_end_l,
            }
        )
    pd.DataFrame(rows).to_csv(result_dir / f"{output_prefix}_assignments.csv", index=False)

    summary_rows = []
    for cluster_id in sorted(set(r.cluster for r in records)):
        members = [i for i, r in enumerate(records) if r.cluster == cluster_id]
        medoid = next(i for i in members if i in medoid_set)
        if distance is not None:
            mean_dist = float(distance[np.ix_(members, members)].mean()) if len(members) > 1 else 0.0
            mean_to_medoid = float(np.mean([distance[i, medoid] for i in members]))
        elif dist_to_medoids is not None:
            mean_dist = float("nan")
            mean_to_medoid = float(np.mean(dist_to_medoids[members, cluster_id]))
        else:
            mean_dist = float("nan")
            mean_to_medoid = float("nan")
        r = records[medoid]
        aligned = r.aligned_xy if r.aligned_xy is not None else r.xy
        stats = cluster_shape_stats(r)
        summary_rows.append(
            {
                "cluster": cluster_id,
                "behavior_label": label_map.get(cluster_id, ""),
                "size": len(members),
                "medoid_record_idx": medoid,
                "medoid_dataset": r.dataset,
                "medoid_clip": r.clip,
                "medoid_t0_us": r.t0_us,
                "medoid_sample_idx": r.sample_idx,
                "mean_intra_frechet": mean_dist,
                "mean_to_medoid_frechet": mean_to_medoid,
                "medoid_aligned_end_x": float(aligned[-1, 0]),
                "medoid_aligned_end_y": float(aligned[-1, 1]),
                "medoid_max_abs_y": stats["max_abs_y"],
                "medoid_final_heading_rad": stats["final_heading_rad"],
                "medoid_end_s": float(r.frenet[-1, 0]) if r.frenet is not None else float("nan"),
                "medoid_end_l": float(r.frenet[-1, 1]) if r.frenet is not None else float("nan"),
            }
        )
    pd.DataFrame(summary_rows).to_csv(result_dir / f"{output_prefix}_summary.csv", index=False)


def plot_mds(
    records: list[TrajectoryRecord],
    embedding: np.ndarray,
    medoids: np.ndarray,
    result_dir: Path,
    output_prefix: str,
    title: str,
) -> None:
    clusters = np.asarray([r.cluster for r in records])
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap="tab10", s=18, alpha=0.75)
    plt.scatter(
        embedding[medoids, 0],
        embedding[medoids, 1],
        c=clusters[medoids],
        cmap="tab10",
        s=180,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        label="Medoids",
    )
    for idx in medoids:
        plt.text(embedding[idx, 0], embedding[idx, 1], f" C{records[idx].cluster}", fontsize=9)
    plt.title(title)
    plt.xlabel("Axis 1")
    plt.ylabel("Axis 2")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(result_dir / f"{output_prefix}_embedding.png", dpi=180)
    plt.close()


def plot_cluster_trajectories(records: list[TrajectoryRecord], medoids: np.ndarray, result_dir: Path, output_prefix: str) -> None:
    cluster_ids = sorted(set(r.cluster for r in records))
    ncols = 3
    nrows = math.ceil(len(cluster_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10")
    medoid_set = set(int(i) for i in medoids)

    for ax, cluster_id in zip(axes.ravel(), cluster_ids):
        members = [r for r in records if r.cluster == cluster_id]
        for r in members:
            ax.plot(r.frenet[:, 1], r.frenet[:, 0], color="0.65", alpha=0.18, linewidth=0.8)
        for idx, r in enumerate(records):
            if idx in medoid_set and r.cluster == cluster_id:
                ax.plot(
                    r.frenet[:, 1],
                    r.frenet[:, 0],
                    color=colors(cluster_id % 10),
                    linewidth=2.6,
                    label=f"medoid idx={idx}",
                )
                ax.scatter(r.frenet[-1, 1], r.frenet[-1, 0], color=colors(cluster_id % 10), s=35)
                break
        ax.set_title(f"Cluster {cluster_id} | n={len(members)}")
        ax.set_xlabel("l: lateral offset (m)")
        ax.set_ylabel("s: longitudinal progress (m)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(cluster_ids):]:
        ax.axis("off")
    fig.suptitle("Clustered Trajectories in History-Fitted Frenet Space", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(result_dir / f"{output_prefix}_frenet_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_xy_clusters(records: list[TrajectoryRecord], medoids: np.ndarray, result_dir: Path, output_prefix: str) -> None:
    cluster_ids = sorted(set(r.cluster for r in records))
    ncols = 3
    nrows = math.ceil(len(cluster_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10")
    medoid_set = set(int(i) for i in medoids)

    for ax, cluster_id in zip(axes.ravel(), cluster_ids):
        members = [r for r in records if r.cluster == cluster_id]
        label = members[0].behavior_label if members else ""
        for r in members:
            xy = r.aligned_xy if r.aligned_xy is not None else r.xy
            ax.plot(xy[:, 1], xy[:, 0], color="0.65", alpha=0.16, linewidth=0.8)
        for idx, r in enumerate(records):
            if idx in medoid_set and r.cluster == cluster_id:
                xy = r.aligned_xy if r.aligned_xy is not None else r.xy
                ax.plot(
                    xy[:, 1],
                    xy[:, 0],
                    color=colors(cluster_id % 10),
                    linewidth=2.8,
                    label=f"{label} medoid idx={idx}",
                )
                ax.scatter(xy[-1, 1], xy[-1, 0], color=colors(cluster_id % 10), s=35)
                break
        ax.axhline(0.0, color="0.2", alpha=0.25, linewidth=0.8)
        ax.axvline(0.0, color="0.2", alpha=0.25, linewidth=0.8)
        ax.set_title(f"Cluster {cluster_id} | {label} | n={len(members)}")
        ax.set_xlabel("y left (m)")
        ax.set_ylabel("x forward (m)")
        ax.axis("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(cluster_ids):]:
        ax.axis("off")
    fig.suptitle("V2 Clustered Trajectories in Per-Trajectory Aligned XY", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(result_dir / f"{output_prefix}_aligned_xy_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_medoid_xy(
    records: list[TrajectoryRecord],
    medoids: np.ndarray,
    result_dir: Path,
    output_prefix: str,
    aligned: bool,
) -> None:
    plt.figure(figsize=(8, 9))
    colors = plt.get_cmap("tab10")
    for idx, r in enumerate(records):
        xy = r.aligned_xy if aligned and r.aligned_xy is not None else r.xy
        plt.plot(xy[:, 1], xy[:, 0], color=colors(r.cluster % 10), alpha=0.08, linewidth=0.7)
    for idx in medoids:
        r = records[int(idx)]
        xy = r.aligned_xy if aligned and r.aligned_xy is not None else r.xy
        label = f"C{r.cluster} {r.behavior_label} medoid #{idx}"
        plt.plot(xy[:, 1], xy[:, 0], color=colors(r.cluster % 10), linewidth=2.8, label=label)
        plt.scatter(xy[-1, 1], xy[-1, 0], color=colors(r.cluster % 10), s=45)
    frame_name = "Aligned Ego XY" if aligned else "Original Ego XY"
    plt.title(f"V2 Cluster Medoids in {frame_name} (x forward, y left)")
    plt.xlabel("y left (m)")
    plt.ylabel("x forward (m)")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    suffix = "medoids_aligned_xy" if aligned else "medoids_original_xy"
    plt.savefig(result_dir / f"{output_prefix}_{suffix}.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir or (args.output_dir / "clustering_results")
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading trajectories from {args.output_dir}")
    records = load_records(args.output_dir, args.use_filtered, args.limit)
    if len(records) < 2:
        raise SystemExit("Need at least two valid trajectories to cluster.")
    print(f"Loaded {len(records)} trajectories.")

    if args.mode == "frenet":
        print("Building history-fitted Frenet features...")
        make_features(records, args.data_root, args.history_steps, args.dt)
        raw_features = np.asarray([r.frenet for r in records], dtype=np.float64)
    else:
        print("Building v2 per-trajectory aligned Cartesian features...")
        make_aligned_xy_features(records, args.dt)
        raw_features = np.asarray([r.feature for r in records], dtype=np.float64)

    weighted = robust_standardize(raw_features, args.weights)
    for rec, feat in zip(records, weighted):
        rec.feature = feat

    curves = [downsample_curve(r.feature, args.frechet_points) for r in records]
    flat = np.asarray([r.feature.reshape(-1) for r in records], dtype=np.float64)

    distance = None
    dist_to_medoids = None
    if len(records) <= args.exact_pairwise_max:
        print(f"Computing exact pairwise discrete Frechet distances with {args.frechet_points} points...")
        distance = pairwise_frechet(curves)
        np.save(result_dir / f"{args.output_prefix}_pairwise_frechet.npy", distance)
        print(f"Running exact K-Medoids with k={args.k}...")
        labels, medoids = kmedoids(distance, args.k, args.max_iter)
        embedding = classical_mds(distance)
    else:
        print(
            "Dataset is large; using scalable Frechet K-Medoids "
            f"(N={len(records)}, exact_pairwise_max={args.exact_pairwise_max})."
        )
        labels, medoids, dist_to_medoids = scalable_frechet_kmedoids(
            curves=curves,
            flat=flat,
            k=args.k,
            max_iter=args.max_iter,
            candidate_cap=args.candidate_cap,
            eval_cap=args.eval_cap,
        )
        np.save(result_dir / f"{args.output_prefix}_distance_to_medoids.npy", dist_to_medoids)
        embedding = pca_embedding(flat)

    for rec, label in zip(records, labels):
        rec.cluster = int(label)

    save_assignments(
        records,
        medoids,
        distance,
        result_dir,
        output_prefix=args.output_prefix,
        dist_to_medoids=dist_to_medoids,
    )
    print("Creating visualizations...")
    plot_mds(
        records,
        embedding,
        medoids,
        result_dir,
        output_prefix=args.output_prefix,
        title=(
            "V2 Aligned-XY Frechet K-Medoids Clusters (2D Feature View)"
            if args.mode == "aligned_xy"
            else "Frenet Frechet K-Medoids Clusters (2D Distance/Feature View)"
        ),
    )
    if args.mode == "aligned_xy":
        plot_aligned_xy_clusters(records, medoids, result_dir, args.output_prefix)
        plot_medoid_xy(records, medoids, result_dir, args.output_prefix, aligned=True)
        plot_medoid_xy(records, medoids, result_dir, args.output_prefix, aligned=False)
    else:
        plot_cluster_trajectories(records, medoids, result_dir, args.output_prefix)
        plot_medoid_xy(records, medoids, result_dir, args.output_prefix, aligned=False)

    print(f"Done. Results written to {result_dir}")
    print("Key files:")
    print(f"  {result_dir / f'{args.output_prefix}_embedding.png'}")
    if args.mode == "aligned_xy":
        print(f"  {result_dir / f'{args.output_prefix}_aligned_xy_trajectories.png'}")
        print(f"  {result_dir / f'{args.output_prefix}_medoids_aligned_xy.png'}")
        print(f"  {result_dir / f'{args.output_prefix}_medoids_original_xy.png'}")
    else:
        print(f"  {result_dir / f'{args.output_prefix}_frenet_trajectories.png'}")
        print(f"  {result_dir / f'{args.output_prefix}_medoids_original_xy.png'}")
    print(f"  {result_dir / f'{args.output_prefix}_summary.csv'}")
    print(f"  {result_dir / f'{args.output_prefix}_assignments.csv'}")


if __name__ == "__main__":
    main()
