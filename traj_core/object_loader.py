"""Load and prepare traffic participant objects for BEV annotation overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


OBJECT_COLUMNS = [
    "timestamp",
    "frame_index",
    "source",
    "object_id",
    "object_type",
    "confidence",
    "x",
    "y",
    "z",
    "x_kf",
    "y_kf",
    "z_kf",
    "heading",
    "origin_heading",
    "length",
    "width",
    "height",
    "vx_rel",
    "vy_rel",
    "vz_rel",
    "vx_abs",
    "vy_abs",
    "vz_abs",
    "life_time",
    "lost_time",
    "object_timestamp",
]


def empty_objects_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=OBJECT_COLUMNS)


def load_objects_for_clip(data_root: str | Path, dataset_name: str, clip_stem: str) -> pd.DataFrame:
    """Load traffic participant parquet for one converted dataset clip."""
    object_path = (
        Path(data_root)
        / str(dataset_name)
        / "data-objects"
        / f"{clip_stem}.objects.parquet"
    )
    if not object_path.exists():
        return empty_objects_frame()
    df = pd.read_parquet(object_path)
    for column in OBJECT_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df[OBJECT_COLUMNS].sort_values(["timestamp", "object_id"]).reset_index(drop=True)


def nearest_objects_at_timestamp(
    objects_df: pd.DataFrame,
    t0_us: int,
    max_delta_us: int = 150_000,
) -> pd.DataFrame:
    """Return all objects from the nearest object frame around t0."""
    if objects_df is None or objects_df.empty or "timestamp" not in objects_df.columns:
        return empty_objects_frame()
    timestamps = pd.Series(objects_df["timestamp"].dropna().astype("int64").unique())
    if timestamps.empty:
        return empty_objects_frame()
    deltas = (timestamps - int(t0_us)).abs()
    nearest_idx = int(deltas.idxmin())
    if int(deltas.loc[nearest_idx]) > int(max_delta_us):
        return empty_objects_frame()
    nearest_timestamp = int(timestamps.loc[nearest_idx])
    return objects_df[objects_df["timestamp"].astype("int64") == nearest_timestamp].reset_index(drop=True)


def _finite_float(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _center_xy(row: Mapping, prefer_filtered: bool = True) -> tuple[float, float]:
    if prefer_filtered:
        x_kf = _finite_float(row.get("x_kf"), default=np.nan)
        y_kf = _finite_float(row.get("y_kf"), default=np.nan)
        if np.isfinite(x_kf) and np.isfinite(y_kf):
            return x_kf, y_kf
    return _finite_float(row.get("x")), _finite_float(row.get("y"))


def object_center_xy(row: Mapping, prefer_filtered: bool = True) -> tuple[float, float]:
    """Return object center as GUI ego-local x-forward/y-left meters.

    Calmcar BEV object points store x as lateral-right and y as backward
    relative to the ego frame used by this GUI.  The annotation GUI uses
    x-forward/y-left, so both axes need a sign flip and the axes are swapped.
    """
    lateral_right, backward = _center_xy(row, prefer_filtered=prefer_filtered)
    return -backward, -lateral_right


def object_footprint_xy(row: Mapping, prefer_filtered: bool = True) -> np.ndarray:
    """Return the four BEV rectangle corners in ego-local x-forward/y-left meters."""
    center_x, center_y = _center_xy(row, prefer_filtered=prefer_filtered)
    heading = _finite_float(row.get("heading"))
    length = max(_finite_float(row.get("length")), 0.2)
    width = max(_finite_float(row.get("width")), 0.2)

    forward = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    left = np.array([-np.sin(heading), np.cos(heading)], dtype=np.float64)
    center = np.array([center_x, center_y], dtype=np.float64)
    half_l = 0.5 * length
    half_w = 0.5 * width
    return np.array(
        [
            center + forward * half_l + left * half_w,
            center + forward * half_l - left * half_w,
            center - forward * half_l - left * half_w,
            center - forward * half_l + left * half_w,
        ],
        dtype=np.float64,
    )


__all__ = [
    "OBJECT_COLUMNS",
    "empty_objects_frame",
    "load_objects_for_clip",
    "nearest_objects_at_timestamp",
    "object_center_xy",
    "object_footprint_xy",
]
