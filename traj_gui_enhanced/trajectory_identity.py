"""Trajectory identity helpers for GUI delete/save workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

TrajectoryKey = tuple[int | None, int]


def normalize_trajectory_source(source: Any) -> str:
    """Normalize source labels written by older and newer GUI actions."""
    try:
        if pd.isna(source):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(source or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "manual": "manual_bezier",
        "manual_curve": "manual_bezier",
        "bezier": "manual_bezier",
        "cluster": "cluster_center",
        "cluster_center_preview": "cluster_center",
        "vlm": "vla",
        "model": "vla",
        "model_generated": "vla",
        "vla_generated": "vla",
    }
    return aliases.get(text, text)


def _record_value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return record.get(key, default)
    except AttributeError:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def trajectory_key_from_record(
    record: Mapping[str, Any],
    fallback_index: int,
    fallback_t0_us: int | None = None,
) -> TrajectoryKey:
    """Return a stable per-sample key for one trajectory row."""
    t0_us = _safe_int(_record_value(record, "t0_us", fallback_t0_us), fallback_t0_us)
    sample_idx = _safe_int(_record_value(record, "sample_idx", fallback_index), fallback_index)
    return t0_us, int(sample_idx if sample_idx is not None else fallback_index)


def is_gt_trajectory_record(record: Mapping[str, Any], fallback_index: int = -1) -> bool:
    """Return whether a trajectory is an explicit protected GT row.

    Real GT is loaded from the source dataset. Output rows are pseudo-GT unless
    they explicitly carry source=gt from legacy GUI writes.
    """
    source = normalize_trajectory_source(_record_value(record, "source", ""))
    return source == "gt"


def is_deletable_trajectory_record(record: Mapping[str, Any], fallback_index: int = -1) -> bool:
    """Return whether the GUI may stage-delete this trajectory."""
    return not is_gt_trajectory_record(record, fallback_index)


def drop_trajectory_rows_by_keys(
    df: pd.DataFrame,
    current_t0_us: int | None,
    deleted_keys: set[TrajectoryKey],
) -> pd.DataFrame:
    """Drop staged deleted rows from a parquet DataFrame."""
    if df.empty or not deleted_keys:
        return df.copy()

    drop_indices = []
    for fallback_index, (row_idx, row) in enumerate(df.iterrows()):
        if "t0_us" in df.columns:
            row_t0 = _safe_int(row.get("t0_us"), None)
            if current_t0_us is not None and row_t0 != int(current_t0_us):
                continue
        key = trajectory_key_from_record(
            row,
            fallback_index=fallback_index,
            fallback_t0_us=current_t0_us,
        )
        if key in deleted_keys:
            drop_indices.append(row_idx)

    if not drop_indices:
        return df.copy()
    return df.drop(index=drop_indices).reset_index(drop=True)


__all__ = [
    "TrajectoryKey",
    "drop_trajectory_rows_by_keys",
    "is_deletable_trajectory_record",
    "is_gt_trajectory_record",
    "normalize_trajectory_source",
    "trajectory_key_from_record",
]
