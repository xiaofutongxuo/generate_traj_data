"""Video-frame t0 indexing helpers shared by inference and GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VideoFrameCandidates:
    """Candidate t0 timestamps built from a clip's master video timestamps."""

    t0_values: list[int]
    total_video_frames: int
    valid_t0_count: int
    frame_stride: int
    first_valid_frame_index: int
    last_valid_frame_index: int


def master_timestamp_file(data_root: str | Path, dataset_name: str, clip_stem: str) -> Path:
    return (
        Path(data_root)
        / dataset_name
        / "data-timestamps"
        / f"{clip_stem}.timestamps.parquet"
    )


def load_master_video_timestamps(
    data_root: str | Path,
    dataset_name: str,
    clip_stem: str,
) -> np.ndarray:
    """Load the 10 Hz master video timestamps for one clip."""
    ts_file = master_timestamp_file(data_root, dataset_name, clip_stem)
    if not ts_file.exists():
        return np.zeros(0, dtype=np.int64)
    df = pd.read_parquet(ts_file, columns=["timestamp"])
    return df["timestamp"].to_numpy(dtype=np.int64)


def valid_video_frame_indices(
    frame_count: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    frame_stride: int = 1,
    require_full_history: bool = False,
    require_full_future: bool = False,
) -> list[int]:
    """Return video frame indices to use as t0 samples."""
    frame_count = max(0, int(frame_count))
    stride = max(1, int(frame_stride))
    if frame_count == 0:
        return []

    start_idx = max(0, int(num_history_steps) - 1) if require_full_history else 0
    end_idx = frame_count - 1 - max(0, int(num_future_steps)) if require_full_future else frame_count - 1
    if start_idx > end_idx:
        return []
    return list(range(start_idx, end_idx + 1, stride))


def build_video_frame_t0_candidates(
    data_root: str | Path,
    dataset_name: str,
    clip_stem: str,
    frame_stride: int = 1,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    require_full_history: bool = False,
    require_full_future: bool = False,
    allowed_timestamps: Optional[set[int]] = None,
) -> VideoFrameCandidates:
    """Build t0 timestamps from every selected master video frame."""
    timestamps = load_master_video_timestamps(data_root, dataset_name, clip_stem)
    indices = valid_video_frame_indices(
        len(timestamps),
        num_history_steps=num_history_steps,
        num_future_steps=num_future_steps,
        frame_stride=frame_stride,
        require_full_history=require_full_history,
        require_full_future=require_full_future,
    )
    t0_values = [int(timestamps[idx]) for idx in indices]
    if allowed_timestamps is not None:
        allowed = {int(value) for value in allowed_timestamps}
        t0_values = [value for value in t0_values if int(value) in allowed]

    return VideoFrameCandidates(
        t0_values=t0_values,
        total_video_frames=int(len(timestamps)),
        valid_t0_count=int(len(t0_values)),
        frame_stride=max(1, int(frame_stride)),
        first_valid_frame_index=int(indices[0]) if indices else -1,
        last_valid_frame_index=int(indices[-1]) if indices else -1,
    )


def video_frame_coverage_summary(
    total_video_t0: int,
    generated_t0: int,
) -> str:
    """Return a compact coverage label for GUI status text."""
    total = max(0, int(total_video_t0))
    generated = max(0, int(generated_t0))
    missing = max(0, total - generated)
    return f"Video t0: {total} | Generated t0: {generated} | Missing: {missing}"


__all__ = [
    "VideoFrameCandidates",
    "build_video_frame_t0_candidates",
    "load_master_video_timestamps",
    "master_timestamp_file",
    "valid_video_frame_indices",
    "video_frame_coverage_summary",
]
