"""SampleIOMixin for the enhanced trajectory GUI."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ..tk_compat import Image, ImageTk, messagebox, tk, ttk

from ..environment import setup_environment
setup_environment()

from data_loader import (
    filter_t0s_with_full_future,
    get_dataset_names,
    get_clip_stems_from_dataset,
    load_data,
    get_t0_candidates,
)
from frame_index import build_video_frame_t0_candidates, video_frame_coverage_summary
from calibration_loader import load_calibration_for_segment
from visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

from ..constants import *
from ..math_utils import *
from ..speed_utils import *
from ..projection_utils import *
from ..cluster_utils import *
from ..dynamics import optimize_pseudo_gt_trajectory, trajectory_components_from_xyz
from ..trajectory_identity import drop_trajectory_rows_by_keys, normalize_trajectory_source

class SampleIOMixin:
    def _output_pseudo_gt_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return output rows that are generated/augmented pseudo-GT, not real GT."""
        if df.empty:
            return df.copy()
        if "source" not in df.columns:
            return df.copy()
        source = df["source"].map(normalize_trajectory_source)
        return df[source != "gt"].copy()

    def _load_sample_index(self, start_dataset: str = "", start_clip: str = "") -> list[tuple[str, str, int]]:
        """Build the navigable sample list from GT or generated trajectory files."""
        if self.index_mode in {"video_frames", "merged"}:
            return self._load_video_frame_sample_index(
                start_dataset=start_dataset,
                start_clip=start_clip,
            )
        if self.gt_only:
            return self._load_gt_sample_index(start_dataset=start_dataset, start_clip=start_clip)
        return self._load_generated_sample_index()

    def _generated_t0_values_for_clip(self, dataset_name: str, clip_stem: str) -> set[int]:
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        if not traj_file.exists():
            return set()
        try:
            df = pd.read_parquet(traj_file)
        except Exception as e:
            print(f"Warning: Could not load generated t0 index {traj_file}: {e}")
            return set()
        df = self._output_pseudo_gt_rows(df)
        if "t0_us" not in df.columns or len(df) == 0:
            return set()
        return {int(value) for value in df["t0_us"].dropna().astype("int64").unique()}

    def _clip_stems_for_video_index(self, dataset_path: Path) -> list[str]:
        clip_stems = get_clip_stems_from_dataset(dataset_path)
        if clip_stems:
            return clip_stems
        timestamps_dir = dataset_path / "data-timestamps"
        if not timestamps_dir.exists():
            return []
        return sorted(
            path.name.replace(".timestamps.parquet", "")
            for path in timestamps_dir.glob("*.timestamps.parquet")
            if "_fovs_" not in path.name
        )

    def _load_video_frame_sample_index(
        self,
        start_dataset: str = "",
        start_clip: str = "",
    ) -> list[tuple[str, str, int]]:
        """Build samples from the master 10 Hz video-frame timestamps."""
        dataset_filter = start_dataset.strip()
        clip_filter = start_clip.strip()
        samples: list[tuple[str, str, int]] = []
        self.video_t0_count_by_clip = {}
        self.generated_t0_count_by_clip = {}
        self.generated_t0_by_clip = {}

        for dataset_name in self.datasets:
            if dataset_filter and dataset_name != dataset_filter:
                continue
            dataset_path = self.data_root / dataset_name
            for clip_stem in self._clip_stems_for_video_index(dataset_path):
                if clip_filter and clip_stem != clip_filter:
                    continue

                candidates = build_video_frame_t0_candidates(
                    self.data_root,
                    dataset_name,
                    clip_stem,
                    frame_stride=self.frame_stride,
                    require_full_history=False,
                    require_full_future=False,
                )
                video_t0_values = filter_t0s_with_full_future(
                    self.data_root,
                    dataset_name,
                    [int(value) for value in candidates.t0_values],
                    clip_stem=clip_stem,
                )
                generated_t0_values = self._generated_t0_values_for_clip(dataset_name, clip_stem)
                key = (dataset_name, clip_stem)
                self.video_t0_count_by_clip[key] = len(video_t0_values)
                self.generated_t0_count_by_clip[key] = len(generated_t0_values)
                self.generated_t0_by_clip[key] = generated_t0_values

                samples.extend((dataset_name, clip_stem, t0_us) for t0_us in video_t0_values)

                if self.index_mode == "merged":
                    video_set = set(video_t0_values)
                    samples.extend(
                        (dataset_name, clip_stem, t0_us)
                        for t0_us in sorted(generated_t0_values - video_set)
                    )

        samples.sort(key=lambda x: (x[0], x[1], x[2]))
        total_video_t0 = sum(self.video_t0_count_by_clip.values())
        total_generated_t0 = sum(self.generated_t0_count_by_clip.values())
        print(
            f"Video-frame sample index: {len(samples)} samples "
            f"(mode={self.index_mode}, frame_stride={self.frame_stride}, "
            f"video_t0={total_video_t0}, generated_t0={total_generated_t0})"
        )
        return samples

    def _load_generated_sample_index(self) -> list[tuple[str, str, int]]:
        samples: list[tuple[str, str, int]] = []
        for dataset_name in self.datasets:
            dataset_output = self.output_dir / dataset_name
            if not dataset_output.exists():
                continue

            for parquet_file in dataset_output.glob("*.egomotion.parquet"):
                clip_stem = str(parquet_file.name).replace(".egomotion.parquet", "")
                try:
                    df = pd.read_parquet(parquet_file)
                    df = self._output_pseudo_gt_rows(df)
                    if len(df) > 0:
                        if "t0_us" in df.columns:
                            for t0_us in sorted(df["t0_us"].dropna().astype("int64").unique()):
                                samples.append((dataset_name, clip_stem, int(t0_us)))
                        else:
                            samples.append((dataset_name, clip_stem, _row_t0_us(df.iloc[0])))
                except Exception as e:
                    print(f"Warning: Could not load {parquet_file}: {e}")
        samples.sort(key=lambda x: (x[0], x[1], x[2]))
        return samples

    def _load_gt_sample_index(self, start_dataset: str = "", start_clip: str = "") -> list[tuple[str, str, int]]:
        """Build GT-only samples directly from data-egomotion at a fixed frame stride."""
        dataset_filter = start_dataset.strip()
        clip_filter = start_clip.strip()
        samples: list[tuple[str, str, int]] = []

        for dataset_name in self.datasets:
            if dataset_filter and dataset_name != dataset_filter:
                continue
            dataset_path = self.data_root / dataset_name
            egomotion_dir = dataset_path / "data-egomotion"
            if not egomotion_dir.exists():
                continue

            clip_stems = get_clip_stems_from_dataset(dataset_path)
            if not clip_stems:
                clip_stems = sorted(
                    path.name.replace(".egomotion.parquet", "")
                    for path in egomotion_dir.glob("*.egomotion.parquet")
                )

            for clip_stem in clip_stems:
                if clip_filter and clip_stem != clip_filter:
                    continue
                ego_file = egomotion_dir / f"{clip_stem}.egomotion.parquet"
                if not ego_file.exists():
                    continue
                try:
                    df = pd.read_parquet(ego_file, columns=["timestamp"])
                except Exception as e:
                    print(f"Warning: Could not load GT timestamps {ego_file}: {e}")
                    continue
                timestamps = df["timestamp"].to_numpy(dtype=np.int64)
                # load_data defaults to 16 history steps and 64 future steps.
                lo = 16
                hi = len(timestamps) - 65
                if lo > hi:
                    continue
                for idx in range(lo, hi + 1, self.gt_stride_frames):
                    samples.append((dataset_name, clip_stem, int(timestamps[idx])))

        samples.sort(key=lambda x: (x[0], x[1], x[2]))
        print(
            f"GT-only sample index: {len(samples)} samples "
            f"(stride={self.gt_stride_frames} frames)"
        )
        return samples

    def _load_cot_index(self):
        """Load CoT sidecar records keyed by dataset, clip, t0, sample."""
        cot_file = self.output_dir / "cot.jsonl"
        cot_index = {}
        if not cot_file.exists():
            return cot_index

        try:
            with open(cot_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (
                        row.get("dataset_name", ""),
                        row.get("clip_id", ""),
                        int(row.get("t0_us", 0)),
                        int(row.get("sample_idx", 0)),
                    )
                    cot_index[key] = row.get("cot", "")
        except Exception as e:
            print(f"Warning: Could not load CoT sidecar {cot_file}: {e}")
        return cot_index

    def _load_viewer_state(self) -> dict:
        """Load the last viewed sample state."""
        if not self.viewer_state_file.exists():
            return {}
        try:
            with open(self.viewer_state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            return state if isinstance(state, dict) else {}
        except Exception as e:
            print(f"Warning: Could not load viewer state {self.viewer_state_file}: {e}")
            return {}

    def _save_viewer_state(self) -> None:
        """Persist the current sample so the next GUI launch can resume here."""
        if not (0 <= self.current_idx < len(self.samples)):
            return
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        state = {
            "sample_index": int(self.current_idx),
            "dataset_name": dataset_name,
            "clip_id": clip_stem,
            "t0_us": int(t0_us),
        }
        try:
            self.viewer_state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.viewer_state_file.with_suffix(".json.tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.write("\n")
            tmp_file.replace(self.viewer_state_file)
        except Exception as e:
            print(f"Warning: Could not save viewer state {self.viewer_state_file}: {e}")

    def _find_sample_index(
        self,
        dataset_name: str = "",
        clip_stem: str = "",
        t0_us: Optional[int] = None,
    ) -> Optional[int]:
        """Find a sample by dataset, clip, and optionally t0."""
        dataset_name = dataset_name.strip()
        clip_stem = clip_stem.strip()
        for idx, (sample_dataset, sample_clip, sample_t0) in enumerate(self.samples):
            if dataset_name and sample_dataset != dataset_name:
                continue
            if clip_stem and sample_clip != clip_stem:
                continue
            if t0_us is not None and int(sample_t0) != int(t0_us):
                continue
            return idx
        return None

    def _resolve_start_index(
        self,
        start_index: Optional[int],
        start_dataset: str,
        start_clip: str,
        start_t0: Optional[int],
        restore_last: bool,
    ) -> int:
        """Resolve the initial sample index from CLI arguments or saved state."""
        has_manual_target = (
            start_index is not None
            or bool(start_dataset.strip())
            or bool(start_clip.strip())
            or start_t0 is not None
        )

        if start_index is not None:
            return int(np.clip(int(start_index) - 1, 0, len(self.samples) - 1))

        if has_manual_target:
            found = self._find_sample_index(start_dataset, start_clip, start_t0)
            if found is not None:
                return found
            print(
                "Warning: Requested start sample not found; falling back to first sample "
                f"(dataset={start_dataset!r}, clip={start_clip!r}, t0={start_t0!r})"
            )
            return 0

        if restore_last:
            state = self._load_viewer_state()
            found = self._find_sample_index(
                str(state.get("dataset_name", "")),
                str(state.get("clip_id", "")),
                int(state["t0_us"]) if "t0_us" in state else None,
            )
            if found is not None:
                return found
            if "sample_index" in state:
                return int(np.clip(int(state["sample_index"]), 0, len(self.samples) - 1))

        return 0

    def _base_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        return self.output_dir / dataset_name / f"{clip_stem}.egomotion.parquet"

    def _active_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        return self._base_traj_file(dataset_name, clip_stem)

    def _first_editable_trajectory_index(self) -> int:
        for idx, traj in enumerate(self.trajectories):
            if not self._is_gt_trajectory(traj, idx):
                return idx
        return 0

    def _load_manual_line_points_index(self) -> dict[tuple[str, str, int], list[dict]]:
        """Load saved manual BEV line points keyed by dataset, clip, and t0."""
        if not self.manual_points_file.exists():
            return {}

        try:
            with open(self.manual_points_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manual line points {self.manual_points_file}: {e}")
            return {}

        records = data.get("samples", []) if isinstance(data, dict) else data
        points_index = {}
        for record in records:
            try:
                key = (
                    record.get("dataset_name", ""),
                    record.get("clip_id", ""),
                    int(record.get("t0_us", 0)),
                )
                points = []
                for point in record.get("line_points", []):
                    points.append({
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                        "z": float(point.get("z", 0.0)),
                    })
                if points:
                    points_index[key] = points
            except Exception as e:
                print(f"Warning: Skipping invalid manual line point record: {e}")
        return points_index

    def _load_manual_camera_line_points_index(self) -> dict[tuple[str, str, int], list[dict]]:
        """Load saved manual image line points keyed by dataset, clip, and t0."""
        if not self.manual_points_file.exists():
            return {}

        try:
            with open(self.manual_points_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manual camera line points {self.manual_points_file}: {e}")
            return {}

        records = data.get("samples", []) if isinstance(data, dict) else data
        points_index = {}
        for record in records:
            try:
                key = (
                    record.get("dataset_name", ""),
                    record.get("clip_id", ""),
                    int(record.get("t0_us", 0)),
                )
                points = []
                for point in record.get("camera_line_points", []):
                    points.append({
                        "camera": str(point.get("camera", "")),
                        "u": float(point.get("u", 0.0)),
                        "v": float(point.get("v", 0.0)),
                    })
                if points:
                    points_index[key] = points
            except Exception as e:
                print(f"Warning: Skipping invalid manual camera line point record: {e}")
        return points_index

    def _load_manual_stop_points_index(self) -> dict[tuple[str, str, int], list[dict]]:
        """Load saved stop markers keyed by dataset, clip, and t0."""
        if not self.manual_points_file.exists():
            return {}

        try:
            with open(self.manual_points_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manual stop points {self.manual_points_file}: {e}")
            return {}

        records = data.get("samples", []) if isinstance(data, dict) else data
        stop_index = {}
        for record in records:
            try:
                key = (
                    record.get("dataset_name", ""),
                    record.get("clip_id", ""),
                    int(record.get("t0_us", 0)),
                )
                stops = []
                for stop in record.get("stop_points", []):
                    stops.append({
                        "fraction": float(np.clip(float(stop.get("fraction", 0.0)), 0.0, 1.0)),
                        "duration_s": float(np.clip(float(stop.get("duration_s", 2.0)), 0.1, 6.0)),
                    })
                if stops:
                    stop_index[key] = sorted(stops, key=lambda item: item["fraction"])
            except Exception as e:
                print(f"Warning: Skipping invalid manual stop point record: {e}")
        return stop_index

    def _current_manual_points_key(self) -> tuple[str, str, int]:
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        return dataset_name, clip_stem, int(t0_us)

    def _current_sample_key(self) -> tuple[str, str, int]:
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        return dataset_name, clip_stem, int(t0_us)

    def _write_manual_points_index(self):
        records = []
        all_keys = (
            set(self.manual_line_points_index)
            | set(self.manual_camera_line_points_index)
            | set(self.manual_stop_points_index)
        )
        for dataset_name, clip_stem, t0_us in sorted(all_keys):
            line_points = self.manual_line_points_index.get((dataset_name, clip_stem, t0_us), [])
            camera_line_points = self.manual_camera_line_points_index.get((dataset_name, clip_stem, t0_us), [])
            stop_points = self.manual_stop_points_index.get((dataset_name, clip_stem, t0_us), [])
            if not line_points and not camera_line_points and not stop_points:
                continue
            records.append({
                "dataset_name": dataset_name,
                "clip_id": clip_stem,
                "t0_us": int(t0_us),
                "line_points": [
                    {
                        "x": float(point["x"]),
                        "y": float(point["y"]),
                        "z": float(point.get("z", 0.0)),
                    }
                    for point in line_points
                ],
                "camera_line_points": [
                    {
                        "camera": point["camera"],
                        "u": float(point["u"]),
                        "v": float(point["v"]),
                    }
                    for point in camera_line_points
                ],
                "stop_points": [
                    {
                        "fraction": float(np.clip(float(point["fraction"]), 0.0, 1.0)),
                        "duration_s": float(max(0.1, point["duration_s"])),
                    }
                    for point in stop_points
                ],
            })

        self.manual_points_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.manual_points_file.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "samples": records}, f, indent=2)
            f.write("\n")
        tmp_file.replace(self.manual_points_file)

    def _load_sample(self, idx: int):
        """Load a sample by index."""
        if idx < 0 or idx >= len(self.samples):
            return
        
        self.current_idx = idx
        dataset_name, clip_stem, t0_us = self.samples[idx]
        self.gt_future_mode = "raw"
        self.speed_hover_frame_idx = None
        self.speed_hover_source = None
        self._cancel_speed_edit(redraw=False)
        self._cancel_gt_speed_edit(redraw=False)
        
        self.trajectories = []
        self._reset_pending_delete_state()
        if not self.gt_only:
            # Load trajectories from parquet
            traj_file = self._active_traj_file(dataset_name, clip_stem)
            if traj_file.exists():
                df = pd.read_parquet(traj_file)
            else:
                df = pd.DataFrame()
            df = self._output_pseudo_gt_rows(df)
            if "t0_us" in df.columns:
                df = df[df["t0_us"].astype("int64") == int(t0_us)]

            for row_idx, (_, row) in enumerate(df.iterrows()):
                sample_idx = int(row["sample_idx"]) if "sample_idx" in row else row_idx
                cot_key = (dataset_name, clip_stem, int(t0_us), sample_idx)
                self.trajectories.append({
                    "sample_idx": sample_idx,
                    "source": normalize_trajectory_source(row.get("source", "")) if hasattr(row, "get") else "",
                    "timestamp": row["timestamp"],
                    "x": np.array(row["x"]),
                    "y": np.array(row["y"]),
                    "z": np.array(row["z"]),
                    "qx": np.array(row["qx"]),
                    "qy": np.array(row["qy"]),
                    "qz": np.array(row["qz"]),
                    "qw": np.array(row["qw"]),
                    "vx": np.array(row["vx"]),
                    "vy": np.array(row["vy"]),
                    "vz": np.array(row["vz"]),
                    "curvature": np.array(row["curvature"]),
                    "cot": self.cot_index.get(cot_key, ""),
                })
        
        # Initialize states (all kept by default)
        self.trajectory_states = {i: True for i in range(len(self.trajectories))}
        self.current_traj_idx = self._first_editable_trajectory_index()
        self.manual_line_points = [
            dict(point)
            for point in self.manual_line_points_index.get((dataset_name, clip_stem, int(t0_us)), [])
        ]
        self.manual_camera_line_points = [
            dict(point)
            for point in self.manual_camera_line_points_index.get((dataset_name, clip_stem, int(t0_us)), [])
        ]
        self.manual_stop_points = [
            dict(point)
            for point in self.manual_stop_points_index.get((dataset_name, clip_stem, int(t0_us)), [])
        ]
        self.manual_point_actions = []
        self.manual_line_points_dirty = False
        self.manual_camera_line_points_dirty = False
        self.manual_stop_points_dirty = False
        sample_key = (dataset_name, clip_stem, int(t0_us))
        self.camera_base_images = {}
        
        # Load original data for visualization
        try:
            cached_visual = self.visual_data_cache.get(sample_key)
            if cached_visual is not None:
                self.visual_data_cache.move_to_end(sample_key)
                self.conv_data, self.calibration = cached_visual
            else:
                # Load only the one frame displayed by the GUI. Inference uses 4 frames,
                # but this viewer only renders frames[cam_idx, 0].
                self.conv_data = load_data(
                    str(self.data_root),
                    clip_stem,
                    dataset_name,
                    t0_us=t0_us,
                    num_frames=1,
                    target_image_hw=(1080, 1920),
                    cameras=self.cameras,
                )
                
                # Load calibration for this segment
                calib_dataset = dataset_name.replace('_converted', '')
                try:
                    self.calibration = load_calibration_for_segment(
                        str(self.calibration_dir),
                        calib_dataset,
                        clip_stem,
                        data_root=str(self.data_root),
                        target_image_hw=(1080, 1920),
                    )
                except Exception as calib_error:
                    print(f"Warning: Could not load calibration: {calib_error}")
                    self.calibration = None
                self.visual_data_cache[sample_key] = (self.conv_data, self.calibration)
                while len(self.visual_data_cache) > self.visual_data_cache_limit:
                    self.visual_data_cache.popitem(last=False)
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
            self.conv_data = None
            self.calibration = None
        self._save_viewer_state()

    def _sample_index_coverage_status(self, dataset_name: str, clip_stem: str, t0_us: int) -> str:
        if self.index_mode not in {"video_frames", "merged"}:
            return ""
        key = (dataset_name, clip_stem)
        total_video_t0 = self.video_t0_count_by_clip.get(key, 0)
        generated_t0 = self.generated_t0_count_by_clip.get(key, 0)
        current_has_generated = int(t0_us) in self.generated_t0_by_clip.get(key, set())
        current_status = "generated" if current_has_generated else "no generated"
        return (
            " | "
            + video_frame_coverage_summary(total_video_t0, generated_t0)
            + f" | Current: {current_status}"
        )

    def _write_gt_trajectory_to_parquet(self, gt_xyz: np.ndarray, speed_optimized: bool = False) -> bool:
        messagebox.showwarning(
            "GT Source Data",
            "Real GT is loaded from the source dataset and is no longer written to output parquet. "
            "Output parquet files only store generated or augmented pseudo-GT trajectories.",
        )
        return False

    def _optimized_pseudo_gt_components_for_save(self, traj: dict) -> Optional[dict[str, np.ndarray]]:
        """Optimize a generated trajectory before persisting it to output parquet."""
        try:
            xyz = np.column_stack([
                np.asarray(traj["x"], dtype=np.float64),
                np.asarray(traj["y"], dtype=np.float64),
                np.asarray(traj["z"], dtype=np.float64),
            ])
            result = optimize_pseudo_gt_trajectory(xyz)
            return trajectory_components_from_xyz(result.xyz)
        except Exception as exc:
            print(f"Warning: could not optimize pseudo-GT trajectory before save: {exc}")
            return None

    def _write_selected_trajectory_to_parquet(self, traj_idx: int) -> bool:
        if not (0 <= traj_idx < len(self.trajectories)):
            return False
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        traj = self.trajectories[traj_idx]
        if self._is_gt_trajectory(traj, traj_idx):
            messagebox.showwarning("Keep GT", "Use the GT speed panel to edit GT. Prediction speed edits only write rule/manual rows.")
            return False
        row_indices = df.index[df["t0_us"].astype("int64") == int(t0_us)].tolist() if "t0_us" in df.columns else list(df.index)
        if "sample_idx" in df.columns:
            sample_idx = int(traj.get("sample_idx", traj_idx))
            matched = df.index[
                (df["t0_us"].astype("int64") == int(t0_us))
                & (df["sample_idx"].astype("int64") == sample_idx)
            ].tolist() if "t0_us" in df.columns else df.index[df["sample_idx"].astype("int64") == sample_idx].tolist()
            if matched:
                row_idx = matched[0]
            elif traj_idx < len(row_indices):
                row_idx = row_indices[traj_idx]
            else:
                messagebox.showwarning("Save Failed", "Could not locate the selected trajectory row.")
                return False
        else:
            if traj_idx >= len(row_indices):
                messagebox.showwarning("Save Failed", "Could not locate the selected trajectory row.")
                return False
            row_idx = row_indices[traj_idx]

        optimized_components = self._optimized_pseudo_gt_components_for_save(traj)
        if optimized_components is None:
            messagebox.showwarning("Save Failed", "Could not optimize this pseudo-GT trajectory before saving.")
            return False
        for key, values in optimized_components.items():
            if key in traj:
                traj[key] = np.asarray(values, dtype=np.float64)

        for key in ("x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "curvature"):
            if key in df.columns and key in traj:
                df.at[row_idx, key] = np.asarray(traj[key]).tolist()

        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(traj_file, index=False)
        return True

    def _save_results(self):
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before writing the parquet.",
            )
            return
        if getattr(self, "traj_geom_edit_active", False):
            messagebox.showwarning(
                "Trajectory Edit Active",
                "Save or cancel the current trajectory geometry edit before writing the parquet.",
            )
            return
        if self.gt_only:
            messagebox.showwarning(
                "GT Only",
                "GT-only mode has no generated trajectory parquet to filter.",
            )
            return
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        if not traj_file.exists():
            messagebox.showwarning(
                "Save Failed",
                f"No generated trajectory parquet exists for this clip yet: {traj_file}",
            )
            return
        
        df = pd.read_parquet(traj_file)
        pending_deleted = set(getattr(self, "pending_deleted_traj_keys", set()))
        df_saved = drop_trajectory_rows_by_keys(
            df,
            current_t0_us=int(t0_us),
            deleted_keys=pending_deleted,
        )
        
        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df_saved.to_parquet(traj_file, index=False)
        self._persist_pending_manual_point_deletes()

        self._load_sample(self.current_idx)
        self._update_display()
        
        messagebox.showinfo(
            "Saved",
            f"Saved {len(df_saved)} trajectories to {traj_file}\n"
            f"(Removed {len(df) - len(df_saved)} trajectories)"
        )
