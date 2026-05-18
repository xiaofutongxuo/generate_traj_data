"""GTControlsMixin for the enhanced trajectory GUI."""

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

from traj_core.data_loader import get_dataset_names, get_clip_stems_from_dataset, load_data, get_t0_candidates
from traj_core.calibration_loader import load_calibration_for_segment
from traj_core.visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

from traj_core.constants import *
from traj_core.math_utils import *
from traj_core.speed_utils import *
from ..projection_utils import *
from traj_core.cluster_utils import *
from traj_core.trajectory_identity import is_gt_trajectory_record, normalize_trajectory_source

class GTControlsMixin:

    def _is_gt_trajectory(self, traj: dict, fallback_index: int = -1) -> bool:
        return is_gt_trajectory_record(traj, fallback_index)

    def _optimized_gt_xyz(self, gt_xyz: np.ndarray) -> Optional[np.ndarray]:
        gt = np.asarray(gt_xyz, dtype=np.float64).reshape(-1, 3)
        if len(gt) < 2:
            return None
        speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        smoothed = _smooth_speed_profile(speed, passes=3)
        sampled = _resample_xyz_by_speed_profile(gt, smoothed)
        if sampled is None:
            return None
        limited = _acceleration_limited_resample_path(sampled)
        if limited is not None:
            sampled = limited
        return sampled.astype(np.float32)

    def _auto_optimize_gt_row_in_loaded_df(
        self,
        df: pd.DataFrame,
        t0_us: int,
    ) -> pd.DataFrame:
        if self.gt_only or not AUTO_OPTIMIZE_GT_ON_LOAD or df.empty:
            return df

        if "t0_us" in df.columns:
            current_mask = df["t0_us"].astype("int64") == int(t0_us)
        else:
            current_mask = pd.Series(True, index=df.index)

        if "source" in df.columns:
            gt_mask = df["source"].map(normalize_trajectory_source) == "gt"
        elif "sample_idx" in df.columns:
            return df
        else:
            return df

        matched = df.index[current_mask & gt_mask].tolist()
        if not matched:
            return df
        row_idx = matched[0]

        if GT_SPEED_OPTIMIZED_COLUMN in df.columns:
            try:
                if bool(df.at[row_idx, GT_SPEED_OPTIMIZED_COLUMN]):
                    return df
            except (TypeError, ValueError):
                pass

        row = df.loc[row_idx]
        try:
            gt_xyz = np.column_stack([
                np.asarray(row["x"], dtype=np.float64),
                np.asarray(row["y"], dtype=np.float64),
                np.asarray(row["z"], dtype=np.float64),
            ])
        except Exception as exc:
            print(f"Warning: could not auto-optimize GT speed for t0={int(t0_us)}: {exc}")
            return df

        optimized = self._optimized_gt_xyz(gt_xyz)
        if optimized is None:
            return df

        df = df.copy()
        if GT_SPEED_OPTIMIZED_COLUMN not in df.columns:
            df[GT_SPEED_OPTIMIZED_COLUMN] = False
        components = self._trajectory_components_from_xyz(optimized)
        for key, values in components.items():
            if key in df.columns:
                df.at[row_idx, key] = np.asarray(values).tolist()
        df.at[row_idx, GT_SPEED_OPTIMIZED_COLUMN] = True
        return df

    def _gt_future_key_for_mode(self, mode: Optional[str] = None) -> str:
        mode = mode or self.gt_future_mode
        if mode == "repaired":
            return "ego_future_xyz_repaired"
        return "ego_future_xyz_raw"

    def _get_source_gt_future_xyz(self, mode: Optional[str] = None):
        """Return source ground-truth future trajectory in ego-local coordinates."""
        if self.conv_data is None:
            return None

        key = self._gt_future_key_for_mode(mode)
        if key not in self.conv_data:
            key = "ego_future_xyz"
        if key not in self.conv_data:
            return None

        gt = self.conv_data[key]
        if hasattr(gt, "detach"):
            gt = gt.detach().cpu().numpy()
        gt = np.asarray(gt).reshape(-1, 3)
        if len(gt) < 2:
            return None
        mask = self.conv_data.get("ego_future_valid_mask")
        if mask is not None:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) == len(gt) and not bool(mask.all()):
                return None
        return gt

    def _gt_trajectory_from_current_parquet(self) -> Optional[np.ndarray]:
        # Real GT is sourced from the raw dataset. Output parquet rows are
        # treated as pseudo-GT and should not override the source GT display.
        return None

    def _get_gt_future_xyz(self, mode: Optional[str] = None):
        """Return active ground-truth future trajectory, preferring the editable parquet GT row."""
        if mode is None:
            if self.gt_edit_active and self.gt_edit_preview_xyz is not None:
                return np.asarray(self.gt_edit_preview_xyz, dtype=np.float32).reshape(-1, 3)
            parquet_gt = self._gt_trajectory_from_current_parquet()
            if parquet_gt is not None:
                return parquet_gt
        return self._get_source_gt_future_xyz(mode)

    def _get_gt_forward_acceleration(self, mode: Optional[str] = None):
        """Return source longitudinal GT acceleration for diagnostics when available."""
        if self.conv_data is None:
            return None
        if mode is None and self._gt_trajectory_from_current_parquet() is not None:
            return None

        mode = mode or self.gt_future_mode
        if mode == "repaired" and "ego_future_forward_acceleration_repaired" in self.conv_data:
            key = "ego_future_forward_acceleration_repaired"
        elif "ego_future_forward_acceleration_raw" in self.conv_data:
            key = "ego_future_forward_acceleration_raw"
        else:
            key = "ego_future_forward_acceleration"
        if key not in self.conv_data:
            return None

        accel = self.conv_data[key]
        if hasattr(accel, "detach"):
            accel = accel.detach().cpu().numpy()
        accel = np.asarray(accel, dtype=np.float64).reshape(-1)
        if len(accel) < 2 or not np.isfinite(accel).all():
            return None
        return accel

    def _get_gt_quality_diagnostics(self, mode: Optional[str] = None) -> Optional[dict[str, object]]:
        gt = self._get_gt_future_xyz(mode)
        if gt is None:
            return None
        return _trajectory_quality_diagnostics(
            gt,
            acceleration_mps2=self._get_gt_forward_acceleration(mode),
        )

    def _repair_gt_future(self):
        """Write the velocity-integrated GT future into the editable parquet GT row."""
        if self.conv_data is None:
            messagebox.showwarning("No GT", "Current sample has no loaded GT data.")
            return
        if "ego_future_xyz_repaired" not in self.conv_data:
            messagebox.showwarning("No Repair", "This sample does not include repaired GT data.")
            return
        if not bool(self.conv_data.get("gt_repair_available", True)):
            error = str(self.conv_data.get("gt_repair_error", "")).strip()
            message = "Velocity-based GT repair is unavailable for this sample."
            if error:
                message += f"\n\n{error}"
            messagebox.showwarning("No Repair", message)
            return

        repaired = self._get_source_gt_future_xyz("repaired")
        if repaired is None:
            messagebox.showwarning("No Repair", "Could not load repaired GT for this sample.")
            return
        if not self._write_gt_trajectory_to_parquet(repaired):
            return
        self.gt_future_mode = "raw"
        self._load_sample(self.current_idx)
        self._update_display()

    def _restore_gt_future(self):
        """Write the raw source GT future back into the editable parquet GT row."""
        raw = self._get_source_gt_future_xyz("raw")
        if raw is None:
            messagebox.showwarning("No GT", "Could not load raw GT for this sample.")
            return
        if not self._write_gt_trajectory_to_parquet(raw):
            return
        self.gt_future_mode = "raw"
        self._load_sample(self.current_idx)
        self._update_display()
