"""NavigationMixin for the enhanced trajectory GUI."""

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

from data_loader import get_dataset_names, get_clip_stems_from_dataset, load_data, get_t0_candidates
from calibration_loader import load_calibration_for_segment
from visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

from ..constants import *
from ..math_utils import *
from ..speed_utils import *
from ..projection_utils import *
from ..cluster_utils import *

class NavigationMixin:
    SCENE_FILTER_NONE = "None"

    def _warn_pending_deletes_before_sample_change(self) -> bool:
        if hasattr(self, "_pending_delete_count") and self._pending_delete_count() > 0:
            messagebox.showwarning(
                "Pending Deletes",
                "Save or undo pending trajectory deletes before changing samples.",
            )
            return False
        return True

    def _warn_active_traj_geometry_edit(self, action: str) -> bool:
        if getattr(self, "traj_geom_edit_active", False):
            messagebox.showwarning(
                "Trajectory Edit Active",
                f"Save or cancel the current trajectory geometry edit before {action}.",
            )
            return False
        return True

    def _sync_sample_selectors(self):
        """Sync dataset/clip/t0 controls to the current sample."""
        if self.dataset_var is None or not (0 <= self.current_idx < len(self.samples)):
            return

        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        self.dataset_var.set(dataset_name)
        self._sync_scene_filter_values(dataset_name)
        if self.clip_combo is not None:
            clips = self.clips_by_dataset.get(dataset_name, [])
            self.clip_combo.configure(values=clips)
        if self.clip_var is not None:
            self.clip_var.set(clip_stem)
        if self.t0_combo is not None:
            t0_values = [str(value) for value in self.t0_by_dataset_clip.get((dataset_name, clip_stem), [])]
            self.t0_combo.configure(values=t0_values)
        if self.t0_var is not None:
            self.t0_var.set(str(int(t0_us)))

    def _on_dataset_combo_selected(self, _event=None):
        dataset_name = self.dataset_var.get() if self.dataset_var is not None else ""
        self._sync_scene_filter_values(dataset_name)
        clips = self.clips_by_dataset.get(dataset_name, [])
        if self.clip_combo is not None:
            self.clip_combo.configure(values=clips)
        if clips and self.clip_var is not None:
            self.clip_var.set(clips[0])
            self._on_clip_combo_selected()

    def _on_clip_combo_selected(self, _event=None):
        dataset_name = self.dataset_var.get() if self.dataset_var is not None else ""
        clip_stem = self.clip_var.get() if self.clip_var is not None else ""
        t0_values = [str(value) for value in self.t0_by_dataset_clip.get((dataset_name, clip_stem), [])]
        if self.t0_combo is not None:
            self.t0_combo.configure(values=t0_values)
        if t0_values and self.t0_var is not None:
            self.t0_var.set(t0_values[0])

    def _scene_filter_values_for_dataset(self, dataset_name: str) -> list[str]:
        scenes = getattr(self, "scenes_by_dataset", {}).get(str(dataset_name), [])
        return [self.SCENE_FILTER_NONE] + [str(scene) for scene in scenes]

    def _current_scene_filter(self) -> str:
        scene_filter_var = getattr(self, "scene_filter_var", None)
        if scene_filter_var is not None:
            value = scene_filter_var.get()
        else:
            value = getattr(self, "scene_filter_value", self.SCENE_FILTER_NONE)
        value = str(value or self.SCENE_FILTER_NONE).strip()
        return value or self.SCENE_FILTER_NONE

    def _sync_scene_filter_values(self, dataset_name: str) -> None:
        values = self._scene_filter_values_for_dataset(dataset_name)
        current = self._current_scene_filter()
        if current not in values:
            current = self.SCENE_FILTER_NONE
        self.scene_filter_value = current
        scene_filter_combo = getattr(self, "scene_filter_combo", None)
        if scene_filter_combo is not None:
            scene_filter_combo.configure(values=values)
        scene_filter_var = getattr(self, "scene_filter_var", None)
        if scene_filter_var is not None:
            scene_filter_var.set(current)

    def _on_scene_filter_selected(self, _event=None) -> None:
        self.scene_filter_value = self._current_scene_filter()
        if self.scene_filter_value == self.SCENE_FILTER_NONE:
            return
        if not (0 <= int(self.current_idx) < len(self.samples)):
            return
        if not self._warn_pending_deletes_before_sample_change():
            return
        if not self._warn_active_traj_geometry_edit("changing samples"):
            return
        if getattr(self, "speed_edit_active", False):
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before changing samples.",
            )
            return
        dataset_name = self.samples[int(self.current_idx)][0]
        target_idx = self._first_scene_sample_index(dataset_name, self.scene_filter_value)
        if target_idx is None or int(target_idx) == int(self.current_idx):
            return
        self._load_sample(target_idx)
        self._update_display()

    def _sample_scene_label(self, idx: int) -> Optional[str]:
        if not (0 <= int(idx) < len(self.samples)):
            return None
        dataset_name, clip_stem, t0_us = self.samples[int(idx)]
        return getattr(self, "scene_label_by_sample", {}).get((dataset_name, clip_stem, int(t0_us)))

    def _first_scene_sample_index(self, dataset_name: str, scene: str) -> Optional[int]:
        scene = str(scene or "").strip()
        if not scene or scene == self.SCENE_FILTER_NONE:
            return None
        for idx, (sample_dataset, _clip_stem, _t0_us) in enumerate(self.samples):
            if sample_dataset == dataset_name and self._sample_scene_label(idx) == scene:
                return idx
        return None

    def _scene_filtered_neighbor_index(self, direction: int) -> Optional[int]:
        direction = 1 if int(direction) >= 0 else -1
        next_idx = int(self.current_idx) + direction
        scene = self._current_scene_filter()
        if scene == self.SCENE_FILTER_NONE:
            if 0 <= next_idx < len(self.samples):
                return next_idx
            return None
        if not (0 <= int(self.current_idx) < len(self.samples)):
            return None
        current_dataset = self.samples[int(self.current_idx)][0]
        idx = next_idx
        while 0 <= idx < len(self.samples):
            dataset_name, _clip_stem, _t0_us = self.samples[idx]
            if dataset_name == current_dataset and self._sample_scene_label(idx) == scene:
                return idx
            idx += direction
        return None

    def _jump_to_selected_sample(self):
        if not self._warn_pending_deletes_before_sample_change():
            return
        if not self._warn_active_traj_geometry_edit("changing samples"):
            return
        dataset_name = self.dataset_var.get() if self.dataset_var is not None else ""
        clip_stem = self.clip_var.get() if self.clip_var is not None else ""
        t0_text = self.t0_var.get() if self.t0_var is not None else ""
        try:
            t0_us = int(t0_text) if t0_text else None
        except ValueError:
            t0_us = None
        target_idx = self._find_sample_index(dataset_name, clip_stem, t0_us)
        if target_idx is None:
            messagebox.showwarning(
                "Sample Not Found",
                f"No sample found for dataset={dataset_name}, clip={clip_stem}, t0={t0_text}",
            )
            return
        self._load_sample(target_idx)
        self._update_display()

    def _on_list_select(self, event):
        selection = self.traj_listbox.curselection()
        if selection:
            traj_idx = self._traj_index_from_listbox_row(selection[0])
            if traj_idx is None:
                return
            if self.speed_edit_active and traj_idx != self.speed_edit_traj_idx:
                messagebox.showwarning(
                    "Speed Edit Active",
                    "Save or discard the current speed adjustment before selecting another trajectory.",
                )
                self.traj_listbox.selection_clear(0, tk.END)
                if self.speed_edit_traj_idx is not None:
                    try:
                        row_idx = self.traj_listbox_to_traj_idx.index(int(self.speed_edit_traj_idx))
                        self.traj_listbox.selection_set(row_idx)
                    except ValueError:
                        pass
                return
            if (
                getattr(self, "traj_geom_edit_active", False)
                and traj_idx != getattr(self, "traj_geom_edit_traj_idx", None)
            ):
                messagebox.showwarning(
                    "Trajectory Edit Active",
                    "Save or cancel the current trajectory geometry edit before selecting another trajectory.",
                )
                self.traj_listbox.selection_clear(0, tk.END)
                if self.traj_geom_edit_traj_idx is not None:
                    try:
                        row_idx = self.traj_listbox_to_traj_idx.index(int(self.traj_geom_edit_traj_idx))
                        self.traj_listbox.selection_set(row_idx)
                    except ValueError:
                        pass
                return
            self.current_traj_idx = traj_idx
            self._update_display()

    def _trajectory_listbox_has_key_event(self, event) -> bool:
        return getattr(event, "widget", None) is getattr(self, "traj_listbox", None)

    def _bind_arrow_keys_for_trajectory_navigation(self, widget) -> None:
        """Route Up/Down on focused dropdowns to trajectory navigation."""
        if widget is None:
            return
        widget.bind("<Up>", self._on_global_prev_traj_key)
        widget.bind("<Down>", self._on_global_next_traj_key)

    def _on_global_prev_traj_key(self, event=None):
        if self._trajectory_listbox_has_key_event(event):
            return None
        self._prev_traj()
        return "break"

    def _on_global_next_traj_key(self, event=None):
        if self._trajectory_listbox_has_key_event(event):
            return None
        self._next_traj()
        return "break"

    def _prev_sample(self):
        if not self._warn_pending_deletes_before_sample_change():
            return
        if not self._warn_active_traj_geometry_edit("changing samples"):
            return
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before changing samples.",
            )
            return
        target_idx = self._scene_filtered_neighbor_index(-1)
        if target_idx is not None:
            self._load_sample(target_idx)
            self._update_display()

    def _next_sample(self):
        if not self._warn_pending_deletes_before_sample_change():
            return
        if not self._warn_active_traj_geometry_edit("changing samples"):
            return
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before changing samples.",
            )
            return
        target_idx = self._scene_filtered_neighbor_index(1)
        if target_idx is not None:
            self._load_sample(target_idx)
            self._update_display()

    def _prev_traj(self):
        if not self._warn_active_traj_geometry_edit("selecting another trajectory"):
            return
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before selecting another trajectory.",
            )
            return
        for idx in range(self.current_traj_idx - 1, -1, -1):
            if self._is_traj_pending_deleted(idx):
                continue
            if not self._is_gt_trajectory(self.trajectories[idx], idx):
                self.current_traj_idx = idx
                break
        else:
            visible = [idx for idx in self._visible_trajectory_indices() if idx < self.current_traj_idx]
            if visible:
                self.current_traj_idx = visible[-1]
        self._update_display()

    def _next_traj(self):
        if not self._warn_active_traj_geometry_edit("selecting another trajectory"):
            return
        if self.speed_edit_active:
            messagebox.showwarning(
                "Speed Edit Active",
                "Save or discard the current speed adjustment before selecting another trajectory.",
            )
            return
        for idx in range(self.current_traj_idx + 1, len(self.trajectories)):
            if self._is_traj_pending_deleted(idx):
                continue
            if not self._is_gt_trajectory(self.trajectories[idx], idx):
                self.current_traj_idx = idx
                break
        else:
            visible = [idx for idx in self._visible_trajectory_indices() if idx > self.current_traj_idx]
            if visible:
                self.current_traj_idx = visible[0]
        self._update_display()
