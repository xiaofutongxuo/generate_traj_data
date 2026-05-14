"""SpeedControlsMixin for the enhanced trajectory GUI."""

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
from ..dynamics import trajectory_components_from_xyz

class SpeedControlsMixin:

    def _trajectory_draw_style(self, traj_index: int, is_selected: bool, is_kept: bool) -> dict:
        """Return shared BEV/camera drawing style for a generated trajectory."""
        is_unsmooth = not bool(
            self.trajectory_smoothness.get(traj_index, {}).get("ok", True)
        )
        if not is_kept:
            rgb = (92, 92, 92)
            if is_selected:
                rgb = _brighten_rgb(rgb, amount=0.25)
            return {
                "rgb": rgb,
                "hex": _rgb_to_hex(rgb),
                "bev_width": 3 if is_selected else 1,
                "camera_width": 3 if is_selected else 1,
                "dash": (5, 5),
                "draw_points": is_selected,
                "point_radius": 5 if is_selected else 3,
            }

        rgb = (235, 64, 64) if is_unsmooth else _trajectory_base_color(
            traj_index,
            len(self.trajectories),
        )
        if is_selected:
            rgb = _brighten_rgb(rgb, amount=0.32)
        return {
            "rgb": rgb,
            "hex": _rgb_to_hex(rgb),
            "bev_width": 5 if is_selected else 2,
            "camera_width": 5 if is_selected else 2,
            "dash": None,
            "draw_points": is_selected,
            "point_radius": 6 if is_selected else 4,
        }

    def _trajectory_speed_profile(self, traj: dict) -> np.ndarray:
        return _speed_profile_from_trajectory(
            traj.get("x", []),
            traj.get("y", []),
            traj.get("z", []),
            traj.get("vx"),
            traj.get("vy"),
            traj.get("vz"),
        )

    def _refresh_trajectory_smoothness(self) -> None:
        self.trajectory_smoothness = {}
        _label, gt_speed, _stops, _color = self._gt_speed_profile_source()
        reference_speed = gt_speed if len(gt_speed) else None
        for idx, traj in enumerate(self.trajectories):
            speed = self._trajectory_speed_profile(traj)
            self.trajectory_smoothness[idx] = _speed_smoothness_diagnostics(
                speed,
                reference_speed=reference_speed,
            )

    def _hide_traj_list_tooltip(self) -> None:
        if self.traj_list_tooltip is not None:
            self.traj_list_tooltip.destroy()
            self.traj_list_tooltip = None

    def _show_traj_list_tooltip(self, event, text: str) -> None:
        self._hide_traj_list_tooltip()
        tooltip = tk.Toplevel(self.root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 10}")
        label = tk.Label(
            tooltip,
            text=text,
            bg="#111111",
            fg="#f6f6f6",
            bd=1,
            relief=tk.SOLID,
            padx=7,
            pady=5,
            font=("Arial", 9),
            justify=tk.LEFT,
        )
        label.pack()
        self.traj_list_tooltip = tooltip

    def _on_traj_list_motion(self, event) -> None:
        if not hasattr(self, "traj_listbox"):
            return
        row_idx = self.traj_listbox.nearest(event.y)
        bbox = self.traj_listbox.bbox(row_idx)
        if bbox is None or not (bbox[1] <= event.y <= bbox[1] + bbox[3]):
            self._hide_traj_list_tooltip()
            return
        traj_idx = self._traj_index_from_listbox_row(row_idx)
        if traj_idx is None:
            self._hide_traj_list_tooltip()
            return
        diagnostics = self.trajectory_smoothness.get(traj_idx, {})
        if event.x <= 24 and not bool(diagnostics.get("ok", True)):
            self._show_traj_list_tooltip(event, "建议删除原因：速度不够平滑")
            return
        self._hide_traj_list_tooltip()

    def _manual_preview_is_active_for_speed(self) -> bool:
        return (
            self.draw_line_enabled
            or self.manual_line_points_dirty
            or self.manual_camera_line_points_dirty
            or self.manual_stop_points_dirty
        )

    def _gt_speed_profile_source(self) -> tuple[str, np.ndarray, list[dict], str]:
        gt = self._get_gt_future_xyz()
        if gt is None:
            return "GT", np.zeros(0, dtype=np.float64), [], GT_COLOR_HEX
        speed = _smoothed_gt_speed_profile_from_xyz(gt)
        return "GT", speed, _detect_stop_segments(speed), GT_COLOR_HEX

    def _history_points_xyz(self, smoothed: bool = True) -> Optional[np.ndarray]:
        if self.conv_data is None or "ego_history_xyz" not in self.conv_data:
            return None
        history = self.conv_data["ego_history_xyz"]
        if hasattr(history, "detach"):
            history = history.detach().cpu().numpy()
        try:
            history = np.asarray(history, dtype=np.float64).reshape(-1, 3)
        except ValueError:
            return None
        mask = self.conv_data.get("ego_history_valid_mask")
        if mask is not None:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) == len(history):
                history = history[mask]
        if len(history) < 2:
            return None
        if smoothed:
            history = _smooth_history_xyz_for_display(history)
        return history

    def _history_speed_profile_source(self) -> tuple[str, np.ndarray, list[dict], str]:
        history = self._history_points_xyz(smoothed=False)
        if history is None:
            return "History", np.zeros(0, dtype=np.float64), [], HISTORY_SPEED_COLOR_HEX
        speed = _smoothed_history_speed_profile_from_xyz(history)
        return "History", speed, _detect_stop_segments(speed), HISTORY_SPEED_COLOR_HEX

    def _selected_speed_profile_source(self) -> tuple[str, np.ndarray, list[dict], str]:
        if self.speed_edit_active and self.speed_edit_speed is not None:
            speed = np.asarray(self.speed_edit_speed, dtype=np.float64)
            label = f"Editing T{self.speed_edit_traj_idx}"
            style = self._trajectory_draw_style(
                int(self.speed_edit_traj_idx),
                True,
                self.trajectory_states.get(int(self.speed_edit_traj_idx), True),
            )
            return label, speed, _detect_stop_segments(speed), style["hex"]

        manual_preview = None
        if self._manual_preview_is_active_for_speed():
            manual_preview = self._build_manual_bezier_trajectory()
        if manual_preview is not None and len(manual_preview) >= 2:
            speed = _speed_profile_from_trajectory(
                manual_preview[:, 0],
                manual_preview[:, 1],
                manual_preview[:, 2],
            )
            return "Manual preview", speed, _detect_stop_segments(speed), MANUAL_TRAJ_COLOR_HEX

        if self.cluster_preview_traj is not None and len(self.cluster_preview_traj) >= 2:
            cluster = np.asarray(self.cluster_preview_traj, dtype=np.float64)
            speed = _speed_profile_from_trajectory(cluster[:, 0], cluster[:, 1], cluster[:, 2])
            return "Cluster preview", speed, _detect_stop_segments(speed), "#f5b041"

        if 0 <= self.current_traj_idx < len(self.trajectories):
            if self._is_traj_pending_deleted(self.current_traj_idx):
                return "", np.zeros(0, dtype=np.float64), [], "#6f6f6f"
            traj = self.trajectories[self.current_traj_idx]
            style = self._trajectory_draw_style(
                self.current_traj_idx,
                True,
                self.trajectory_states.get(self.current_traj_idx, True),
            )
            speed = self._trajectory_speed_profile(traj)
            return f"T{self.current_traj_idx}", speed, _detect_stop_segments(speed), style["hex"]
        return "", np.zeros(0, dtype=np.float64), [], "#6f6f6f"

    def _speed_profile_source(self, source: str) -> tuple[str, np.ndarray, list[dict], str]:
        if source == "history":
            return self._history_speed_profile_source()
        if source == "gt":
            return self._gt_speed_profile_source()
        return self._selected_speed_profile_source()

    def _selected_trajectory_points_xyz(self):
        if self._manual_preview_is_active_for_speed():
            manual_preview = self._build_manual_bezier_trajectory()
            if manual_preview is not None and len(manual_preview) >= 2:
                return manual_preview
        if self.cluster_preview_traj is not None and len(self.cluster_preview_traj) >= 2:
            return np.asarray(self.cluster_preview_traj, dtype=np.float64)
        if 0 <= self.current_traj_idx < len(self.trajectories):
            if self._is_traj_pending_deleted(self.current_traj_idx):
                return None
            traj = self.trajectories[self.current_traj_idx]
            if self._is_gt_trajectory(traj, self.current_traj_idx):
                return None
            return np.column_stack([traj["x"], traj["y"], traj["z"]])
        return None

    def _trajectory_points_for_speed_source(self, source: str):
        if source == "history":
            return self._history_points_xyz()
        if source == "gt":
            return self._get_gt_future_xyz()
        return self._selected_trajectory_points_xyz()

    def _speed_frame_bounds(self, future_speed: np.ndarray) -> tuple[int, int]:
        _history_label, history_speed, _history_stops, _history_color = (
            self._history_speed_profile_source()
        )
        min_frame = -(len(history_speed) - 1) if len(history_speed) else 0
        future_len = len(future_speed) if future_speed is not None else 0
        gt_future_len = 0
        gt_getter = getattr(self, "_get_gt_future_xyz", None)
        if callable(gt_getter):
            gt = gt_getter()
            if gt is not None:
                try:
                    gt_future_len = len(np.asarray(gt, dtype=np.float64).reshape(-1, 3))
                except ValueError:
                    gt_future_len = len(np.asarray(gt))
        max_frame = max(0, future_len - 1, gt_future_len - 1)
        return int(min_frame), int(max_frame)

    def _speed_frame_label_for_source(self, source: str, frame_idx: int) -> int:
        if source == "history":
            points = self._history_points_xyz()
            history_len = len(points) if points is not None else 0
            return int(frame_idx) - max(history_len - 1, 0)
        return int(frame_idx)

    def _speed_plot_geometry(self) -> dict[str, float]:
        margin_left = 44
        margin_right = 14
        margin_top = 22
        margin_bottom = 30
        plot_w = self.speed_canvas_width - margin_left - margin_right
        plot_h = self.speed_canvas_height - margin_top - margin_bottom
        return {
            "left": float(margin_left),
            "top": float(margin_top),
            "right": float(margin_left + plot_w),
            "bottom": float(margin_top + plot_h),
            "width": float(plot_w),
            "height": float(plot_h),
        }

    def _set_speed_hover_frame(self, source: str, frame_idx: Optional[int]) -> None:
        if frame_idx is None:
            next_idx = None
            next_source = None
        else:
            _label, speed, _stops, _color = self._speed_profile_source(source)
            if len(speed) == 0:
                next_idx = None
                next_source = None
            else:
                next_idx = int(np.clip(int(frame_idx), 0, len(speed) - 1))
                next_source = source
        if self.speed_hover_frame_idx == next_idx and self.speed_hover_source == next_source:
            return
        self.speed_hover_frame_idx = next_idx
        self.speed_hover_source = next_source
        self._draw_trajectories()
        self._draw_speed_profile()
        self._draw_gt_speed_profile()
        self._draw_camera_images()

    def _speed_hover_target_for_canvas_x(
        self,
        source: str,
        x: float,
        rect: dict[str, float],
    ) -> tuple[Optional[str], Optional[int]]:
        _label, future_speed, _stops, _color = self._speed_profile_source(source)
        _history_label, history_speed, _history_stops, _history_color = (
            self._history_speed_profile_source()
        )
        if len(future_speed) == 0 and len(history_speed) == 0:
            return None, None

        min_frame, max_frame = self._speed_frame_bounds(future_speed)
        if max_frame <= min_frame:
            frame_label = max_frame
        else:
            fraction = (float(x) - rect["left"]) / max(rect["width"], 1.0)
            frame_label = int(round(min_frame + np.clip(fraction, 0.0, 1.0) * (max_frame - min_frame)))

        if frame_label < 0 and len(history_speed):
            history_idx = int(np.clip(frame_label + len(history_speed) - 1, 0, len(history_speed) - 1))
            return "history", history_idx
        if len(future_speed) == 0:
            return None, None
        future_idx = int(np.clip(frame_label, 0, len(future_speed) - 1))
        return source, future_idx

    def _on_speed_canvas_motion(self, event, source: str) -> None:
        if source == "gt":
            rect = self.gt_speed_plot_rect or self._speed_plot_geometry()
        else:
            rect = self.speed_plot_rect or self._speed_plot_geometry()
        inside = (
            rect["left"] <= float(event.x) <= rect["right"]
            and rect["top"] <= float(event.y) <= rect["bottom"]
        )
        if not inside:
            self._set_speed_hover_frame(source, None)
            return

        hover_source, frame_idx = self._speed_hover_target_for_canvas_x(source, event.x, rect)
        self._set_speed_hover_frame(hover_source or source, frame_idx)

    def _speed_canvas_frame_from_x(self, x: float) -> Optional[int]:
        rect = self.speed_plot_rect or self._speed_plot_geometry()
        if not (rect["left"] <= float(x) <= rect["right"]):
            return None
        hover_source, frame_idx = self._speed_hover_target_for_canvas_x("pred", x, rect)
        if hover_source != "pred":
            return None
        return frame_idx

    def _speed_canvas_value_from_y(self, y: float) -> float:
        rect = self.speed_plot_rect or self._speed_plot_geometry()
        _label, speed, _stops, _color = self._selected_speed_profile_source()
        _history_label, history_speed, _history_stops, _history_color = (
            self._history_speed_profile_source()
        )
        finite_speed = np.concatenate([
            speed[np.isfinite(speed)],
            history_speed[np.isfinite(history_speed)],
        ])
        max_speed = float(np.nanmax(finite_speed)) if len(finite_speed) else 1.0
        y_max = max(1.0, np.ceil(max_speed * 1.15))
        fraction = (rect["bottom"] - float(y)) / max(rect["height"], 1.0)
        return float(np.clip(fraction, 0.0, 1.0) * y_max)

    def _on_speed_canvas_right_click(self, event) -> None:
        menu = tk.Menu(self.root, tearoff=0)
        can_edit = (
            not self.gt_only
            and 0 <= self.current_traj_idx < len(self.trajectories)
            and not self._is_traj_pending_deleted(self.current_traj_idx)
            and not self._is_gt_trajectory(self.trajectories[self.current_traj_idx], self.current_traj_idx)
            and not self._manual_preview_is_active_for_speed()
        )
        if not self.speed_edit_active:
            menu.add_command(
                label="进行调整",
                command=self._start_speed_edit,
                state=tk.NORMAL if can_edit else tk.DISABLED,
            )
        else:
            menu.add_command(label="保存更改", command=self._save_speed_edit)
            menu.add_command(label="不保存，恢复原状", command=lambda: self._cancel_speed_edit(redraw=True))
        menu.tk_popup(event.x_root, event.y_root)

    def _pack_pred_speed_actions(self) -> None:
        if self.pred_speed_action_frame is not None:
            if hasattr(self, "gt_speed_header_frame") and self.gt_speed_header_frame is not None:
                self.pred_speed_action_frame.pack(
                    before=self.gt_speed_header_frame,
                    pady=(0, 6),
                )
            else:
                self.pred_speed_action_frame.pack(pady=(0, 6))

    def _hide_pred_speed_actions(self) -> None:
        if getattr(self, "pred_speed_action_frame", None) is not None:
            self.pred_speed_action_frame.pack_forget()

    def _pack_gt_speed_actions(self) -> None:
        if self.gt_speed_action_frame is not None:
            self.gt_speed_action_frame.pack(pady=(0, 6))
        if self.gt_stop_action_frame is not None:
            self.gt_stop_action_frame.pack_forget()

    def _pack_gt_stop_actions(self) -> None:
        if self.gt_speed_action_frame is not None:
            self.gt_speed_action_frame.pack_forget()
        if self.gt_stop_action_frame is not None:
            self.gt_stop_action_frame.pack(pady=(0, 6))

    def _hide_gt_action_frames(self) -> None:
        if self.gt_speed_action_frame is not None:
            self.gt_speed_action_frame.pack_forget()
        if self.gt_stop_action_frame is not None:
            self.gt_stop_action_frame.pack_forget()

    def _optimize_pred_speed_curve(self) -> None:
        if self.gt_only or not (0 <= self.current_traj_idx < len(self.trajectories)):
            messagebox.showwarning("No Trajectory", "Select a diversity trajectory before optimizing speed.")
            return
        if self._is_gt_trajectory(self.trajectories[self.current_traj_idx], self.current_traj_idx):
            messagebox.showwarning("Keep GT", "Select a rule/manual trajectory; GT is optimized from the GT panel.")
            return
        if self._manual_preview_is_active_for_speed():
            messagebox.showwarning("Manual Preview", "Finish manual Bezier editing before optimizing a saved trajectory.")
            return
        self._start_speed_edit()
        if not self.speed_edit_active or self.speed_edit_speed is None:
            return
        self.speed_edit_speed = _smooth_speed_profile(self.speed_edit_speed, passes=3)
        self.speed_edit_dirty = True
        if not self._apply_speed_edit_to_trajectory():
            self._cancel_speed_edit(redraw=False)
            messagebox.showwarning("Optimize Failed", "Could not optimize this trajectory speed curve.")
            return
        self._pack_pred_speed_actions()
        self._update_display()

    def _optimize_gt_speed_curve(self) -> None:
        gt = self._get_gt_future_xyz()
        if gt is None:
            messagebox.showwarning("No GT", "Current sample has no GT trajectory.")
            return
        self._cancel_gt_speed_edit(redraw=False)
        speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        smoothed = _smooth_speed_profile(speed, passes=3)
        sampled = _resample_xyz_by_speed_profile(gt, smoothed)
        if sampled is None:
            messagebox.showwarning("Optimize Failed", "Could not optimize GT speed curve.")
            return
        self.gt_edit_active = True
        self.gt_edit_mode = "optimize"
        self.gt_edit_original_xyz = np.asarray(gt, dtype=np.float32).copy()
        self.gt_edit_preview_xyz = sampled.astype(np.float32)
        self.gt_stop_frame_idx = None
        self._pack_gt_speed_actions()
        self._update_display()

    def _start_gt_stop_add(self) -> None:
        gt = self._get_gt_future_xyz()
        if gt is None:
            messagebox.showwarning("No GT", "Current sample has no GT trajectory.")
            return
        self._cancel_gt_speed_edit(redraw=False)
        self.gt_edit_active = True
        self.gt_edit_mode = "stop"
        self.gt_edit_original_xyz = np.asarray(gt, dtype=np.float32).copy()
        self.gt_edit_preview_xyz = np.asarray(gt, dtype=np.float32).copy()
        self.gt_stop_frame_idx = None
        self._pack_gt_stop_actions()
        self._update_display()

    def _on_gt_speed_canvas_click(self, event) -> None:
        if not self.gt_edit_active or self.gt_edit_mode != "stop":
            return
        rect = self.gt_speed_plot_rect or self._speed_plot_geometry()
        if not (
            rect["left"] <= float(event.x) <= rect["right"]
            and rect["top"] <= float(event.y) <= rect["bottom"]
        ):
            return
        gt = self.gt_edit_original_xyz
        if gt is None or len(gt) == 0:
            return
        if len(gt) == 1:
            frame_idx = 0
        else:
            fraction = (float(event.x) - rect["left"]) / max(rect["width"], 1.0)
            frame_idx = int(round(np.clip(fraction, 0.0, 1.0) * (len(gt) - 1)))
        preview = np.asarray(gt, dtype=np.float32).copy()
        stop_anchor_idx = max(0, frame_idx - 1)
        preview[frame_idx:] = preview[stop_anchor_idx]
        self.gt_edit_preview_xyz = preview
        self.gt_stop_frame_idx = frame_idx
        self._update_display()

    def _undo_gt_stop_add(self) -> None:
        if not self.gt_edit_active or self.gt_edit_original_xyz is None:
            return
        self.gt_edit_preview_xyz = np.asarray(self.gt_edit_original_xyz, dtype=np.float32).copy()
        self.gt_stop_frame_idx = None
        self._update_display()

    def _save_gt_speed_edit(self) -> None:
        if not self.gt_edit_active or self.gt_edit_preview_xyz is None:
            return
        if not self._write_gt_trajectory_to_parquet(
            np.asarray(self.gt_edit_preview_xyz, dtype=np.float32),
            speed_optimized=self.gt_edit_mode == "optimize",
        ):
            return
        self.gt_edit_active = False
        self.gt_edit_mode = None
        self.gt_edit_original_xyz = None
        self.gt_edit_preview_xyz = None
        self.gt_stop_frame_idx = None
        self._hide_gt_action_frames()
        self._load_sample(self.current_idx)
        self._update_display()

    def _cancel_gt_speed_edit(self, redraw: bool = True) -> None:
        self.gt_edit_active = False
        self.gt_edit_mode = None
        self.gt_edit_original_xyz = None
        self.gt_edit_preview_xyz = None
        self.gt_stop_frame_idx = None
        self._hide_gt_action_frames()
        if redraw and hasattr(self, "root"):
            self._update_display()

    def _on_speed_canvas_left_down(self, event) -> None:
        if not self.speed_edit_active:
            return
        self.speed_edit_last_frame = self._speed_canvas_frame_from_x(event.x)
        self._apply_speed_canvas_edit(event)

    def _on_speed_canvas_left_drag(self, event) -> None:
        if self.speed_edit_active:
            self._apply_speed_canvas_edit(event)

    def _on_speed_canvas_left_release(self, _event) -> None:
        self.speed_edit_last_frame = None

    def _apply_speed_canvas_edit(self, event) -> None:
        if not self.speed_edit_active or self.speed_edit_speed is None:
            return
        frame_idx = self._speed_canvas_frame_from_x(event.x)
        if frame_idx is None:
            return
        new_speed = self._speed_canvas_value_from_y(event.y)
        editable = np.asarray(self.speed_edit_speed, dtype=np.float64).copy()
        center = int(frame_idx)
        radius = min(SPEED_EDIT_LOCAL_RADIUS_FRAMES, max(len(editable) // 10, 2))
        start = max(0, center - radius)
        end = min(len(editable) - 1, center + radius)
        for idx in range(start, end + 1):
            distance = abs(idx - center) / max(float(radius), 1.0)
            weight = 0.5 * (1.0 + np.cos(np.pi * distance))
            editable[idx] = editable[idx] * (1.0 - weight) + new_speed * weight
        if self.speed_edit_last_frame is not None and abs(center - self.speed_edit_last_frame) > 1:
            bridge_start = int(min(self.speed_edit_last_frame, center))
            bridge_end = int(max(self.speed_edit_last_frame, center))
            for idx in range(bridge_start, bridge_end + 1):
                distance = abs(idx - center) / max(float(radius), 1.0)
                if distance > 1.0:
                    continue
                weight = 0.35 * 0.5 * (1.0 + np.cos(np.pi * distance))
                editable[idx] = editable[idx] * (1.0 - weight) + new_speed * weight
        self.speed_edit_last_frame = frame_idx
        editable = _enforce_speed_acceleration_limits(editable)
        self.speed_edit_speed = editable
        self.speed_edit_dirty = True
        self._apply_speed_edit_to_trajectory()
        self.speed_hover_source = "pred"
        self.speed_hover_frame_idx = frame_idx
        self._draw_trajectories()
        self._draw_speed_profile()
        self._draw_camera_images()

    def _start_speed_edit(self) -> None:
        if self.gt_only or not (0 <= self.current_traj_idx < len(self.trajectories)):
            return
        if getattr(self, "traj_geom_edit_active", False):
            if hasattr(self, "root"):
                messagebox.showwarning(
                    "Trajectory Edit Active",
                    "Save or cancel the current trajectory geometry edit before editing speed.",
                )
            return
        if self._is_traj_pending_deleted(self.current_traj_idx):
            return
        self._cancel_speed_edit(redraw=False)
        traj = self.trajectories[self.current_traj_idx]
        if self._is_gt_trajectory(traj, self.current_traj_idx):
            messagebox.showwarning("Keep GT", "Select a rule/manual trajectory; GT is optimized from the GT panel.")
            return
        self.speed_edit_active = True
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = int(self.current_traj_idx)
        self.speed_edit_original_traj = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in traj.items()
        }
        self.speed_edit_original_xyz = np.column_stack([traj["x"], traj["y"], traj["z"]]).copy()
        self.speed_edit_speed = self._trajectory_speed_profile(traj).copy()
        self.speed_edit_last_frame = None
        self._update_display()

    def _cancel_speed_edit(self, redraw: bool = True) -> None:
        if self.speed_edit_active and self.speed_edit_original_traj is not None:
            idx = self.speed_edit_traj_idx
            if idx is not None and 0 <= int(idx) < len(self.trajectories):
                self.trajectories[int(idx)] = {
                    key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                    for key, value in self.speed_edit_original_traj.items()
                }
        self.speed_edit_active = False
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = None
        self.speed_edit_original_traj = None
        self.speed_edit_original_xyz = None
        self.speed_edit_speed = None
        self.speed_edit_last_frame = None
        self._hide_pred_speed_actions()
        if redraw and hasattr(self, "root"):
            self._update_display()

    def _trajectory_components_from_xyz(self, traj_xyz: np.ndarray) -> dict:
        return trajectory_components_from_xyz(traj_xyz)

    def _apply_components_to_traj(self, traj: dict, components: dict) -> None:
        for key, values in components.items():
            if key in traj:
                traj[key] = np.asarray(values, dtype=np.float64)

    def _apply_speed_edit_to_trajectory(self) -> bool:
        if (
            self.speed_edit_original_xyz is None
            or self.speed_edit_speed is None
            or self.speed_edit_traj_idx is None
            or not (0 <= int(self.speed_edit_traj_idx) < len(self.trajectories))
        ):
            return False
        sampled = _resample_xyz_by_speed_profile(
            self.speed_edit_original_xyz,
            self.speed_edit_speed,
        )
        if sampled is None:
            return False
        limited = _acceleration_limited_resample_path(sampled)
        if limited is not None:
            sampled = limited
        traj = self.trajectories[int(self.speed_edit_traj_idx)]
        self._apply_components_to_traj(traj, self._trajectory_components_from_xyz(sampled))
        self.speed_edit_speed = self._trajectory_speed_profile(traj)
        self._refresh_trajectory_smoothness()
        return True

    def _save_speed_edit(self) -> None:
        if not self.speed_edit_active or self.speed_edit_traj_idx is None:
            return
        if not self.speed_edit_dirty:
            self._cancel_speed_edit(redraw=True)
            return
        edited_idx = int(self.speed_edit_traj_idx)
        edited_sample_idx = int(self.trajectories[edited_idx].get("sample_idx", edited_idx))
        if not self._write_selected_trajectory_to_parquet(edited_idx):
            return
        self.speed_edit_active = False
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = None
        self.speed_edit_original_traj = None
        self.speed_edit_original_xyz = None
        self.speed_edit_speed = None
        self.speed_edit_last_frame = None
        self._hide_pred_speed_actions()
        self._load_sample(self.current_idx)
        for idx, traj in enumerate(self.trajectories):
            if int(traj.get("sample_idx", idx)) == edited_sample_idx:
                self.current_traj_idx = idx
                break
        self._update_display()
