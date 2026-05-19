"""DrawBevMixin for the enhanced trajectory GUI."""

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
from traj_core.object_loader import object_center_xy
from traj_core.visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

from traj_core.constants import *
from traj_core.math_utils import *
from traj_core.speed_utils import *
from ..projection_utils import *
from traj_core.cluster_utils import *

class DrawBevMixin:

    def _world_to_canvas(self, wx, wy, scale=None, offset=None):
        """Convert ego-local meters to BEV canvas pixels."""
        forward_scale = self.bev_forward_scale if scale is None else scale
        lateral_scale = self.bev_lateral_scale if scale is None else scale
        offset = self.bev_origin if offset is None else offset
        # Ego x is forward, ego y is left. Draw forward upward and left leftward.
        return offset[0] - wy * lateral_scale, offset[1] - wx * forward_scale

    def _canvas_to_world(self, cx, cy, scale=None, offset=None):
        """Convert BEV canvas pixels to ego-local meters."""
        forward_scale = self.bev_forward_scale if scale is None else scale
        lateral_scale = self.bev_lateral_scale if scale is None else scale
        offset = self.bev_origin if offset is None else offset
        return (offset[1] - cy) / forward_scale, (offset[0] - cx) / lateral_scale

    def _toggle_object_overlay(self):
        var = getattr(self, "show_objects_var", None)
        self.show_objects_enabled = bool(var.get()) if var is not None else not bool(
            getattr(self, "show_objects_enabled", True)
        )
        if hasattr(self, "_draw_trajectories"):
            self._draw_trajectories()
        if hasattr(self, "_draw_camera_images"):
            self._draw_camera_images()

    def _traffic_object_color(self, object_type: int) -> str:
        if object_type in {1, 2, 3, 4}:
            return "#ffd166"
        if object_type in {5, 6, 7, 8}:
            return "#f8961e"
        return "#8ecae6"

    def _draw_traffic_participants(self):
        """Draw traffic participants from data-objects parquet on BEV."""
        var = getattr(self, "show_objects_var", None)
        if var is not None:
            enabled = bool(var.get())
        else:
            enabled = bool(getattr(self, "show_objects_enabled", True))
        if not enabled:
            return

        objects_df = getattr(self, "current_objects", None)
        if objects_df is None or len(objects_df) == 0:
            return

        for _, row in objects_df.iterrows():
            try:
                center_x, center_y = object_center_xy(row)
            except Exception:
                continue
            if not (np.isfinite(center_x) and np.isfinite(center_y)):
                continue
            object_type = int(row.get("object_type", 0) or 0)
            color = self._traffic_object_color(object_type)
            center_px, center_py = self._world_to_canvas(center_x, center_y)
            radius = 4
            self.traj_canvas.create_oval(
                center_px - radius,
                center_py - radius,
                center_px + radius,
                center_py + radius,
                fill=color,
                outline="#111111",
                width=1,
                tags=("traffic_object",),
            )

    def _draw_bev_grid(self):
        """Draw a meter grid and ego axes on the BEV canvas."""
        width = self.bev_canvas_width
        height = self.bev_canvas_height
        grid_color = "#2f2f2f"
        major_color = "#3c3c3c"
        axis_color = "#6f6f6f"
        label_color = "#8a8a8a"
        label_font = ("Arial", 8)

        for forward_m in range(-10, 106, 5):
            _, y = self._world_to_canvas(forward_m, 0)
            if 0 <= y <= height:
                color = major_color if forward_m % 10 == 0 else grid_color
                self.traj_canvas.create_line(0, y, width, y, fill=color)
                if forward_m % 10 == 0:
                    self.traj_canvas.create_text(
                        10, y - 7, text=f"{forward_m}m",
                        fill=label_color, font=label_font, anchor=tk.W,
                    )

        for left_m in np.arange(-30.0, 30.01, 5.0):
            x, _ = self._world_to_canvas(0, left_m)
            if 0 <= x <= width:
                is_major = abs((left_m / 10.0) - round(left_m / 10.0)) < 1e-6
                color = major_color if is_major else grid_color
                self.traj_canvas.create_line(x, 0, x, height, fill=color)
                if is_major and abs(left_m) > 1e-6:
                    self.traj_canvas.create_text(
                        x, height - 12, text=f"{left_m:g}m",
                        fill=label_color, font=label_font,
                    )

        origin_x, origin_y = self._world_to_canvas(0, 0)
        self.traj_canvas.create_line(origin_x, 0, origin_x, height, fill=axis_color)
        self.traj_canvas.create_line(0, origin_y, width, origin_y, fill=axis_color)
        self.traj_canvas.create_text(
            origin_x + 6, origin_y + 10, text="0m",
            fill=label_color, font=label_font, anchor=tk.W,
        )

    def _draw_history_trajectory(self):
        """Draw the ego history loaded for the current sample."""
        hist = self._history_points_xyz()
        if hist is None:
            return

        points = []
        for x, y, _z in hist:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])

        self.traj_canvas.create_line(
            *points,
            fill="#b0b0b0",
            width=3,
            dash=(6, 4),
            smooth=True,
        )
        start_x, start_y = self._world_to_canvas(float(hist[0, 0]), float(hist[0, 1]))
        self.traj_canvas.create_oval(
            start_x - 3, start_y - 3, start_x + 3, start_y + 3,
            fill="#b0b0b0", outline="",
        )

    def _draw_gt_future_trajectory(self):
        """Draw ground-truth future trajectory on the BEV canvas."""
        gt = self._get_gt_future_xyz()
        if gt is None:
            return
        diagnostics = self._get_gt_quality_diagnostics() or _trajectory_quality_diagnostics(gt)
        has_problem = not bool(diagnostics.get("ok", True))

        points = []
        for x, y, _z in gt:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])

        self.traj_canvas.create_line(
            *points,
            fill="#ff6b6b" if has_problem else GT_COLOR_HEX,
            width=4,
            smooth=True,
        )
        for idx in diagnostics.get("bad_accel_indices", []):
            if 0 <= idx < len(gt):
                px, py = self._world_to_canvas(float(gt[idx, 0]), float(gt[idx, 1]))
                self.traj_canvas.create_oval(
                    px - 6, py - 6, px + 6, py + 6,
                    fill="#ff3333",
                    outline="white",
                    width=2,
                )
        for idx in diagnostics.get("jump_indices", []):
            if 0 <= idx < len(gt):
                px, py = self._world_to_canvas(float(gt[idx, 0]), float(gt[idx, 1]))
                self.traj_canvas.create_rectangle(
                    px - 6, py - 6, px + 6, py + 6,
                    fill="#d36bff",
                    outline="white",
                    width=2,
                )
        end_x, end_y = self._world_to_canvas(float(gt[-1, 0]), float(gt[-1, 1]))
        self.traj_canvas.create_oval(
            end_x - 5, end_y - 5, end_x + 5, end_y + 5,
            fill="#ff6b6b" if has_problem else GT_COLOR_HEX,
            outline="black",
            width=1,
        )
        gt_speed = _smoothed_gt_speed_profile_from_xyz(gt)
        for segment in _detect_stop_segments(gt_speed):
            marker_idx = min(int(segment["end"]), len(gt) - 1)
            px, py = self._world_to_canvas(float(gt[marker_idx, 0]), float(gt[marker_idx, 1]))
            radius = STOP_MARKER_RADIUS_PX
            self.traj_canvas.create_oval(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                fill="#ff2020",
                outline="white",
                width=1,
            )
            self.stop_marker_hitboxes.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius + 6),
                "text": (
                    "GT STOP\n"
                    f"Duration: {float(segment['duration_s']):.1f}s "
                    f"({int(segment['frames'])} frames)\n"
                    f"Frames: {int(segment['start'])}-{int(segment['end'])}\n"
                    f"Mean speed: {float(segment['mean_speed_mps']):.2f} m/s"
                ),
            })
        if has_problem:
            self.traj_canvas.create_text(
                end_x + 8,
                end_y + 10,
                text=(
                    f"GT acc {len(diagnostics.get('bad_accel_indices', []))} "
                    f"jump {len(diagnostics.get('jump_indices', []))}"
                ),
                fill="#ffb3b3",
                font=("Arial", 9, "bold"),
                anchor=tk.W,
            )

    def _draw_manual_bezier_preview(self):
        """Draw the generated manual Bezier trajectory preview on the BEV canvas."""
        trajectory_xyz = self._build_manual_bezier_trajectory()
        if trajectory_xyz is None or len(trajectory_xyz) < 2:
            return

        points = []
        for x, y, _z in trajectory_xyz:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])

        self.traj_canvas.create_line(
            *points,
            fill=MANUAL_TRAJ_COLOR_HEX,
            width=4,
            smooth=True,
        )
        end_x, end_y = self._world_to_canvas(
            float(trajectory_xyz[-1, 0]),
            float(trajectory_xyz[-1, 1]),
        )
        self.traj_canvas.create_oval(
            end_x - 5, end_y - 5, end_x + 5, end_y + 5,
            fill=MANUAL_TRAJ_COLOR_HEX,
            outline="black",
            width=1,
        )

    def _draw_cluster_center_preview(self):
        """Draw the selected cluster center preview on the BEV canvas."""
        if self.cluster_preview_traj is None or len(self.cluster_preview_traj) < 2:
            return

        points = []
        for x, y, _z in self.cluster_preview_traj:
            px, py = self._world_to_canvas(float(x), float(y))
            points.extend([px, py])
        self.traj_canvas.create_line(
            *points,
            fill=CLUSTER_TRAJ_COLOR_HEX,
            width=5,
            smooth=True,
        )
        end_x, end_y = self._world_to_canvas(
            float(self.cluster_preview_traj[-1, 0]),
            float(self.cluster_preview_traj[-1, 1]),
        )
        self.traj_canvas.create_oval(
            end_x - 6, end_y - 6, end_x + 6, end_y + 6,
            fill=CLUSTER_TRAJ_COLOR_HEX,
            outline="black",
            width=2,
        )
        label = self._current_cluster_preview_label()
        self.traj_canvas.create_text(
            end_x + 8,
            end_y - 8,
            text=label,
            fill=CLUSTER_TRAJ_COLOR_HEX,
            font=("Arial", 9, "bold"),
            anchor=tk.W,
        )

    def _draw_manual_stop_markers(self):
        """Draw stop markers on top of the Bezier path."""
        if not self.manual_stop_points:
            return
        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            return

        for idx, stop in enumerate(self.manual_stop_points):
            point = _point_at_path_fraction(base_traj, float(stop.get("fraction", 0.0)))
            if point is None:
                continue
            px, py = self._world_to_canvas(float(point[0]), float(point[1]))
            radius = STOP_MARKER_RADIUS_PX
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="#ff5c5c", outline="white", width=2,
            )
            self.stop_marker_hitboxes.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius + 6),
                "text": (
                    "Manual STOP\n"
                    f"Duration: {float(stop.get('duration_s', 0.0)):.1f}s"
                ),
            })

    def _draw_manual_line(self):
        """Draw manually clicked BEV control points."""
        if not self.manual_line_points:
            return

        for idx, point in enumerate(self.manual_line_points):
            px, py = self._world_to_canvas(float(point["x"]), float(point["y"]))
            radius = 5
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="#00d4ff", outline="black", width=2,
            )
            self.traj_canvas.create_text(
                px + 8, py + 9, text=f"L{idx + 1}",
                fill="#00d4ff", font=("Arial", 9, "bold"), anchor=tk.W,
            )

    def _draw_manual_camera_line_on_bev(self):
        """Draw camera image control points back-projected onto the BEV ground plane."""
        points = self._manual_camera_line_points_to_canvas()
        if not points:
            return

        for idx in range(0, len(points), 2):
            px, py = points[idx], points[idx + 1]
            line_idx = idx // 2 + 1
            radius = 5
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill="#00d4ff", outline="black", width=2,
            )
            self.traj_canvas.create_text(
                px + 8, py + 9, text=f"CL{line_idx}",
                fill="#00d4ff", font=("Arial", 8, "bold"), anchor=tk.W,
            )

    def _draw_generated_stop_markers(
        self,
        traj: dict,
        traj_index: int,
        is_selected: bool,
        is_kept: bool,
    ) -> None:
        """Draw detected stop actions for a generated trajectory on BEV."""
        speed = self._trajectory_speed_profile(traj)
        stop_segments = _detect_stop_segments(speed)
        if not stop_segments:
            return

        x_coords = np.asarray(traj.get("x", []), dtype=np.float64).reshape(-1)
        y_coords = np.asarray(traj.get("y", []), dtype=np.float64).reshape(-1)
        if len(x_coords) == 0 or len(y_coords) != len(x_coords):
            return

        for segment in stop_segments:
            marker_idx = min(int(segment["end"]), len(x_coords) - 1)
            px, py = self._world_to_canvas(float(x_coords[marker_idx]), float(y_coords[marker_idx]))
            radius = STOP_MARKER_RADIUS_PX + (1 if is_selected else 0)
            fill = "#ff2020" if is_kept else "#8a2020"
            self.traj_canvas.create_oval(
                px - radius, py - radius, px + radius, py + radius,
                fill=fill,
                outline="white" if is_selected else "#1e1e1e",
                width=2 if is_selected else 1,
            )
            self.stop_marker_hitboxes.append({
                "x": float(px),
                "y": float(py),
                "radius": float(radius + 6),
                "text": (
                    f"T{traj_index} STOP\n"
                    f"Duration: {float(segment['duration_s']):.1f}s "
                    f"({int(segment['frames'])} frames)\n"
                    f"Frames: {int(segment['start'])}-{int(segment['end'])}\n"
                    f"Mean speed: {float(segment['mean_speed_mps']):.2f} m/s"
                ),
            })

    def _hide_stop_tooltip(self) -> None:
        if not hasattr(self, "traj_canvas"):
            return
        for item in self.stop_tooltip_items:
            self.traj_canvas.delete(item)
        self.stop_tooltip_items = []

    def _show_stop_tooltip(self, canvas_x: float, canvas_y: float, text: str) -> None:
        self._hide_stop_tooltip()
        padding = 7
        tooltip_x = min(float(canvas_x) + 14, self.bev_canvas_width - 190)
        tooltip_y = min(float(canvas_y) + 14, self.bev_canvas_height - 78)
        tooltip_x = max(8, tooltip_x)
        tooltip_y = max(8, tooltip_y)

        text_item = self.traj_canvas.create_text(
            tooltip_x + padding,
            tooltip_y + padding,
            text=text,
            fill="#f6f6f6",
            font=("Arial", 9),
            anchor=tk.NW,
            justify=tk.LEFT,
            tags=("stop_tooltip",),
        )
        bbox = self.traj_canvas.bbox(text_item)
        if bbox is None:
            return
        rect_item = self.traj_canvas.create_rectangle(
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
            fill="#111111",
            outline="#ff5c5c",
            width=1,
            tags=("stop_tooltip",),
        )
        self.traj_canvas.tag_raise(text_item, rect_item)
        self.stop_tooltip_items = [rect_item, text_item]

    def _on_traj_canvas_motion(self, event) -> None:
        for marker in reversed(self.stop_marker_hitboxes):
            dx = float(event.x) - float(marker["x"])
            dy = float(event.y) - float(marker["y"])
            if (dx * dx + dy * dy) ** 0.5 <= float(marker["radius"]):
                self._show_stop_tooltip(event.x, event.y, str(marker["text"]))
                return
        self._hide_stop_tooltip()

    def _draw_speed_hover_marker_on_bev(self, source: str) -> None:
        if self.speed_hover_frame_idx is None or self.speed_hover_source != source:
            return
        points_xyz = self._trajectory_points_for_speed_source(source)
        if points_xyz is None or len(points_xyz) == 0:
            return
        frame_idx = int(np.clip(self.speed_hover_frame_idx, 0, len(points_xyz) - 1))
        point = points_xyz[frame_idx]
        frame_label = self._speed_frame_label_for_source(source, frame_idx)
        px, py = self._world_to_canvas(float(point[0]), float(point[1]))
        radius = 8
        self.traj_canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            fill=HOVER_FRAME_COLOR_HEX,
            outline="black",
            width=2,
        )
        self.traj_canvas.create_text(
            px + radius + 6,
            py + radius,
            text=f"F{frame_label}",
            fill=HOVER_FRAME_COLOR_HEX,
            font=("Arial", 9, "bold"),
            anchor=tk.W,
        )

    def _draw_trajectories(self):
        """Draw all trajectories on the canvas."""
        self.traj_canvas.delete("all")
        self.stop_marker_hitboxes = []
        self.stop_tooltip_items = []
        self._draw_bev_grid()
        self._draw_history_trajectory()
        self._draw_gt_future_trajectory()
        self._draw_traffic_participants()
        self._draw_manual_bezier_preview()

        if not self.trajectories:
            self._draw_cluster_center_preview()
            self._draw_manual_camera_line_on_bev()
            self._draw_manual_line()
            self._draw_manual_stop_markers()
            self._draw_speed_hover_marker_on_bev("history")
            self._draw_speed_hover_marker_on_bev("pred")
            self._draw_speed_hover_marker_on_bev("gt")
            return

        for i, traj in enumerate(self.trajectories):
            if self._is_traj_pending_deleted(i):
                continue
            x_coords = traj["x"]
            y_coords = traj["y"]

            if len(x_coords) < 2:
                continue

            is_selected = (i == self.current_traj_idx)
            is_kept = self.trajectory_states.get(i, True)
            style = self._trajectory_draw_style(i, is_selected, is_kept)
            color = style["hex"]

            points = []
            for x, y in zip(x_coords, y_coords):
                px, py = self._world_to_canvas(x, y)
                points.extend([px, py])

            if len(points) >= 4:
                self.traj_canvas.create_line(
                    *points,
                    fill=color,
                    width=style["bev_width"],
                    dash=style["dash"],
                    smooth=True,
                )

            # Draw start point
            if len(x_coords) > 0:
                px, py = self._world_to_canvas(x_coords[0], y_coords[0])
                r = style["point_radius"] if is_selected else 4
                self.traj_canvas.create_oval(
                    px - r, py - r, px + r, py + r,
                    fill=color, outline="white" if is_selected else "black",
                    width=2 if is_selected else 1,
                )

            # Draw trajectory ID
            if len(x_coords) > 5:
                px, py = self._world_to_canvas(x_coords[5], y_coords[5])
                self.traj_canvas.create_text(
                    px, py, text=f"T{i}", fill=color,
                    font=("Arial", 10, "bold"),
                )

            self._draw_generated_stop_markers(
                traj,
                i,
                is_selected=is_selected,
                is_kept=is_kept,
            )

        if hasattr(self, "_draw_saved_trajectory_edit_handles"):
            self._draw_saved_trajectory_edit_handles()

        # Origin marker
        px, py = self._world_to_canvas(0, 0)
        self.traj_canvas.create_oval(
            px - 4, py - 4, px + 4, py + 4,
            fill="#2ecc71", outline="white",
        )
        self.traj_canvas.create_line(
            px, py, px, py - 24, fill="#2ecc71", width=3, arrow=tk.LAST,
        )
        self._draw_manual_camera_line_on_bev()
        self._draw_manual_line()
        self._draw_manual_stop_markers()
        self._draw_speed_hover_marker_on_bev("history")
        self._draw_speed_hover_marker_on_bev("pred")
        self._draw_speed_hover_marker_on_bev("gt")
        self._draw_cluster_center_preview()
