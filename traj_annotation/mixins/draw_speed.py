"""DrawSpeedMixin for the enhanced trajectory GUI."""

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

class DrawSpeedMixin:

    def _draw_speed_profile_on_canvas(self, canvas, source: str) -> None:
        """Draw a speed-over-frame chart for one trajectory source."""
        if canvas is None:
            return
        canvas.delete("all")
        # Always use the real rendered size of the canvas; fall back to the
        # initially configured dimensions when the canvas hasn't been mapped yet.
        _cw = canvas.winfo_width()
        _ch = canvas.winfo_height()
        width = _cw if _cw > 10 else self.speed_canvas_width
        height = _ch if _ch > 10 else self.speed_canvas_height
        rect = self._speed_plot_geometry()
        if source == "gt":
            self.gt_speed_plot_rect = rect
        else:
            self.speed_plot_rect = rect
        margin_left = rect["left"]
        margin_top = rect["top"]
        plot_w = rect["width"]
        plot_h = rect["height"]

        label, speed, stop_segments, color = self._speed_profile_source(source)
        history_label, history_speed, _history_stops, history_color = (
            self._history_speed_profile_source()
        )
        axis_color = "#5a5a5a"
        grid_color = "#2c2c2c"
        text_color = "#b8b8b8"
        canvas.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_w,
            margin_top + plot_h,
            outline=axis_color,
            fill="#171717",
        )
        canvas.create_text(
            8,
            8,
            text=f"{label or 'No trajectory'} speed",
            fill=text_color,
            font=("Arial", 9, "bold"),
            anchor=tk.NW,
        )

        if len(speed) == 0 and len(history_speed) == 0:
            canvas.create_text(
                margin_left + plot_w / 2,
                margin_top + plot_h / 2,
                text="No speed data",
                fill="#777777",
                font=("Arial", 10),
            )
            return

        finite_speed = np.concatenate([
            speed[np.isfinite(speed)],
            history_speed[np.isfinite(history_speed)],
        ])
        max_speed = float(np.nanmax(finite_speed)) if len(finite_speed) else 0.0
        y_max = max(1.0, np.ceil(max_speed * 1.15))
        min_frame, max_frame = self._speed_frame_bounds(speed)

        for fraction in (0.25, 0.5, 0.75):
            y = margin_top + plot_h * (1.0 - fraction)
            canvas.create_line(
                margin_left, y, margin_left + plot_w, y,
                fill=grid_color,
            )
            canvas.create_text(
                margin_left - 6,
                y,
                text=f"{y_max * fraction:.1f}",
                fill=text_color,
                font=("Arial", 8),
                anchor=tk.E,
            )

        canvas.create_text(
            margin_left - 6,
            margin_top + plot_h,
            text="0",
            fill=text_color,
            font=("Arial", 8),
            anchor=tk.E,
        )
        canvas.create_text(
            margin_left + plot_w / 2,
            height - 8,
            text="frame",
            fill=text_color,
            font=("Arial", 8),
        )
        canvas.create_text(
            10,
            margin_top + plot_h / 2,
            text="m/s",
            fill=text_color,
            font=("Arial", 8),
        )

        def frame_to_x(frame_label: int) -> float:
            if max_frame <= min_frame:
                return margin_left
            return margin_left + (
                (float(frame_label) - float(min_frame)) / float(max_frame - min_frame)
            ) * plot_w

        def speed_to_y(value: float) -> float:
            return margin_top + plot_h - (float(value) / y_max) * plot_h

        for segment in stop_segments:
            x0 = frame_to_x(int(segment["start"]))
            x1 = frame_to_x(int(segment["end"]))
            canvas.create_rectangle(
                x0,
                margin_top,
                x1,
                margin_top + plot_h,
                fill="#3a1717",
                outline="",
            )

        zero_x = frame_to_x(0)
        canvas.create_line(
            zero_x,
            margin_top,
            zero_x,
            margin_top + plot_h,
            fill="#7f8c8d",
            dash=(3, 4),
            width=2,
        )
        canvas.create_text(
            zero_x + 4,
            margin_top + 4,
            text="0",
            fill="#c7d0d8",
            font=("Arial", 8, "bold"),
            anchor=tk.NW,
        )

        history_points = []
        history_start_frame = -(len(history_speed) - 1) if len(history_speed) else 0
        for idx, value in enumerate(history_speed):
            if not np.isfinite(value):
                continue
            history_points.extend([frame_to_x(history_start_frame + idx), speed_to_y(value)])
        if len(history_points) >= 4:
            canvas.create_line(
                *history_points,
                fill=history_color,
                width=3,
                smooth=True,
            )

        points = []
        for idx, value in enumerate(speed):
            if not np.isfinite(value):
                continue
            points.extend([frame_to_x(idx), speed_to_y(value)])
        if len(points) >= 4:
            canvas.create_line(
                *points,
                fill=color,
                width=3,
                smooth=True,
            )

        canvas.create_line(
            margin_left,
            speed_to_y(STOP_SPEED_THRESHOLD_MPS),
            margin_left + plot_w,
            speed_to_y(STOP_SPEED_THRESHOLD_MPS),
            fill="#ff5c5c",
            dash=(4, 3),
        )
        canvas.create_text(
            margin_left + plot_w - 2,
            speed_to_y(STOP_SPEED_THRESHOLD_MPS) - 8,
            text=f"stop < {STOP_SPEED_THRESHOLD_MPS:.1f}m/s",
            fill="#ff9b9b",
            font=("Arial", 8),
            anchor=tk.E,
        )

        canvas.create_text(
            margin_left,
            margin_top + plot_h + 12,
            text=str(min_frame),
            fill=text_color,
            font=("Arial", 8),
            anchor=tk.N,
        )
        if min_frame < 0 < max_frame:
            canvas.create_text(
                zero_x,
                margin_top + plot_h + 12,
                text="0",
                fill="#c7d0d8",
                font=("Arial", 8, "bold"),
                anchor=tk.N,
            )
        canvas.create_text(
            margin_left + plot_w,
            margin_top + plot_h + 12,
            text=str(max_frame),
            fill=text_color,
            font=("Arial", 8),
            anchor=tk.N,
        )

        hover_speed = None
        hover_label = None
        hover_frame_label = None
        if self.speed_hover_frame_idx is not None and self.speed_hover_source == "history":
            if len(history_speed):
                hover_idx = int(np.clip(self.speed_hover_frame_idx, 0, len(history_speed) - 1))
                hover_speed = float(history_speed[hover_idx])
                hover_frame_label = history_start_frame + hover_idx
                hover_label = f"F{hover_frame_label}"
        elif self.speed_hover_frame_idx is not None and self.speed_hover_source == source:
            if len(speed):
                hover_idx = int(np.clip(self.speed_hover_frame_idx, 0, len(speed) - 1))
                hover_speed = float(speed[hover_idx])
                hover_frame_label = hover_idx
                hover_label = f"F{hover_idx}"

        if hover_speed is not None and hover_label is not None and hover_frame_label is not None:
            hover_x = frame_to_x(int(hover_frame_label))
            hover_y = speed_to_y(hover_speed) if np.isfinite(hover_speed) else margin_top + plot_h
            canvas.create_line(
                hover_x,
                margin_top,
                hover_x,
                margin_top + plot_h,
                fill=HOVER_FRAME_COLOR_HEX,
                width=2,
            )
            canvas.create_oval(
                hover_x - 5,
                hover_y - 5,
                hover_x + 5,
                hover_y + 5,
                fill=HOVER_FRAME_COLOR_HEX,
                outline="black",
                width=1,
            )
            canvas.create_text(
                min(hover_x + 8, margin_left + plot_w - 78),
                margin_top + 6,
                text=f"{hover_label} {hover_speed:.2f}m/s",
                fill=HOVER_FRAME_COLOR_HEX,
                font=("Arial", 8, "bold"),
                anchor=tk.NW,
            )

    def _draw_speed_profile(self) -> None:
        if not hasattr(self, "speed_canvas"):
            return
        self._draw_speed_profile_on_canvas(self.speed_canvas, "pred")

    def _draw_gt_speed_profile(self) -> None:
        if not hasattr(self, "gt_speed_canvas"):
            return
        self._draw_speed_profile_on_canvas(self.gt_speed_canvas, "gt")
