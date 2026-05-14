"""WidgetLayoutMixin for the enhanced trajectory GUI."""

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

class WidgetLayoutMixin:

    def _create_widgets(self):
        """Create GUI widgets."""
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar
        title_frame = tk.Frame(main_frame, bg="#2b2b2b")
        title_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            title_frame, text="Trajectory Viewer",
            font=("Arial", 16, "bold"), fg="white", bg="#2b2b2b",
        )
        self.title_label.pack(side=tk.LEFT)
        
        self.nav_label = tk.Label(
            title_frame, text="",
            font=("Arial", 12), fg="#888888", bg="#2b2b2b",
        )
        self.nav_label.pack(side=tk.RIGHT)

        jump_frame = tk.Frame(main_frame, bg="#2b2b2b")
        jump_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Label(
            jump_frame, text="Dataset",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.dataset_var,
            values=self.datasets,
            state="readonly",
            width=30,
        )
        self.dataset_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_combo_selected)

        tk.Label(
            jump_frame, text="Clip",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.clip_var = tk.StringVar()
        self.clip_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.clip_var,
            state="readonly",
            width=24,
        )
        self.clip_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.clip_combo.bind("<<ComboboxSelected>>", self._on_clip_combo_selected)

        tk.Label(
            jump_frame, text="t0",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=(0, 4))
        self.t0_var = tk.StringVar()
        self.t0_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.t0_var,
            state="readonly",
            width=22,
        )
        self.t0_combo.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(
            jump_frame, text="Jump", command=self._jump_to_selected_sample,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            jump_frame, text="Current", command=self._sync_sample_selectors,
        ).pack(side=tk.LEFT)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg="#2b2b2b")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Bird's eye view
        left_frame = tk.Frame(content_frame, bg="#1e1e1e")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(
            left_frame, text="Bird's Eye View",
            font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e",
        ).pack(pady=5)
        
        self.traj_canvas = tk.Canvas(
            left_frame, width=self.bev_canvas_width, height=self.bev_canvas_height,
            bg="#1e1e1e", highlightthickness=0,
        )
        self.traj_canvas.pack(padx=5, pady=5)
        self.traj_canvas.bind("<Button-1>", self._on_canvas_click)
        self.traj_canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.traj_canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.traj_canvas.bind("<Button-3>", self._on_canvas_right_down)
        self.traj_canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.traj_canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.traj_canvas.bind("<Motion>", self._on_traj_canvas_motion)
        self.traj_canvas.bind("<Leave>", lambda _event: self._hide_stop_tooltip())

        pred_speed_header = tk.Frame(left_frame, bg="#1e1e1e")
        pred_speed_header.pack(fill=tk.X, pady=(8, 2))
        tk.Label(
            pred_speed_header, text="Diversity Speed Profile",
            font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e",
        ).pack(side=tk.LEFT)
        tk.Button(
            pred_speed_header,
            text="优化速度曲线",
            command=self._optimize_pred_speed_curve,
            bg="#34495e",
            fg="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=(0, 5))
        self.speed_canvas = tk.Canvas(
            left_frame,
            width=self.speed_canvas_width,
            height=self.speed_canvas_height,
            bg="#171717",
            highlightthickness=0,
        )
        self.speed_canvas.pack(padx=5, pady=(0, 5))
        self.speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "pred"))
        self.speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("pred", None))
        self.pred_speed_action_frame = tk.Frame(left_frame, bg="#1e1e1e")
        tk.Button(
            self.pred_speed_action_frame,
            text="接受",
            command=self._save_speed_edit,
            bg="#1f8f5f",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.pred_speed_action_frame,
            text="取消",
            command=lambda: self._cancel_speed_edit(redraw=True),
            bg="#7f8c8d",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)

        self.gt_speed_header_frame = tk.Frame(left_frame, bg="#1e1e1e")
        self.gt_speed_header_frame.pack(fill=tk.X, pady=(6, 2))
        tk.Label(
            self.gt_speed_header_frame, text="GT Speed Profile",
            font=("Arial", 12, "bold"), fg="white", bg="#1e1e1e",
        ).pack(side=tk.LEFT)
        tk.Button(
            self.gt_speed_header_frame,
            text="优化速度曲线",
            command=self._optimize_gt_speed_curve,
            bg="#5d6d7e",
            fg="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=(0, 5))
        tk.Button(
            self.gt_speed_header_frame,
            text="停车添加",
            command=self._start_gt_stop_add,
            bg="#8e3b35",
            fg="white",
            padx=8,
            pady=2,
        ).pack(side=tk.RIGHT, padx=(0, 5))
        self.gt_speed_canvas = tk.Canvas(
            left_frame,
            width=self.speed_canvas_width,
            height=self.speed_canvas_height,
            bg="#171717",
            highlightthickness=0,
        )
        self.gt_speed_canvas.pack(padx=5, pady=(0, 5))
        self.gt_speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "gt"))
        self.gt_speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("gt", None))
        self.gt_speed_canvas.bind("<Button-1>", self._on_gt_speed_canvas_click)
        self.gt_speed_action_frame = tk.Frame(left_frame, bg="#1e1e1e")
        tk.Button(
            self.gt_speed_action_frame,
            text="接受",
            command=self._save_gt_speed_edit,
            bg="#1f8f5f",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            self.gt_speed_action_frame,
            text="取消",
            command=lambda: self._cancel_gt_speed_edit(redraw=True),
            bg="#7f8c8d",
            fg="white",
            padx=16,
        ).pack(side=tk.LEFT, padx=5)
        self.gt_stop_action_frame = tk.Frame(left_frame, bg="#1e1e1e")
        tk.Button(
            self.gt_stop_action_frame,
            text="保存",
            command=self._save_gt_speed_edit,
            bg="#1f8f5f",
            fg="white",
            padx=14,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.gt_stop_action_frame,
            text="取消",
            command=lambda: self._cancel_gt_speed_edit(redraw=True),
            bg="#7f8c8d",
            fg="white",
            padx=14,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self.gt_stop_action_frame,
            text="撤回",
            command=self._undo_gt_stop_add,
            bg="#a56a22",
            fg="white",
            padx=14,
        ).pack(side=tk.LEFT, padx=4)
        
        # Middle panel - Camera images with projection
        middle_frame = tk.Frame(content_frame, bg="#2b2b2b")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10)
        
        # Camera selection
        cam_select_frame = tk.Frame(middle_frame, bg="#2b2b2b")
        cam_select_frame.pack(fill=tk.X)
        
        tk.Label(
            cam_select_frame, text="Projection Camera:",
            font=("Arial", 10), fg="white", bg="#2b2b2b",
        ).pack(side=tk.LEFT, padx=5)
        
        self.cam_var = tk.StringVar(value=self.current_cam_for_projection)
        for cam in ["FL", "FC", "FR", "RL", "RC", "RR"]:
            tk.Radiobutton(
                cam_select_frame, text=cam, variable=self.cam_var, value=cam,
                command=lambda c=cam: self._set_projection_camera(c),
                bg="#2b2b2b", fg="white", selectcolor="#444444",
                activebackground="#2b2b2b", activeforeground="white",
            ).pack(side=tk.LEFT, padx=2)
        
        # Camera image labels. Rear cameras sit side-by-side above the larger FC view.
        self.camera_labels = {}

        def create_camera_panel(parent, cam, side=tk.TOP, expand=False):
            cam_frame = tk.Frame(parent, bg="#1e1e1e")
            cam_frame.pack(side=side, expand=expand, padx=4, pady=4)

            tk.Label(
                cam_frame, text=cam,
                font=("Arial", 10, "bold"), fg="white", bg="#1e1e1e",
            ).pack()

            self.camera_labels[cam] = tk.Label(cam_frame, bg="#1e1e1e")
            self.camera_labels[cam].pack()
            self.camera_labels[cam].bind(
                "<Button-1>",
                lambda event, camera=cam: self._on_camera_click(event, camera),
            )
            self.camera_labels[cam].bind(
                "<Button-3>",
                lambda event, camera=cam: self._on_camera_right_down(event, camera),
            )
            self.camera_labels[cam].bind(
                "<B3-Motion>",
                lambda event, camera=cam: self._on_camera_right_drag(event, camera),
            )
            self.camera_labels[cam].bind("<ButtonRelease-3>", self._on_right_release)

        top_camera_row = tk.Frame(middle_frame, bg="#2b2b2b")
        top_camera_row.pack(fill=tk.X, pady=(5, 4))
        main_camera_area = tk.Frame(middle_frame, bg="#2b2b2b")
        main_camera_area.pack(fill=tk.BOTH, expand=True)

        top_cameras = [cam for cam in ("RL", "RR") if cam in self.cameras]
        remaining_top_cameras = [
            cam for cam in self.cameras
            if cam not in top_cameras and cam != "FC"
        ]
        top_cameras.extend(remaining_top_cameras)

        for cam in top_cameras:
            create_camera_panel(top_camera_row, cam, side=tk.LEFT, expand=True)

        if "FC" in self.cameras:
            create_camera_panel(main_camera_area, "FC", side=tk.TOP, expand=False)
        
        # Right panel - Trajectory list
        right_frame = tk.Frame(content_frame, bg="#2b2b2b", width=460)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)
        
        tk.Label(
            right_frame, text="Trajectories:",
            font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
        ).pack(anchor=tk.W, pady=(0, 5))
        
        list_frame = tk.Frame(right_frame, bg="#2b2b2b")
        list_frame.pack(fill=tk.X, pady=(0, 6))
        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.traj_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scroll.set,
            font=("Courier", 10), bg="#1e1e1e", fg="white",
            selectbackground="#444444", selectforeground="white", height=20,
            width=56,
            justify=tk.LEFT,
        )
        self.traj_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.traj_listbox.yview)
        self.traj_listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.traj_listbox.bind("<Motion>", self._on_traj_list_motion)
        self.traj_listbox.bind("<Leave>", lambda _event: self._hide_traj_list_tooltip())

        tk.Label(
            right_frame, text="CoT:",
            font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
        ).pack(anchor=tk.W, pady=(10, 5))

        self.cot_text = tk.Text(
            right_frame,
            height=12,
            width=42,
            wrap=tk.WORD,
            font=("Arial", 9),
            bg="#1e1e1e",
            fg="#dddddd",
            insertbackground="white",
        )
        self.cot_text.pack(fill=tk.BOTH, expand=False)
        self.cot_text.configure(state=tk.DISABLED)

        cluster_frame = tk.Frame(right_frame, bg="#2b2b2b")
        cluster_frame.pack(fill=tk.X, pady=(10, 4))
        tk.Label(
            cluster_frame, text="Cluster Centers:",
            font=("Arial", 12, "bold"), fg="white", bg="#2b2b2b",
        ).pack(anchor=tk.W, pady=(0, 5))

        cluster_select_frame = tk.Frame(cluster_frame, bg="#2b2b2b")
        cluster_select_frame.pack(fill=tk.X)
        cluster_button_frame = tk.Frame(cluster_frame, bg="#2b2b2b")
        cluster_button_frame.pack(fill=tk.X, pady=(5, 0))

        self.cluster_category_var = tk.StringVar(value="stop")
        self.cluster_category_combo = ttk.Combobox(
            cluster_select_frame,
            textvariable=self.cluster_category_var,
            values=CLUSTER_CATEGORY_ORDER,
            state="readonly",
            width=12,
        )
        self.cluster_category_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_category_combo.bind(
            "<<ComboboxSelected>>",
            self._on_cluster_category_selected,
        )

        self.cluster_choice_var = tk.StringVar()
        self.cluster_choice_combo = ttk.Combobox(
            cluster_select_frame,
            textvariable=self.cluster_choice_var,
            state="readonly",
            width=26,
        )
        self.cluster_choice_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_choice_combo.bind(
            "<<ComboboxSelected>>",
            self._on_cluster_choice_selected,
        )

        tk.Button(
            cluster_button_frame, text="-", command=lambda: self._cycle_selected_cluster_center(-1),
            bg="#7f8c8d", fg="white", padx=8, width=2,
        ).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(
            cluster_button_frame, text="+", command=lambda: self._cycle_selected_cluster_center(1),
            bg="#7f8c8d", fg="white", padx=8, width=2,
        ).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(
            cluster_button_frame, text="Confirm Save", command=self._save_selected_cluster_center_trajectory,
            bg="#ba6f1e", fg="white", padx=8,
        ).pack(side=tk.LEFT)
        self._refresh_cluster_choice_values()
        
        # Status bar
        status_frame = tk.Frame(main_frame, bg="#2b2b2b")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = tk.Label(
            status_frame, text="",
            font=("Arial", 10), fg="#888888", bg="#2b2b2b",
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Buttons
        controls_frame = tk.Frame(main_frame, bg="#2b2b2b")
        controls_frame.pack(fill=tk.X, pady=(8, 0))
        btn_frame = tk.Frame(controls_frame, bg="#2b2b2b")
        btn_frame.pack(anchor=tk.E, pady=(0, 6))
        btn_frame_2 = tk.Frame(controls_frame, bg="#2b2b2b")
        btn_frame_2.pack(anchor=tk.E)

        control_font = ("Arial", 10, "bold")
        control_button_opts = {
            "font": control_font,
            "padx": 14,
            "pady": 6,
        }

        self.draw_line_var = tk.BooleanVar(value=self.draw_line_enabled)
        tk.Checkbutton(
            btn_frame, text="Draw Bezier", variable=self.draw_line_var,
            command=self._toggle_draw_line,
            bg="#2b2b2b", fg="white", selectcolor="#444444",
            activebackground="#2b2b2b", activeforeground="white",
            font=control_font,
            padx=8,
            pady=4,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="Add Final Stop", command=self._add_final_stop_point,
            bg="#b03a2e", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Label(
            btn_frame, text="Stop Time(s)", bg="#2b2b2b", fg="#dddddd",
            font=control_font,
        ).pack(side=tk.LEFT, padx=(4, 2))
        self.stop_duration_var = tk.DoubleVar(value=self.stop_duration_seconds)
        stop_spin = tk.Spinbox(
            btn_frame,
            from_=0.5,
            to=5.0,
            increment=0.5,
            width=4,
            textvariable=self.stop_duration_var,
            command=self._update_stop_duration_seconds,
            bg="#1e1e1e",
            fg="white",
            insertbackground="white",
            font=control_font,
        )
        stop_spin.pack(side=tk.LEFT, padx=(0, 5))
        stop_spin.bind("<Return>", lambda _event: self._update_stop_duration_seconds())
        stop_spin.bind("<FocusOut>", lambda _event: self._update_stop_duration_seconds())

        tk.Button(
            btn_frame, text="Undo Control", command=self._undo_manual_point,
            bg="#7f8c8d", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame_2, text="Save Curve Traj", command=self._save_manual_bezier_trajectory,
            bg="#16a085", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Save Bezier Center", command=self._save_manual_bezier_as_cluster_center,
            bg="#d68910", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Delete Current Bezier Center", command=self._delete_current_bezier_cluster_center,
            bg="#a93226", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame_2, text="Edit Traj", command=self._start_saved_trajectory_edit,
            bg="#1f77b4", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Save Edit", command=self._save_saved_trajectory_edit,
            bg="#1e8449", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Cancel Edit", command=self._cancel_saved_trajectory_edit,
            bg="#7f8c8d", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Restore Edit", command=self._restore_saved_trajectory_edit,
            bg="#566573", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame_2, text="Repair GT", command=self._repair_gt_future,
            bg="#9b59b6", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            btn_frame_2, text="Restore GT", command=self._restore_gt_future,
            bg="#566573", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame_2, text="Delete (Del)", command=self._delete_traj,
            bg="#c0392b", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame_2, text="Undo Delete", command=lambda: self._undo_delete_traj(redraw=True),
            bg="#27ae60", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame_2, text="Confirm Save (Ctrl+S)", command=self._save_results,
            bg="#2980b9", fg="white", **control_button_opts,
        ).pack(side=tk.LEFT, padx=5)
        
        # Help
        help_frame = tk.Frame(main_frame, bg="#2b2b2b")
        help_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            help_frame,
            text="Controls: ←/→ Samples | ↑/↓ Trajectories | +/- Cluster centers | Draw Bezier adds controls | Right-drag controls in BEV/FC edits the same curve | Edit Traj shows BEV handles for saved pseudo-GT | Add Final Stop uses the Bezier endpoint and Stop Time(s) | Repair/Restore GT toggles raw vs velocity-integrated GT | Ctrl+S Save | Q Quit",
            font=("Arial", 9), fg="#666666", bg="#2b2b2b",
        ).pack()

    def _set_projection_camera(self, cam):
        """Set camera for trajectory projection."""
        self.current_cam_for_projection = cam
        self._update_display()

    def _toggle_projection_camera(self):
        """Toggle to next camera for projection."""
        cams = ["FL", "FC", "FR", "RL", "RC", "RR"]
        current_idx = cams.index(self.current_cam_for_projection)
        next_idx = (current_idx + 1) % len(cams)
        self.current_cam_for_projection = cams[next_idx]
        self.cam_var.set(self.current_cam_for_projection)
        self._update_display()

    def _toggle_draw_line(self):
        self.draw_line_enabled = bool(self.draw_line_var.get())
        self._update_draw_cursor()

    def _update_stop_duration_seconds(self):
        try:
            value = float(self.stop_duration_var.get())
        except (TypeError, ValueError, tk.TclError):
            value = self.stop_duration_seconds
        self.stop_duration_seconds = float(np.clip(value, 0.5, 5.0))
        if self.stop_duration_var is not None:
            self.stop_duration_var.set(self.stop_duration_seconds)

    def _update_draw_cursor(self):
        cursor = "crosshair" if self.draw_line_enabled else ("hand2" if getattr(self, "traj_geom_edit_active", False) else "")
        self.traj_canvas.config(cursor=cursor)
        for label in self.camera_labels.values():
            label.config(cursor=cursor)

    def _update_display(self):
        """Update the display."""
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        mode_parts = []
        if self.gt_only:
            mode_parts.append(f"GT-only stride={self.gt_stride_frames}")
        if self.index_mode in {"video_frames", "merged"}:
            mode_parts.append(f"index={self.index_mode} stride={self.frame_stride}")
        mode_suffix = f" | {' | '.join(mode_parts)}" if mode_parts else ""
        
        self.title_label.config(
            text=f"Dataset: {dataset_name} | Clip: {clip_stem}{mode_suffix}"
        )
        self.nav_label.config(
            text=f"{self.current_idx + 1} / {len(self.samples)}"
        )
        self._sync_sample_selectors()
        
        visible_indices = self._visible_trajectory_indices()
        if self.trajectories and self.current_traj_idx not in visible_indices:
            self._select_nearest_visible_trajectory(self.current_traj_idx)
            visible_indices = self._visible_trajectory_indices()
        pending_delete_count = self._pending_delete_count()
        kept = len(visible_indices)
        edit_suffix = ""
        if getattr(self, "traj_geom_edit_active", False):
            dirty = "*" if getattr(self, "traj_geom_edit_dirty", False) else ""
            edit_suffix = f" | Traj edit: T{self.traj_geom_edit_traj_idx}{dirty}"
        points_suffix = (
            "*"
            if self.manual_line_points_dirty
            or self.manual_camera_line_points_dirty
            or self.manual_stop_points_dirty
            else ""
        )
        total_line_points = len(self.manual_line_points) + len(self.manual_camera_line_points)
        coverage_status = self._sample_index_coverage_status(dataset_name, clip_stem, int(t0_us))
        gt_diag = self._get_gt_quality_diagnostics()
        gt_status = "GT: n/a"
        if gt_diag is not None:
            bad_accel_count = len(gt_diag.get("bad_accel_indices", []))
            jump_count = len(gt_diag.get("jump_indices", []))
            gt_status = (
                f"GT({self.gt_future_mode}): acc[{gt_diag.get('min_accel', 0.0):.1f},"
                f"{gt_diag.get('max_accel', 0.0):.1f}] "
                f"jump={gt_diag.get('max_step_m', 0.0):.2f}m"
            )
            if bad_accel_count or jump_count:
                gt_status += f" WARN acc={bad_accel_count} jump={jump_count}"
            if self.gt_future_mode == "raw" and self.conv_data is not None:
                repaired_diag = self._get_gt_quality_diagnostics("repaired")
                if repaired_diag is not None:
                    repaired_acc = len(repaired_diag.get("bad_accel_indices", []))
                    repaired_jump = len(repaired_diag.get("jump_indices", []))
                    gt_status += f" | repaired acc={repaired_acc} jump={repaired_jump}"
        self.status_label.config(
            text=(
                f"Trajectories: {len(self.trajectories)} | Visible: {kept} | "
                f"Pending delete: {pending_delete_count} | Bezier controls: {total_line_points}{points_suffix} | "
                f"Stops: {len(self.manual_stop_points)}{edit_suffix} | t0={int(t0_us)}{coverage_status} | {gt_status}"
            )
        )
        
        # Update listbox
        self._refresh_trajectory_smoothness()
        self.traj_listbox.delete(0, tk.END)
        self.traj_listbox_to_traj_idx = []
        for i in visible_indices:
            traj = self.trajectories[i]
            is_kept = self.trajectory_states.get(i, True)
            diagnostics = self.trajectory_smoothness.get(i, {})
            is_gt = self._is_gt_trajectory(traj, i)
            status = "GT" if is_gt else ("×" if not bool(diagnostics.get("ok", True)) else ("√" if is_kept else "×"))
            x_end = traj["x"][-1] if len(traj["x"]) > 0 else 0
            y_end = traj["y"][-1] if len(traj["y"]) > 0 else 0
            source = str(traj.get("source", "") or "traj")
            text = f"[{status}] T{i:<2} {source:<12} end=({x_end:6.1f}, {y_end:6.1f})"
            self.traj_listbox.insert(tk.END, text)
            row_idx = len(self.traj_listbox_to_traj_idx)
            self.traj_listbox_to_traj_idx.append(i)
            if is_gt:
                self.traj_listbox.itemconfig(row_idx, foreground=GT_COLOR_HEX)
            elif not bool(diagnostics.get("ok", True)):
                self.traj_listbox.itemconfig(row_idx, foreground="#ff8a8a")
        
        if self.current_traj_idx in self.traj_listbox_to_traj_idx:
            row_idx = self.traj_listbox_to_traj_idx.index(self.current_traj_idx)
            self.traj_listbox.selection_clear(0, tk.END)
            self.traj_listbox.selection_set(row_idx)
            self.traj_listbox.see(row_idx)
        
        self._draw_trajectories()
        self._draw_speed_profile()
        self._draw_gt_speed_profile()
        self._draw_camera_images()
        self._update_cot_text()

    def _update_cot_text(self):
        """Display CoT for the selected trajectory when available."""
        cot = ""
        if 0 <= self.current_traj_idx < len(self.trajectories):
            cot = self.trajectories[self.current_traj_idx].get("cot", "")
        if not cot:
            cot = "No CoT saved for this trajectory."

        self.cot_text.configure(state=tk.NORMAL)
        self.cot_text.delete("1.0", tk.END)
        self.cot_text.insert(tk.END, cot)
        self.cot_text.configure(state=tk.DISABLED)
