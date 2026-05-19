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

from traj_core.data_loader import get_dataset_names, get_clip_stems_from_dataset, load_data, get_t0_candidates
from traj_core.calibration_loader import load_calibration_for_segment
from traj_core.visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

from traj_core.constants import *
from traj_core.math_utils import *
from traj_core.speed_utils import *
from ..projection_utils import *
from traj_core.cluster_utils import *

# Color Palette constants
BG_MAIN = "#1E1E1E"
BG_PANEL = "#252526"
BG_HEADER = "#333333"
FG_PRIMARY = "#FFFFFF"
FG_SECONDARY = "#CCCCCC"

def FlatButton(parent, text, command, bg, fg="white", **kwargs):
    """Helper for modern flat buttons."""
    return tk.Button(
        parent, text=text, command=command, bg=bg, fg=fg,
        activebackground=bg, activeforeground="white",
        relief=tk.FLAT, borderwidth=0, cursor="hand2", **kwargs
    )

class WidgetLayoutMixin:

    def _create_scrollable_main_frame(self, bg_color: str):
        # Use a fixed root-sized frame; its direct children use grid rows so
        # the bottom controls remain visible when the visual content shrinks.
        main_frame = tk.Frame(self.root, bg=bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True)
        content_pad = getattr(self, "responsive_content_pad", 12)
        main_frame.configure(padx=content_pad, pady=content_pad)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        self.main_scroll_frame = main_frame
        return main_frame

    def _create_widgets(self):
        """Create GUI widgets with a modern dark theme, using modularized builders."""
        # Style configuration
        self.style = ttk.Style()
        if 'clam' in self.style.theme_names():
            self.style.theme_use('clam')
        self.style.configure("Dark.TLabelframe", background=BG_PANEL, foreground=FG_PRIMARY)
        self.style.configure("Dark.TLabelframe.Label", background=BG_PANEL, foreground=FG_PRIMARY, font=("Segoe UI", 10, "bold"))

        self.root.configure(bg=BG_MAIN)
        self.main_frame = self._create_scrollable_main_frame(BG_MAIN)

        self._build_header_panel(self.main_frame)
        self._build_center_workspace(self.main_frame)
        self._build_bottom_controls(self.main_frame)
        self._build_footer_panel(self.main_frame)

        # Bind a single resize handler on the root window.
        self.root.bind("<Configure>", self._on_window_configure, add="+")

    def _build_header_panel(self, parent):
        header_frame = tk.Frame(parent, bg=BG_PANEL, relief=tk.FLAT, bd=0)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        title_frame = tk.Frame(header_frame, bg=BG_PANEL)
        title_frame.pack(fill=tk.X, padx=10, pady=8)

        self.title_label = tk.Label(
            title_frame, text="Trajectory Viewer",
            font=("Segoe UI", 14, "bold"), fg=FG_PRIMARY, bg=BG_PANEL,
        )
        self.title_label.pack(side=tk.LEFT)

        self.nav_label = tk.Label(
            title_frame, text="",
            font=("Segoe UI", 12), fg=FG_SECONDARY, bg=BG_PANEL,
        )
        self.nav_label.pack(side=tk.RIGHT)

        jump_frame = tk.Frame(header_frame, bg=BG_PANEL)
        jump_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Label(jump_frame, text="Dataset", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            jump_frame, textvariable=self.dataset_var, values=self.datasets,
            state="readonly", width=getattr(self, "dataset_combo_width", 30),
        )
        self.dataset_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_combo_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.dataset_combo)

        tk.Label(jump_frame, text="Clip", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.clip_var = tk.StringVar()
        self.clip_combo = ttk.Combobox(
            jump_frame, textvariable=self.clip_var, state="readonly",
            width=getattr(self, "clip_combo_width", 24),
        )
        self.clip_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.clip_combo.bind("<<ComboboxSelected>>", self._on_clip_combo_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.clip_combo)

        tk.Label(jump_frame, text="t0", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.t0_var = tk.StringVar()
        self.t0_combo = ttk.Combobox(
            jump_frame, textvariable=self.t0_var, state="readonly",
            width=getattr(self, "t0_combo_width", 22),
        )
        self.t0_combo.pack(side=tk.LEFT, padx=(0, 16))
        self._bind_arrow_keys_for_trajectory_navigation(self.t0_combo)

        tk.Label(jump_frame, text="Scene", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.scene_filter_var = tk.StringVar(value=getattr(self, "scene_filter_value", "None"))
        self.scene_filter_combo = ttk.Combobox(
            jump_frame, textvariable=self.scene_filter_var, state="readonly",
            width=getattr(self, "scene_combo_width", 18),
        )
        self.scene_filter_combo.pack(side=tk.LEFT, padx=(0, 16))
        self.scene_filter_combo.bind("<<ComboboxSelected>>", self._on_scene_filter_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.scene_filter_combo)

        FlatButton(jump_frame, "Jump", self._jump_to_selected_sample, bg="#2196F3", padx=16, pady=4).pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(jump_frame, "Current", self._sync_sample_selectors, bg="#607D8B", padx=16, pady=4).pack(side=tk.LEFT)

    def _build_center_workspace(self, parent):
        content_frame = tk.Frame(parent, bg=BG_MAIN)
        content_frame.grid(row=1, column=0, sticky="nsew")

        horizontal_paned = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        horizontal_paned.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(horizontal_paned)
        self._build_middle_panel(horizontal_paned)
        self._build_right_panel(horizontal_paned)

    def _build_left_panel(self, paned_window):
        left_outer_frame = tk.Frame(paned_window, bg=BG_PANEL)
        paned_window.add(left_outer_frame, weight=0)  # Weight 0 so it wraps tightly to canvas width

        left_paned = ttk.PanedWindow(left_outer_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        # Pane 1: BEV
        bev_frame = tk.Frame(left_paned, bg=BG_PANEL)
        left_paned.add(bev_frame, weight=3)

        bev_header = tk.Frame(bev_frame, bg=BG_PANEL)
        bev_header.pack(fill=tk.X, pady=(0, 5))
        tk.Label(
            bev_header, text="Bird's Eye View",
            font=("Segoe UI", 12, "bold"), fg=FG_PRIMARY, bg=BG_PANEL,
        ).pack(side=tk.LEFT)
        
        self.show_objects_var = tk.BooleanVar(value=bool(getattr(self, "show_objects_enabled", True)))
        tk.Checkbutton(
            bev_header, text="交通参与者", variable=self.show_objects_var,
            command=self._toggle_object_overlay, bg=BG_PANEL, fg=FG_SECONDARY,
            selectcolor=BG_MAIN, activebackground=BG_PANEL, activeforeground=FG_PRIMARY,
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT)

        self.traj_canvas = tk.Canvas(
            bev_frame, width=self.bev_canvas_width, height=self.bev_canvas_height,
            bg="#121212", highlightthickness=1, highlightbackground="#333333",
        )
        self.traj_canvas.pack(fill=tk.BOTH, expand=True)
        self.traj_canvas.bind("<Button-1>", self._on_canvas_click)
        self.traj_canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.traj_canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.traj_canvas.bind("<Button-3>", self._on_canvas_right_down)
        self.traj_canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.traj_canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.traj_canvas.bind("<Motion>", self._on_traj_canvas_motion)
        self.traj_canvas.bind("<Leave>", lambda _event: self._hide_stop_tooltip())

        # Pane 2: Pred Speed
        pred_frame = tk.Frame(left_paned, bg=BG_PANEL)
        left_paned.add(pred_frame, weight=1)

        pred_speed_header = tk.Frame(pred_frame, bg=BG_PANEL)
        pred_speed_header.pack(fill=tk.X, pady=(5, 2))
        tk.Label(pred_speed_header, text="Diversity Speed Profile", font=("Segoe UI", 11, "bold"), fg=FG_PRIMARY, bg=BG_PANEL).pack(side=tk.LEFT)
        FlatButton(pred_speed_header, "优化速度曲线", self._optimize_pred_speed_curve, bg="#34495e", padx=10, pady=2).pack(side=tk.RIGHT)

        self.speed_canvas = tk.Canvas(
            pred_frame, width=self.speed_canvas_width, height=self.speed_canvas_height,
            bg="#121212", highlightthickness=1, highlightbackground="#333333",
        )
        self.speed_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "pred"))
        self.speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("pred", None))
        self.speed_canvas.bind("<Button-1>", self._on_speed_canvas_left_down)
        self.speed_canvas.bind("<B1-Motion>", self._on_speed_canvas_left_drag)
        self.speed_canvas.bind("<ButtonRelease-1>", self._on_speed_canvas_left_release)

        self.pred_speed_action_frame = tk.Frame(pred_frame, bg=BG_PANEL)
        FlatButton(self.pred_speed_action_frame, "接受", self._save_speed_edit, bg="#4CAF50", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.pred_speed_action_frame, "取消", lambda: self._cancel_speed_edit(redraw=True), bg="#F44336", padx=16).pack(side=tk.LEFT)

        # Pane 3: GT Speed
        gt_frame = tk.Frame(left_paned, bg=BG_PANEL)
        left_paned.add(gt_frame, weight=1)

        gt_speed_header_frame = tk.Frame(gt_frame, bg=BG_PANEL)
        gt_speed_header_frame.pack(fill=tk.X, pady=(5, 2))
        tk.Label(gt_speed_header_frame, text="GT Speed Profile", font=("Segoe UI", 11, "bold"), fg=FG_PRIMARY, bg=BG_PANEL).pack(side=tk.LEFT)
        FlatButton(gt_speed_header_frame, "优化速度曲线", self._optimize_gt_speed_curve, bg="#5d6d7e", padx=10, pady=2).pack(side=tk.RIGHT)
        FlatButton(gt_speed_header_frame, "停车添加", self._start_gt_stop_add, bg="#D32F2F", padx=10, pady=2).pack(side=tk.RIGHT, padx=(0, 5))

        self.gt_speed_canvas = tk.Canvas(
            gt_frame, width=self.speed_canvas_width, height=self.speed_canvas_height,
            bg="#121212", highlightthickness=1, highlightbackground="#333333",
        )
        self.gt_speed_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.gt_speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "gt"))
        self.gt_speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("gt", None))
        self.gt_speed_canvas.bind("<Button-1>", self._on_gt_speed_canvas_click)

        self.gt_speed_action_frame = tk.Frame(gt_frame, bg=BG_PANEL)
        FlatButton(self.gt_speed_action_frame, "接受", self._save_gt_speed_edit, bg="#4CAF50", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.gt_speed_action_frame, "取消", lambda: self._cancel_gt_speed_edit(redraw=True), bg="#F44336", padx=16).pack(side=tk.LEFT)

        self.gt_stop_action_frame = tk.Frame(gt_frame, bg=BG_PANEL)
        FlatButton(self.gt_stop_action_frame, "保存", self._save_gt_speed_edit, bg="#4CAF50", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.gt_stop_action_frame, "取消", lambda: self._cancel_gt_speed_edit(redraw=True), bg="#F44336", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.gt_stop_action_frame, "撤回", self._undo_gt_stop_add, bg="#FF9800", padx=16).pack(side=tk.LEFT)

    def _build_middle_panel(self, paned_window):
        middle_frame = tk.Frame(paned_window, bg=BG_PANEL)
        self.middle_frame = middle_frame
        paned_window.add(middle_frame, weight=1)

        cam_select_frame = tk.Frame(middle_frame, bg=BG_PANEL)
        cam_select_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(cam_select_frame, text="Projection Camera:", font=("Segoe UI", 10, "bold"), fg=FG_PRIMARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 10))

        self.cam_var = tk.StringVar(value=self.current_cam_for_projection)
        for cam in ["FL", "FC", "FR", "RL", "RC", "RR"]:
            tk.Radiobutton(
                cam_select_frame, text=cam, variable=self.cam_var, value=cam,
                command=lambda c=cam: self._set_projection_camera(c),
                bg=BG_PANEL, fg=FG_SECONDARY, selectcolor=BG_MAIN,
                activebackground=BG_PANEL, activeforeground=FG_PRIMARY, font=("Segoe UI", 9)
            ).pack(side=tk.LEFT, padx=4)

        self.camera_labels = {}
        self.camera_frames = {}

        def create_camera_panel(parent, cam, row, col, rowspan=1, colspan=1):
            cam_frame = tk.Frame(parent, bg=BG_MAIN, highlightthickness=1, highlightbackground="#333333")
            cam_frame.grid(row=row, column=col, rowspan=rowspan, columnspan=colspan, sticky="nsew", padx=4, pady=4)
            cam_frame.grid_propagate(False)

            cam_frame.grid_rowconfigure(1, weight=1)
            cam_frame.grid_columnconfigure(0, weight=1)

            tk.Label(cam_frame, text=cam, font=("Segoe UI", 10, "bold"), fg=FG_PRIMARY, bg=BG_HEADER).grid(row=0, column=0, sticky="ew")
            
            label = tk.Label(cam_frame, bg=BG_MAIN)
            label.grid(row=1, column=0, sticky="nsew")
            
            self.camera_labels[cam] = label
            self.camera_frames[cam] = cam_frame

            label.bind("<Button-1>", lambda event, camera=cam: self._on_camera_click(event, camera))
            label.bind("<Button-3>", lambda event, camera=cam: self._on_camera_right_down(event, camera))
            label.bind("<B3-Motion>", lambda event, camera=cam: self._on_camera_right_drag(event, camera))
            label.bind("<ButtonRelease-3>", self._on_right_release)

            cam_frame.bind("<Configure>", lambda e, c=cam: self._on_camera_frame_configure(e, c))

        cameras_container = tk.Frame(middle_frame, bg=BG_PANEL)
        cameras_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        cameras_container.grid_rowconfigure(0, weight=1)
        cameras_container.grid_rowconfigure(1, weight=2)

        top_cameras = [cam for cam in ("RL", "RR") if cam in self.cameras]
        remaining_top_cameras = [cam for cam in self.cameras if cam not in top_cameras and cam != "FC"]
        top_cameras.extend(remaining_top_cameras)
        num_top_cams = max(1, len(top_cameras))

        for i in range(num_top_cams):
            cameras_container.grid_columnconfigure(i, weight=1)

        for idx, cam in enumerate(top_cameras):
            create_camera_panel(cameras_container, cam, row=0, col=idx)

        if "FC" in self.cameras:
            create_camera_panel(cameras_container, "FC", row=1, col=0, colspan=num_top_cams)

    def _build_right_panel(self, paned_window):
        right_frame = tk.Frame(
            paned_window, bg=BG_PANEL,
            width=getattr(self, "right_panel_width", 460),
        )
        paned_window.add(right_frame, weight=0)
        right_frame.pack_propagate(False)

        vertical_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        vertical_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        # Pane 1: Trajectories
        traj_pane = ttk.LabelFrame(vertical_paned, text="Trajectories", style="Dark.TLabelframe")
        vertical_paned.add(traj_pane, weight=3)

        list_frame = tk.Frame(traj_pane, bg=BG_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.traj_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scroll.set,
            font=("Consolas", 10), bg="#121212", fg=FG_PRIMARY,
            selectbackground="#2196F3", selectforeground="white", height=12,
            width=getattr(self, "trajectory_listbox_width", 56),
            justify=tk.LEFT, relief=tk.FLAT, highlightthickness=1, highlightbackground="#333333"
        )
        self.traj_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.traj_listbox.yview)
        self.traj_listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.traj_listbox.bind("<Motion>", self._on_traj_list_motion)
        self.traj_listbox.bind("<Leave>", lambda _event: self._hide_traj_list_tooltip())

        # Pane 2: Chain of Thought
        cot_pane = ttk.LabelFrame(vertical_paned, text="Chain of Thought", style="Dark.TLabelframe")
        vertical_paned.add(cot_pane, weight=2)

        self.cot_text = tk.Text(
            cot_pane, height=8, width=42, wrap=tk.WORD,
            font=("Segoe UI", 9), bg="#121212", fg=FG_SECONDARY,
            insertbackground="white", relief=tk.FLAT, highlightthickness=1, highlightbackground="#333333"
        )
        self.cot_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.cot_text.configure(state=tk.DISABLED)

        # Pane 3: Cluster Centers
        cluster_pane = ttk.LabelFrame(vertical_paned, text="Cluster Centers", style="Dark.TLabelframe")
        vertical_paned.add(cluster_pane, weight=0)

        cluster_inner = tk.Frame(cluster_pane, bg=BG_PANEL)
        cluster_inner.pack(fill=tk.X, padx=8, pady=8)

        cluster_select_frame = tk.Frame(cluster_inner, bg=BG_PANEL)
        cluster_select_frame.pack(fill=tk.X)
        cluster_button_frame = tk.Frame(cluster_inner, bg=BG_PANEL)
        cluster_button_frame.pack(fill=tk.X, pady=(10, 0))

        self.cluster_category_var = tk.StringVar(value="stop")
        self.cluster_category_combo = ttk.Combobox(cluster_select_frame, textvariable=self.cluster_category_var, values=CLUSTER_CATEGORY_ORDER, state="readonly", width=12)
        self.cluster_category_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_category_combo.bind("<<ComboboxSelected>>", self._on_cluster_category_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.cluster_category_combo)

        self.cluster_choice_var = tk.StringVar()
        self.cluster_choice_combo = ttk.Combobox(cluster_select_frame, textvariable=self.cluster_choice_var, state="readonly", width=26)
        self.cluster_choice_combo.pack(side=tk.LEFT, padx=(0, 5))
        self.cluster_choice_combo.bind("<<ComboboxSelected>>", self._on_cluster_choice_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.cluster_choice_combo)

        FlatButton(cluster_button_frame, "-", lambda: self._cycle_selected_cluster_center(-1), bg="#607D8B", padx=12).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(cluster_button_frame, "+", lambda: self._cycle_selected_cluster_center(1), bg="#607D8B", padx=12).pack(side=tk.LEFT, padx=(0, 10))
        FlatButton(cluster_button_frame, "Confirm Save", self._save_selected_cluster_center_trajectory, bg="#FF9800", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(cluster_button_frame, "Hide", self._hide_cluster_preview, bg="#607D8B", padx=14).pack(side=tk.LEFT)
        self._refresh_cluster_choice_values()

    def _build_bottom_controls(self, parent):
        controls_outer_frame = tk.Frame(parent, bg=BG_MAIN)
        controls_outer_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        controls_outer_frame.grid_columnconfigure(0, weight=1)
        
        compact_controls = bool(
            getattr(getattr(self, "responsive_layout", None), "window_width", 1900) < 1700
        )

        control_font = ("Segoe UI", 9 if compact_controls else 10, "bold")
        btn_padx, btn_pady = (8, 4) if compact_controls else (12, 6)

        def place_control_group(widget, row: int, is_last: bool = False):
            if compact_controls:
                widget.grid(row=row, column=0, sticky="w", pady=(0, 4 if not is_last else 0))
            else:
                widget.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10 if not is_last else 0))

        # Group 1: Drawing & Bezier Tools
        draw_group = ttk.LabelFrame(controls_outer_frame, text="Drawing & Bezier Tools", style="Dark.TLabelframe")
        place_control_group(draw_group, 0)
        draw_inner = tk.Frame(draw_group, bg=BG_PANEL)
        draw_inner.pack(padx=8, pady=8)

        self.draw_line_var = tk.BooleanVar(value=self.draw_line_enabled)
        tk.Checkbutton(
            draw_inner, text="Draw Bezier", variable=self.draw_line_var, command=self._toggle_draw_line,
            bg=BG_PANEL, fg=FG_PRIMARY, selectcolor=BG_MAIN, activebackground=BG_PANEL, activeforeground=FG_PRIMARY,
            font=control_font
        ).pack(side=tk.LEFT, padx=(0, 8))

        FlatButton(draw_inner, "Add Final Stop", self._add_final_stop_point, bg="#D32F2F", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)

        tk.Label(draw_inner, text="Stop(s):", bg=BG_PANEL, fg=FG_SECONDARY, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(4, 2))
        self.stop_duration_var = tk.DoubleVar(value=self.stop_duration_seconds)
        stop_spin = tk.Spinbox(
            draw_inner, from_=0.5, to=5.0, increment=0.5, width=4, textvariable=self.stop_duration_var,
            command=self._update_stop_duration_seconds, bg="#121212", fg=FG_PRIMARY, insertbackground="white", font=control_font, relief=tk.FLAT
        )
        stop_spin.pack(side=tk.LEFT, padx=(0, 8))
        stop_spin.bind("<Return>", lambda _event: self._update_stop_duration_seconds())
        stop_spin.bind("<FocusOut>", lambda _event: self._update_stop_duration_seconds())

        FlatButton(draw_inner, "Undo Control", self._undo_manual_point, bg="#757575", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        ttk.Separator(draw_inner, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        FlatButton(draw_inner, "Save Curve Traj", self._save_manual_bezier_trajectory, bg="#009688", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(draw_inner, "Save Bezier Center", self._save_manual_bezier_as_cluster_center, bg="#FF9800", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(draw_inner, "Delete Bezier Center", self._delete_current_bezier_cluster_center, bg="#F44336", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)

        # Group 2: Trajectory Editing
        edit_group = ttk.LabelFrame(controls_outer_frame, text="Trajectory Editing", style="Dark.TLabelframe")
        place_control_group(edit_group, 1)
        edit_inner = tk.Frame(edit_group, bg=BG_PANEL)
        edit_inner.pack(padx=8, pady=8)

        FlatButton(edit_inner, "Edit Traj", self._start_saved_trajectory_edit, bg="#2196F3", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(edit_inner, "Save Edit", self._save_saved_trajectory_edit, bg="#4CAF50", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(edit_inner, "Cancel Edit", self._cancel_saved_trajectory_edit, bg="#9E9E9E", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(edit_inner, "Restore Edit", self._restore_saved_trajectory_edit, bg="#607D8B", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)

        # Group 3: GT & Global Actions
        action_group = ttk.LabelFrame(controls_outer_frame, text="GT & Global Actions", style="Dark.TLabelframe")
        place_control_group(action_group, 2, is_last=True)
        action_inner = tk.Frame(action_group, bg=BG_PANEL)
        action_inner.pack(padx=8, pady=8)

        FlatButton(action_inner, "Repair GT", self._repair_gt_future, bg="#9C27B0", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(action_inner, "Restore GT", self._restore_gt_future, bg="#795548", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        ttk.Separator(action_inner, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        FlatButton(action_inner, "Delete (Del)", self._delete_traj, bg="#F44336", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(action_inner, "Undo Delete", lambda: self._undo_delete_traj(redraw=True), bg="#4CAF50", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(action_inner, "Confirm Save (Ctrl+S)", self._save_results, bg="#2196F3", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(action_inner, "View Log", self._show_edit_log_window, bg="#607D8B", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(action_inner, "Restore Backup", self._restore_latest_current_clip_backup, bg="#795548", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)

    def _build_footer_panel(self, parent):
        footer_frame = tk.Frame(parent, bg=BG_MAIN)
        footer_frame.grid(row=3, column=0, sticky="ew", pady=(6, 0))

        self.status_label = tk.Label(footer_frame, text="", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_MAIN)
        self.status_label.pack(side=tk.LEFT)

        tk.Label(
            footer_frame,
            text="Shortcuts: ←/→ Samples | ↑/↓ Trajectories | +/- Cluster | Ctrl+S Save | Q Quit | Del Delete | Tab Camera",
            font=("Segoe UI", 9), fg="#757575", bg=BG_MAIN,
        ).pack(side=tk.RIGHT)

    def _set_projection_camera(self, cam):
        self.current_cam_for_projection = cam
        self._update_display()

    def _toggle_projection_camera(self):
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
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        mode_parts = []
        if self.gt_only:
            mode_parts.append(f"GT-only stride={self.gt_stride_frames}")
        if self.index_mode in {"video_frames", "merged"}:
            mode_parts.append(f"index={self.index_mode} stride={self.frame_stride}")
        mode_suffix = f" | {' | '.join(mode_parts)}" if mode_parts else ""

        self.title_label.config(
            text=f"DATASET:  {dataset_name}    |    CLIP:  {clip_stem}{mode_suffix}"
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
        cot = ""
        if 0 <= self.current_traj_idx < len(self.trajectories):
            cot = self.trajectories[self.current_traj_idx].get("cot", "")
        if not cot:
            cot = "No CoT saved for this trajectory."

        self.cot_text.configure(state=tk.NORMAL)
        self.cot_text.delete("1.0", tk.END)
        self.cot_text.insert(tk.END, cot)
        self.cot_text.configure(state=tk.DISABLED)

    def _on_camera_frame_configure(self, event, cam):
        if not hasattr(self, 'camera_display_meta'):
            self.camera_display_meta = {}
            
        if cam not in self.camera_display_meta:
            self.camera_display_meta[cam] = {}

        target_w = event.width - 4
        target_h = event.height - 24

        old_w = self.camera_display_meta[cam].get("target_width", 0)
        old_h = self.camera_display_meta[cam].get("target_height", 0)
        
        if abs(target_w - old_w) > 5 or abs(target_h - old_h) > 5:
            self.camera_display_meta[cam]["target_width"] = target_w
            self.camera_display_meta[cam]["target_height"] = target_h
            
            if hasattr(self, '_resize_after_id') and self._resize_after_id is not None:
                self.root.after_cancel(self._resize_after_id)
            self._resize_after_id = self.root.after(150, self._on_window_resize_deferred)

    def _on_window_configure(self, event):
        if getattr(self, "_closing", False):
            return
        if event.widget is not self.root:
            return
        if hasattr(self, '_resize_after_id') and self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(200, self._on_window_resize_deferred)

    def _on_window_resize_deferred(self):
        if getattr(self, "_closing", False):
            return
        self._resize_after_id = None

        if hasattr(self, 'traj_canvas'):
            try:
                w = self.traj_canvas.winfo_width()
                h = self.traj_canvas.winfo_height()
                if w > 10 and h > 10:
                    self.bev_canvas_width = w
                    self.bev_canvas_height = h
                    self.bev_forward_scale = 6.2 * h / 700.0
                    self.bev_lateral_scale = 10.0 * w / 560.0
                    self.bev_origin = (
                        w / 2,
                        h - max(40, int(65 * (h / 700.0))),
                    )
            except Exception:
                pass

        if hasattr(self, 'speed_canvas'):
            try:
                w = self.speed_canvas.winfo_width()
                h = self.speed_canvas.winfo_height()
                if w > 10 and h > 10:
                    self.speed_canvas_width = w
                    self.speed_canvas_height = h
            except Exception:
                pass
                
        if hasattr(self, 'middle_frame'):
            try:
                w = self.middle_frame.winfo_width()
                if w > 10:
                    self.middle_panel_width = w
            except Exception:
                pass

        if hasattr(self, 'trajectories') and self.trajectories:
            self._draw_trajectories()
            self._draw_speed_profile()
            self._draw_gt_speed_profile()
            self._draw_camera_images()
