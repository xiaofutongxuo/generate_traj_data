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

class WidgetLayoutMixin:

    def _create_scrollable_main_frame(self, bg_color: str):
        scroll_container = tk.Frame(self.root, bg=bg_color)
        scroll_container.pack(fill=tk.BOTH, expand=True)

        self.main_scroll_canvas = tk.Canvas(
            scroll_container,
            bg=bg_color,
            highlightthickness=0,
            borderwidth=0,
        )
        v_scroll = ttk.Scrollbar(
            scroll_container,
            orient=tk.VERTICAL,
            command=self.main_scroll_canvas.yview,
        )
        h_scroll = ttk.Scrollbar(
            scroll_container,
            orient=tk.HORIZONTAL,
            command=self.main_scroll_canvas.xview,
        )
        self.main_scroll_canvas.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
        )

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.main_scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main_frame = tk.Frame(self.main_scroll_canvas, bg=bg_color)
        self.main_scroll_frame = main_frame
        self.main_scroll_window = self.main_scroll_canvas.create_window(
            (0, 0),
            window=main_frame,
            anchor=tk.NW,
        )
        main_frame.pack_propagate(True)

        main_frame.bind("<Configure>", self._sync_main_scrollregion)
        self.main_scroll_canvas.bind("<Configure>", self._sync_main_scrollregion)
        self.root.bind("<MouseWheel>", self._on_main_scroll_mousewheel, add="+")
        self.root.bind("<Shift-MouseWheel>", self._on_main_scroll_shift_mousewheel, add="+")
        self.root.bind("<Button-4>", self._on_main_scroll_linux_wheel, add="+")
        self.root.bind("<Button-5>", self._on_main_scroll_linux_wheel, add="+")

        content_pad = getattr(self, "responsive_content_pad", 12)
        main_frame.configure(padx=content_pad, pady=content_pad)
        return main_frame

    def _sync_main_scrollregion(self, _event=None) -> None:
        if not hasattr(self, "main_scroll_canvas"):
            return
        canvas = self.main_scroll_canvas
        canvas.update_idletasks()
        canvas_width = max(canvas.winfo_width(), 1)
        canvas_height = max(canvas.winfo_height(), 1)
        frame = self.main_scroll_frame
        target_width = max(frame.winfo_reqwidth(), canvas_width)
        target_height = max(frame.winfo_reqheight(), canvas_height)
        canvas.itemconfigure(
            self.main_scroll_window,
            width=target_width,
            height=target_height,
        )
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _widget_owns_mousewheel(self, widget) -> bool:
        try:
            widget_class = str(widget.winfo_class())
        except tk.TclError:
            return False
        return widget_class in {"Text", "Listbox", "TCombobox", "Combobox", "Spinbox"}

    def _mousewheel_units(self, event) -> int:
        delta = getattr(event, "delta", 0)
        if delta:
            units = int(-1 * (delta / 120))
            if units == 0:
                units = -1 if delta > 0 else 1
            return units
        return 0

    def _on_main_scroll_mousewheel(self, event):
        if self._widget_owns_mousewheel(event.widget):
            return None
        if getattr(event, "state", 0) & 0x0001:
            return self._on_main_scroll_shift_mousewheel(event)
        units = self._mousewheel_units(event)
        if units:
            self.main_scroll_canvas.yview_scroll(units, "units")
            return "break"
        return None

    def _on_main_scroll_shift_mousewheel(self, event):
        if self._widget_owns_mousewheel(event.widget):
            return None
        units = self._mousewheel_units(event)
        if units:
            self.main_scroll_canvas.xview_scroll(units, "units")
            return "break"
        return None

    def _on_main_scroll_linux_wheel(self, event):
        if self._widget_owns_mousewheel(event.widget):
            return None
        units = -3 if getattr(event, "num", None) == 4 else 3
        self.main_scroll_canvas.yview_scroll(units, "units")
        return "break"

    def _create_widgets(self):
        """Create GUI widgets with a modern dark theme."""
        # Style configuration
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        # Color Palette
        BG_MAIN = "#1E1E1E"
        BG_PANEL = "#252526"
        BG_HEADER = "#333333"
        FG_PRIMARY = "#FFFFFF"
        FG_SECONDARY = "#CCCCCC"

        # Configure global window background
        self.root.configure(bg=BG_MAIN)

        main_frame = self._create_scrollable_main_frame(BG_MAIN)

        # --- Top Header Bar ---
        header_frame = tk.Frame(main_frame, bg=BG_PANEL, relief=tk.FLAT, bd=0)
        header_frame.pack(fill=tk.X, pady=(0, 10))

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

        # Helper for modern flat buttons
        def FlatButton(parent, text, command, bg, fg="white", **kwargs):
            return tk.Button(
                parent, text=text, command=command, bg=bg, fg=fg,
                activebackground=bg, activeforeground="white",
                relief=tk.FLAT, borderwidth=0, cursor="hand2", **kwargs
            )

        tk.Label(jump_frame, text="Dataset", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.dataset_var,
            values=self.datasets,
            state="readonly",
            width=getattr(self, "dataset_combo_width", 30),
        )
        self.dataset_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_combo_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.dataset_combo)

        tk.Label(jump_frame, text="Clip", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.clip_var = tk.StringVar()
        self.clip_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.clip_var,
            state="readonly",
            width=getattr(self, "clip_combo_width", 24),
        )
        self.clip_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.clip_combo.bind("<<ComboboxSelected>>", self._on_clip_combo_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.clip_combo)

        tk.Label(jump_frame, text="t0", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.t0_var = tk.StringVar()
        self.t0_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.t0_var,
            state="readonly",
            width=getattr(self, "t0_combo_width", 22),
        )
        self.t0_combo.pack(side=tk.LEFT, padx=(0, 16))
        self._bind_arrow_keys_for_trajectory_navigation(self.t0_combo)

        tk.Label(jump_frame, text="Scene", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_PANEL).pack(side=tk.LEFT, padx=(0, 4))
        self.scene_filter_var = tk.StringVar(value=getattr(self, "scene_filter_value", "None"))
        self.scene_filter_combo = ttk.Combobox(
            jump_frame,
            textvariable=self.scene_filter_var,
            state="readonly",
            width=getattr(self, "scene_combo_width", 18),
        )
        self.scene_filter_combo.pack(side=tk.LEFT, padx=(0, 16))
        self.scene_filter_combo.bind("<<ComboboxSelected>>", self._on_scene_filter_selected)
        self._bind_arrow_keys_for_trajectory_navigation(self.scene_filter_combo)

        FlatButton(jump_frame, "Jump", self._jump_to_selected_sample, bg="#2196F3", padx=16, pady=4).pack(side=tk.LEFT, padx=(0, 8))
        FlatButton(jump_frame, "Current", self._sync_sample_selectors, bg="#607D8B", padx=16, pady=4).pack(side=tk.LEFT)

        # --- Content Area ---
        content_frame = tk.Frame(main_frame, bg=BG_MAIN)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # 1. Left panel - Bird's eye view & Speeds
        left_frame = tk.Frame(content_frame, bg=BG_PANEL)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        tk.Label(left_frame, text="Bird's Eye View", font=("Segoe UI", 12, "bold"), fg=FG_PRIMARY, bg=BG_PANEL).pack(pady=(10, 5))

        self.traj_canvas = tk.Canvas(
            left_frame, width=self.bev_canvas_width, height=self.bev_canvas_height,
            bg="#121212", highlightthickness=1, highlightbackground="#333333",
        )
        self.traj_canvas.pack(padx=10, pady=(0, 10))
        self.traj_canvas.bind("<Button-1>", self._on_canvas_click)
        self.traj_canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.traj_canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.traj_canvas.bind("<Button-3>", self._on_canvas_right_down)
        self.traj_canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.traj_canvas.bind("<ButtonRelease-3>", self._on_right_release)
        self.traj_canvas.bind("<Motion>", self._on_traj_canvas_motion)
        self.traj_canvas.bind("<Leave>", lambda _event: self._hide_stop_tooltip())

        # Speed Profiles
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        pred_speed_header = tk.Frame(left_frame, bg=BG_PANEL)
        pred_speed_header.pack(fill=tk.X, padx=10, pady=(5, 2))
        tk.Label(pred_speed_header, text="Diversity Speed Profile", font=("Segoe UI", 11, "bold"), fg=FG_PRIMARY, bg=BG_PANEL).pack(side=tk.LEFT)
        FlatButton(pred_speed_header, "优化速度曲线", self._optimize_pred_speed_curve, bg="#34495e", padx=10, pady=2).pack(side=tk.RIGHT)

        self.speed_canvas = tk.Canvas(
            left_frame, width=self.speed_canvas_width, height=self.speed_canvas_height,
            bg="#121212", highlightthickness=1, highlightbackground="#333333",
        )
        self.speed_canvas.pack(padx=10, pady=(0, 5))
        self.speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "pred"))
        self.speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("pred", None))
        self.speed_canvas.bind("<Button-1>", self._on_speed_canvas_left_down)
        self.speed_canvas.bind("<B1-Motion>", self._on_speed_canvas_left_drag)
        self.speed_canvas.bind("<ButtonRelease-1>", self._on_speed_canvas_left_release)

        self.pred_speed_action_frame = tk.Frame(left_frame, bg=BG_PANEL)
        FlatButton(self.pred_speed_action_frame, "接受", self._save_speed_edit, bg="#4CAF50", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.pred_speed_action_frame, "取消", lambda: self._cancel_speed_edit(redraw=True), bg="#F44336", padx=16).pack(side=tk.LEFT)

        gt_speed_header_frame = tk.Frame(left_frame, bg=BG_PANEL)
        gt_speed_header_frame.pack(fill=tk.X, padx=10, pady=(10, 2))
        tk.Label(gt_speed_header_frame, text="GT Speed Profile", font=("Segoe UI", 11, "bold"), fg=FG_PRIMARY, bg=BG_PANEL).pack(side=tk.LEFT)
        FlatButton(gt_speed_header_frame, "优化速度曲线", self._optimize_gt_speed_curve, bg="#5d6d7e", padx=10, pady=2).pack(side=tk.RIGHT)
        FlatButton(gt_speed_header_frame, "停车添加", self._start_gt_stop_add, bg="#D32F2F", padx=10, pady=2).pack(side=tk.RIGHT, padx=(0, 5))

        self.gt_speed_canvas = tk.Canvas(
            left_frame, width=self.speed_canvas_width, height=self.speed_canvas_height,
            bg="#121212", highlightthickness=1, highlightbackground="#333333",
        )
        self.gt_speed_canvas.pack(padx=10, pady=(0, 5))
        self.gt_speed_canvas.bind("<Motion>", lambda event: self._on_speed_canvas_motion(event, "gt"))
        self.gt_speed_canvas.bind("<Leave>", lambda _event: self._set_speed_hover_frame("gt", None))
        self.gt_speed_canvas.bind("<Button-1>", self._on_gt_speed_canvas_click)

        self.gt_speed_action_frame = tk.Frame(left_frame, bg=BG_PANEL)
        FlatButton(self.gt_speed_action_frame, "接受", self._save_gt_speed_edit, bg="#4CAF50", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.gt_speed_action_frame, "取消", lambda: self._cancel_gt_speed_edit(redraw=True), bg="#F44336", padx=16).pack(side=tk.LEFT)

        self.gt_stop_action_frame = tk.Frame(left_frame, bg=BG_PANEL)
        FlatButton(self.gt_stop_action_frame, "保存", self._save_gt_speed_edit, bg="#4CAF50", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.gt_stop_action_frame, "取消", lambda: self._cancel_gt_speed_edit(redraw=True), bg="#F44336", padx=16).pack(side=tk.LEFT, padx=(0, 5))
        FlatButton(self.gt_stop_action_frame, "撤回", self._undo_gt_stop_add, bg="#FF9800", padx=16).pack(side=tk.LEFT)

        # 2. Middle panel - Camera images with projection
        middle_frame = tk.Frame(content_frame, bg=BG_PANEL)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

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

        def create_camera_panel(parent, cam, side=tk.TOP, expand=False):
            cam_frame = tk.Frame(parent, bg=BG_MAIN, highlightthickness=1, highlightbackground="#333333")
            cam_frame.pack(side=side, expand=expand, fill=tk.BOTH, padx=4, pady=4)
            tk.Label(cam_frame, text=cam, font=("Segoe UI", 10, "bold"), fg=FG_PRIMARY, bg=BG_HEADER).pack(fill=tk.X)
            self.camera_labels[cam] = tk.Label(cam_frame, bg=BG_MAIN)
            self.camera_labels[cam].pack(expand=True, fill=tk.BOTH)
            self.camera_labels[cam].bind("<Button-1>", lambda event, camera=cam: self._on_camera_click(event, camera))
            self.camera_labels[cam].bind("<Button-3>", lambda event, camera=cam: self._on_camera_right_down(event, camera))
            self.camera_labels[cam].bind("<B3-Motion>", lambda event, camera=cam: self._on_camera_right_drag(event, camera))
            self.camera_labels[cam].bind("<ButtonRelease-3>", self._on_right_release)

        top_camera_row = tk.Frame(middle_frame, bg=BG_PANEL)
        top_camera_row.pack(fill=tk.X, padx=6)
        main_camera_area = tk.Frame(middle_frame, bg=BG_PANEL)
        main_camera_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        top_cameras = [cam for cam in ("RL", "RR") if cam in self.cameras]
        remaining_top_cameras = [cam for cam in self.cameras if cam not in top_cameras and cam != "FC"]
        top_cameras.extend(remaining_top_cameras)

        for cam in top_cameras:
            create_camera_panel(top_camera_row, cam, side=tk.LEFT, expand=True)

        if "FC" in self.cameras:
            create_camera_panel(main_camera_area, "FC", side=tk.TOP, expand=True)

        # 3. Right panel - Trajectory list, CoT, Clusters using PanedWindow
        right_frame = tk.Frame(
            content_frame,
            bg=BG_PANEL,
            width=getattr(self, "right_panel_width", 460),
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_frame.pack_propagate(False)

        style.configure("Dark.TLabelframe", background=BG_PANEL, foreground=FG_PRIMARY)
        style.configure("Dark.TLabelframe.Label", background=BG_PANEL, foreground=FG_PRIMARY, font=("Segoe UI", 10, "bold"))

        # We use a TPanedwindow to allow resizing sections
        paned_window = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        # Pane 1: Trajectories
        traj_pane = ttk.LabelFrame(paned_window, text="Trajectories", style="Dark.TLabelframe")
        paned_window.add(traj_pane, weight=3)

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
        cot_pane = ttk.LabelFrame(paned_window, text="Chain of Thought", style="Dark.TLabelframe")
        paned_window.add(cot_pane, weight=2)

        self.cot_text = tk.Text(
            cot_pane, height=8, width=42, wrap=tk.WORD,
            font=("Segoe UI", 9), bg="#121212", fg=FG_SECONDARY,
            insertbackground="white", relief=tk.FLAT, highlightthickness=1, highlightbackground="#333333"
        )
        self.cot_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.cot_text.configure(state=tk.DISABLED)

        # Pane 3: Cluster Centers
        cluster_pane = ttk.LabelFrame(paned_window, text="Cluster Centers", style="Dark.TLabelframe")
        paned_window.add(cluster_pane, weight=0)

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

        # --- Bottom Control Panels ---
        controls_outer_frame = tk.Frame(main_frame, bg=BG_MAIN)
        controls_outer_frame.pack(fill=tk.X, pady=(10, 0))

        control_font = ("Segoe UI", 10, "bold")
        btn_padx, btn_pady = 12, 6

        # Group 1: Drawing & Bezier Tools
        draw_group = ttk.LabelFrame(controls_outer_frame, text="Drawing & Bezier Tools", style="Dark.TLabelframe")
        draw_group.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
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
        edit_group.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        edit_inner = tk.Frame(edit_group, bg=BG_PANEL)
        edit_inner.pack(padx=8, pady=8)

        FlatButton(edit_inner, "Edit Traj", self._start_saved_trajectory_edit, bg="#2196F3", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(edit_inner, "Save Edit", self._save_saved_trajectory_edit, bg="#4CAF50", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(edit_inner, "Cancel Edit", self._cancel_saved_trajectory_edit, bg="#9E9E9E", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)
        FlatButton(edit_inner, "Restore Edit", self._restore_saved_trajectory_edit, bg="#607D8B", font=control_font, padx=btn_padx, pady=btn_pady).pack(side=tk.LEFT, padx=4)

        # Group 3: GT & Global Actions
        action_group = ttk.LabelFrame(controls_outer_frame, text="GT & Global Actions", style="Dark.TLabelframe")
        action_group.pack(side=tk.LEFT, fill=tk.Y)
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

        # --- Status & Help ---
        footer_frame = tk.Frame(main_frame, bg=BG_MAIN)
        footer_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(footer_frame, text="", font=("Segoe UI", 10), fg=FG_SECONDARY, bg=BG_MAIN)
        self.status_label.pack(side=tk.LEFT)

        tk.Label(
            footer_frame,
            text="Shortcuts: ←/→ Samples | ↑/↓ Trajectories | +/- Cluster | Ctrl+S Save | Q Quit | Del Delete | Tab Camera",
            font=("Segoe UI", 9), fg="#757575", bg=BG_MAIN,
        ).pack(side=tk.RIGHT)

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
