"""Main viewer class for the enhanced trajectory GUI."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Optional


from .tk_compat import tk

from .environment import setup_environment
setup_environment()

from traj_core.data_loader import get_dataset_names

from traj_core.constants import *
from .mixins.cluster_controls import ClusterControlsMixin
from .mixins.delete_controls import DeleteControlsMixin
from .mixins.draw_bev import DrawBevMixin
from .mixins.draw_camera import DrawCameraMixin
from .mixins.draw_speed import DrawSpeedMixin
from .mixins.gt_controls import GTControlsMixin
from .mixins.manual_editing import ManualEditingMixin
from .mixins.manual_events import ManualEventsMixin
from .mixins.navigation import NavigationMixin
from .mixins.sample_io import SampleIOMixin
from .mixins.saved_traj_editing import SavedTrajectoryEditingMixin
from .mixins.speed_controls import SpeedControlsMixin
from .mixins.widget_layout import WidgetLayoutMixin
from .responsive_layout import ResponsiveLayout, compute_responsive_layout
from .scene_labels import build_scene_label_index


class TrajectoryViewerEnhanced(
    SampleIOMixin,
    WidgetLayoutMixin,
    DeleteControlsMixin,
    NavigationMixin,
    ManualEditingMixin,
    SavedTrajectoryEditingMixin,
    ManualEventsMixin,
    ClusterControlsMixin,
    GTControlsMixin,
    SpeedControlsMixin,
    DrawBevMixin,
    DrawCameraMixin,
    DrawSpeedMixin,
):
    """Enhanced GUI for viewing VLM-generated trajectories with camera projections."""

    def __init__(
        self,
        data_root: str,
        output_dir: str,
        calibration_dir: str,
        cameras: list[str] = None,
        start_index: Optional[int] = None,
        start_dataset: str = "",
        start_clip: str = "",
        start_t0: Optional[int] = None,
        restore_last: bool = True,
        gt_only: bool = False,
        gt_stride_frames: int = 3,
        index_mode: str = "video_frames",
        frame_stride: int = 1,
    ):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.calibration_dir = Path(calibration_dir)
        self._closing = False
        self.gt_only = bool(gt_only)
        self.gt_stride_frames = max(1, int(gt_stride_frames))
        self.index_mode = str(index_mode or "generated")
        if self.index_mode not in {"generated", "video_frames", "merged"}:
            self.index_mode = "generated"
        self.frame_stride = max(1, int(frame_stride))
        self.video_t0_count_by_clip = {}
        self.generated_t0_count_by_clip = {}
        self.generated_t0_by_clip = {}
        self.viewer_state_file = self.output_dir / ".trajectory_gui_state.json"
        # Default: show RL, FC, RR
        self.cameras = cameras or ["RL", "FC", "RR"]
        self.current_cam_for_projection = "FC"
        self.bev_canvas_width = 560
        self.bev_canvas_height = 700
        self.speed_canvas_width = self.bev_canvas_width
        self.speed_canvas_height = 180
        self.bev_forward_scale = 6.2
        self.bev_lateral_scale = 10.0
        self.bev_origin = (self.bev_canvas_width / 2, self.bev_canvas_height - 65)
        self.responsive_layout = None
        self.camera_fc_display_height = 720
        self.camera_aux_display_height = 300
        self.camera_fc_display_height_many = 600
        self.camera_aux_display_height_many = 240
        self.right_panel_width = 460
        self.trajectory_listbox_width = 56
        self.dataset_combo_width = 30
        self.clip_combo_width = 24
        self.t0_combo_width = 22
        self.scene_combo_width = 18
        self.middle_panel_width = 800
        self.stop_marker_hitboxes = []
        self.stop_tooltip_items = []
        self.speed_hover_frame_idx = None
        self.speed_hover_source = None
        self.speed_plot_rect = None
        self.gt_speed_plot_rect = None
        self.speed_edit_active = False
        self.speed_edit_dirty = False
        self.speed_edit_traj_idx = None
        self.speed_edit_original_traj = None
        self.speed_edit_original_xyz = None
        self.speed_edit_speed = None
        self.speed_edit_last_frame = None
        self.speed_edit_undo_stack = []
        self.speed_edit_redo_stack = []
        self.speed_edit_drag_snapshot = None
        self.traj_geom_edit_active = False
        self.traj_geom_edit_dirty = False
        self.traj_geom_edit_traj_idx = None
        self.traj_geom_edit_original_traj = None
        self.traj_geom_edit_original_xyz = None
        self.traj_geom_edit_undo_stack = []
        self.traj_geom_edit_redo_stack = []
        self.pred_speed_action_frame = None
        self.gt_speed_action_frame = None
        self.gt_stop_action_frame = None
        self.gt_edit_active = False
        self.gt_edit_mode = None
        self.gt_edit_original_xyz = None
        self.gt_edit_preview_xyz = None
        self.gt_stop_frame_idx = None
        self.trajectory_smoothness = {}
        self.traj_list_tooltip = None
        self.pending_deleted_traj_keys = set()
        self.pending_delete_stack = []
        self.traj_listbox_to_traj_idx = []

        # Load dataset info
        self.datasets = get_dataset_names(str(self.data_root))
        self.samples = self._load_sample_index(
            start_dataset=start_dataset,
            start_clip=start_clip,
        )

        self.samples.sort(key=lambda x: (x[0], x[1], x[2]))

        if not self.samples:
            if self.gt_only:
                raise ValueError(f"No GT samples found in {self.data_root}")
            if self.index_mode in {"video_frames", "merged"}:
                raise ValueError(f"No video-frame samples found in {self.data_root}")
            raise ValueError(f"No trajectory files found in {self.output_dir}")

        self.current_idx = self._resolve_start_index(
            start_index=start_index,
            start_dataset=start_dataset,
            start_clip=start_clip,
            start_t0=start_t0,
            restore_last=restore_last,
        )
        self.scene_label_by_sample, self.scenes_by_dataset = build_scene_label_index(
            self.data_root,
            self.samples,
        )
        self.scene_filter_value = "None"
        self.trajectories = []
        self.trajectory_states = {}
        self.current_traj_idx = 0
        self.image_tk = {cam: None for cam in self.cameras}
        self.cot_index = self._load_cot_index()
        self.manual_points_file = self.output_dir / "manual_points.json"
        self.manual_line_points_index = self._load_manual_line_points_index()
        self.manual_camera_line_points_index = self._load_manual_camera_line_points_index()
        self.manual_stop_points_index = self._load_manual_stop_points_index()
        self.manual_line_points = []
        self.manual_camera_line_points = []
        self.manual_stop_points = []
        self.manual_point_actions = []
        self.manual_line_points_dirty = False
        self.manual_camera_line_points_dirty = False
        self.manual_stop_points_dirty = False
        self.bezier_cluster_center_ids = self._load_bezier_cluster_center_ids()
        self.cluster_center_library = self._load_cluster_center_library()
        self.cluster_preview_traj = None
        self.cluster_preview_record = None
        self.cluster_preview_is_edited = False
        self.cluster_category_var = None
        self.cluster_choice_var = None
        self.cluster_category_combo = None
        self.cluster_choice_combo = None
        self.camera_display_meta = {}
        self.draw_line_enabled = False
        self.draw_line_var = None
        self.stop_duration_seconds = 2.0
        self.stop_duration_var = None
        self.drag_state = None
        self.visual_data_cache = OrderedDict()
        self.visual_data_cache_limit = 8
        self.object_data_cache = OrderedDict()
        self.object_data_cache_limit = 12
        self.current_objects = None
        self.show_objects_enabled = True
        self.show_objects_var = None
        self.camera_base_images = {}
        self.gt_future_mode = "raw"
        self.samples_by_dataset = {}
        self.clips_by_dataset = {}
        self.t0_by_dataset_clip = {}
        for sample_dataset, sample_clip, sample_t0 in self.samples:
            self.samples_by_dataset.setdefault(sample_dataset, []).append(
                (sample_clip, int(sample_t0))
            )
            self.clips_by_dataset.setdefault(sample_dataset, [])
            if sample_clip not in self.clips_by_dataset[sample_dataset]:
                self.clips_by_dataset[sample_dataset].append(sample_clip)
            self.t0_by_dataset_clip.setdefault((sample_dataset, sample_clip), []).append(int(sample_t0))
        for clips in self.clips_by_dataset.values():
            clips.sort()
        for t0_values in self.t0_by_dataset_clip.values():
            t0_values.sort()
        self.dataset_var = None
        self.clip_var = None
        self.t0_var = None
        self.scene_filter_var = None
        self.dataset_combo = None
        self.clip_combo = None
        self.t0_combo = None
        self.scene_filter_combo = None

        # Current camera frame for projection (default to FC)
        self.current_cam_for_projection = "FC"

        # Load initial sample
        self._load_sample(self.current_idx)

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Trajectory Annotator")
        self._configure_responsive_window()
        self.root.configure(bg="#2b2b2b")

        self._create_widgets()
        self._update_display()
        self.root.protocol("WM_DELETE_WINDOW", self._shutdown_gui)

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_sample())
        self.root.bind("<Right>", lambda e: self._next_sample())
        self.root.bind("<Up>", self._on_global_prev_traj_key)
        self.root.bind("<Down>", self._on_global_next_traj_key)
        self.root.bind("<Delete>", lambda e: self._delete_traj())
        self.root.bind("<BackSpace>", lambda e: self._delete_traj())
        self.root.bind("<Control-s>", lambda e: self._save_results())
        self.root.bind("<Control-z>", self._on_undo_saved_trajectory_edit_key)
        self.root.bind("<Control-Z>", self._on_undo_saved_trajectory_edit_key)
        self.root.bind("<Control-y>", self._on_redo_saved_trajectory_edit_key)
        self.root.bind("<Control-Y>", self._on_redo_saved_trajectory_edit_key)
        self.root.bind("<q>", lambda e: self._shutdown_gui())
        self.root.bind("<Tab>", lambda e: self._toggle_projection_camera())
        self.root.bind("<KeyPress-minus>", lambda e: self._cycle_selected_cluster_center(-1))
        self.root.bind("<KeyPress-underscore>", lambda e: self._cycle_selected_cluster_center(-1))
        self.root.bind("<KeyPress-plus>", lambda e: self._cycle_selected_cluster_center(1))
        self.root.bind("<KeyPress-equal>", lambda e: self._cycle_selected_cluster_center(1))

        try:
            self.root.mainloop()
        finally:
            if not getattr(self, "_closing", False):
                self._shutdown_gui()

    def _shutdown_gui(self):
        """Release Tk state cleanly before the GUI process exits."""
        if getattr(self, "_closing", False):
            return "break"
        self._closing = True

        self.drag_state = None
        self.draw_line_enabled = False

        root = getattr(self, "root", None)
        if root is not None:
            after_id = getattr(self, "_resize_after_id", None)
            if after_id is not None:
                try:
                    root.after_cancel(after_id)
                except Exception:
                    pass
                self._resize_after_id = None

            try:
                grab_widget = root.grab_current()
            except Exception:
                grab_widget = None
            if grab_widget is not None:
                try:
                    grab_widget.grab_release()
                except Exception:
                    pass

        for cleanup_name in ("_hide_traj_list_tooltip", "_hide_stop_tooltip"):
            cleanup = getattr(self, cleanup_name, None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass

        if root is not None:
            try:
                root.quit()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass
        return "break"

    def _configure_responsive_window(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        layout = compute_responsive_layout(screen_width, screen_height)
        self._apply_responsive_layout(layout)
        self.root.geometry(layout.geometry)
        self.root.minsize(layout.min_width, layout.min_height)
        if layout.start_maximized:
            self.root.after_idle(self._try_maximize_root)

    def _apply_responsive_layout(self, layout: ResponsiveLayout) -> None:
        self.responsive_layout = layout
        self.bev_canvas_width = layout.bev_canvas_width
        self.bev_canvas_height = layout.bev_canvas_height
        self.speed_canvas_width = layout.speed_canvas_width
        self.speed_canvas_height = layout.speed_canvas_height
        self.bev_forward_scale = layout.bev_forward_scale
        self.bev_lateral_scale = layout.bev_lateral_scale
        self.bev_origin = (
            self.bev_canvas_width / 2,
            self.bev_canvas_height - layout.bev_origin_bottom_margin,
        )
        self.camera_fc_display_height = layout.camera_fc_height
        self.camera_aux_display_height = layout.camera_aux_height
        self.camera_fc_display_height_many = layout.camera_fc_height_many
        self.camera_aux_display_height_many = layout.camera_aux_height_many
        self.right_panel_width = layout.right_panel_width
        self.trajectory_listbox_width = layout.trajectory_listbox_width
        self.dataset_combo_width = layout.dataset_combo_width
        self.clip_combo_width = layout.clip_combo_width
        self.t0_combo_width = layout.t0_combo_width
        self.scene_combo_width = layout.scene_combo_width
        self.middle_panel_width = layout.middle_panel_width

    def _try_maximize_root(self) -> None:
        if getattr(self, "_closing", False):
            return
        maximized = False
        try:
            self.root.state("zoomed")
            maximized = True
        except tk.TclError:
            pass
        if not maximized:
            try:
                self.root.attributes("-zoomed", True)
                maximized = True
            except tk.TclError:
                pass
        # After the window manager applies the maximized state, force a
        # full redraw so all canvases and camera panels use the real screen size.
        self.root.after(300, self._deferred_fullscreen_refresh)

    def _deferred_fullscreen_refresh(self) -> None:
        """Called after the window manager applies the maximized state.
        We synthetically fire the resize pipeline instead of calling _update_display
        directly, so the debounced handler updates widget dimensions first.
        """
        if getattr(self, "_closing", False):
            return
        try:
            self.root.update_idletasks()
            # Trigger the debounced resize pipeline by scheduling the deferred
            # handler directly (the root Configure event may not re-fire after
            # after_idle in some WMs).
            if hasattr(self, "_on_window_resize_deferred"):
                if hasattr(self, '_resize_after_id') and self._resize_after_id is not None:
                    self.root.after_cancel(self._resize_after_id)
                self._resize_after_id = self.root.after(100, self._on_window_resize_deferred)
        except Exception:
            pass


__all__ = ["TrajectoryViewerEnhanced"]
