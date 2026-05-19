import math
import os
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from traj_core.frame_index import (
    build_video_frame_t0_candidates,
    load_master_video_timestamps,
    valid_video_frame_indices,
)
from traj_core.data_loader import filter_t0s_with_full_future
from traj_core.object_loader import (
    OBJECT_COLUMNS as GUI_OBJECT_COLUMNS,
    object_center_xy,
    load_objects_for_clip,
    nearest_objects_at_timestamp,
    object_footprint_xy,
)
from traj_core.calibration_loader import load_calibration_for_segment
from traj_core.constants import HISTORY_SPEED_COLOR_HEX, SPEED_EDIT_LOCAL_RADIUS_FRAMES
from traj_core.dynamics import (
    DynamicsLimits,
    deform_trajectory_by_keyframe_drag,
    diagnose_trajectory_dynamics,
    editable_trajectory_keyframes,
    optimize_pseudo_gt_trajectory,
    trajectory_components_from_xyz,
)
from traj_annotation.environment import setup_environment
from traj_core.math_utils import _rgb_to_hex, _trajectory_base_color, _resample_curve_by_distance
from traj_annotation.mixins.cluster_controls import ClusterControlsMixin
from traj_annotation.mixins.delete_controls import DeleteControlsMixin
from traj_annotation.mixins.draw_bev import DrawBevMixin
from traj_annotation.mixins.draw_camera import DrawCameraMixin
from traj_annotation.mixins.navigation import NavigationMixin
from traj_annotation.mixins.sample_io import SampleIOMixin
from traj_annotation.mixins.saved_traj_editing import SavedTrajectoryEditingMixin
from traj_annotation.responsive_layout import compute_responsive_layout
from traj_annotation.viewer import TrajectoryViewerEnhanced
from traj_annotation.save_audit import (
    apply_gui_edit_metadata,
    restore_file_from_backup_with_audit,
    write_text_file_with_audit,
    write_parquet_with_audit,
)
from traj_annotation.mixins.speed_controls import SpeedControlsMixin
from traj_core.speed_utils import (
    TRAJ_ACCEL_MAX_MPS2,
    TRAJ_ACCEL_MIN_MPS2,
    TRAJ_DT_SECONDS,
    _detect_stop_segments,
    _enforce_speed_acceleration_limits,
    _history_speed_profile_from_xyz,
    _resample_xyz_by_speed_profile,
    _smooth_history_xyz_for_display,
    _smooth_speed_profile,
    _speed_profile_from_trajectory,
    _smoothed_history_speed_profile_from_xyz,
    _smoothed_gt_speed_profile_from_xyz,
    _speed_smoothness_diagnostics,
)
from traj_core.trajectory_identity import (
    drop_trajectory_rows_by_keys,
    is_deletable_trajectory_record,
    is_gt_trajectory_record,
    normalize_trajectory_source,
    trajectory_key_from_record,
)


class GuiHelperTests(unittest.TestCase):
    def _write_egomotion_file(self, path: Path, timestamps: list[int]) -> None:
        ts = np.asarray(timestamps, dtype=np.int64)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "timestamp": ts,
                "x": ts.astype(np.float64) / 100_000.0,
                "y": np.zeros(len(ts), dtype=np.float64),
                "z": np.zeros(len(ts), dtype=np.float64),
                "qx": np.zeros(len(ts), dtype=np.float64),
                "qy": np.zeros(len(ts), dtype=np.float64),
                "qz": np.zeros(len(ts), dtype=np.float64),
                "qw": np.ones(len(ts), dtype=np.float64),
                "vx": np.ones(len(ts), dtype=np.float64),
                "vy": np.zeros(len(ts), dtype=np.float64),
                "vz": np.zeros(len(ts), dtype=np.float64),
            }
        ).to_parquet(path)

    def _write_egomotion_xyz_file(
        self,
        path: Path,
        timestamps: list[int],
        x_values: list[float],
    ) -> None:
        ts = np.asarray(timestamps, dtype=np.int64)
        x = np.asarray(x_values, dtype=np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "timestamp": ts,
                "x": x,
                "y": np.zeros(len(ts), dtype=np.float64),
                "z": np.zeros(len(ts), dtype=np.float64),
                "qx": np.zeros(len(ts), dtype=np.float64),
                "qy": np.zeros(len(ts), dtype=np.float64),
                "qz": np.zeros(len(ts), dtype=np.float64),
                "qw": np.ones(len(ts), dtype=np.float64),
                "vx": np.ones(len(ts), dtype=np.float64),
                "vy": np.zeros(len(ts), dtype=np.float64),
                "vz": np.zeros(len(ts), dtype=np.float64),
            }
        ).to_parquet(path)

    def test_local_dataset_xml_calibration_takes_priority_over_global_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "train_data"
            local_fc = data_root / "demo_converted" / "calibration" / "fc120"
            local_fc.mkdir(parents=True)
            (local_fc / "cameraIntrinsic.xml").write_text(
                """<?xml version="1.0"?>
<opencv_storage>
<camIntrinsicMat type_id="opencv-matrix">
  <rows>3</rows><cols>3</cols><dt>d</dt>
  <data>2000 0 1900 0 2200 1000 0 0 1</data>
</camIntrinsicMat>
<distortion_coefficients type_id="opencv-matrix">
  <rows>8</rows><cols>1</cols><dt>d</dt>
  <data>1 2 3 4 5 6 7 8</data>
</distortion_coefficients>
</opencv_storage>
""",
                encoding="utf-8",
            )
            (local_fc / "cameraExtrinsic.xml").write_text(
                """<?xml version="1.0"?>
<opencv_storage>
<extrinsicMatrix type_id="opencv-matrix">
  <rows>4</rows><cols>4</cols><dt>d</dt>
  <data>1 0 0 1000 0 1 0 2000 0 0 1 3000 0 0 0 1</data>
</extrinsicMatrix>
</opencv_storage>
""",
                encoding="utf-8",
            )

            global_dir = root / "global_calibration"
            global_dir.mkdir()
            (global_dir / "manifest.json").write_text(
                '{"camera_models": {"FC": {"fx": 10, "fy": 10, "cx": 10, "cy": 10}}}',
                encoding="utf-8",
            )
            (global_dir / "demo_roll_only_3d_raw_distorted_extrinsics.jsonl").write_text(
                (
                    '{"dataset":"demo","segment":"clip","cameras":{'
                    '"FC":{"T_bev_m_to_camera_m_3x4":[1,0,0,9,0,1,0,9,0,0,1,9]}}}\n'
                ),
                encoding="utf-8",
            )

            calibs = load_calibration_for_segment(
                str(global_dir),
                "demo_converted",
                "clip",
                data_root=str(data_root),
            )

        self.assertIn("FC", calibs)
        np.testing.assert_allclose(calibs["FC"].T_bev_to_camera[:, 3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(calibs["FC"].fx, 1000.0)
        self.assertAlmostEqual(calibs["FC"].fy, 1100.0)
        self.assertEqual(calibs["FC"].image_width, 1920)
        self.assertEqual(calibs["FC"].image_height, 1080)

    def test_global_jsonl_calibration_remains_available_without_local_data_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            calib_dir = Path(tmpdir)
            (calib_dir / "manifest.json").write_text(
                '{"camera_models": {"FC": {"fx": 20, "fy": 30, "cx": 40, "cy": 50, '
                '"raw_image_width": 1920, "raw_image_height": 1080}}}',
                encoding="utf-8",
            )
            (calib_dir / "demo_roll_only_3d_raw_distorted_extrinsics.jsonl").write_text(
                (
                    '{"dataset":"demo","segment":"clip","cameras":{'
                    '"FC":{"T_bev_m_to_camera_m_3x4":[1,0,0,4,0,1,0,5,0,0,1,6]}}}\n'
                ),
                encoding="utf-8",
            )

            calibs = load_calibration_for_segment(str(calib_dir), "demo", "clip")

        np.testing.assert_allclose(calibs["FC"].T_bev_to_camera[:, 3], [4.0, 5.0, 6.0])
        self.assertEqual(calibs["FC"].fx, 20)

    def test_annotation_entrypoint_is_nonempty_wrapper(self):
        entrypoint = Path(__file__).resolve().parents[1] / "trajectory_annotator.py"

        text = entrypoint.read_text(encoding="utf-8")

        self.assertIn("traj_annotation.cli", text)
        self.assertIn("TrajectoryViewerEnhanced", text)
        self.assertGreater(len(text.splitlines()), 10)

    def test_setup_environment_makes_tkinter_importable(self):
        setup_environment()
        import tkinter

        self.assertGreaterEqual(float(tkinter.TkVersion), 8.6)

    def test_gui_cli_defaults_are_cross_platform(self):
        from traj_annotation import cli as gui_cli

        with mock.patch.dict(os.environ, {}, clear=True):
            defaults = [
                gui_cli.default_data_root(),
                gui_cli.default_output_dir(),
                gui_cli.default_calibration_dir(),
            ]

        self.assertEqual(defaults, ["train_data", "output", "calibration"])
        self.assertFalse(any(value.startswith("/home/") for value in defaults))

    def test_gui_cli_defaults_use_environment_variables(self):
        from traj_annotation import cli as gui_cli

        with mock.patch.dict(
            os.environ,
            {
                "GENERATE_TRAJ_GUI_DATA_ROOT": r"D:\traj\train_data",
                "GENERATE_TRAJ_GUI_OUTPUT_DIR": r"D:\traj\output",
                "GENERATE_TRAJ_GUI_CALIBRATION_DIR": r"D:\traj\calibration",
            },
            clear=True,
        ):
            self.assertEqual(gui_cli.default_data_root(), r"D:\traj\train_data")
            self.assertEqual(gui_cli.default_output_dir(), r"D:\traj\output")
            self.assertEqual(gui_cli.default_calibration_dir(), r"D:\traj\calibration")

    def test_load_objects_for_clip_returns_empty_schema_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = load_objects_for_clip(tmpdir, "demo_converted", "missing_clip")

        self.assertEqual(list(df.columns), GUI_OBJECT_COLUMNS)
        self.assertEqual(len(df), 0)

    def test_nearest_objects_at_timestamp_filters_by_time_window(self):
        df = pd.DataFrame(
            {
                "timestamp": [1_000_000, 1_100_000, 1_100_000, 1_400_000],
                "frame_index": [0, 1, 1, 4],
                "source": ["bev_object"] * 4,
                "object_id": [1, 2, 3, 4],
                "object_type": [1, 1, 5, 1],
                "confidence": [0.9, 0.8, 0.7, 0.6],
                "x": [10.0, 20.0, 21.0, 40.0],
                "y": [0.0, 1.0, -1.0, 3.0],
                "z": [0.0, 0.0, 0.0, 0.0],
                "x_kf": [10.0, 20.0, 21.0, 40.0],
                "y_kf": [0.0, 1.0, -1.0, 3.0],
                "z_kf": [0.0, 0.0, 0.0, 0.0],
                "heading": [0.0, 0.0, 0.0, 0.0],
                "origin_heading": [0.0, 0.0, 0.0, 0.0],
                "length": [4.0, 4.0, 1.0, 4.0],
                "width": [2.0, 2.0, 0.6, 2.0],
                "height": [1.5, 1.5, 1.7, 1.5],
                "vx_rel": [0.0, 0.0, 0.0, 0.0],
                "vy_rel": [0.0, 0.0, 0.0, 0.0],
                "vz_rel": [0.0, 0.0, 0.0, 0.0],
                "vx_abs": [0.0, 0.0, 0.0, 0.0],
                "vy_abs": [0.0, 0.0, 0.0, 0.0],
                "vz_abs": [0.0, 0.0, 0.0, 0.0],
                "life_time": [1, 1, 1, 1],
                "lost_time": [0, 0, 0, 0],
                "object_timestamp": [1_000_000, 1_100_000, 1_100_000, 1_400_000],
            }
        )

        near = nearest_objects_at_timestamp(df, 1_120_000, max_delta_us=50_000)
        far = nearest_objects_at_timestamp(df, 1_250_000, max_delta_us=50_000)

        self.assertEqual(near["object_id"].tolist(), [2, 3])
        self.assertEqual(len(far), 0)

    def test_object_footprint_xy_uses_center_heading_and_dimensions(self):
        row = {
            "x": 10.0,
            "y": 1.0,
            "x_kf": 10.0,
            "y_kf": 1.0,
            "heading": 0.0,
            "length": 4.0,
            "width": 2.0,
        }

        footprint = object_footprint_xy(row)

        np.testing.assert_allclose(
            footprint,
            np.array([
                [12.0, 2.0],
                [12.0, 0.0],
                [8.0, 0.0],
                [8.0, 2.0],
            ]),
        )

    def test_object_center_xy_converts_csd_axes_to_gui_ego_axes(self):
        row = {
            "x": 2.0,
            "y": -12.0,
            "x_kf": 3.0,
            "y_kf": -14.0,
        }

        center = object_center_xy(row)

        self.assertEqual(center, (14.0, -3.0))

    def test_traffic_participants_draw_position_markers_without_direction_cues(self):
        class Canvas:
            def __init__(self):
                self.calls = []

            def create_oval(self, *args, **kwargs):
                self.calls.append(("oval", args, kwargs))

            def create_line(self, *args, **kwargs):
                self.calls.append(("line", args, kwargs))

            def create_polygon(self, *args, **kwargs):
                self.calls.append(("polygon", args, kwargs))

            def create_text(self, *args, **kwargs):
                self.calls.append(("text", args, kwargs))

        class Viewer(DrawBevMixin):
            def __init__(self):
                self.traj_canvas = Canvas()
                self.bev_forward_scale = 10.0
                self.bev_lateral_scale = 10.0
                self.bev_origin = (100.0, 200.0)
                self.show_objects_enabled = True
                self.show_objects_var = None
                self.current_objects = pd.DataFrame(
                    {
                        "object_id": [7],
                        "object_type": [1],
                        "x": [2.0],
                        "y": [-10.0],
                        "x_kf": [3.0],
                        "y_kf": [-12.0],
                        "heading": [1.57],
                        "length": [4.5],
                        "width": [1.8],
                        "vx_rel": [3.0],
                        "vy_rel": [0.0],
                    }
                )

        viewer = Viewer()

        viewer._draw_traffic_participants()

        call_names = [name for name, _args, _kwargs in viewer.traj_canvas.calls]
        self.assertEqual(call_names, ["oval"])
        oval_args = viewer.traj_canvas.calls[0][1]
        self.assertEqual(oval_args, (126.0, 76.0, 134.0, 84.0))
        oval_kwargs = viewer.traj_canvas.calls[0][2]
        self.assertEqual(oval_kwargs["tags"], ("traffic_object",))

    def test_traffic_participants_project_to_fc_camera_when_enabled(self):
        class Calibration:
            image_width = 100
            image_height = 80

            def __init__(self):
                self.projected_points = None

            def project_bev_to_image(self, points):
                self.projected_points = np.asarray(points, dtype=np.float64)
                return (
                    np.asarray([20.0], dtype=np.float64),
                    np.asarray([30.0], dtype=np.float64),
                    np.asarray([5.0], dtype=np.float64),
                )

            def is_point_visible(self, u, v, z):
                return (u >= 0) & (v >= 0) & (z > 0)

        class Viewer(DrawCameraMixin):
            def __init__(self):
                self.show_objects_enabled = True
                self.show_objects_var = None
                self.current_objects = pd.DataFrame(
                    {
                        "object_id": [7],
                        "object_type": [1],
                        "x": [2.0],
                        "y": [-10.0],
                        "x_kf": [3.0],
                        "y_kf": [-12.0],
                    }
                )
                self.calibration = {"FC": Calibration()}

        viewer = Viewer()
        image = np.zeros((80, 100, 3), dtype=np.uint8)

        out = viewer._draw_traffic_participants_on_image(image, "FC")

        np.testing.assert_allclose(viewer.calibration["FC"].projected_points, [[3.0, 12.0, 0.0]])
        self.assertTrue(np.any(out[30, 20] > 0))
        self.assertTrue(np.array_equal(out[0, 0], [0, 0, 0]))

    def test_object_overlay_toggle_redraws_bev_and_camera_images(self):
        class ToggleVar:
            def get(self):
                return False

        class Viewer(DrawBevMixin):
            def __init__(self):
                self.show_objects_enabled = True
                self.show_objects_var = ToggleVar()
                self.bev_redraws = 0
                self.camera_redraws = 0

            def _draw_trajectories(self):
                self.bev_redraws += 1

            def _draw_camera_images(self):
                self.camera_redraws += 1

        viewer = Viewer()

        viewer._toggle_object_overlay()

        self.assertFalse(viewer.show_objects_enabled)
        self.assertEqual(viewer.bev_redraws, 1)
        self.assertEqual(viewer.camera_redraws, 1)

    def test_viewer_shutdown_releases_tk_state_and_destroys_root(self):
        class GrabWidget:
            def __init__(self):
                self.released = 0

            def grab_release(self):
                self.released += 1

        class Root:
            def __init__(self):
                self.cancelled = []
                self.quit_calls = 0
                self.destroy_calls = 0
                self.grab_widget = GrabWidget()

            def after_cancel(self, after_id):
                self.cancelled.append(after_id)

            def grab_current(self):
                return self.grab_widget

            def quit(self):
                self.quit_calls += 1

            def destroy(self):
                self.destroy_calls += 1

        class Tooltip:
            def __init__(self):
                self.destroy_calls = 0

            def destroy(self):
                self.destroy_calls += 1

        viewer = TrajectoryViewerEnhanced.__new__(TrajectoryViewerEnhanced)
        viewer.root = Root()
        viewer._resize_after_id = "resize-1"
        viewer.traj_list_tooltip = Tooltip()
        viewer.drag_state = {"type": "manual_control"}
        viewer.draw_line_enabled = True

        viewer._shutdown_gui()
        viewer._shutdown_gui()

        self.assertTrue(viewer._closing)
        self.assertIsNone(viewer._resize_after_id)
        self.assertEqual(viewer.root.cancelled, ["resize-1"])
        self.assertEqual(viewer.root.grab_widget.released, 1)
        self.assertEqual(viewer.root.quit_calls, 1)
        self.assertEqual(viewer.root.destroy_calls, 1)
        self.assertIsNone(viewer.traj_list_tooltip)
        self.assertIsNone(viewer.drag_state)
        self.assertFalse(viewer.draw_line_enabled)

    def test_window_resize_callback_ignores_events_after_shutdown_starts(self):
        class Root:
            def __init__(self):
                self.cancelled = []
                self.after_calls = []

            def after_cancel(self, after_id):
                self.cancelled.append(after_id)

            def after(self, delay_ms, callback):
                self.after_calls.append((delay_ms, callback))
                return "new-after-id"

        class Event:
            def __init__(self, widget):
                self.widget = widget

        viewer = TrajectoryViewerEnhanced.__new__(TrajectoryViewerEnhanced)
        viewer.root = Root()
        viewer._closing = True
        viewer._resize_after_id = "old-after-id"

        viewer._on_window_configure(Event(viewer.root))

        self.assertEqual(viewer.root.cancelled, [])
        self.assertEqual(viewer.root.after_calls, [])
        self.assertEqual(viewer._resize_after_id, "old-after-id")

    def test_responsive_layout_scales_down_for_small_windows_screens(self):
        layout = compute_responsive_layout(1366, 768)

        self.assertLessEqual(layout.window_width, 1366)
        self.assertLessEqual(layout.window_height, 768)
        self.assertLess(layout.bev_canvas_height, 700)
        self.assertLess(layout.speed_canvas_height, 180)
        self.assertLess(layout.camera_fc_height, 720)
        self.assertGreaterEqual(layout.bev_canvas_width, 400)
        self.assertTrue(layout.start_maximized)

    def test_responsive_layout_keeps_bottom_controls_visible_on_laptop_height(self):
        layout = compute_responsive_layout(1366, 768)

        visual_stack_height = layout.bev_canvas_height + 2 * layout.speed_canvas_height

        self.assertLessEqual(visual_stack_height, 480)
        self.assertLessEqual(layout.camera_fc_height + layout.camera_aux_height, 520)

    def test_responsive_layout_keeps_full_visual_size_on_large_screens(self):
        layout = compute_responsive_layout(2560, 1440)

        self.assertEqual(layout.bev_canvas_width, 560)
        self.assertEqual(layout.bev_canvas_height, 700)
        self.assertEqual(layout.speed_canvas_height, 180)
        self.assertEqual(layout.camera_fc_height, 720)
        self.assertEqual(layout.right_panel_width, 456)

    def test_responsive_layout_never_exceeds_tiny_screen_size(self):
        layout = compute_responsive_layout(800, 600)

        self.assertLessEqual(layout.window_width, 800)
        self.assertLessEqual(layout.window_height, 600)
        self.assertLessEqual(layout.min_width, layout.window_width)
        self.assertLessEqual(layout.min_height, layout.window_height)

    def test_setup_environment_on_windows_skips_linux_display_and_bundled_tk(self):
        from traj_annotation import environment
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tkinter":
                raise ModuleNotFoundError("tkinter")
            return original_import(name, *args, **kwargs)

        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with mock.patch.object(environment, "_configure_bundled_tk") as configure_bundled_tk:
                    environment.setup_environment(platform_name="win32")
                    display_value = os.environ.get("DISPLAY")
                    use_torch_value = os.environ.get("GENERATE_TRAJ_USE_TORCH")

        self.assertIsNone(display_value)
        self.assertEqual(use_torch_value, "0")
        configure_bundled_tk.assert_not_called()

    def test_data_loader_does_not_unconditionally_prepend_linux_alpamayo_src(self):
        self.assertNotIn("/home/tsingyu/lxh/alpamayo_1.5/src", sys.path[:5])

    def test_cluster_category_file_still_points_to_project_kmeans_dir(self):
        class Viewer(ClusterControlsMixin):
            pass

        path = Viewer()._cluster_category_file("stop")

        self.assertEqual(path.name, "stop.txt")
        self.assertEqual(path.parent.name, "k_means")
        self.assertEqual(path.parent.parent.name, "generate_traj_data")

    def test_cluster_preview_hide_clears_unsaved_preview_only(self):
        class Viewer(ClusterControlsMixin):
            def __init__(self):
                self.update_count = 0

            def _update_display(self):
                self.update_count += 1

        viewer = Viewer()
        viewer.cluster_preview_record = {"id": 3}
        viewer.cluster_preview_traj = np.ones((4, 3), dtype=np.float32)
        viewer.cluster_preview_is_edited = True
        viewer.trajectories = [{"source": "cluster_center", "x": np.array([1.0])}]

        viewer._hide_cluster_preview()

        self.assertIsNone(viewer.cluster_preview_record)
        self.assertIsNone(viewer.cluster_preview_traj)
        self.assertFalse(viewer.cluster_preview_is_edited)
        self.assertEqual(len(viewer.trajectories), 1)
        self.assertEqual(viewer.trajectories[0]["source"], "cluster_center")
        np.testing.assert_allclose(viewer.trajectories[0]["x"], [1.0])
        self.assertEqual(viewer.update_count, 1)

    def test_global_trajectory_arrow_keys_defer_to_trajectory_listbox(self):
        class Viewer(NavigationMixin):
            def __init__(self):
                self.prev_calls = 0
                self.next_calls = 0
                self.traj_listbox = object()

            def _prev_traj(self):
                self.prev_calls += 1

            def _next_traj(self):
                self.next_calls += 1

        class Event:
            def __init__(self, widget):
                self.widget = widget

        viewer = Viewer()

        self.assertIsNone(viewer._on_global_prev_traj_key(Event(viewer.traj_listbox)))
        self.assertIsNone(viewer._on_global_next_traj_key(Event(viewer.traj_listbox)))
        self.assertEqual(viewer.prev_calls, 0)
        self.assertEqual(viewer.next_calls, 0)

        self.assertEqual(viewer._on_global_prev_traj_key(Event(object())), "break")
        self.assertEqual(viewer._on_global_next_traj_key(Event(object())), "break")
        self.assertEqual(viewer.prev_calls, 1)
        self.assertEqual(viewer.next_calls, 1)

    def test_dropdown_arrow_keys_bind_to_trajectory_navigation(self):
        class Viewer(NavigationMixin):
            pass

        class Dropdown:
            def __init__(self):
                self.bindings = {}

            def bind(self, sequence, callback):
                self.bindings[sequence] = callback

        viewer = Viewer()
        dropdown = Dropdown()

        viewer._bind_arrow_keys_for_trajectory_navigation(dropdown)

        self.assertEqual(dropdown.bindings["<Up>"], viewer._on_global_prev_traj_key)
        self.assertEqual(dropdown.bindings["<Down>"], viewer._on_global_next_traj_key)

    def test_scene_filtered_sample_navigation_stays_in_dataset_and_scene(self):
        class Viewer(NavigationMixin):
            pass

        viewer = Viewer()
        viewer.samples = [
            ("dataset_a", "clip_1", 100),
            ("dataset_a", "clip_1", 200),
            ("dataset_a", "clip_2", 300),
            ("dataset_b", "clip_1", 400),
            ("dataset_a", "clip_3", 500),
        ]
        viewer.scene_filter_var = None
        viewer.scene_label_by_sample = {
            ("dataset_a", "clip_1", 100): "straight",
            ("dataset_a", "clip_1", 200): "turn",
            ("dataset_a", "clip_2", 300): "straight",
            ("dataset_b", "clip_1", 400): "straight",
            ("dataset_a", "clip_3", 500): "straight",
        }
        viewer.current_idx = 0

        self.assertEqual(viewer._scene_filtered_neighbor_index(1), 1)

        viewer.scene_filter_value = "straight"
        self.assertEqual(viewer._scene_filtered_neighbor_index(1), 2)

        viewer.current_idx = 2
        self.assertEqual(viewer._scene_filtered_neighbor_index(1), 4)
        self.assertEqual(viewer._scene_filtered_neighbor_index(-1), 0)

    def test_scene_filtered_sample_navigation_none_uses_normal_order(self):
        class Viewer(NavigationMixin):
            pass

        viewer = Viewer()
        viewer.samples = [
            ("dataset_a", "clip_1", 100),
            ("dataset_a", "clip_1", 200),
            ("dataset_b", "clip_1", 300),
        ]
        viewer.scene_filter_value = "None"
        viewer.scene_label_by_sample = {}
        viewer.current_idx = 1

        self.assertEqual(viewer._scene_filtered_neighbor_index(1), 2)
        self.assertEqual(viewer._scene_filtered_neighbor_index(-1), 0)

    def test_scene_filter_selection_jumps_to_first_matching_scene_in_current_dataset(self):
        class Var:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        class Viewer(NavigationMixin):
            def _load_sample(self, idx):
                self.current_idx = int(idx)
                self.loaded_idx = int(idx)

            def _update_display(self):
                self.updated = True

        viewer = Viewer()
        viewer.samples = [
            ("dataset_a", "clip_1", 100),
            ("dataset_a", "clip_1", 200),
            ("dataset_a", "clip_2", 300),
            ("dataset_b", "clip_1", 400),
            ("dataset_a", "clip_3", 500),
        ]
        viewer.scene_label_by_sample = {
            ("dataset_a", "clip_1", 100): "straight",
            ("dataset_a", "clip_1", 200): "turn",
            ("dataset_a", "clip_2", 300): "straight",
            ("dataset_b", "clip_1", 400): "straight",
            ("dataset_a", "clip_3", 500): "straight",
        }
        viewer.current_idx = 2
        viewer.scene_filter_var = Var("straight")
        viewer.scene_filter_value = "None"
        viewer.speed_edit_active = False
        viewer.loaded_idx = None
        viewer.updated = False

        viewer._on_scene_filter_selected()

        self.assertEqual(viewer.scene_filter_value, "straight")
        self.assertEqual(viewer.loaded_idx, 0)
        self.assertEqual(viewer.current_idx, 0)
        self.assertTrue(viewer.updated)

    def test_scene_filter_selection_none_keeps_current_sample(self):
        class Var:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

        class Viewer(NavigationMixin):
            def _load_sample(self, idx):
                self.loaded_idx = int(idx)

            def _update_display(self):
                self.updated = True

        viewer = Viewer()
        viewer.samples = [("dataset_a", "clip_1", 100), ("dataset_a", "clip_1", 200)]
        viewer.scene_label_by_sample = {("dataset_a", "clip_1", 100): "straight"}
        viewer.current_idx = 1
        viewer.scene_filter_var = Var("None")
        viewer.scene_filter_value = "straight"
        viewer.loaded_idx = None
        viewer.updated = False

        viewer._on_scene_filter_selected()

        self.assertEqual(viewer.scene_filter_value, "None")
        self.assertIsNone(viewer.loaded_idx)
        self.assertFalse(viewer.updated)

    def test_scene_label_index_loads_labels_for_samples(self):
        from traj_annotation.scene_labels import build_scene_label_index

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset_a"
            dataset_dir.mkdir()
            (dataset_dir / "scene_labels.json").write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "clip": "clip_1",
                                "points": [
                                    {
                                        "timestamp": 100,
                                        "scenario_type": "straight",
                                    },
                                    {
                                        "timestamp": 200,
                                        "scenario_type": "lane_change",
                                    },
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            sample_scene, scenes_by_dataset = build_scene_label_index(
                root,
                [
                    ("dataset_a", "clip_1", 100),
                    ("dataset_a", "clip_1", 200),
                    ("dataset_a", "clip_1", 300),
                ],
            )

        self.assertEqual(sample_scene[("dataset_a", "clip_1", 100)], "straight")
        self.assertEqual(sample_scene[("dataset_a", "clip_1", 200)], "lane_change")
        self.assertNotIn(("dataset_a", "clip_1", 300), sample_scene)
        self.assertEqual(scenes_by_dataset["dataset_a"], ["lane_change", "straight"])

    def test_valid_video_frame_indices_default_covers_every_video_frame(self):
        indices = valid_video_frame_indices(
            frame_count=8,
            num_history_steps=4,
            num_future_steps=3,
            frame_stride=1,
        )

        self.assertEqual(indices, list(range(8)))

    def test_valid_video_frame_indices_can_require_full_windows(self):
        indices = valid_video_frame_indices(
            frame_count=10,
            num_history_steps=4,
            num_future_steps=3,
            frame_stride=2,
            require_full_history=True,
            require_full_future=True,
        )

        self.assertEqual(indices, [3, 5])

    def test_video_frame_candidates_read_master_timestamps_and_keep_all_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ts_dir = root / "dataset_converted" / "data-timestamps"
            ts_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": [1000, 1100, 1200, 1300],
                    "frame_index": [0, 1, 2, 3],
                }
            ).to_parquet(ts_dir / "clip.timestamps.parquet")

            timestamps = load_master_video_timestamps(root, "dataset_converted", "clip")
            candidates = build_video_frame_t0_candidates(
                root,
                "dataset_converted",
                "clip",
                frame_stride=1,
            )

        np.testing.assert_array_equal(timestamps, np.array([1000, 1100, 1200, 1300]))
        self.assertEqual(candidates.t0_values, [1000, 1100, 1200, 1300])
        self.assertEqual(candidates.total_video_frames, 4)
        self.assertEqual(candidates.valid_t0_count, 4)

    def test_sample_io_video_frame_index_keeps_frames_without_generated_rows(self):
        class Viewer(SampleIOMixin):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_root = root / "data"
            output_dir = root / "output"
            ts_dir = data_root / "dataset_converted" / "data-timestamps"
            ego_dir = data_root / "dataset_converted" / "data-egomotion"
            out_dir = output_dir / "dataset_converted"
            ts_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)
            pd.DataFrame({"timestamp": [1_000_000, 1_100_000, 1_200_000, 1_300_000]}).to_parquet(
                ts_dir / "clip.timestamps.parquet"
            )
            self._write_egomotion_file(
                ego_dir / "clip.egomotion.parquet",
                [
                    1_000_000,
                    1_100_000,
                    1_200_000,
                    1_300_000,
                    1_400_000,
                    1_500_000,
                    1_600_000,
                    1_700_000,
                    1_800_000,
                    1_900_000,
                    2_000_000,
                    2_100_000,
                    2_200_000,
                    2_300_000,
                    2_400_000,
                    2_500_000,
                    2_600_000,
                    2_700_000,
                    2_800_000,
                    2_900_000,
                    3_000_000,
                    3_100_000,
                    3_200_000,
                    3_300_000,
                    3_400_000,
                    3_500_000,
                    3_600_000,
                    3_700_000,
                    3_800_000,
                    3_900_000,
                    4_000_000,
                    4_100_000,
                    4_200_000,
                    4_300_000,
                    4_400_000,
                    4_500_000,
                    4_600_000,
                    4_700_000,
                    4_800_000,
                    4_900_000,
                    5_000_000,
                    5_100_000,
                    5_200_000,
                    5_300_000,
                    5_400_000,
                    5_500_000,
                    5_600_000,
                    5_700_000,
                    5_800_000,
                    5_900_000,
                    6_000_000,
                    6_100_000,
                    6_200_000,
                    6_300_000,
                    6_400_000,
                    6_500_000,
                    6_600_000,
                    6_700_000,
                    6_800_000,
                    6_900_000,
                    7_000_000,
                    7_100_000,
                    7_200_000,
                    7_300_000,
                    7_400_000,
                    7_500_000,
                    7_600_000,
                    7_700_000,
                ],
            )
            pd.DataFrame({"t0_us": [1_100_000, 1_300_000], "sample_idx": [0, 0]}).to_parquet(
                out_dir / "clip.egomotion.parquet"
            )

            viewer = Viewer()
            viewer.data_root = data_root
            viewer.output_dir = output_dir
            viewer.datasets = ["dataset_converted"]
            viewer.index_mode = "video_frames"
            viewer.frame_stride = 1

            samples = viewer._load_video_frame_sample_index()
            status = viewer._sample_index_coverage_status(
                "dataset_converted",
                "clip",
                1_000_000,
            )

        self.assertEqual(
            samples,
            [
                ("dataset_converted", "clip", 1_000_000),
                ("dataset_converted", "clip", 1_100_000),
                ("dataset_converted", "clip", 1_200_000),
                ("dataset_converted", "clip", 1_300_000),
            ],
        )
        self.assertIn("Video t0: 4", status)
        self.assertIn("Generated t0: 2", status)
        self.assertIn("Current: no generated", status)

    def test_full_future_filter_keeps_small_missing_frame_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            self._write_egomotion_file(
                ego_dir / "clip.egomotion.parquet",
                [0, 100_000, 300_000, 400_000],
            )

            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [0],
                num_future_steps=3,
                time_step=0.1,
                max_gap_seconds=0.3,
            )

        self.assertEqual(filtered, [0])

    def test_full_future_filter_rejects_large_egomotion_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            self._write_egomotion_file(
                ego_dir / "clip.egomotion.parquet",
                [0, 100_000, 500_000],
            )

            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [0],
                num_future_steps=3,
                time_step=0.1,
                max_gap_seconds=0.3,
            )

        self.assertEqual(filtered, [])

    def test_full_future_filter_rejects_t0_inside_large_egomotion_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            self._write_egomotion_file(
                ego_dir / "clip.egomotion.parquet",
                [0, 100_000, 500_000, 600_000, 700_000, 800_000],
            )

            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [450_000],
                num_future_steps=3,
                time_step=0.1,
                max_gap_seconds=0.3,
            )

        self.assertEqual(filtered, [])

    def test_full_future_filter_connects_continuous_next_clip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            self._write_egomotion_file(
                ego_dir / "clip_a.egomotion.parquet",
                [0, 100_000, 200_000],
            )
            self._write_egomotion_file(
                ego_dir / "clip_b.egomotion.parquet",
                [300_000, 400_000, 500_000],
            )

            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [100_000],
                num_future_steps=4,
                time_step=0.1,
                max_gap_seconds=0.3,
                clip_stem="clip_a",
            )

        self.assertEqual(filtered, [100_000])

    def test_full_future_filter_rejects_spatial_jump_between_clips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            self._write_egomotion_xyz_file(
                ego_dir / "clip_a.egomotion.parquet",
                [0, 100_000, 200_000],
                [0.0, 1.0, 2.0],
            )
            self._write_egomotion_xyz_file(
                ego_dir / "clip_b.egomotion.parquet",
                [300_000, 400_000, 500_000],
                [300.0, 301.0, 302.0],
            )

            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [100_000],
                num_future_steps=4,
                time_step=0.1,
                max_gap_seconds=0.3,
                clip_stem="clip_a",
            )

        self.assertEqual(filtered, [])

    def test_full_future_filter_drops_tail_without_enough_future(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            self._write_egomotion_file(
                ego_dir / "clip.egomotion.parquet",
                [0, 100_000, 200_000],
            )

            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [100_000],
                num_future_steps=2,
                time_step=0.1,
                max_gap_seconds=0.3,
            )

        self.assertEqual(filtered, [])

    def test_full_future_filter_large_index_is_vectorized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            timestamps = np.arange(0, 80_000 * 100_000, 100_000, dtype=np.int64)
            self._write_egomotion_file(
                ego_dir / "clip.egomotion.parquet",
                timestamps.tolist(),
            )
            t0_values = timestamps[:1200].tolist()

            started = time.perf_counter()
            filtered = filter_t0s_with_full_future(
                root,
                "dataset_converted",
                [int(value) for value in t0_values],
                num_future_steps=64,
                time_step=0.1,
                max_gap_seconds=0.3,
            )
            elapsed = time.perf_counter() - started

        self.assertEqual(filtered, [int(value) for value in t0_values])
        self.assertLess(elapsed, 1.0)

    def test_trajectory_identity_normalizes_source_and_protects_gt(self):
        self.assertEqual(normalize_trajectory_source(" Manual-Bezier "), "manual_bezier")
        self.assertEqual(normalize_trajectory_source("cluster center"), "cluster_center")
        self.assertEqual(normalize_trajectory_source("VLA"), "vla")
        self.assertEqual(normalize_trajectory_source(np.nan), "")

        self.assertTrue(is_gt_trajectory_record({"source": "gt", "sample_idx": 9}, 9))
        self.assertFalse(is_gt_trajectory_record({"source": "", "sample_idx": 0}, 0))
        self.assertFalse(is_deletable_trajectory_record({"source": "gt", "sample_idx": 0}, 0))
        self.assertTrue(is_deletable_trajectory_record({"source": "", "sample_idx": 0}, 0))
        self.assertTrue(is_deletable_trajectory_record({"source": "vla", "sample_idx": 0}, 0))
        self.assertTrue(is_deletable_trajectory_record({"source": "manual_bezier", "sample_idx": 4}, 4))
        self.assertTrue(is_deletable_trajectory_record({"source": "", "sample_idx": 7}, 7))

    def test_sample_io_counts_only_non_gt_output_rows_as_generated(self):
        class Viewer(SampleIOMixin):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "output" / "dataset_converted"
            out_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"t0_us": 1000, "sample_idx": 0, "source": "gt"},
                    {"t0_us": 1100, "sample_idx": 0, "source": "vla"},
                    {"t0_us": 1200, "sample_idx": 0, "source": ""},
                    {"t0_us": 1300, "sample_idx": 1, "source": "manual_bezier"},
                ]
            ).to_parquet(out_dir / "clip.egomotion.parquet")

            viewer = Viewer()
            viewer.output_dir = root / "output"

            values = viewer._generated_t0_values_for_clip("dataset_converted", "clip")

        self.assertEqual(values, {1100, 1200, 1300})

    def test_sample_io_optimizes_non_gt_trajectory_before_parquet_write(self):
        class Viewer(SampleIOMixin):
            def _is_gt_trajectory(self, traj, fallback_index=-1):
                return False

        xyz = np.column_stack(
            [
                np.linspace(0.05, 12.0, 64),
                np.zeros(64),
                np.zeros(64),
            ]
        )
        xyz[30, 1] = 4.0
        xyz[31, 1] = -3.0
        components = trajectory_components_from_xyz(xyz)
        row = {
            "t0_us": 1000,
            "sample_idx": 0,
            "source": "vla",
            "timestamp": [1000 + int((i + 1) * 100000) for i in range(len(xyz))],
        }
        row.update({key: value.tolist() for key, value in components.items()})

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_dir = root / "output" / "dataset_converted"
            out_dir.mkdir(parents=True)
            traj_file = out_dir / "clip.egomotion.parquet"
            pd.DataFrame([row]).to_parquet(traj_file, index=False)

            viewer = Viewer()
            viewer.output_dir = root / "output"
            viewer.samples = [("dataset_converted", "clip", 1000)]
            viewer.current_idx = 0
            viewer.trajectories = [
                {
                    "t0_us": 1000,
                    "sample_idx": 0,
                    "source": "vla",
                    **components,
                }
            ]

            before = diagnose_trajectory_dynamics(xyz).metrics["max_abs_curvature_1pm"]
            self.assertTrue(viewer._write_selected_trajectory_to_parquet(0))
            saved = pd.read_parquet(traj_file).iloc[0]
            log_file = root / "output" / "edit_log.jsonl"
            log_exists = log_file.exists()
            backup_files = list((root / "output" / ".backups").glob("**/*.egomotion.parquet"))

        saved_xyz = np.column_stack([saved["x"], saved["y"], saved["z"]])
        after = diagnose_trajectory_dynamics(saved_xyz).metrics["max_abs_curvature_1pm"]

        self.assertLess(after, before)
        self.assertEqual(saved["edit_version"], 1)
        self.assertEqual(saved["edited_by_gui"], True)
        self.assertEqual(saved["edit_operation"], "edit_selected_trajectory")
        self.assertTrue(log_exists)
        self.assertEqual(len(backup_files), 1)

    def test_gui_edit_metadata_increments_only_affected_rows(self):
        df = pd.DataFrame(
            [
                {"sample_idx": 0, "source": "vla", "edit_version": 2},
                {"sample_idx": 1, "source": "manual_bezier", "edit_version": None},
                {"sample_idx": 2, "source": "cluster_center", "edit_version": 5},
            ]
        )

        updated = apply_gui_edit_metadata(
            df,
            row_indices=[0, 1],
            operation="unit_test_edit",
            edit_time="2026-05-18T12:00:00Z",
        )

        self.assertEqual(updated.loc[0, "edit_version"], 3)
        self.assertEqual(updated.loc[1, "edit_version"], 1)
        self.assertEqual(updated.loc[2, "edit_version"], 5)
        self.assertEqual(updated.loc[0, "edited_by_gui"], True)
        self.assertEqual(updated.loc[1, "edited_by_gui"], True)
        self.assertTrue(pd.isna(updated.loc[2, "edited_by_gui"]))
        self.assertEqual(updated.loc[0, "edit_time"], "2026-05-18T12:00:00Z")
        self.assertEqual(updated.loc[1, "edit_operation"], "unit_test_edit")

    def test_write_parquet_with_audit_creates_backup_and_jsonl_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            traj_dir = output_dir / "dataset_converted"
            traj_dir.mkdir(parents=True)
            traj_file = traj_dir / "clip.egomotion.parquet"
            original = pd.DataFrame([{"t0_us": 1000, "sample_idx": 0, "source": "vla"}])
            replacement = pd.DataFrame([{"t0_us": 1000, "sample_idx": 1, "source": "manual_bezier"}])
            original.to_parquet(traj_file, index=False)

            record = write_parquet_with_audit(
                traj_file,
                replacement,
                output_dir=output_dir,
                operation="unit_test_write",
                dataset_name="dataset_converted",
                clip_stem="clip",
                t0_us=1000,
                affected_rows=1,
                edit_time="2026-05-18T12:00:00Z",
            )

            saved = pd.read_parquet(traj_file)
            backup = pd.read_parquet(record.backup_file)
            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(saved.iloc[0]["sample_idx"], 1)
        self.assertEqual(backup.iloc[0]["sample_idx"], 0)
        self.assertEqual(len(log_rows), 1)
        self.assertEqual(log_rows[0]["operation"], "unit_test_write")
        self.assertEqual(log_rows[0]["dataset_name"], "dataset_converted")
        self.assertEqual(log_rows[0]["clip_stem"], "clip")
        self.assertEqual(log_rows[0]["t0_us"], 1000)
        self.assertEqual(log_rows[0]["rows_before"], 1)
        self.assertEqual(log_rows[0]["rows_after"], 1)
        self.assertEqual(log_rows[0]["affected_rows"], 1)
        self.assertTrue(log_rows[0]["backup_file"].endswith(".egomotion.parquet"))

    def test_write_text_file_with_audit_creates_backup_and_jsonl_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True)
            target_file = output_dir / "manual_points.json"
            target_file.write_text('{"version": 1, "samples": []}\n', encoding="utf-8")

            record = write_text_file_with_audit(
                target_file,
                '{"version": 1, "samples": [{"t0_us": 1000}]}\n',
                output_dir=output_dir,
                operation="save_manual_points",
                dataset_name="dataset_converted",
                clip_stem="clip",
                t0_us=1000,
                affected_rows=1,
                edit_time="2026-05-18T12:01:00Z",
                backup_group="manual_points",
            )

            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            saved_text = target_file.read_text(encoding="utf-8")
            backup_text = record.backup_file.read_text(encoding="utf-8")

        self.assertEqual(saved_text, '{"version": 1, "samples": [{"t0_us": 1000}]}\n')
        self.assertEqual(backup_text, '{"version": 1, "samples": []}\n')
        self.assertEqual(len(log_rows), 1)
        self.assertEqual(log_rows[0]["operation"], "save_manual_points")
        self.assertEqual(log_rows[0]["file_kind"], "text")
        self.assertEqual(log_rows[0]["target_file"], "manual_points.json")
        self.assertTrue(log_rows[0]["backup_file"].endswith(".json"))
        self.assertGreater(log_rows[0]["bytes_after"], log_rows[0]["bytes_before"])

    def test_restore_file_from_backup_with_audit_replaces_target_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True)
            target_file = output_dir / "dataset_converted" / "clip.egomotion.parquet"
            target_file.parent.mkdir(parents=True)
            target_file.write_text("active version\n", encoding="utf-8")
            backup_file = output_dir / ".backups" / "dataset_converted" / "clip" / "old.egomotion.parquet"
            backup_file.parent.mkdir(parents=True)
            backup_file.write_text("restored version\n", encoding="utf-8")

            record = restore_file_from_backup_with_audit(
                target_file,
                backup_file,
                output_dir=output_dir,
                operation="restore_parquet_backup",
                dataset_name="dataset_converted",
                clip_stem="clip",
                t0_us=1000,
                edit_time="2026-05-18T12:02:00Z",
                backup_group="dataset_converted/clip",
            )

            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            restored_text = target_file.read_text(encoding="utf-8")
            pre_restore_backup_text = record.backup_file.read_text(encoding="utf-8")

        self.assertEqual(restored_text, "restored version\n")
        self.assertEqual(pre_restore_backup_text, "active version\n")
        self.assertEqual(len(log_rows), 1)
        self.assertEqual(log_rows[0]["operation"], "restore_parquet_backup")
        self.assertEqual(log_rows[0]["target_file"], "dataset_converted/clip.egomotion.parquet")
        self.assertTrue(log_rows[0]["backup_file"].endswith(".egomotion.parquet"))
        self.assertTrue(log_rows[0]["metadata"]["restored_from_backup"].endswith("old.egomotion.parquet"))

    def test_manual_points_index_write_is_audited(self):
        class Viewer(SampleIOMixin):
            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True)
            viewer = Viewer()
            viewer.output_dir = output_dir
            viewer.manual_points_file = output_dir / "manual_points.json"
            viewer.manual_points_file.write_text('{"version": 1, "samples": []}\n', encoding="utf-8")
            key = ("dataset_converted", "clip", 1000)
            viewer.manual_line_points_index = {key: [{"x": 1.0, "y": 2.0, "z": 0.0}]}
            viewer.manual_camera_line_points_index = {}
            viewer.manual_stop_points_index = {}

            viewer._write_manual_points_index()

            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            backup_files = list((output_dir / ".backups" / "files" / "manual_points").glob("*.json"))
            saved = json.loads(viewer.manual_points_file.read_text(encoding="utf-8"))

        self.assertEqual(saved["samples"][0]["dataset_name"], "dataset_converted")
        self.assertEqual(len(backup_files), 1)
        self.assertEqual(log_rows[0]["operation"], "save_manual_points")
        self.assertEqual(log_rows[0]["affected_rows"], 1)

    def test_cluster_center_library_file_writes_are_audited(self):
        class Viewer(ClusterControlsMixin):
            def _cluster_category_file(self, category):
                return self.kmeans_dir / f"{category}_centers.txt"

            def _bezier_cluster_center_meta_file(self):
                return self.kmeans_dir / "bezier_centers.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            kmeans_dir = Path(tmpdir) / "k_means"
            output_dir.mkdir(parents=True)
            kmeans_dir.mkdir(parents=True)
            viewer = Viewer()
            viewer.output_dir = output_dir
            viewer.kmeans_dir = kmeans_dir
            viewer.bezier_cluster_center_ids = {"straight": {2}}
            category_file = viewer._cluster_category_file("straight")
            meta_file = viewer._bezier_cluster_center_meta_file()
            category_file.write_text("old centers\n", encoding="utf-8")
            meta_file.write_text('{"version": 1, "centers": {"straight": [1]}}\n', encoding="utf-8")

            viewer._write_cluster_category_file(
                "straight",
                [{"id": 2, "count": 1, "trajectory": np.zeros((2, 3), dtype=np.float32)}],
            )
            viewer._write_bezier_cluster_center_ids()

            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            backup_files = list((output_dir / ".backups" / "files" / "k_means").glob("**/*"))

        self.assertEqual([row["operation"] for row in log_rows], ["write_cluster_category_file", "write_bezier_cluster_center_ids"])
        self.assertTrue(any(path.suffix == ".txt" for path in backup_files))
        self.assertTrue(any(path.suffix == ".json" for path in backup_files))

    def test_sample_io_restores_latest_current_clip_parquet_backup(self):
        class Viewer(SampleIOMixin):
            def _active_traj_file(self, dataset_name, clip_stem):
                return self.output_dir / dataset_name / f"{clip_stem}.egomotion.parquet"

            def _load_sample(self, idx):
                self.loaded_idx = int(idx)

            def _update_display(self):
                self.updated = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            traj_dir = output_dir / "dataset_converted"
            traj_dir.mkdir(parents=True)
            traj_file = traj_dir / "clip.egomotion.parquet"
            original = pd.DataFrame([{"t0_us": 1000, "sample_idx": 0, "source": "vla"}])
            replacement = pd.DataFrame([{"t0_us": 1000, "sample_idx": 1, "source": "manual_bezier"}])
            original.to_parquet(traj_file, index=False)
            write_parquet_with_audit(
                traj_file,
                replacement,
                output_dir=output_dir,
                operation="unit_test_write",
                dataset_name="dataset_converted",
                clip_stem="clip",
                t0_us=1000,
                affected_rows=1,
                edit_time="2026-05-18T12:03:00Z",
            )

            viewer = Viewer()
            viewer.output_dir = output_dir
            viewer.samples = [("dataset_converted", "clip", 1000)]
            viewer.current_idx = 0

            with mock.patch("traj_annotation.mixins.sample_io.messagebox.askyesno", return_value=True), \
                 mock.patch("traj_annotation.mixins.sample_io.messagebox.showinfo"):
                viewer._restore_latest_current_clip_backup()

            restored = pd.read_parquet(traj_file)
            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(restored.iloc[0]["sample_idx"], 0)
        self.assertEqual(log_rows[-1]["operation"], "restore_parquet_backup")
        self.assertTrue(log_rows[-1]["metadata"]["restored_from_backup"].endswith(".egomotion.parquet"))
        self.assertEqual(viewer.loaded_idx, 0)
        self.assertTrue(viewer.updated)

    def test_sample_io_save_results_audits_delete_write(self):
        class Viewer(SampleIOMixin):
            def _active_traj_file(self, dataset_name, clip_stem):
                return self.output_dir / dataset_name / f"{clip_stem}.egomotion.parquet"

            def _persist_pending_manual_point_deletes(self):
                self.persisted_manual_deletes = True

            def _load_sample(self, idx):
                self.loaded_idx = int(idx)

            def _update_display(self):
                self.updated = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            traj_dir = output_dir / "dataset_converted"
            traj_dir.mkdir(parents=True)
            traj_file = traj_dir / "clip.egomotion.parquet"
            pd.DataFrame(
                [
                    {"t0_us": 1000, "sample_idx": 0, "source": "vla"},
                    {"t0_us": 1000, "sample_idx": 1, "source": "manual_bezier"},
                ]
            ).to_parquet(traj_file, index=False)

            viewer = Viewer()
            viewer.output_dir = output_dir
            viewer.samples = [("dataset_converted", "clip", 1000)]
            viewer.current_idx = 0
            viewer.speed_edit_active = False
            viewer.traj_geom_edit_active = False
            viewer.gt_only = False
            viewer.pending_deleted_traj_keys = {(1000, 1)}

            with mock.patch("traj_annotation.mixins.sample_io.messagebox.showinfo"):
                viewer._save_results()

            saved = pd.read_parquet(traj_file)
            log_rows = [
                json.loads(line)
                for line in (output_dir / "edit_log.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            backup_files = list((output_dir / ".backups").glob("**/*.egomotion.parquet"))

        self.assertEqual(saved["sample_idx"].tolist(), [0])
        self.assertEqual(len(backup_files), 1)
        self.assertEqual(log_rows[0]["operation"], "delete_trajectories")
        self.assertEqual(log_rows[0]["affected_rows"], 1)
        self.assertEqual(log_rows[0]["metadata"]["deleted_keys"], [{"t0_us": 1000, "sample_idx": 1}])

    def test_trajectory_key_and_drop_rows_use_t0_and_sample_idx(self):
        key = trajectory_key_from_record({"t0_us": 111, "sample_idx": 4, "source": "manual_bezier"}, 0)
        df = pd.DataFrame(
            [
                {"t0_us": 111, "sample_idx": 0, "source": "gt"},
                {"t0_us": 111, "sample_idx": 4, "source": "manual_bezier"},
                {"t0_us": 222, "sample_idx": 4, "source": "manual_bezier"},
            ]
        )

        kept = drop_trajectory_rows_by_keys(df, current_t0_us=111, deleted_keys={key})

        self.assertEqual(len(kept), 2)
        self.assertEqual(kept.iloc[0]["sample_idx"], 0)
        self.assertEqual(kept.iloc[1]["t0_us"], 222)

    def test_delete_controls_stage_and_undo_hide_visible_indices(self):
        class Viewer(DeleteControlsMixin):
            pass

        viewer = Viewer()
        viewer.trajectories = [
            {"t0_us": 111, "sample_idx": 0, "source": "gt"},
            {"t0_us": 111, "sample_idx": 4, "source": "manual_bezier"},
            {"t0_us": 111, "sample_idx": 5, "source": "cluster_center"},
        ]
        viewer.current_traj_idx = 1
        viewer._reset_pending_delete_state()

        self.assertTrue(viewer._stage_delete_traj_idx(1))

        self.assertEqual(viewer._visible_trajectory_indices(), [0, 2])
        self.assertTrue(viewer._is_traj_pending_deleted(1))
        self.assertEqual(viewer._pending_delete_count(), 1)

        self.assertTrue(viewer._undo_delete_traj(redraw=False))

        self.assertEqual(viewer._visible_trajectory_indices(), [0, 1, 2])
        self.assertFalse(viewer._is_traj_pending_deleted(1))
        self.assertEqual(viewer.current_traj_idx, 1)

    def test_delete_controls_stage_and_undo_manual_points_in_memory(self):
        class Viewer(DeleteControlsMixin):
            def _selected_trajectory_matches_manual_curve(self):
                return True

        viewer = Viewer()
        viewer.trajectories = [
            {"t0_us": 111, "sample_idx": 1, "source": "manual_bezier"},
        ]
        viewer.current_traj_idx = 0
        viewer.manual_line_points = [{"x": 1.0, "y": 2.0}]
        viewer.manual_camera_line_points = []
        viewer.manual_stop_points = [{"fraction": 1.0, "duration_s": 2.0}]
        viewer.manual_point_actions = [("line", 0)]
        viewer.manual_line_points_dirty = False
        viewer.manual_camera_line_points_dirty = False
        viewer.manual_stop_points_dirty = False
        viewer._reset_pending_delete_state()

        self.assertTrue(viewer._stage_delete_traj_idx(0))

        self.assertEqual(viewer.manual_line_points, [])
        self.assertEqual(viewer.manual_stop_points, [])
        self.assertTrue(viewer.pending_manual_points_delete)

        self.assertTrue(viewer._undo_delete_traj(redraw=False))

        self.assertEqual(viewer.manual_line_points, [{"x": 1.0, "y": 2.0}])
        self.assertEqual(viewer.manual_stop_points, [{"fraction": 1.0, "duration_s": 2.0}])
        self.assertFalse(viewer.pending_manual_points_delete)

    def test_delete_controls_block_delete_while_saved_geometry_edit_active(self):
        class Viewer(DeleteControlsMixin):
            pass

        viewer = Viewer()
        viewer.trajectories = [
            {"t0_us": 111, "sample_idx": 1, "source": "vla"},
        ]
        viewer.current_traj_idx = 0
        viewer.traj_geom_edit_active = True
        viewer._reset_pending_delete_state()

        self.assertFalse(viewer._stage_delete_traj_idx(0))
        self.assertEqual(viewer._pending_delete_count(), 0)
        self.assertFalse(viewer._is_traj_pending_deleted(0))

    def test_rgb_helpers_clip_and_generate_stable_extended_colors(self):
        self.assertEqual(_rgb_to_hex((300, -4, 16)), "#ff0010")

        color_a = _trajectory_base_color(8, 12)
        color_b = _trajectory_base_color(8, 12)

        self.assertEqual(color_a, color_b)
        self.assertEqual(len(color_a), 3)
        self.assertTrue(all(0 <= channel <= 255 for channel in color_a))

    def test_speed_profile_prefers_velocity_columns_over_position_diffs(self):
        speed = _speed_profile_from_trajectory(
            [0.0, 100.0],
            [0.0, 0.0],
            [0.0, 0.0],
            vx=[3.0, 4.0],
            vy=[4.0, 3.0],
            vz=[0.0, 12.0],
        )

        np.testing.assert_allclose(speed, [5.0, 13.0])

    def test_history_speed_profile_uses_history_diffs_and_keeps_t0_speed(self):
        history_xyz = np.column_stack(
            [
                np.linspace(-1.5, 0.0, 16),
                np.zeros(16),
                np.zeros(16),
            ]
        )

        speed = _history_speed_profile_from_xyz(history_xyz, dt_seconds=0.1)

        self.assertEqual(len(speed), 16)
        np.testing.assert_allclose(speed, np.ones(16), atol=1e-9)

    def test_smooth_history_xyz_preserves_current_and_reduces_middle_noise(self):
        history_xyz = np.column_stack(
            [
                np.linspace(-1.5, 0.0, 16),
                np.zeros(16),
                np.zeros(16),
            ]
        )
        history_xyz[7, 1] = 2.0

        smoothed = _smooth_history_xyz_for_display(history_xyz, passes=2)

        np.testing.assert_allclose(smoothed[0], history_xyz[0])
        np.testing.assert_allclose(smoothed[-1], history_xyz[-1])
        self.assertLess(float(np.max(np.abs(smoothed[:, 1]))), 2.0)
        self.assertEqual(smoothed.shape, history_xyz.shape)

    def test_smoothed_history_speed_profile_reduces_diff_noise(self):
        history_xyz = np.column_stack(
            [
                np.linspace(-1.5, 0.0, 16),
                np.zeros(16),
                np.zeros(16),
            ]
        )
        history_xyz[7, 1] = 1.5

        raw_speed = _history_speed_profile_from_xyz(history_xyz, dt_seconds=0.1)
        smoothed_speed = _smoothed_history_speed_profile_from_xyz(history_xyz, dt_seconds=0.1)

        self.assertEqual(len(smoothed_speed), len(raw_speed))
        self.assertLess(float(np.max(smoothed_speed)), float(np.max(raw_speed)))

    def test_smoothed_history_speed_profile_preserves_t0_transition_speed(self):
        history_xyz = np.column_stack(
            [
                np.array([
                    -2.10, -1.90, -1.70, -1.50,
                    -1.30, -1.10, -0.90, -0.70,
                    -0.50, -0.35, -0.22, -0.12,
                    -0.06, -0.03, -0.01, 0.00,
                ]),
                np.zeros(16),
                np.zeros(16),
            ]
        )

        smoothed_xyz = _smooth_history_xyz_for_display(history_xyz, passes=2)
        expected_t0_speed = _history_speed_profile_from_xyz(
            smoothed_xyz,
            dt_seconds=0.1,
        )[-1]
        smoothed_speed = _smoothed_history_speed_profile_from_xyz(
            history_xyz,
            dt_seconds=0.1,
            xyz_passes=2,
            speed_passes=1,
        )

        self.assertEqual(len(smoothed_speed), len(history_xyz))
        np.testing.assert_allclose(smoothed_speed[-1], expected_t0_speed, atol=1e-9)

    def test_speed_controls_expose_history_speed_source_and_points(self):
        class Viewer(SpeedControlsMixin):
            pass

        history_xyz = np.column_stack(
            [
                np.linspace(-3.0, 0.0, 16),
                np.zeros(16),
                np.zeros(16),
            ]
        )
        viewer = Viewer()
        viewer.conv_data = {"ego_history_xyz": history_xyz.reshape(1, 1, 16, 3)}

        label, speed, stops, color = viewer._history_speed_profile_source()
        points = viewer._trajectory_points_for_speed_source("history")

        self.assertEqual(label, "History")
        self.assertEqual(color, HISTORY_SPEED_COLOR_HEX)
        self.assertEqual(stops, [])
        np.testing.assert_allclose(speed, np.full(16, 2.0))
        np.testing.assert_allclose(points, history_xyz)

    def test_speed_controls_history_points_respect_valid_mask(self):
        class Viewer(SpeedControlsMixin):
            pass

        history_xyz = np.column_stack(
            [
                np.arange(4, dtype=np.float64),
                np.zeros(4),
                np.zeros(4),
            ]
        )
        viewer = Viewer()
        viewer.conv_data = {
            "ego_history_xyz": history_xyz.reshape(1, 1, 4, 3),
            "ego_history_valid_mask": np.array([False, False, True, True]).reshape(1, 1, 4),
        }

        points = viewer._history_points_xyz()

        np.testing.assert_allclose(points, history_xyz[-2:])

        viewer.conv_data["ego_history_valid_mask"] = np.array(
            [False, False, False, True]
        ).reshape(1, 1, 4)
        self.assertIsNone(viewer._history_points_xyz())

    def test_speed_controls_keeps_velocity_history_when_position_history_is_short(self):
        class Viewer(SpeedControlsMixin):
            pass

        viewer = Viewer()
        viewer.conv_data = {
            "ego_history_xyz": np.zeros((1, 1, 4, 3), dtype=np.float32),
            "ego_history_valid_mask": np.array(
                [False, False, False, True],
                dtype=bool,
            ).reshape(1, 1, 4),
            "ego_history_speed_mps": np.array(
                [3.0, 4.0, 5.0, 6.0],
                dtype=np.float32,
            ).reshape(1, 1, 4),
            "ego_history_speed_valid_mask": np.ones((1, 1, 4), dtype=bool),
        }

        label, speed, _stops, color = viewer._history_speed_profile_source()

        self.assertEqual(label, "History")
        np.testing.assert_allclose(speed, [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(color, HISTORY_SPEED_COLOR_HEX)
        self.assertIsNone(viewer._history_points_xyz(smoothed=False))

    def test_selected_diversity_speed_profile_follows_selected_geometry(self):
        class Viewer(SpeedControlsMixin):
            def _manual_preview_is_active_for_speed(self):
                return False

            def _is_traj_pending_deleted(self, _traj_idx):
                return False

            def _is_gt_trajectory(self, _traj, _fallback_index=-1):
                return False

        def traj_with_stale_velocity(step_m: float) -> dict:
            x = np.arange(1, 5, dtype=np.float64) * float(step_m)
            return {
                "x": x,
                "y": np.zeros_like(x),
                "z": np.zeros_like(x),
                "vx": np.full_like(x, 99.0),
                "vy": np.zeros_like(x),
                "vz": np.zeros_like(x),
            }

        viewer = Viewer()
        viewer.trajectories = [
            traj_with_stale_velocity(0.1),
            traj_with_stale_velocity(0.3),
        ]
        viewer.trajectory_states = {0: True, 1: True}
        viewer.trajectory_smoothness = {}
        viewer.cluster_preview_traj = None
        viewer.current_traj_idx = 0
        viewer.speed_edit_active = False
        viewer.speed_edit_speed = None

        _label0, speed0, _stops0, _color0 = viewer._selected_speed_profile_source()
        viewer.current_traj_idx = 1
        _label1, speed1, _stops1, _color1 = viewer._selected_speed_profile_source()

        np.testing.assert_allclose(speed0, np.full(4, 1.0))
        np.testing.assert_allclose(speed1, np.full(4, 3.0))
        self.assertFalse(np.allclose(speed0, speed1))

    def test_cluster_preview_speed_profile_follows_current_preview_geometry(self):
        class Viewer(SpeedControlsMixin):
            def _manual_preview_is_active_for_speed(self):
                return False

            def _is_traj_pending_deleted(self, _traj_idx):
                return False

            def _is_gt_trajectory(self, _traj, _fallback_index=-1):
                return False

        viewer = Viewer()
        viewer.trajectories = [{
            "x": np.arange(1, 5, dtype=np.float64),
            "y": np.zeros(4),
            "z": np.zeros(4),
        }]
        viewer.trajectory_states = {0: True}
        viewer.trajectory_smoothness = {}
        viewer.current_traj_idx = 0
        viewer.speed_edit_active = False
        viewer.speed_edit_speed = None

        viewer.cluster_preview_traj = np.column_stack([
            np.arange(1, 5, dtype=np.float64) * 0.2,
            np.zeros(4),
            np.zeros(4),
        ])
        label0, speed0, _stops0, _color0 = viewer._selected_speed_profile_source()

        viewer.cluster_preview_traj = np.column_stack([
            np.arange(1, 5, dtype=np.float64) * 0.4,
            np.zeros(4),
            np.zeros(4),
        ])
        label1, speed1, _stops1, _color1 = viewer._selected_speed_profile_source()

        self.assertEqual(label0, "Cluster preview")
        self.assertEqual(label1, "Cluster preview")
        np.testing.assert_allclose(speed0, np.full(4, 2.0))
        np.testing.assert_allclose(speed1, np.full(4, 4.0))

    def test_speed_controls_history_points_are_display_smoothed_without_mutating_source(self):
        class Viewer(SpeedControlsMixin):
            pass

        history_xyz = np.column_stack(
            [
                np.linspace(-1.5, 0.0, 16),
                np.zeros(16),
                np.zeros(16),
            ]
        )
        history_xyz[7, 1] = 1.5
        viewer = Viewer()
        viewer.conv_data = {"ego_history_xyz": history_xyz.reshape(1, 1, 16, 3).copy()}

        raw_points = viewer._history_points_xyz(smoothed=False)
        display_points = viewer._history_points_xyz()

        np.testing.assert_allclose(raw_points, history_xyz)
        np.testing.assert_allclose(display_points[-1], history_xyz[-1])
        self.assertLess(float(np.max(np.abs(display_points[:, 1]))), 1.5)
        stored = np.asarray(viewer.conv_data["ego_history_xyz"]).reshape(-1, 3)
        np.testing.assert_allclose(stored, history_xyz)

    def test_speed_hover_mapping_uses_negative_frames_for_history(self):
        class Viewer(SpeedControlsMixin):
            pass

        viewer = Viewer()
        viewer.conv_data = {
            "ego_history_xyz": np.column_stack(
                [
                    np.linspace(-1.5, 0.0, 16),
                    np.zeros(16),
                    np.zeros(16),
                ]
            ).reshape(1, 1, 16, 3)
        }
        viewer._selected_speed_profile_source = lambda: (
            "T0",
            np.ones(64, dtype=np.float64),
            [],
            "#ffffff",
        )
        rect = {"left": 0.0, "right": 78.0, "width": 78.0}

        self.assertEqual(
            viewer._speed_hover_target_for_canvas_x("pred", 0.0, rect),
            ("history", 0),
        )
        self.assertEqual(
            viewer._speed_hover_target_for_canvas_x("pred", 14.0, rect),
            ("history", 14),
        )
        self.assertEqual(
            viewer._speed_hover_target_for_canvas_x("pred", 15.0, rect),
            ("pred", 0),
        )
        self.assertEqual(
            viewer._speed_hover_target_for_canvas_x("pred", 78.0, rect),
            ("pred", 63),
        )

    def test_speed_frame_bounds_use_gt_horizon_when_pred_is_empty(self):
        class Viewer(SpeedControlsMixin):
            def _get_gt_future_xyz(self):
                return np.zeros((64, 3), dtype=np.float64)

        viewer = Viewer()
        viewer.conv_data = {
            "ego_history_xyz": np.column_stack(
                [
                    np.linspace(-1.5, 0.0, 16),
                    np.zeros(16),
                    np.zeros(16),
                ]
            ).reshape(1, 1, 16, 3)
        }

        self.assertEqual(
            viewer._speed_frame_bounds(np.zeros(0, dtype=np.float64)),
            (-15, 63),
        )

    def test_stop_detection_returns_segments_with_duration_and_mean_speed(self):
        speed = np.array([0.2, 0.05, 0.04, 0.03, 0.02, 0.01, 0.3])

        segments = _detect_stop_segments(speed, threshold_mps=0.1, min_frames=3, dt_seconds=0.1)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 1)
        self.assertEqual(segments[0]["end"], 5)
        self.assertEqual(segments[0]["frames"], 5)
        self.assertAlmostEqual(segments[0]["duration_s"], 0.5)
        self.assertAlmostEqual(segments[0]["mean_speed_mps"], 0.03)

    def test_speed_smoothness_allows_gt_like_curves_but_flags_large_spikes(self):
        gt_like = np.array([1.0, 1.1, 1.05, 1.12, 1.08, 1.1])
        self.assertTrue(_speed_smoothness_diagnostics(gt_like, gt_like)["ok"])

        spiky = np.array([1.0, 1.0, 15.0, 1.0, 1.0, 1.0])
        diagnostics = _speed_smoothness_diagnostics(spiky)

        self.assertFalse(diagnostics["ok"])
        self.assertEqual(diagnostics["reason"], "速度不够平滑")

    def test_acceleration_limiter_clamps_every_transition(self):
        limited = _enforce_speed_acceleration_limits(np.array([0.0, 10.0, 0.0, 10.0]))
        accel = np.diff(limited) / TRAJ_DT_SECONDS

        self.assertTrue(np.all(accel <= TRAJ_ACCEL_MAX_MPS2 + 1e-9))
        self.assertTrue(np.all(accel >= TRAJ_ACCEL_MIN_MPS2 - 1e-9))

    def test_speed_smoothing_reduces_spikes_without_negative_values(self):
        speed = np.array([2.0, 2.0, 12.0, 2.0, 2.0, 2.0, 2.0])

        smoothed = _smooth_speed_profile(speed, passes=2)

        self.assertEqual(len(smoothed), len(speed))
        self.assertGreater(smoothed[2], 0.0)
        self.assertLess(smoothed[2], speed[2])
        self.assertTrue(np.all(smoothed >= 0.0))

    def test_smoothed_gt_speed_profile_reduces_position_diff_noise(self):
        gt = np.column_stack(
            [
                np.arange(12, dtype=np.float64),
                np.zeros(12, dtype=np.float64),
                np.zeros(12, dtype=np.float64),
            ]
        )
        gt[5, 1] = 0.8

        raw_speed = _speed_profile_from_trajectory(gt[:, 0], gt[:, 1], gt[:, 2])
        smoothed = _smoothed_gt_speed_profile_from_xyz(gt)

        self.assertEqual(len(smoothed), len(raw_speed))
        self.assertLess(np.nanmax(np.abs(np.diff(smoothed))), np.nanmax(np.abs(np.diff(raw_speed))))
        self.assertGreater(smoothed[5], 0.0)
        self.assertTrue(np.all(smoothed >= 0.0))

    def test_distance_and_speed_resampling_preserve_endpoint(self):
        dense = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        by_distance = _resample_curve_by_distance(
            dense,
            num_steps=5,
            initial_speed_mps=0.0,
            dt_seconds=TRAJ_DT_SECONDS,
        )

        np.testing.assert_allclose(by_distance[-1], dense[-1])

        original = np.column_stack(
            [
                np.linspace(1.0, 10.0, 5),
                np.zeros(5),
                np.zeros(5),
            ]
        )
        by_speed = _resample_xyz_by_speed_profile(original, np.ones(5))

        self.assertIsNotNone(by_speed)
        np.testing.assert_allclose(by_speed[-1], original[-1])
        self.assertTrue(math.isclose(float(by_speed[0, 1]), 0.0))

    def test_dynamics_diagnostics_flags_speed_accel_and_curvature(self):
        xyz = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [5.5, 0.0, 0.0],
                [5.7, 2.5, 0.0],
                [5.9, -2.5, 0.0],
                [6.1, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        limits = DynamicsLimits(
            max_speed_mps=8.0,
            max_step_m=0.8,
            max_curvature_1pm=0.6,
            max_accel_mps2=2.0,
            min_accel_mps2=-6.0,
        )

        diagnostics = diagnose_trajectory_dynamics(xyz, limits=limits)

        self.assertFalse(diagnostics.ok)
        self.assertIn("speed", diagnostics.violations)
        self.assertIn("step", diagnostics.violations)
        self.assertIn("acceleration", diagnostics.violations)
        self.assertIn("curvature", diagnostics.violations)

    def test_dynamics_optimizer_reduces_spikes_and_preserves_endpoint(self):
        xyz = np.column_stack(
            [
                np.linspace(0.05, 12.0, 64),
                np.zeros(64),
                np.zeros(64),
            ]
        )
        xyz[30, 1] = 4.0
        xyz[31, 1] = -3.0
        limits = DynamicsLimits(max_curvature_1pm=1.0)

        before = diagnose_trajectory_dynamics(xyz, limits=limits)
        result = optimize_pseudo_gt_trajectory(xyz, limits=limits)
        after = diagnose_trajectory_dynamics(result.xyz, limits=limits)

        self.assertTrue(result.ok)
        self.assertEqual(result.xyz.shape, xyz.shape)
        np.testing.assert_allclose(result.xyz[-1], xyz[-1], atol=0.25)
        self.assertLess(after.metrics["max_abs_curvature_1pm"], before.metrics["max_abs_curvature_1pm"])
        self.assertLessEqual(after.metrics["max_abs_curvature_1pm"], limits.max_curvature_1pm)
        self.assertLessEqual(after.metrics["max_speed_mps"], limits.max_speed_mps + 1e-6)
        self.assertLessEqual(after.metrics["max_step_m"], limits.max_step_m + 1e-6)

    def test_saved_trajectory_geometry_edit_keyframes_skip_origin_and_include_endpoint(self):
        keyframes = editable_trajectory_keyframes(64, interval=8)

        self.assertEqual(keyframes, [8, 16, 24, 32, 40, 48, 56, 63])

    def test_saved_trajectory_edit_mixin_uses_four_position_handles(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def _selected_saved_traj_xyz(self):
                return np.column_stack(
                    [
                        np.linspace(0.05, 12.0, 64),
                        np.zeros(64),
                        np.zeros(64),
                    ]
                )

        viewer = Viewer()

        self.assertEqual(viewer._saved_trajectory_edit_keyframes(), [16, 32, 48, 63])

    def test_speed_drag_during_saved_trajectory_edit_preserves_endpoint(self):
        class Viewer(SpeedControlsMixin, SavedTrajectoryEditingMixin):
            def __init__(self):
                self.gt_only = False
                self.current_traj_idx = 0
                self.traj_geom_edit_active = True
                self.traj_geom_edit_dirty = False
                self.traj_geom_edit_traj_idx = 0
                self.traj_geom_edit_original_traj = None
                self.traj_geom_edit_original_xyz = None
                self.speed_edit_active = False
                self.speed_edit_dirty = False
                self.speed_edit_traj_idx = None
                self.speed_edit_original_traj = None
                self.speed_edit_original_xyz = None
                self.speed_edit_speed = None
                self.speed_edit_last_frame = None
                self.speed_canvas_width = 240
                self.speed_canvas_height = 140
                self.speed_plot_rect = None
                self.speed_hover_source = None
                self.speed_hover_frame_idx = None
                self.conv_data = None
                self.trajectory_states = {0: True}
                self.trajectory_smoothness = {}
                x = np.linspace(0.25, 16.0, 64)
                y = 1.2 * np.sin(np.linspace(0.0, np.pi, 64))
                z = np.zeros(64)
                self.trajectories = [{"x": x.copy(), "y": y.copy(), "z": z.copy()}]
                self.draw_calls = 0

            def _is_traj_pending_deleted(self, _traj_idx):
                return False

            def _is_gt_trajectory(self, _traj, _fallback_index=-1):
                return False

            def _refresh_trajectory_smoothness(self):
                self.trajectory_smoothness = {0: {"ok": True}}

            def _hide_pred_speed_actions(self):
                pass

            def _update_display(self):
                pass

            def _draw_trajectories(self):
                self.draw_calls += 1

            def _draw_speed_profile(self):
                self.draw_calls += 1

            def _draw_camera_images(self):
                self.draw_calls += 1

        class Event:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        viewer = Viewer()
        original_endpoint = viewer._selected_saved_traj_xyz()[-1].copy()
        rect = viewer._speed_plot_geometry()

        viewer._on_speed_canvas_left_down(
            Event(
                x=rect["left"] + rect["width"] * 0.45,
                y=rect["top"] + rect["height"] * 0.25,
            )
        )

        edited_xyz = viewer._selected_saved_traj_xyz()
        self.assertTrue(viewer.speed_edit_active)
        self.assertTrue(viewer.speed_edit_dirty)
        self.assertTrue(viewer.traj_geom_edit_dirty)
        np.testing.assert_allclose(edited_xyz[-1], original_endpoint, atol=1e-6)
        self.assertGreater(viewer.draw_calls, 0)

    def test_speed_edit_undo_redo_restores_speed_and_geometry_snapshots(self):
        class Viewer(SpeedControlsMixin):
            def __init__(self):
                self.speed_edit_active = True
                self.speed_edit_dirty = False
                self.speed_edit_traj_idx = 0
                self.speed_edit_undo_stack = []
                self.speed_edit_redo_stack = []
                self.traj_geom_edit_active = True
                self.traj_geom_edit_traj_idx = 0
                self.traj_geom_edit_dirty = False
                self.trajectory_smoothness = {}
                self.update_count = 0
                self.refresh_count = 0
                base = np.column_stack([
                    np.linspace(0.25, 16.0, 64),
                    np.zeros(64),
                    np.zeros(64),
                ])
                self.trajectories = [trajectory_components_from_xyz(base)]
                self.speed_edit_original_xyz = base.copy()
                self.speed_edit_speed = self._trajectory_speed_profile(self.trajectories[0]).copy()

            def _refresh_trajectory_smoothness(self):
                self.refresh_count += 1

            def _update_display(self):
                self.update_count += 1

        viewer = Viewer()
        before = viewer._speed_edit_snapshot()
        after_xyz = np.column_stack([
            np.linspace(0.25, 16.0, 64),
            np.linspace(0.0, 1.5, 64),
            np.zeros(64),
        ])
        after = {
            "speed": np.linspace(2.0, 5.0, 64),
            "xyz": after_xyz,
        }
        viewer._apply_speed_edit_snapshot(after)
        viewer._record_speed_edit_snapshot(before, after)

        self.assertTrue(viewer._undo_speed_edit())
        undone = viewer._speed_edit_snapshot()
        np.testing.assert_allclose(undone["speed"], before["speed"])
        np.testing.assert_allclose(undone["xyz"], before["xyz"])

        self.assertTrue(viewer._redo_speed_edit())
        redone = viewer._speed_edit_snapshot()
        np.testing.assert_allclose(redone["speed"], after["speed"])
        np.testing.assert_allclose(redone["xyz"], after["xyz"])
        self.assertEqual(len(viewer.speed_edit_undo_stack), 1)
        self.assertEqual(len(viewer.speed_edit_redo_stack), 0)
        self.assertGreaterEqual(viewer.update_count, 2)
        self.assertGreaterEqual(viewer.refresh_count, 2)

    def test_speed_drag_records_single_undo_step_on_release(self):
        class Viewer(SpeedControlsMixin, SavedTrajectoryEditingMixin):
            def __init__(self):
                self.gt_only = False
                self.current_traj_idx = 0
                self.traj_geom_edit_active = True
                self.traj_geom_edit_dirty = False
                self.traj_geom_edit_traj_idx = 0
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
                self.speed_canvas_width = 240
                self.speed_canvas_height = 140
                self.speed_plot_rect = None
                self.speed_hover_source = None
                self.speed_hover_frame_idx = None
                self.conv_data = None
                self.trajectory_states = {0: True}
                self.trajectory_smoothness = {}
                x = np.linspace(0.25, 16.0, 64)
                y = 1.2 * np.sin(np.linspace(0.0, np.pi, 64))
                z = np.zeros(64)
                self.trajectories = [{"x": x.copy(), "y": y.copy(), "z": z.copy()}]
                self.draw_calls = 0

            def _is_traj_pending_deleted(self, _traj_idx):
                return False

            def _is_gt_trajectory(self, _traj, _fallback_index=-1):
                return False

            def _refresh_trajectory_smoothness(self):
                self.trajectory_smoothness = {0: {"ok": True}}

            def _hide_pred_speed_actions(self):
                pass

            def _update_display(self):
                pass

            def _draw_trajectories(self):
                self.draw_calls += 1

            def _draw_speed_profile(self):
                self.draw_calls += 1

            def _draw_camera_images(self):
                self.draw_calls += 1

        class Event:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        viewer = Viewer()
        rect = viewer._speed_plot_geometry()
        viewer._start_speed_edit_for_canvas_interaction()
        before = viewer._speed_edit_snapshot()
        viewer._reset_speed_edit_state()

        viewer._on_speed_canvas_left_down(
            Event(
                x=rect["left"] + rect["width"] * 0.45,
                y=rect["top"] + rect["height"] * 0.25,
            )
        )
        viewer._on_speed_canvas_left_drag(
            Event(
                x=rect["left"] + rect["width"] * 0.55,
                y=rect["top"] + rect["height"] * 0.20,
            )
        )
        viewer._on_speed_canvas_left_release(Event(x=0, y=0))

        self.assertEqual(len(viewer.speed_edit_undo_stack), 1)
        self.assertEqual(len(viewer.speed_edit_redo_stack), 0)
        self.assertTrue(viewer._undo_speed_edit())
        undone = viewer._speed_edit_snapshot()
        np.testing.assert_allclose(undone["speed"], before["speed"])
        np.testing.assert_allclose(undone["xyz"], before["xyz"])

    def test_ctrl_z_y_prioritizes_active_speed_edit(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def __init__(self):
                self.speed_edit_active = True
                self.speed_undo_calls = 0
                self.speed_redo_calls = 0
                self.geometry_undo_calls = 0
                self.geometry_redo_calls = 0

            def _undo_speed_edit(self):
                self.speed_undo_calls += 1
                return True

            def _redo_speed_edit(self):
                self.speed_redo_calls += 1
                return True

            def _undo_saved_trajectory_edit(self):
                self.geometry_undo_calls += 1
                return True

            def _redo_saved_trajectory_edit(self):
                self.geometry_redo_calls += 1
                return True

        viewer = Viewer()

        self.assertEqual(viewer._on_undo_saved_trajectory_edit_key(), "break")
        self.assertEqual(viewer._on_redo_saved_trajectory_edit_key(), "break")
        self.assertEqual(viewer.speed_undo_calls, 1)
        self.assertEqual(viewer.speed_redo_calls, 1)
        self.assertEqual(viewer.geometry_undo_calls, 0)
        self.assertEqual(viewer.geometry_redo_calls, 0)

    def test_saved_trajectory_edit_undo_redo_restores_drag_snapshots(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def __init__(self):
                self.current_traj_idx = 0
                self.traj_geom_edit_active = True
                self.traj_geom_edit_dirty = False
                self.traj_geom_edit_traj_idx = 0
                self.traj_geom_edit_undo_stack = []
                self.traj_geom_edit_redo_stack = []
                self.update_count = 0
                self.refresh_count = 0
                base = np.column_stack([
                    np.linspace(0.25, 16.0, 64),
                    np.zeros(64),
                    np.zeros(64),
                ])
                self.trajectories = [trajectory_components_from_xyz(base)]

            def _update_display(self):
                self.update_count += 1

            def _refresh_trajectory_smoothness(self):
                self.refresh_count += 1

        viewer = Viewer()
        before = viewer._selected_saved_traj_xyz().copy()
        after = before.copy()
        after[16, 1] = 2.0
        viewer._apply_saved_trajectory_edit_xyz_snapshot(after)
        viewer._record_saved_trajectory_edit_snapshot(before, after)

        self.assertTrue(viewer._undo_saved_trajectory_edit())
        np.testing.assert_allclose(viewer._selected_saved_traj_xyz(), before)

        self.assertTrue(viewer._redo_saved_trajectory_edit())
        np.testing.assert_allclose(viewer._selected_saved_traj_xyz(), after)
        self.assertEqual(len(viewer.traj_geom_edit_undo_stack), 1)
        self.assertEqual(len(viewer.traj_geom_edit_redo_stack), 0)
        self.assertGreaterEqual(viewer.update_count, 2)
        self.assertGreaterEqual(viewer.refresh_count, 2)

    def test_saved_trajectory_drag_records_single_undo_step_on_release(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def __init__(self):
                self.current_traj_idx = 0
                self.traj_geom_edit_active = True
                self.traj_geom_edit_dirty = False
                self.traj_geom_edit_traj_idx = 0
                self.traj_geom_edit_undo_stack = []
                self.traj_geom_edit_redo_stack = []
                base = np.column_stack([
                    np.linspace(0.25, 16.0, 64),
                    np.zeros(64),
                    np.zeros(64),
                ])
                self.trajectories = [trajectory_components_from_xyz(base)]
                self.drag_state = {
                    "type": "saved_traj_keyframe",
                    "traj_idx": 0,
                    "frame_idx": 16,
                    "base_xyz": base.copy(),
                }

            def _canvas_to_world(self, canvas_x, canvas_y):
                return float(canvas_x), float(canvas_y)

            def _update_display(self):
                pass

            def _refresh_trajectory_smoothness(self):
                pass

        viewer = Viewer()
        before = viewer.drag_state["base_xyz"].copy()

        self.assertTrue(viewer._drag_saved_trajectory_keyframe(4.0, 2.0))
        self.assertTrue(viewer._finish_saved_trajectory_keyframe_drag())
        self.assertEqual(len(viewer.traj_geom_edit_undo_stack), 1)
        np.testing.assert_allclose(viewer.traj_geom_edit_undo_stack[0], before)

        self.assertTrue(viewer._undo_saved_trajectory_edit())
        np.testing.assert_allclose(viewer._selected_saved_traj_xyz(), before)

    def test_speed_edit_uses_broader_local_influence_window(self):
        class Viewer(SpeedControlsMixin):
            pass

        viewer = Viewer()

        self.assertEqual(SPEED_EDIT_LOCAL_RADIUS_FRAMES, 48)
        self.assertEqual(viewer._speed_edit_local_radius(64), 48)

    def test_speed_edit_postprocess_smooths_dragged_speed_profile(self):
        class Viewer(SpeedControlsMixin):
            pass

        viewer = Viewer()
        speed = np.full(64, 4.0, dtype=np.float64)
        speed[32] = 12.0

        smoothed = viewer._postprocess_edited_speed_profile(speed)

        self.assertEqual(len(smoothed), len(speed))
        self.assertLess(float(smoothed[32]), 12.0)
        self.assertGreater(float(smoothed[31]), 4.0)
        self.assertGreater(float(smoothed[33]), 4.0)
        self.assertTrue(np.all(np.isfinite(smoothed)))

    def test_speed_edit_postprocess_anchors_future_speed_to_t0_history_speed(self):
        class Viewer(SpeedControlsMixin):
            def _speed_edit_initial_speed_mps(self):
                return 8.0

        viewer = Viewer()
        speed = np.full(64, 2.0, dtype=np.float64)
        speed[0] = 0.0

        smoothed = viewer._postprocess_edited_speed_profile(speed)
        first_accel = (smoothed[0] - 8.0) / TRAJ_DT_SECONDS

        self.assertGreaterEqual(first_accel, TRAJ_ACCEL_MIN_MPS2 - 1e-9)
        self.assertLessEqual(first_accel, TRAJ_ACCEL_MAX_MPS2 + 1e-9)

    def test_speed_edit_resampled_trajectory_respects_t0_history_speed(self):
        class Viewer(SpeedControlsMixin):
            def __init__(self):
                self.speed_edit_original_xyz = np.column_stack([
                    np.linspace(0.8, 22.0, 64),
                    np.zeros(64),
                    np.zeros(64),
                ])
                self.speed_edit_speed = np.full(64, 2.0, dtype=np.float64)
                self.speed_edit_traj_idx = 0
                components = trajectory_components_from_xyz(self.speed_edit_original_xyz)
                self.trajectories = [{"sample_idx": 0, "source": "vla", **components}]
                self.trajectory_smoothness = {}

            def _speed_edit_initial_speed_mps(self):
                return 8.0

            def _refresh_trajectory_smoothness(self):
                pass

        viewer = Viewer()

        self.assertTrue(viewer._apply_speed_edit_to_trajectory())
        edited_xyz = np.column_stack([
            viewer.trajectories[0]["x"],
            viewer.trajectories[0]["y"],
            viewer.trajectories[0]["z"],
        ])
        future_speed0 = float(np.linalg.norm(edited_xyz[0, :2]) / TRAJ_DT_SECONDS)
        first_accel = (future_speed0 - 8.0) / TRAJ_DT_SECONDS

        self.assertGreaterEqual(first_accel, TRAJ_ACCEL_MIN_MPS2 - 0.2)
        self.assertLessEqual(first_accel, TRAJ_ACCEL_MAX_MPS2 + 0.2)
        np.testing.assert_allclose(edited_xyz[-1], viewer.speed_edit_original_xyz[-1], atol=1e-6)

    def test_saved_trajectory_geometry_edit_deforms_locally_and_preserves_start(self):
        xyz = np.column_stack(
            [
                np.linspace(0.05, 12.0, 64),
                np.zeros(64),
                np.zeros(64),
            ]
        )

        edited = deform_trajectory_by_keyframe_drag(
            xyz,
            frame_idx=63,
            target_xy=(12.0, 4.0),
            influence_radius_frames=12,
        )

        self.assertEqual(edited.shape, xyz.shape)
        np.testing.assert_allclose(edited[0], xyz[0])
        np.testing.assert_allclose(edited[-1, :2], np.array([12.0, 4.0]))
        self.assertGreater(float(edited[50, 1]), 0.0)
        self.assertLess(float(edited[20, 1]), float(edited[50, 1]))

    def test_saved_trajectory_edit_mixin_drag_updates_components_and_cancel_restores(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def _is_gt_trajectory(self, traj, fallback_index=-1):
                return False

            def _is_traj_pending_deleted(self, traj_idx):
                return False

        xyz = np.column_stack(
            [
                np.linspace(0.05, 12.0, 64),
                np.zeros(64),
                np.zeros(64),
            ]
        )
        components = trajectory_components_from_xyz(xyz)
        viewer = Viewer()
        viewer.trajectories = [{"sample_idx": 0, "source": "vla", **components}]
        viewer.current_traj_idx = 0
        viewer.gt_only = False
        viewer.speed_edit_active = False
        viewer.traj_geom_edit_active = False
        viewer.traj_geom_edit_dirty = False
        viewer.traj_geom_edit_traj_idx = None
        viewer.traj_geom_edit_original_traj = None
        viewer.traj_geom_edit_original_xyz = None

        self.assertTrue(viewer._start_saved_trajectory_edit(redraw=False))
        self.assertTrue(viewer._apply_saved_trajectory_keyframe_drag(63, (12.0, 4.0)))
        edited = viewer.trajectories[0]

        self.assertTrue(viewer.traj_geom_edit_dirty)
        self.assertGreater(float(edited["y"][-1]), 0.0)
        for key in ("vx", "vy", "qz", "qw", "curvature"):
            self.assertTrue(np.all(np.isfinite(edited[key])))

        viewer._cancel_saved_trajectory_edit(redraw=False)

        self.assertFalse(viewer.traj_geom_edit_active)
        np.testing.assert_allclose(viewer.trajectories[0]["y"], components["y"])

    def test_saved_trajectory_edit_mixin_rejects_second_edit_until_resolved(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def _is_gt_trajectory(self, traj, fallback_index=-1):
                return False

            def _is_traj_pending_deleted(self, traj_idx):
                return False

        xyz = np.column_stack(
            [
                np.linspace(0.05, 12.0, 64),
                np.zeros(64),
                np.zeros(64),
            ]
        )
        components = trajectory_components_from_xyz(xyz)
        viewer = Viewer()
        viewer.trajectories = [
            {"sample_idx": 0, "source": "vla", **components},
            {"sample_idx": 1, "source": "manual_bezier", **components},
        ]
        viewer.current_traj_idx = 0
        viewer.gt_only = False
        viewer.speed_edit_active = False
        viewer.traj_geom_edit_active = False

        self.assertTrue(viewer._start_saved_trajectory_edit(redraw=False))
        viewer.current_traj_idx = 1

        self.assertFalse(viewer._start_saved_trajectory_edit(redraw=False))
        self.assertEqual(viewer.traj_geom_edit_traj_idx, 0)

    def test_saved_trajectory_edit_save_writes_row_and_resets_state(self):
        class Viewer(SavedTrajectoryEditingMixin):
            def _write_selected_trajectory_to_parquet(self, traj_idx):
                self.saved_traj_idx = int(traj_idx)
                return True

            def _load_sample(self, current_idx):
                self.loaded_current_idx = int(current_idx)

            def _update_display(self):
                self.updated = True

        viewer = Viewer()
        viewer.trajectories = [{"sample_idx": 4, "source": "vla"}]
        viewer.current_idx = 9
        viewer.traj_geom_edit_active = True
        viewer.traj_geom_edit_dirty = True
        viewer.traj_geom_edit_traj_idx = 0
        viewer.traj_geom_edit_original_traj = None
        viewer.traj_geom_edit_original_xyz = None
        viewer.saved_traj_idx = None
        viewer.loaded_current_idx = None
        viewer.updated = False

        viewer._save_saved_trajectory_edit()

        self.assertEqual(viewer.saved_traj_idx, 0)
        self.assertEqual(viewer.loaded_current_idx, 9)
        self.assertTrue(viewer.updated)
        self.assertFalse(viewer.traj_geom_edit_active)
        self.assertFalse(viewer.traj_geom_edit_dirty)
        self.assertIsNone(viewer.traj_geom_edit_traj_idx)

    def test_speed_edit_does_not_start_when_geometry_edit_targets_another_trajectory(self):
        class Viewer(SpeedControlsMixin):
            def _is_traj_pending_deleted(self, traj_idx):
                return False

            def _is_gt_trajectory(self, traj, fallback_index=-1):
                return False

            def _update_display(self):
                self.updated = True

        xyz = np.column_stack(
            [
                np.linspace(0.05, 12.0, 64),
                np.zeros(64),
                np.zeros(64),
            ]
        )
        components = trajectory_components_from_xyz(xyz)
        viewer = Viewer()
        viewer.gt_only = False
        viewer.current_traj_idx = 0
        viewer.trajectories = [{"sample_idx": 0, "source": "vla", **components}]
        viewer.traj_geom_edit_active = True
        viewer.traj_geom_edit_traj_idx = 1
        viewer.speed_edit_active = False
        viewer.speed_edit_original_traj = None
        viewer.speed_edit_traj_idx = None

        viewer._start_speed_edit()

        self.assertFalse(viewer.speed_edit_active)
        self.assertIsNone(viewer.speed_edit_traj_idx)

    def test_trajectory_components_recompute_parquet_fields(self):
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.1, 0.0],
                [0.3, 0.1, 0.0],
            ],
            dtype=np.float64,
        )

        components = trajectory_components_from_xyz(xyz)

        self.assertEqual(
            set(components),
            {"x", "y", "z", "vx", "vy", "vz", "qx", "qy", "qz", "qw", "curvature"},
        )
        for values in components.values():
            self.assertEqual(len(values), len(xyz))
            self.assertTrue(np.all(np.isfinite(values)))
        np.testing.assert_allclose(components["x"], xyz[:, 0])
        np.testing.assert_allclose(components["y"], xyz[:, 1])
        np.testing.assert_allclose(components["z"], xyz[:, 2])


if __name__ == "__main__":
    unittest.main()
