import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from frame_index import (
    build_video_frame_t0_candidates,
    load_master_video_timestamps,
    valid_video_frame_indices,
)
from data_loader import filter_t0s_with_full_future
from calibration_loader import load_calibration_for_segment
from traj_gui_enhanced.constants import HISTORY_SPEED_COLOR_HEX
from traj_gui_enhanced.dynamics import (
    DynamicsLimits,
    deform_trajectory_by_keyframe_drag,
    diagnose_trajectory_dynamics,
    editable_trajectory_keyframes,
    optimize_pseudo_gt_trajectory,
    trajectory_components_from_xyz,
)
from traj_gui_enhanced.environment import setup_environment
from traj_gui_enhanced.math_utils import _rgb_to_hex, _trajectory_base_color, _resample_curve_by_distance
from traj_gui_enhanced.mixins.cluster_controls import ClusterControlsMixin
from traj_gui_enhanced.mixins.delete_controls import DeleteControlsMixin
from traj_gui_enhanced.mixins.sample_io import SampleIOMixin
from traj_gui_enhanced.mixins.saved_traj_editing import SavedTrajectoryEditingMixin
from traj_gui_enhanced.mixins.speed_controls import SpeedControlsMixin
from traj_gui_enhanced.speed_utils import (
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
from traj_gui_enhanced.trajectory_identity import (
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

    def test_legacy_entrypoint_is_nonempty_wrapper(self):
        entrypoint = Path(__file__).resolve().parents[1] / "trajectory_gui_enhanced.py"

        text = entrypoint.read_text(encoding="utf-8")

        self.assertIn("traj_gui_enhanced.cli", text)
        self.assertIn("TrajectoryViewerEnhanced", text)
        self.assertGreater(len(text.splitlines()), 10)

    def test_setup_environment_makes_tkinter_importable(self):
        setup_environment()
        import tkinter

        self.assertGreaterEqual(float(tkinter.TkVersion), 8.6)

    def test_cluster_category_file_still_points_to_project_kmeans_dir(self):
        class Viewer(ClusterControlsMixin):
            pass

        path = Viewer()._cluster_category_file("stop")

        self.assertEqual(path.name, "stop.txt")
        self.assertEqual(path.parent.name, "k_means")
        self.assertEqual(path.parent.parent.name, "generate_traj_data")

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
            )

        self.assertEqual(filtered, [100_000])

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

        saved_xyz = np.column_stack([saved["x"], saved["y"], saved["z"]])
        after = diagnose_trajectory_dynamics(saved_xyz).metrics["max_abs_curvature_1pm"]

        self.assertLess(after, before)

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

    def test_speed_edit_does_not_start_while_geometry_edit_active(self):
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
