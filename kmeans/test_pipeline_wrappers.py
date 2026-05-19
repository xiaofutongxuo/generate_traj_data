#!/usr/bin/env python3
"""Regression tests for the pipeline wrapper scripts."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def load_module(filename: str):
    if filename == "visualize_acc_curvature_smoke.py":
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is not None and not hasattr(pyplot, "subplots"):
            sys.modules.pop("matplotlib.pyplot", None)
            sys.modules.pop("matplotlib", None)
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExportFiveEightWrapperTest(unittest.TestCase):
    def test_default_args_target_explicit_5_8_output(self) -> None:
        module = load_module("export_5_8_future_txt.py")

        args = module.parse_args([])

        self.assertEqual(args.dataset, "data_26_5_8_converted")
        self.assertEqual(args.output_txt.name, "future_trajectories_5_8_xy.txt")
        self.assertEqual(args.steps, 64)
        self.assertEqual(args.t0_stride, 1)
        self.assertEqual(args.min_forward_acc_mps2, -6.0)
        self.assertEqual(args.max_forward_acc_mps2, 2.0)

    def test_export_future_xy_includes_clip_start_frames(self) -> None:
        module = load_module("kmeans_cluster.py")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            ego_dir = root / "dataset_converted" / "data-egomotion"
            ego_dir.mkdir(parents=True)
            base = {
                "y": [0.0] * 6,
                "z": [0.0] * 6,
                "qx": [0.0] * 6,
                "qy": [0.0] * 6,
                "qz": [0.0] * 6,
                "qw": [1.0] * 6,
                "vx": [2.0] * 6,
                "vy": [0.0] * 6,
                "vz": [0.0] * 6,
                "ax": [0.0] * 6,
                "ay": [0.0] * 6,
                "az": [0.0] * 6,
            }
            module.pd.DataFrame(
                {"timestamp": [100000, 200000, 300000, 400000, 500000, 600000], "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], **base}
            ).to_parquet(ego_dir / "clip_a.egomotion.parquet")
            module.pd.DataFrame(
                {"timestamp": [700000, 800000, 900000, 1000000, 1100000, 1200000], "x": [1.2, 1.4, 1.6, 1.8, 2.0, 2.2], **base}
            ).to_parquet(ego_dir / "clip_b.egomotion.parquet")
            out = root / "future.txt"

            stats = module.export_future_xy_txt(
                root,
                out,
                steps=2,
                history_steps=4,
                t0_stride=1,
                min_speed_mps=0.0,
                min_forward_acc_mps2=-6.0,
                max_forward_acc_mps2=2.0,
                max_step_speed_mps=15.0,
                max_step_acc_mps2=0.0,
                allow_backward=False,
            )

            rows = [
                line.split()
                for line in out.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]

        self.assertEqual(stats.kept, 5)
        self.assertEqual(
            [(parts[2], int(parts[4])) for parts in rows],
            [
                ("clip_a", 3),
                ("clip_b", 0),
                ("clip_b", 1),
                ("clip_b", 2),
                ("clip_b", 3),
            ],
        )


class ClusterFiveEightWrapperTest(unittest.TestCase):
    def test_write_center_file_records_dynamic_center_and_medoid_xy(self) -> None:
        module = load_module("cluster_5_8_500.py")
        rows = [
            module.source.TrajectoryRow(10, "dataset", "clip", 1000, 3, np.ones((2, 2))),
            module.source.TrajectoryRow(11, "dataset", "clip", 1100, 4, np.ones((2, 2)) * 2.0),
        ]
        labels = np.asarray([0, 0], dtype=np.int64)
        centers = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        medoid_indices = np.asarray([1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp_name:
            out_path = Path(tmp_name) / "centers.txt"
            module.write_center_file(
                out_path,
                rows,
                labels,
                centers,
                medoid_indices,
                steps=2,
                feature_stats={"mean": np.zeros(4), "std": np.ones(4)},
                metadata={"n_clusters": 1},
            )
            text = out_path.read_text(encoding="utf-8")

        self.assertIn("# acc_curvature_kmeans_centers_v1", text)
        self.assertIn("# n_clusters=1", text)
        self.assertIn("CENTER 0 2 11", text)
        self.assertIn("CENTER_MEDOID_XY 0 2.000000 2.000000 2.000000 2.000000", text)

    def test_default_args_use_native_curvature_and_expected_smoothing(self) -> None:
        module = load_module("cluster_5_8_500.py")

        args = module.parse_args([])

        self.assertEqual(args.curvature_min_speed_mps, 0.0)
        self.assertEqual(args.speed_smooth_passes, 1)
        self.assertEqual(args.acceleration_smooth_passes, 2)
        self.assertEqual(args.curvature_smooth_passes, 1)
        self.assertEqual(args.n_clusters, 500)


class TargetDiverseWrapperTest(unittest.TestCase):
    def test_default_args_use_acc_curvature_top2_clusters_with_two_members_each(self) -> None:
        module = load_module("generate_target_diverse_gt.py")

        args = module.parse_args([])

        self.assertEqual(args.top_clusters, 2)
        self.assertEqual(args.members_per_cluster, 2)
        self.assertEqual(args.n_clusters, 500)
        self.assertEqual(args.t0_stride, 1)
        self.assertEqual(args.curvature_min_speed_mps, 0.0)
        self.assertEqual(args.output_dir.name, "output")
        self.assertEqual(args.endpoint_constraint_longitudinal_m, 4.0)
        self.assertEqual(args.endpoint_constraint_short_longitudinal_m, 5.0)
        self.assertIn("data_26_3_24_1_converted", args.datasets)
        self.assertIn("data_26_3_25_2_converted", args.datasets)

    def test_target_loader_accepts_longer_compare_horizon_and_truncates_to_steps(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        with tempfile.TemporaryDirectory() as tmp_name:
            path = Path(tmp_name) / "target.txt"
            values = " ".join(str(float(i)) for i in range(6))
            path.write_text(f"7 dataset clip 123 4 {values}\n", encoding="utf-8")

            rows = module.load_target_future_rows(path, steps=2)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].traj_id, 7)
        self.assertEqual(rows[0].xy.shape, (2, 2))
        self.assertEqual(rows[0].xy[-1, -1], 3.0)

    def test_parquet_row_includes_train_data_style_history_when_provided(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        row = module.acc.TrajectoryRow(
            1,
            "dataset",
            "clip",
            1000,
            2,
            np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64),
        )
        history = {
            "timestamp": [800, 900],
            "qx": [0.0, 0.0],
            "qy": [0.0, 0.0],
            "qz": [0.0, 0.0],
            "qw": [1.0, 1.0],
            "x": [-1.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "vx": [1.0, 1.0],
            "vy": [0.0, 0.0],
            "vz": [0.0, 0.0],
            "ax": [0.0, 0.0],
            "ay": [0.0, 0.0],
            "az": [0.0, 0.0],
            "curvature": [0.0, 0.0],
        }

        out = module.trajectory_to_parquet_row(row, row.xy, 0, "gt", history_egomotion=history)

        for name in module.EGOMOTION_COLUMNS:
            self.assertIn(name, out)
            self.assertIn(f"history_{name}", out)
        self.assertEqual(out["history_x"], [-1.0, 0.0])
        self.assertEqual(len(out["ax"]), 2)

    def test_history_egomotion_crosses_clip_boundary_and_derives_dynamics_from_positions(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            ego_dir = root / "dataset" / "data-egomotion"
            ts_dir = root / "dataset" / "data-timestamps"
            ego_dir.mkdir(parents=True)
            ts_dir.mkdir(parents=True)
            common = {
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
                "qx": [0.0, 0.0, 0.0],
                "qy": [0.0, 0.0, 0.0],
                "qz": [0.0, 0.0, 0.0],
                "qw": [1.0, 1.0, 1.0],
                "vx": [100.0, 0.0, 100.0],
                "vy": [0.0, 0.0, 0.0],
                "vz": [0.0, 0.0, 0.0],
                "ax": [1000.0, 0.0, 1000.0],
                "ay": [0.0, 0.0, 0.0],
                "az": [0.0, 0.0, 0.0],
                "curvature": [9.0, 9.0, 9.0],
            }
            module.pd.DataFrame({"timestamp": [100, 200, 300], "x": [0.0, 0.2, 0.4], **common}).to_parquet(
                ego_dir / "clip_a.egomotion.parquet"
            )
            module.pd.DataFrame({"timestamp": [400, 500, 600], "x": [0.6, 0.8, 1.0], **common}).to_parquet(
                ego_dir / "clip_b.egomotion.parquet"
            )
            module.pd.DataFrame({"timestamp": [100, 200, 300]}).to_parquet(ts_dir / "clip_a.timestamps.parquet")
            module.pd.DataFrame({"timestamp": [400, 500, 600]}).to_parquet(ts_dir / "clip_b.timestamps.parquet")
            row = module.acc.TrajectoryRow(
                1,
                "dataset",
                "clip_b",
                400,
                0,
                np.array([[0.2, 0.0], [0.4, 0.0]], dtype=np.float64),
            )

            history = module.load_history_egomotion(row, root, history_steps=4, cache={})

        self.assertIsNotNone(history)
        assert history is not None
        self.assertEqual(history["timestamp"], [100, 200, 300, 400])
        self.assertEqual(history["x"][-1], 0.0)
        self.assertTrue(max(abs(v) for v in history["vx"]) < 5.0)
        self.assertTrue(max(abs(v) for v in history["ax"]) < 20.0)

    def test_history_egomotion_reuses_clip_stem_cache(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            ego_dir = root / "dataset" / "data-egomotion"
            ts_dir = root / "dataset" / "data-timestamps"
            ego_dir.mkdir(parents=True)
            ts_dir.mkdir(parents=True)
            common = {
                "y": [0.0, 0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0, 0.0],
                "qx": [0.0, 0.0, 0.0, 0.0],
                "qy": [0.0, 0.0, 0.0, 0.0],
                "qz": [0.0, 0.0, 0.0, 0.0],
                "qw": [1.0, 1.0, 1.0, 1.0],
                "vx": [0.0, 0.0, 0.0, 0.0],
                "vy": [0.0, 0.0, 0.0, 0.0],
                "vz": [0.0, 0.0, 0.0, 0.0],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [0.0, 0.0, 0.0, 0.0],
                "curvature": [0.0, 0.0, 0.0, 0.0],
            }
            module.pd.DataFrame({"timestamp": [100, 200, 300, 400], "x": [0.0, 0.2, 0.4, 0.6], **common}).to_parquet(
                ego_dir / "clip.egomotion.parquet"
            )
            module.pd.DataFrame({"timestamp": [100, 200, 300, 400]}).to_parquet(ts_dir / "clip.timestamps.parquet")
            rows = [
                module.acc.TrajectoryRow(1, "dataset", "clip", 300, 2, np.ones((2, 2))),
                module.acc.TrajectoryRow(2, "dataset", "clip", 400, 3, np.ones((2, 2))),
            ]
            calls = {"count": 0}
            original = module.dataset_clip_stems_by_time

            def counted(*args, **kwargs):
                calls["count"] += 1
                return original(*args, **kwargs)

            module.dataset_clip_stems_by_time = counted
            try:
                stem_cache = {}
                history_cache = {}
                for row in rows:
                    self.assertIsNotNone(
                        module.load_history_egomotion(
                            row,
                            root,
                            history_steps=2,
                            cache=history_cache,
                            clip_stems_cache=stem_cache,
                        )
                    )
            finally:
                module.dataset_clip_stems_by_time = original

        self.assertEqual(calls["count"], 1)

    def test_endpoint_speed_cache_reuses_nearby_speed_bins(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        acceleration = np.zeros((2, 3), dtype=np.float64)
        curvature = np.zeros((2, 3), dtype=np.float64)
        calls = {"count": 0}
        original = module.acc.integrate_acc_curvature_endpoints_batch

        def counted(_acceleration, _curvature, initial_speed_mps):
            calls["count"] += 1
            return np.full((2, 2), float(initial_speed_mps), dtype=np.float64)

        module.acc.integrate_acc_curvature_endpoints_batch = counted
        try:
            cache = {}
            first, first_speed = module.cached_member_endpoint_xy(
                acceleration,
                curvature,
                initial_speed_mps=3.021,
                endpoint_speed_bin_mps=0.05,
                cache=cache,
            )
            second, second_speed = module.cached_member_endpoint_xy(
                acceleration,
                curvature,
                initial_speed_mps=3.024,
                endpoint_speed_bin_mps=0.05,
                cache=cache,
            )
            third, third_speed = module.cached_member_endpoint_xy(
                acceleration,
                curvature,
                initial_speed_mps=3.080,
                endpoint_speed_bin_mps=0.05,
                cache=cache,
            )
        finally:
            module.acc.integrate_acc_curvature_endpoints_batch = original

        self.assertEqual(calls["count"], 2)
        self.assertEqual(first_speed, second_speed)
        self.assertNotEqual(second_speed, third_speed)
        np.testing.assert_allclose(first, second)

    def test_endpoint_cache_flips_native_curvature_for_ego_local_rollout(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        acceleration = np.zeros((1, 5), dtype=np.float64)
        native_curvature = np.full((1, 5), -0.5, dtype=np.float64)

        endpoints, _speed = module.cached_member_endpoint_xy(
            acceleration,
            native_curvature,
            initial_speed_mps=2.0,
            endpoint_speed_bin_mps=0.0,
            cache={},
        )

        self.assertGreater(float(endpoints[0, 1]), 0.0)

    def test_candidate_rollout_flips_native_curvature_for_ego_local_output(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        acceleration = np.zeros((1, 5), dtype=np.float64)
        native_curvature = np.full((1, 5), -0.5, dtype=np.float64)

        rollouts = module.candidate_rollouts_from_members(
            [0],
            acceleration,
            native_curvature,
            initial_speed_mps=2.0,
        )

        self.assertGreater(float(rollouts[0][-1, 1]), 0.0)

    def test_top_cluster_first_selection_does_not_skip_to_endpoint_feasible_third_cluster(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        args = module.parse_args([])
        args.top_clusters = 2
        args.members_per_cluster = 1
        args.endpoint_constraint_lateral_m = 0.5
        args.endpoint_constraint_longitudinal_m = 0.5
        args.endpoint_speed_bin_mps = 0.05
        features = np.array(
            [
                [0.00, 0.0],
                [0.20, 0.0],
                [5.00, 0.0],
            ],
            dtype=np.float64,
        )
        cluster_centers = features.copy()
        valid_clusters = np.array([0, 1, 2], dtype=np.int64)
        cluster_member_indices = [
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            np.array([2], dtype=np.int64),
        ]
        gt_feature = np.array([0.0, 0.0], dtype=np.float64)
        gt_xy = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        endpoint_cache = {
            1.0: np.array(
                [
                    [20.0, 0.0],
                    [2.0, 0.0],
                    [2.0, 0.0],
                ],
                dtype=np.float64,
            )
        }

        chosen, details, clusters, distances = module.select_candidate_members(
            args,
            gt_feature,
            gt_xy,
            features,
            np.zeros((3, 2), dtype=np.float64),
            np.zeros((3, 2), dtype=np.float64),
            cluster_centers,
            valid_clusters,
            cluster_member_indices,
            gt_speed=1.0,
            endpoint_cache=endpoint_cache,
        )

        self.assertEqual(clusters, [0, 1])
        self.assertEqual(chosen, [1])
        self.assertEqual([detail["cluster"] for detail in details], [1])
        self.assertEqual([round(float(value), 3) for value in distances], [0.0, 0.2])

    def test_top_cluster_member_selection_minimizes_feature_plus_endpoint_cost(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        args = module.parse_args([])
        args.top_clusters = 1
        args.members_per_cluster = 1
        args.endpoint_constraint_lateral_m = 20.0
        args.endpoint_constraint_longitudinal_m = 20.0
        args.endpoint_speed_bin_mps = 0.05
        args.member_endpoint_weight = 1.0
        features = np.array(
            [
                [0.00, 0.0],
                [0.20, 0.0],
            ],
            dtype=np.float64,
        )
        gt_feature = np.array([0.0, 0.0], dtype=np.float64)
        gt_xy = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        endpoint_cache = {
            1.0: np.array(
                [
                    [12.0, 0.0],
                    [2.0, 0.0],
                ],
                dtype=np.float64,
            )
        }

        chosen, details, clusters, _distances = module.select_candidate_members(
            args,
            gt_feature,
            gt_xy,
            features,
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
            np.array([[0.0, 0.0]], dtype=np.float64),
            np.array([0], dtype=np.int64),
            [np.array([0, 1], dtype=np.int64)],
            gt_speed=1.0,
            endpoint_cache=endpoint_cache,
        )

        self.assertEqual(clusters, [0])
        self.assertEqual(chosen, [1])
        self.assertAlmostEqual(float(details[0]["selection_score"]), 0.2)

    def test_first_dataset_clip_start_without_history_keeps_gt_only(self) -> None:
        module = load_module("generate_target_diverse_gt.py")
        row = module.acc.TrajectoryRow(
            1,
            "dataset",
            "first_clip",
            100,
            0,
            np.array([[0.2, 0.0], [0.4, 0.0]], dtype=np.float64),
        )

        self.assertFalse(
            module.should_expand_candidates(
                row,
                history_egomotion=None,
                history_steps=4,
            )
        )
        self.assertTrue(
            module.should_expand_candidates(
                row,
                history_egomotion={"x": [-0.2, 0.0]},
                history_steps=4,
            )
        )

    def test_default_dataset_list_contains_six_datasets(self) -> None:
        module = load_module("generate_target_diverse_gt.py")

        args = module.parse_args([])
        datasets = [item for item in args.datasets.split(",") if item]

        self.assertEqual(len(datasets), 6)
        self.assertIn("data_26_5_8_converted", datasets)


class SmokeVisualizationTest(unittest.TestCase):
    def test_draw_smoke_visualization_writes_png(self) -> None:
        module = load_module("visualize_acc_curvature_smoke.py")
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            row = {
                "t0_us": 1000,
                "sample_idx": 0,
                "source": "gt",
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "z": [0.0, 0.0],
                "history_x": [-1.0, 0.0],
                "history_y": [0.0, 0.0],
            }
            cand = dict(row)
            cand["sample_idx"] = 1
            cand["source"] = "acc_curvature_cluster"
            cand["y"] = [0.0, 0.2]
            module.pd.DataFrame([row, cand]).to_parquet(dataset_dir / "clip.egomotion.parquet", index=False)
            out = root / "smoke.png"

            sample = module.load_first_sample(root)
            module.draw_sample(sample, out)

            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
