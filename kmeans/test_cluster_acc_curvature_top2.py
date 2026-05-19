#!/usr/bin/env python3
"""Regression tests for acceleration/curvature clustering helpers."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).with_name("cluster_acc_curvature_top2.py")
TRAIN_DATA_ROOT = Path("/home/ubuntu/Public/train_data")


def load_module():
    """Load the experiment script without optional visualization/cluster deps."""
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))

    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)

    sklearn = types.ModuleType("sklearn")
    cluster = types.ModuleType("sklearn.cluster")
    cluster.KMeans = object
    sys.modules.setdefault("sklearn", sklearn)
    sys.modules.setdefault("sklearn.cluster", cluster)

    spec = importlib.util.spec_from_file_location("cluster_acc_curvature_top2", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class NativeCurvatureSignTest(unittest.TestCase):
    def test_parser_defaults_use_1000_clusters_and_native_curvature_without_speed_gate(self) -> None:
        module = load_module()
        old_argv = sys.argv
        try:
            sys.argv = [str(SCRIPT_PATH)]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.n_clusters, 1000)
        self.assertEqual(args.curvature_min_speed_mps, 0.0)
        self.assertEqual(args.top_k, 2)
        self.assertEqual(args.dataset_filter, "data_26_5_8_converted")
        self.assertEqual(args.candidate_representation, "acc_curvature_nearest_member_rollout")
        self.assertEqual(args.feature_source, "native_parquet")
        self.assertEqual(args.train_data_root, TRAIN_DATA_ROOT)
        self.assertIsNone(args.smooth_passes)
        smoothing = module.resolve_smoothing_config(args)
        self.assertEqual(smoothing.speed_passes, 1)
        self.assertEqual(smoothing.acceleration_passes, 2)
        self.assertEqual(smoothing.curvature_passes, 1)
        self.assertEqual(args.member_endpoint_weight, 0.15)
        self.assertEqual(args.members_per_cluster, 1)
        self.assertEqual(args.endpoint_constraint_mode, "scan_clusters")
        self.assertEqual(args.endpoint_constraint_lateral_m, 2.0)
        self.assertEqual(args.endpoint_constraint_longitudinal_m, 4.0)
        self.assertEqual(args.endpoint_constraint_short_longitudinal_m, 5.0)
        self.assertIsNone(args.match_data_txt)
        self.assertEqual(args.match_dataset_filter, "")

        try:
            sys.argv = [str(SCRIPT_PATH), "--top_k", "20"]
            args = module.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.top_k, 20)

    def test_native_curvature_uses_parquet_values_not_xy_difference(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_dir = root / "dataset" / "data-egomotion"
            parquet_dir.mkdir(parents=True)
            module.pd.DataFrame(
                {
                    "timestamp": [100, 200, 300, 400, 500, 600],
                    "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "y": [0.0] * 6,
                    "z": [0.0] * 6,
                    "curvature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                }
            ).to_parquet(parquet_dir / "clip.egomotion.parquet")
            row = module.TrajectoryRow(
                1,
                "dataset",
                "clip",
                200,
                1,
                np.array([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]], dtype=np.float64),
            )

            (
                kept_rows,
                _features,
                _acceleration,
                native_curvature,
                _raw_acceleration,
                raw_curvature,
                _speed,
                _stats,
            ) = module.build_native_acc_curvature_features(
                [row],
                root,
                steps=3,
                curvature_smooth_passes=0,
            )

        self.assertEqual([item.traj_id for item in kept_rows], [1])
        self.assertTrue(np.allclose(raw_curvature[0], [2.0, 3.0, 4.0]))
        self.assertTrue(np.allclose(native_curvature[0], [2.0, 3.0, 4.0]))

    def test_native_window_uses_t0_timestamp_when_index_is_placeholder(self) -> None:
        module = load_module()
        row = module.TrajectoryRow(
            1,
            "dataset",
            "clip",
            300,
            0,
            np.zeros((2, 2), dtype=np.float64),
        )
        df = module.pd.DataFrame({"timestamp": [100, 200, 300, 400, 500, 600]})

        start = module.native_future_window_start(df, row, steps=2)

        self.assertEqual(start, 3)


class XyDistanceSelectionTest(unittest.TestCase):
    def test_selects_rollouts_by_xy_distance_not_dynamic_distance(self) -> None:
        module = load_module()
        gt_xy = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float64)
        rollouts = [
            np.array([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]], dtype=np.float64),
            np.array([[1.1, 0.0], [2.1, 0.0], [3.1, 0.0]], dtype=np.float64),
            np.array([[1.2, 0.0], [2.2, 0.0], [3.2, 0.0]], dtype=np.float64),
        ]
        dynamic_distances = [0.1, 5.0, 4.0]

        chosen_positions, xy_distances = module.select_top_rollouts_by_xy_distance(
            gt_xy,
            rollouts,
            dynamic_distances,
            top_k=2,
        )

        self.assertEqual(chosen_positions, [1, 2])
        self.assertLess(xy_distances[0], xy_distances[1])

    def test_trajectory_xy_distance_is_mean_per_step_distance(self) -> None:
        module = load_module()

        distance = module.trajectory_xy_distance(
            np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64),
            np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float64),
        )

        self.assertEqual(distance, 7.5)

    def test_candidate_match_label_includes_average_and_endpoint_errors(self) -> None:
        module = load_module()

        label = module.candidate_match_label(
            rank=2,
            cluster_id=7,
            distance_label="dyn_dist",
            distance=1.234,
            average_distance_m=3.456,
            endpoint_distance_m=5.678,
        )

        self.assertEqual(label, "Top2 cluster=7 dyn_dist=1.23 avg=3.5m end=5.7m")

    def test_endpoint_constraint_uses_gt_endpoint_yaw_frame_not_t0_frame(self) -> None:
        module = load_module()
        gt_xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float64)

        inside = module.endpoint_constraint_metrics(
            gt_xy,
            np.array([8.5, 13.5], dtype=np.float64),
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
        )
        outside_lateral = module.endpoint_constraint_metrics(
            gt_xy,
            np.array([7.5, 10.0], dtype=np.float64),
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
        )
        outside_longitudinal = module.endpoint_constraint_metrics(
            gt_xy,
            np.array([10.0, 14.5], dtype=np.float64),
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
        )

        self.assertTrue(inside["constraint_satisfied"])
        self.assertEqual(inside["endpoint_longitudinal_error_m"], 3.5)
        self.assertEqual(inside["endpoint_lateral_error_m"], 1.5)
        self.assertFalse(outside_lateral["constraint_satisfied"])
        self.assertFalse(outside_longitudinal["constraint_satisfied"])

    def test_endpoint_constraint_allows_shorter_longitudinal_margin(self) -> None:
        module = load_module()
        gt_xy = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float64)

        shorter_inside = module.endpoint_constraint_metrics(
            gt_xy,
            np.array([5.5, 0.0], dtype=np.float64),
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
            max_short_longitudinal_m=5.0,
        )
        too_short = module.endpoint_constraint_metrics(
            gt_xy,
            np.array([4.5, 0.0], dtype=np.float64),
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
            max_short_longitudinal_m=5.0,
        )
        too_long = module.endpoint_constraint_metrics(
            gt_xy,
            np.array([14.5, 0.0], dtype=np.float64),
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
            max_short_longitudinal_m=5.0,
        )

        self.assertEqual(shorter_inside["endpoint_longitudinal_error_m"], -4.5)
        self.assertTrue(shorter_inside["constraint_satisfied"])
        self.assertFalse(too_short["constraint_satisfied"])
        self.assertFalse(too_long["constraint_satisfied"])


class CandidateTrajectorySelectionTest(unittest.TestCase):
    def test_uses_medoid_xy_when_requested(self) -> None:
        module = load_module()
        center_rollouts = [
            np.array([[10.0, 0.0], [11.0, 0.0]], dtype=np.float64),
            np.array([[20.0, 0.0], [21.0, 0.0]], dtype=np.float64),
        ]
        medoid_rows = [
            module.TrajectoryRow(1, "d", "c", 0, 0, np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(2, "d", "c", 0, 0, np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float64)),
        ]

        selected = module.candidate_trajectories_for_representation(
            "medoid_xy",
            center_rollouts,
            medoid_rows,
        )

        self.assertTrue(np.array_equal(selected[0], medoid_rows[0].xy))
        self.assertTrue(np.array_equal(selected[1], medoid_rows[1].xy))

    def test_selects_gt_nearest_member_inside_each_chosen_cluster(self) -> None:
        module = load_module()
        rows = [
            module.TrajectoryRow(0, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(1, "d", "c", 0, 0, np.array([[8.0, 0.0], [9.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(2, "d", "c", 0, 0, np.array([[0.1, 0.0], [1.1, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(3, "d", "c", 0, 0, np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(4, "d", "c", 0, 0, np.array([[0.2, 0.0], [1.2, 0.0]], dtype=np.float64)),
        ]
        labels = np.array([0, 1, 1, 2, 2], dtype=np.int64)

        selected = module.choose_gt_nearest_member_indices(
            rows,
            labels,
            chosen_clusters=[1, 2],
            gt_xy=rows[0].xy,
        )

        self.assertEqual(selected, [2, 4])

    def test_selects_feature_nearest_member_inside_each_chosen_cluster(self) -> None:
        module = load_module()
        features = np.array(
            [
                [0.0, 0.0],
                [8.0, 8.0],
                [0.2, 0.1],
                [3.0, 3.0],
                [0.1, 0.2],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 1, 2, 2], dtype=np.int64)

        selected = module.choose_feature_nearest_member_indices(
            features,
            labels,
            chosen_clusters=[1, 2],
            gt_feature=features[0],
        )

        self.assertEqual(selected, [2, 4])

    def test_endpoint_first_selection_skips_infeasible_dynamic_cluster_and_keeps_two_members(self) -> None:
        module = load_module()
        rows = [
            module.TrajectoryRow(i, "d", "c", 0, i, np.zeros((2, 2), dtype=np.float64))
            for i in range(6)
        ]
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        valid_clusters = np.array([0, 1, 2], dtype=np.int64)
        dynamic_distances = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        gt_xy = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        member_endpoint_xy = np.array(
            [
                [20.0, 20.0],
                [21.0, 20.0],
                [2.1, 0.1],
                [2.2, -0.1],
                [2.3, 0.1],
                [2.4, -0.1],
            ],
            dtype=np.float64,
        )
        filled_clusters: list[int] = []

        selected, details, clusters, distances = module.select_endpoint_first_top_cluster_members(
            features=features,
            labels=labels,
            valid_clusters=valid_clusters,
            dynamic_distances=dynamic_distances,
            gt_feature=np.array([0.0, 0.0], dtype=np.float64),
            rows=rows,
            gt_xy=gt_xy,
            member_endpoint_xy=member_endpoint_xy,
            fill_member_rollout_endpoints=lambda cluster_id: filled_clusters.append(int(cluster_id)),
            max_lateral_m=1.0,
            max_longitudinal_m=1.0,
            top_clusters=2,
            members_per_cluster=2,
        )

        self.assertEqual(clusters, [1, 2])
        self.assertEqual(selected, [2, 3, 4, 5])
        self.assertEqual([item["cluster"] for item in details], [1, 1, 2, 2])
        self.assertEqual(distances, [0.2, 0.3])
        self.assertEqual(filled_clusters, [0, 1, 2])

    def test_batch_endpoint_rollout_matches_scalar_integrator(self) -> None:
        module = load_module()
        acceleration = np.array(
            [
                [0.0, 0.2, 0.1],
                [0.0, -0.1, -0.2],
            ],
            dtype=np.float64,
        )
        curvature = np.array(
            [
                [0.0, 0.01, 0.02],
                [0.0, -0.01, -0.02],
            ],
            dtype=np.float64,
        )

        endpoints = module.integrate_acc_curvature_endpoints_batch(
            acceleration,
            curvature,
            initial_speed_mps=3.0,
        )

        expected = np.stack(
            [
                module.integrate_acc_curvature(acceleration[i], curvature[i], 3.0)[-1]
                for i in range(2)
            ],
            axis=0,
        )
        self.assertTrue(np.allclose(endpoints, expected))

    def test_fast_endpoint_first_selection_matches_expected_members(self) -> None:
        module = load_module()
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [1.0, 0.0],
                [1.1, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        cluster_members = module.build_cluster_member_indices(labels, n_clusters=3)
        valid_clusters = np.array([0, 1, 2], dtype=np.int64)
        dynamic_distances = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        gt_xy = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        member_endpoint_xy = np.array(
            [
                [20.0, 20.0],
                [21.0, 20.0],
                [2.1, 0.1],
                [2.2, -0.1],
                [2.3, 0.1],
                [2.4, -0.1],
            ],
            dtype=np.float64,
        )

        selected, details, clusters, distances = module.select_endpoint_first_top_cluster_members_fast(
            features=features,
            cluster_member_indices=cluster_members,
            valid_clusters=valid_clusters,
            dynamic_distances=dynamic_distances,
            gt_feature=np.array([0.0, 0.0], dtype=np.float64),
            gt_xy=gt_xy,
            member_endpoint_xy=member_endpoint_xy,
            max_lateral_m=1.0,
            max_longitudinal_m=1.0,
            top_clusters=2,
            members_per_cluster=2,
        )

        self.assertEqual(clusters, [1, 2])
        self.assertEqual(selected, [2, 3, 4, 5])
        self.assertEqual([item["cluster"] for item in details], [1, 1, 2, 2])
        self.assertEqual(distances, [0.2, 0.3])

    def test_feature_member_selection_can_use_endpoint_tiebreak(self) -> None:
        module = load_module()
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
            ],
            dtype=np.float64,
        )
        rows = [
            module.TrajectoryRow(0, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(1, "d", "c", 0, 0, np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(2, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
        ]
        labels = np.array([0, 1, 1], dtype=np.int64)

        selected, details = module.choose_feature_nearest_member_indices_with_endpoint(
            features,
            labels,
            chosen_clusters=[1],
            gt_feature=features[0],
            rows=rows,
            gt_xy=rows[0].xy,
            endpoint_weight=0.05,
        )

        self.assertEqual(selected, [2])
        self.assertEqual(details[0]["member_index"], 2)
        self.assertEqual(details[0]["endpoint_distance_m"], 0.0)
        self.assertAlmostEqual(details[0]["selection_score"], 0.2)

    def test_feature_member_endpoint_selection_can_use_rollout_endpoints(self) -> None:
        module = load_module()
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
            ],
            dtype=np.float64,
        )
        rows = [
            module.TrajectoryRow(0, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(1, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(2, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
        ]
        endpoint_xy_by_row = np.array(
            [
                [1.0, 0.0],
                [100.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 1], dtype=np.int64)

        selected, details = module.choose_feature_nearest_member_indices_with_endpoint(
            features,
            labels,
            chosen_clusters=[1],
            gt_feature=features[0],
            rows=rows,
            gt_xy=rows[0].xy,
            endpoint_weight=0.05,
            endpoint_xy_by_row=endpoint_xy_by_row,
        )

        self.assertEqual(selected, [2])
        self.assertEqual(details[0]["endpoint_distance_m"], 0.0)

    def test_feature_member_selection_can_return_two_members_per_cluster(self) -> None:
        module = load_module()
        features = np.array(
            [
                [0.0, 0.0],
                [0.3, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
            ],
            dtype=np.float64,
        )
        rows = [
            module.TrajectoryRow(0, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(1, "d", "c", 0, 0, np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(2, "d", "c", 0, 0, np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)),
            module.TrajectoryRow(3, "d", "c", 0, 0, np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)),
        ]
        labels = np.array([0, 1, 1, 1], dtype=np.int64)

        selected, details = module.choose_feature_nearest_member_indices_with_endpoint(
            features,
            labels,
            chosen_clusters=[1],
            gt_feature=features[0],
            rows=rows,
            gt_xy=rows[0].xy,
            endpoint_weight=0.1,
            members_per_cluster=2,
        )

        self.assertEqual(selected, [2, 3])
        self.assertEqual([item["member_rank_in_cluster"] for item in details], [1, 2])

    def test_endpoint_constrained_member_selection_filters_before_ranking(self) -> None:
        module = load_module()
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.2, 0.0],
                [0.3, 0.0],
            ],
            dtype=np.float64,
        )
        rows = [
            module.TrajectoryRow(0, "d", "c", 0, 0, np.zeros((2, 2), dtype=np.float64)),
            module.TrajectoryRow(1, "d", "c", 0, 0, np.zeros((2, 2), dtype=np.float64)),
            module.TrajectoryRow(2, "d", "c", 0, 0, np.zeros((2, 2), dtype=np.float64)),
            module.TrajectoryRow(3, "d", "c", 0, 0, np.zeros((2, 2), dtype=np.float64)),
        ]
        labels = np.array([0, 1, 1, 1], dtype=np.int64)
        gt_xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float64)
        endpoint_xy_by_row = np.array(
            [
                [10.0, 10.0],
                [10.0, 20.0],
                [8.0, 13.0],
                [9.0, 12.0],
            ],
            dtype=np.float64,
        )

        selected, details = module.choose_endpoint_constrained_feature_members(
            features,
            labels,
            chosen_clusters=[1],
            gt_feature=features[0],
            rows=rows,
            gt_xy=gt_xy,
            endpoint_xy_by_row=endpoint_xy_by_row,
            max_lateral_m=2.0,
            max_longitudinal_m=4.0,
            members_per_cluster=2,
        )

        self.assertEqual(selected, [2, 3])
        self.assertTrue(all(item["endpoint_constraint_satisfied"] for item in details))


class DatasetFilterTest(unittest.TestCase):
    def test_filters_rows_to_requested_dataset(self) -> None:
        module = load_module()
        rows = [
            module.TrajectoryRow(1, "data_26_5_8_converted", "a", 0, 0, np.zeros((2, 2))),
            module.TrajectoryRow(2, "other_dataset", "b", 0, 0, np.zeros((2, 2))),
            module.TrajectoryRow(3, "data_26_5_8_converted", "c", 0, 0, np.zeros((2, 2))),
        ]

        filtered = module.filter_rows_by_dataset(rows, "data_26_5_8_converted")

        self.assertEqual([row.traj_id for row in filtered], [1, 3])


class SampleSelectionTest(unittest.TestCase):
    def test_selects_sample_by_traj_id_or_filtered_index(self) -> None:
        module = load_module()
        rows = [
            module.TrajectoryRow(10, "d", "c", 0, 0, np.zeros((2, 2))),
            module.TrajectoryRow(20, "d", "c", 0, 0, np.zeros((2, 2))),
        ]

        self.assertEqual(module.select_sample_index(rows, 20), 1)
        self.assertEqual(module.select_sample_index(rows, 0), 0)

        with self.assertRaises(ValueError):
            module.select_sample_index(rows, 99)


class SpeedProfileTest(unittest.TestCase):
    def test_speed_profile_from_xy_uses_origin_to_first_point(self) -> None:
        module = load_module()
        xy = np.array(
            [
                [0.3, 0.4],
                [0.9, 1.2],
                [0.9, 1.2],
            ],
            dtype=np.float64,
        )

        speed = module.speed_profile_from_xy(xy)

        self.assertTrue(np.allclose(speed, [5.0, 10.0, 0.0]))

    def test_t0_average_speed_uses_timestamp_when_index_is_placeholder(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_dir = root / "dataset" / "data-egomotion"
            parquet_dir.mkdir(parents=True)
            module.pd.DataFrame(
                {
                    "timestamp": [100, 200, 300, 400, 500],
                    "x": [0.0, 0.2, 0.4, 0.6, 0.8],
                    "y": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "z": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "vx": [1.0, 10.0, 100.0, 30.0, 40.0],
                    "vy": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "vz": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "curvature": [0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ).to_parquet(parquet_dir / "clip.egomotion.parquet")
            row = module.TrajectoryRow(
                1,
                "dataset",
                "clip",
                300,
                0,
                np.zeros((2, 2), dtype=np.float64),
            )

            speed = module.t0_average_speed(
                row,
                root,
                root / "missing-output",
                speed_window=1,
                cache={},
            )

        self.assertAlmostEqual(speed, 2.0)


class NativeAccelerationSmoothingTest(unittest.TestCase):
    def test_native_features_derive_speed_from_positions_not_velocity_columns(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet_dir = root / "dataset" / "data-egomotion"
            parquet_dir.mkdir(parents=True)
            module.pd.DataFrame(
                {
                    "timestamp": [100, 200, 300, 400, 500, 600],
                    "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "vx": [0.0, 100.0, 0.0, 100.0, 0.0, 100.0],
                    "vy": [0.0] * 6,
                    "vz": [0.0] * 6,
                    "curvature": [0.0] * 6,
                }
            ).to_parquet(parquet_dir / "clip.egomotion.parquet")
            row = module.TrajectoryRow(
                1,
                "dataset",
                "clip",
                200,
                1,
                np.array([[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]], dtype=np.float64),
            )

            (
                kept_rows,
                _features,
                acceleration,
                curvature,
                _raw_acceleration,
                _raw_curvature,
                speed,
                _stats,
            ) = module.build_native_acc_curvature_features(
                [row],
                root,
                steps=3,
                speed_smooth_passes=1,
                acceleration_smooth_passes=1,
                curvature_smooth_passes=1,
            )

        self.assertEqual([item.traj_id for item in kept_rows], [1])
        self.assertTrue(np.all(speed[0] < 5.0))
        self.assertTrue(np.all(np.abs(acceleration[0]) < 10.0))
        self.assertTrue(np.allclose(curvature[0], 0.0))

    def test_smooths_speed_before_diff_then_smooths_acceleration(self) -> None:
        module = load_module()
        speed = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        smoothed_speed = module.smooth_trace(speed, passes=1)
        expected_raw_acceleration = np.zeros_like(smoothed_speed)
        expected_raw_acceleration[1:] = np.diff(smoothed_speed) / module.DT_SECONDS
        expected_acceleration = module.smooth_trace(expected_raw_acceleration, passes=1)

        raw_acceleration, acceleration = module.acceleration_from_smoothed_speed(
            speed,
            smooth_passes=1,
        )

        self.assertTrue(np.allclose(raw_acceleration, expected_raw_acceleration))
        self.assertTrue(np.allclose(acceleration, expected_acceleration))


class SpeedVisualizationTest(unittest.TestCase):
    def test_gt_speed_visualization_profile_is_smoothed(self) -> None:
        module = load_module()
        raw_speed = np.array([0.0, 0.0, 8.0, 8.0, 8.0], dtype=np.float64)

        profile = module.gt_speed_profile_for_visualization(raw_speed, smooth_passes=1)

        self.assertTrue(np.allclose(profile, module.smooth_trace(raw_speed, passes=1)))


if __name__ == "__main__":
    unittest.main()
