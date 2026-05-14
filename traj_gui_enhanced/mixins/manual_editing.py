"""ManualEditingMixin for the enhanced trajectory GUI."""

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
from ..dynamics import optimize_pseudo_gt_trajectory, trajectory_components_from_xyz

class ManualEditingMixin:

    def _manual_line_points_to_canvas(self) -> list[float]:
        points = []
        for point in self.manual_line_points:
            px, py = self._world_to_canvas(float(point["x"]), float(point["y"]))
            points.extend([px, py])
        return points

    def _camera_curve_control_indices(self) -> list[int]:
        """Return camera controls currently used to build the editable curve."""
        fc_indices = [
            idx for idx, point in enumerate(self.manual_camera_line_points)
            if point.get("camera") == "FC"
        ]
        if len(fc_indices) >= 2:
            return fc_indices
        return []

    def _active_manual_control_refs(self) -> list[dict]:
        """Return the control handles that define the current manual curve."""
        camera_indices = self._camera_curve_control_indices()
        if camera_indices:
            return [
                {"type": "camera_line", "index": index}
                for index in camera_indices
            ]
        return [
            {"type": "line", "index": index}
            for index in range(len(self.manual_line_points))
        ]

    def _manual_control_ref_to_ego(self, ref: dict):
        """Resolve a control handle to an ego-ground point."""
        point_type = ref.get("type")
        index = int(ref.get("index", -1))
        if point_type == "line":
            if not (0 <= index < len(self.manual_line_points)):
                return None
            point = self.manual_line_points[index]
            return (
                float(point["x"]),
                float(point["y"]),
                float(point.get("z", 0.0)),
            )

        if point_type == "camera_line":
            if not (0 <= index < len(self.manual_camera_line_points)):
                return None
            point = self.manual_camera_line_points[index]
            return self._camera_image_point_to_ego_ground(
                point.get("camera", ""),
                float(point["u"]),
                float(point["v"]),
            )
        return None

    def _ego_ground_to_source_image_point(self, cam_name: str, ego_point):
        """Project an ego-ground point to source image coordinates."""
        if self.calibration is None or cam_name not in self.calibration:
            return None

        calib = self.calibration[cam_name]
        point = np.asarray(ego_point, dtype=np.float32).reshape(1, 3)
        bev_point = ego_to_bev_points(point)
        u, v, z = calib.project_bev_to_image(bev_point)
        visible = calib.is_point_visible(u, v, z)
        if not bool(visible[0]):
            return None
        return float(u[0]), float(v[0])

    def _camera_image_point_to_ego_ground(self, cam_name: str, u: float, v: float):
        """Back-project an image point to the ego ground plane using calibration."""
        if self.calibration is None or cam_name not in self.calibration:
            return None

        calib = self.calibration[cam_name]
        camera_matrix = np.array(
            [
                [calib.fx, 0.0, calib.cx],
                [0.0, calib.fy, calib.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        image_point = np.array([[[float(u), float(v)]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(
            image_point,
            camera_matrix,
            calib.distortion_coeffs.astype(np.float64),
        )
        ray_cam = np.array(
            [undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0],
            dtype=np.float64,
        )

        transform = np.asarray(calib.T_bev_to_camera, dtype=np.float64)
        rotation = transform[:, :3]
        translation = transform[:, 3]
        try:
            camera_to_bev = np.linalg.inv(rotation)
        except np.linalg.LinAlgError:
            return None

        camera_center_bev = -camera_to_bev @ translation
        ray_bev = camera_to_bev @ ray_cam
        if abs(ray_bev[2]) < 1e-9:
            return None

        scale = -camera_center_bev[2] / ray_bev[2]
        if scale <= 0:
            return None

        point_bev = camera_center_bev + scale * ray_bev
        ego_x = float(point_bev[1])
        ego_y = float(-point_bev[0])
        return ego_x, ego_y, 0.0

    def _manual_camera_line_points_to_canvas(self) -> list[float]:
        points = []
        for point in self.manual_camera_line_points:
            ego_point = self._camera_image_point_to_ego_ground(
                point.get("camera", ""),
                float(point["u"]),
                float(point["v"]),
            )
            if ego_point is None:
                continue
            px, py = self._world_to_canvas(ego_point[0], ego_point[1])
            points.extend([px, py])
        return points

    def _add_manual_line_point(self, canvas_x: float, canvas_y: float):
        wx, wy = self._canvas_to_world(canvas_x, canvas_y)
        self.manual_line_points.append({
            "x": round(float(wx), 3),
            "y": round(float(wy), 3),
            "z": 0.0,
        })
        self.manual_point_actions.append(("line", len(self.manual_line_points) - 1))
        self.manual_line_points_dirty = True
        self._update_display()

    def _display_to_source_image_point(self, cam_name: str, display_x: float, display_y: float):
        meta = self.camera_display_meta.get(cam_name)
        if not meta:
            return None

        display_w = max(float(meta["display_width"]), 1.0)
        display_h = max(float(meta["display_height"]), 1.0)
        u = display_x * float(meta["source_width"]) / display_w
        v = display_y * float(meta["source_height"]) / display_h
        return u, v

    def _add_manual_camera_line_point(self, cam_name: str, display_x: float, display_y: float):
        source_point = self._display_to_source_image_point(cam_name, display_x, display_y)
        if source_point is None:
            return
        u, v = source_point

        self.manual_camera_line_points.append({
            "camera": cam_name,
            "u": round(float(u), 2),
            "v": round(float(v), 2),
        })
        self.manual_point_actions.append(("camera_line", len(self.manual_camera_line_points) - 1))
        self.manual_camera_line_points_dirty = True
        self._update_display()

    def _base_manual_bezier_trajectory(self) -> Optional[np.ndarray]:
        return self._build_manual_bezier_trajectory(include_stops=False)

    def _add_manual_stop_point_from_ego(self, ego_x: float, ego_y: float):
        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            messagebox.showwarning(
                "No Bezier Path",
                "Draw at least two Bezier control points before adding a stop marker.",
            )
            return

        fraction = _nearest_path_fraction(base_traj, (ego_x, ego_y))
        if fraction is None:
            return
        self._update_stop_duration_seconds()
        stop = {
            "fraction": round(float(fraction), 6),
            "duration_s": round(float(self.stop_duration_seconds), 2),
        }
        self.manual_stop_points.append(stop)
        self.manual_point_actions.append(("stop", len(self.manual_stop_points) - 1))
        self.manual_stop_points_dirty = True
        self._update_display()

    def _add_final_stop_point(self):
        """Add one stop marker at the final point of the current Bezier path."""
        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            messagebox.showwarning(
                "No Bezier Path",
                "Draw at least two Bezier control points before adding a final stop.",
            )
            return

        self._update_stop_duration_seconds()
        stop = {
            "fraction": 1.0,
            "duration_s": round(float(self.stop_duration_seconds), 2),
        }
        candidate = self._build_manual_bezier_trajectory(
            include_stops=True,
            stop_points=[stop],
        )
        if candidate is None:
            messagebox.showwarning(
                "Invalid Stop",
                "Cannot fit this final stop within the speed and acceleration limits. Try a shorter stop time or a longer path.",
            )
            return

        diagnostics = _trajectory_quality_diagnostics(candidate)
        if not bool(diagnostics.get("ok", False)):
            messagebox.showwarning(
                "Invalid Stop",
                (
                    "Final stop exceeds dynamics limits: "
                    f"acc={len(diagnostics.get('bad_accel_indices', []))}, "
                    f"jump={len(diagnostics.get('jump_indices', []))}."
                ),
            )
            return

        self.manual_stop_points = [stop]
        self.manual_point_actions.append(("stop", 0))
        self.manual_stop_points_dirty = True
        self._update_display()

    def _undo_manual_line_point(self) -> bool:
        while self.manual_point_actions:
            point_type, index = self.manual_point_actions.pop()
            if point_type == "line" and index == len(self.manual_line_points) - 1:
                self.manual_line_points.pop()
                self.manual_line_points_dirty = True
                self._update_display()
                return True
            if (
                point_type == "camera_line"
                and index == len(self.manual_camera_line_points) - 1
            ):
                self.manual_camera_line_points.pop()
                self.manual_camera_line_points_dirty = True
                self._update_display()
                return True
            if point_type == "stop" and index < len(self.manual_stop_points):
                self.manual_stop_points.pop(index)
                self.manual_stop_points_dirty = True
                self._update_display()
                return True

        if self.manual_camera_line_points:
            self.manual_camera_line_points.pop()
            self.manual_camera_line_points_dirty = True
            self._update_display()
            return True
        if self.manual_line_points:
            self.manual_line_points.pop()
            self.manual_line_points_dirty = True
            self._update_display()
            return True
        if self.manual_stop_points:
            self.manual_stop_points.pop()
            self.manual_stop_points_dirty = True
            self._update_display()
            return True
        return False

    def _undo_manual_point(self):
        self._undo_manual_line_point()

    def _save_manual_points(self):
        self._persist_current_manual_points()
        self._update_display()

        messagebox.showinfo(
            "Saved Controls",
            (
                f"Saved {len(self.manual_line_points)} BEV controls and "
                f"{len(self.manual_camera_line_points)} image controls and "
                f"{len(self.manual_stop_points)} stop markers to {self.manual_points_file}"
            ),
        )

    def _persist_current_manual_points(self):
        key = self._current_manual_points_key()
        if self.manual_line_points:
            self.manual_line_points_index[key] = [
                dict(point) for point in self.manual_line_points
            ]
        else:
            self.manual_line_points_index.pop(key, None)
        if self.manual_camera_line_points:
            self.manual_camera_line_points_index[key] = [
                dict(point) for point in self.manual_camera_line_points
            ]
        else:
            self.manual_camera_line_points_index.pop(key, None)
        if self.manual_stop_points:
            self.manual_stop_points_index[key] = [
                dict(point) for point in self.manual_stop_points
            ]
        else:
            self.manual_stop_points_index.pop(key, None)

        self._write_manual_points_index()
        self.manual_line_points_dirty = False
        self.manual_camera_line_points_dirty = False
        self.manual_stop_points_dirty = False

    def _estimate_t0_speed_mps(self) -> float:
        """Estimate current speed from the local history in the current sample."""
        if self.conv_data is None or "ego_history_xyz" not in self.conv_data:
            return 2.0

        hist = self.conv_data["ego_history_xyz"]
        if hasattr(hist, "detach"):
            hist = hist.detach().cpu().numpy()
        hist = np.asarray(hist, dtype=np.float64).reshape(-1, 3)
        mask = self.conv_data.get("ego_history_valid_mask")
        if mask is not None:
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) == len(hist):
                hist = hist[mask]
        if len(hist) < 2:
            return 2.0

        hist = _smooth_history_xyz_for_display(hist)
        tail_delta = hist[-1, :2] - hist[-2, :2]
        distance = float(np.linalg.norm(tail_delta))
        if not np.isfinite(distance):
            return 2.0
        speed = distance / TRAJ_DT_SECONDS
        return float(np.clip(speed, 0.0, 12.0))

    def _manual_curve_ground_points(self) -> list[tuple[float, float, float]]:
        """Return the active manual controls as ego-ground points."""
        ground_points = []
        for ref in self._active_manual_control_refs():
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is not None:
                ground_points.append(ego_point)
        return ground_points

    def _build_manual_bezier_trajectory(
        self,
        include_stops: bool = True,
        stop_points: Optional[list[dict]] = None,
    ) -> Optional[np.ndarray]:
        """Build a smooth 64-step future trajectory from manual control points."""
        clicked_points = self._manual_curve_ground_points()
        if len(clicked_points) < 2:
            return None

        clicked = np.asarray(clicked_points, dtype=np.float64)
        clicked[:, 2] = 0.0

        first_xy = clicked[0, :2]
        first_distance = float(np.linalg.norm(first_xy))
        initial_speed = self._estimate_t0_speed_mps()
        anchor_distance = max(initial_speed * 0.45, first_distance * 0.35, 1.0)
        anchor_distance = min(anchor_distance, max(first_distance * 0.8, 1.0), 6.0)
        start_tangent = np.array([anchor_distance, 0.0, 0.0], dtype=np.float64)

        waypoints = np.vstack([np.zeros((1, 3), dtype=np.float64), clicked])
        dense = _sample_cubic_bezier_chain(
            waypoints,
            samples_per_segment=96,
            start_tangent=start_tangent,
        )
        active_stop_points = self.manual_stop_points if stop_points is None else stop_points
        if include_stops and active_stop_points:
            final_stop_points = [
                stop for stop in active_stop_points
                if float(stop.get("fraction", 0.0)) >= 0.999
            ]
            if len(final_stop_points) == len(active_stop_points):
                duration_s = max(float(stop.get("duration_s", 0.0)) for stop in final_stop_points)
                sampled = _resample_curve_with_final_stop(
                    dense,
                    num_steps=FUTURE_TRAJ_STEPS,
                    initial_speed_mps=initial_speed,
                    dt_seconds=TRAJ_DT_SECONDS,
                    duration_s=duration_s,
                )
                if sampled is None:
                    return None
                return sampled.astype(np.float32)
            sampled = _resample_curve_with_stops(
                dense,
                num_steps=FUTURE_TRAJ_STEPS,
                initial_speed_mps=initial_speed,
                dt_seconds=TRAJ_DT_SECONDS,
                stop_points=active_stop_points,
            )
        else:
            sampled = _resample_curve_by_distance(
                dense,
                num_steps=FUTURE_TRAJ_STEPS,
                initial_speed_mps=initial_speed,
                dt_seconds=TRAJ_DT_SECONDS,
            )
            sampled = _smooth_xy(sampled, passes=2)
        sampled[:, 2] = 0.0
        limited = _acceleration_limited_resample_path(sampled)
        if limited is None:
            return None
        sampled = limited
        return sampled.astype(np.float32)

    def _manual_trajectory_to_row(
        self,
        trajectory_xyz: np.ndarray,
        df: pd.DataFrame,
        source: str = "manual_bezier",
    ) -> dict:
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_xyz = np.asarray(trajectory_xyz, dtype=np.float64)
        optimization = optimize_pseudo_gt_trajectory(traj_xyz)
        traj_xyz = optimization.xyz
        num_steps = len(traj_xyz)
        components = trajectory_components_from_xyz(traj_xyz)

        if "t0_us" in df.columns:
            current_df = df[df["t0_us"].astype("int64") == int(t0_us)]
        else:
            current_df = df
        if "sample_idx" in current_df.columns and len(current_df) > 0:
            sample_idx = int(current_df["sample_idx"].max()) + 1
        else:
            sample_idx = len(current_df)

        return {
            "t0_us": int(t0_us),
            "sample_idx": sample_idx,
            "source": source,
            "timestamp": [
                int(t0_us) + int((i + 1) * TRAJ_DT_SECONDS * 1_000_000)
                for i in range(num_steps)
            ],
            "qx": components["qx"].tolist(),
            "qy": components["qy"].tolist(),
            "qz": components["qz"].tolist(),
            "qw": components["qw"].tolist(),
            "x": components["x"].tolist(),
            "y": components["y"].tolist(),
            "z": components["z"].tolist(),
            "vx": components["vx"].tolist(),
            "vy": components["vy"].tolist(),
            "vz": components["vz"].tolist(),
            "curvature": components["curvature"].tolist(),
        }

    def _save_manual_bezier_trajectory(self):
        if self.gt_only:
            messagebox.showwarning(
                "GT Only",
                "GT-only mode does not load or append generated trajectory parquet files.",
            )
            return
        trajectory_xyz = self._build_manual_bezier_trajectory()
        if trajectory_xyz is None:
            messagebox.showwarning(
                "No Curve",
                "Use Draw Bezier to add at least two control points, and keep the curve within acceleration limits.",
            )
            return

        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        if "source" not in df.columns:
            df["source"] = ""
        row = self._manual_trajectory_to_row(trajectory_xyz, df, source="manual_bezier")

        for column in df.columns:
            if column not in row:
                row[column] = None
        new_row = pd.DataFrame([{column: row[column] for column in df.columns}])
        df_appended = pd.concat([df, new_row], ignore_index=True)

        traj_file.parent.mkdir(parents=True, exist_ok=True)
        df_appended.to_parquet(traj_file, index=False)

        self._persist_current_manual_points()
        self._load_sample(self.current_idx)
        self._update_display()

        messagebox.showinfo(
            "Saved Curve Trajectory",
            (
                f"Appended manual Bezier trajectory sample_idx={row['sample_idx']} "
                f"for t0={int(t0_us)} to {traj_file}"
            ),
        )

    def _save_manual_bezier_as_cluster_center(self):
        trajectory_xyz = self._build_manual_bezier_trajectory()
        if trajectory_xyz is None:
            messagebox.showwarning(
                "No Valid Center",
                "Draw at least two Bezier control points and keep the curve within acceleration limits.",
            )
            return

        diagnostics = _trajectory_quality_diagnostics(trajectory_xyz)
        if not bool(diagnostics.get("ok", False)):
            messagebox.showwarning(
                "Invalid Center",
                (
                    "Bezier center exceeds limits: "
                    f"acc={len(diagnostics.get('bad_accel_indices', []))}, "
                    f"jump={len(diagnostics.get('jump_indices', []))}."
                ),
            )
            return

        category = self.cluster_category_var.get() if self.cluster_category_var is not None else "straight"
        if category not in CLUSTER_CATEGORY_FILES:
            category = "straight"

        records = list(self.cluster_center_library.get(category, []))
        next_id = (max((int(record["id"]) for record in records), default=-1) + 1)
        record = {
            "id": next_id,
            "label": "",
            "trajectory": np.asarray(trajectory_xyz, dtype=np.float32),
            "source": CLUSTER_CATEGORY_FILES[category],
            "count": 1,
            "category": category,
            "is_bezier_added": True,
        }
        final_right = -float(record["trajectory"][-1, 1])
        record["label"] = (
            f"B{next_id:02d} n=1 right={final_right:.1f} "
            f"fwd={float(record['trajectory'][-1, 0]):.1f}"
        )
        records.append(record)
        self.cluster_center_library[category] = records
        self.bezier_cluster_center_ids.setdefault(category, set()).add(int(next_id))
        self._write_cluster_category_file(category, records)
        self._write_bezier_cluster_center_ids()

        if self.cluster_category_var is not None:
            self.cluster_category_var.set(category)
        self._refresh_cluster_choice_values()
        if self.cluster_choice_var is not None:
            self.cluster_choice_var.set(record["label"])
        self.cluster_preview_record = record
        self.cluster_preview_traj = np.asarray(record["trajectory"], dtype=np.float32)
        self.cluster_preview_is_edited = False
        self._update_display()

        messagebox.showinfo(
            "Saved Bezier Center",
            f"Saved Bezier center as {category}/{record['label']} to {self._cluster_category_file(category)}",
        )

    def _remove_manual_point_action(self, point_type: str, removed_index: int):
        updated_actions = []
        for action_type, action_index in self.manual_point_actions:
            if action_type != point_type:
                updated_actions.append((action_type, action_index))
            elif action_index < removed_index:
                updated_actions.append((action_type, action_index))
            elif action_index > removed_index:
                updated_actions.append((action_type, action_index - 1))
        self.manual_point_actions = updated_actions

    def _selected_trajectory_matches_manual_curve(self) -> bool:
        """Return whether the selected trajectory is the current manual curve."""
        if not (0 <= self.current_traj_idx < len(self.trajectories)):
            return False

        manual_curve = self._build_manual_bezier_trajectory()
        if manual_curve is None or len(manual_curve) == 0:
            return False

        traj = self.trajectories[self.current_traj_idx]
        selected_curve = np.column_stack([traj["x"], traj["y"], traj["z"]])
        if len(selected_curve) == 0:
            return False

        count = min(len(selected_curve), len(manual_curve))
        deltas = np.linalg.norm(
            selected_curve[:count, :2] - manual_curve[:count, :2],
            axis=1,
        )
        return bool(float(np.nanmean(deltas)) <= 0.25)

    def _clear_current_manual_points(self):
        """Clear manual controls and stops for the current sample."""
        self.manual_line_points = []
        self.manual_camera_line_points = []
        self.manual_stop_points = []
        self.manual_point_actions = []
        self.manual_line_points_dirty = True
        self.manual_camera_line_points_dirty = True
        self.manual_stop_points_dirty = True
        self._persist_current_manual_points()
