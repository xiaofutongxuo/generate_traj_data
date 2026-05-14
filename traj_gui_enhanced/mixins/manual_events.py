"""ManualEventsMixin for the enhanced trajectory GUI."""

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

class ManualEventsMixin:

    def _nearest_manual_line_point_at_canvas(self, canvas_x: float, canvas_y: float):
        hit_index = None
        hit_distance = None
        for idx, point in enumerate(self.manual_line_points):
            px, py = self._world_to_canvas(float(point["x"]), float(point["y"]))
            distance = ((canvas_x - px) ** 2 + (canvas_y - py) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_index = idx
                hit_distance = distance
        return hit_index

    def _nearest_manual_camera_line_point_at_display(
        self,
        cam_name: str,
        display_x: float,
        display_y: float,
    ):
        meta = self.camera_display_meta.get(cam_name)
        if not meta:
            return None

        display_w = max(float(meta["display_width"]), 1.0)
        display_h = max(float(meta["display_height"]), 1.0)
        source_w = max(float(meta["source_width"]), 1.0)
        source_h = max(float(meta["source_height"]), 1.0)

        hit_index = None
        hit_distance = None
        for idx, point in enumerate(self.manual_camera_line_points):
            if point.get("camera") != cam_name:
                continue
            point_x = float(point["u"]) * display_w / source_w
            point_y = float(point["v"]) * display_h / source_h
            distance = ((display_x - point_x) ** 2 + (display_y - point_y) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_index = idx
                hit_distance = distance
        return hit_index

    def _nearest_active_manual_control_at_canvas(self, canvas_x: float, canvas_y: float):
        """Find the active manual control nearest to a BEV canvas point."""
        hit_ref = None
        hit_distance = None
        for order, ref in enumerate(self._active_manual_control_refs()):
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is None:
                continue
            px, py = self._world_to_canvas(float(ego_point[0]), float(ego_point[1]))
            distance = ((canvas_x - px) ** 2 + (canvas_y - py) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_ref = dict(ref)
                hit_ref["order"] = order
                hit_distance = distance
        return hit_ref

    def _nearest_cluster_endpoint_at_canvas(self, canvas_x: float, canvas_y: float) -> bool:
        """Return whether a canvas point hits the current cluster preview endpoint."""
        if self.cluster_preview_traj is None or len(self.cluster_preview_traj) == 0:
            return False
        end_x, end_y = self._world_to_canvas(
            float(self.cluster_preview_traj[-1, 0]),
            float(self.cluster_preview_traj[-1, 1]),
        )
        distance = ((canvas_x - end_x) ** 2 + (canvas_y - end_y) ** 2) ** 0.5
        return distance <= CLUSTER_ENDPOINT_HIT_RADIUS_PX

    def _nearest_active_manual_control_at_display(
        self,
        cam_name: str,
        display_x: float,
        display_y: float,
    ):
        """Find the active manual control nearest to a camera display point."""
        meta = self.camera_display_meta.get(cam_name)
        if not meta:
            return None

        display_w = max(float(meta["display_width"]), 1.0)
        display_h = max(float(meta["display_height"]), 1.0)
        source_w = max(float(meta["source_width"]), 1.0)
        source_h = max(float(meta["source_height"]), 1.0)

        hit_ref = None
        hit_distance = None
        for order, ref in enumerate(self._active_manual_control_refs()):
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is None:
                continue
            image_point = self._ego_ground_to_source_image_point(cam_name, ego_point)
            if image_point is None:
                continue
            point_x = image_point[0] * display_w / source_w
            point_y = image_point[1] * display_h / source_h
            distance = ((display_x - point_x) ** 2 + (display_y - point_y) ** 2) ** 0.5
            if distance <= 16 and (hit_distance is None or distance < hit_distance):
                hit_ref = dict(ref)
                hit_ref["order"] = order
                hit_distance = distance
        return hit_ref

    def _set_manual_control_ref_from_ego(self, ref: dict, ego_point) -> bool:
        """Move a manual control handle to a new ego-ground point."""
        point_type = ref.get("type")
        index = int(ref.get("index", -1))
        if point_type == "line":
            if not (0 <= index < len(self.manual_line_points)):
                return False
            self.manual_line_points[index] = {
                "x": round(float(ego_point[0]), 3),
                "y": round(float(ego_point[1]), 3),
                "z": 0.0,
            }
            self.manual_line_points_dirty = True
            return True

        if point_type == "camera_line":
            if not (0 <= index < len(self.manual_camera_line_points)):
                return False
            cam_name = self.manual_camera_line_points[index].get("camera", "")
            image_point = self._ego_ground_to_source_image_point(cam_name, ego_point)
            if image_point is None:
                return False
            self.manual_camera_line_points[index] = {
                "camera": cam_name,
                "u": round(float(image_point[0]), 2),
                "v": round(float(image_point[1]), 2),
            }
            self.manual_camera_line_points_dirty = True
            return True
        return False

    def _convert_active_camera_controls_to_bev_controls(self, selected_order: int) -> int:
        """Convert FC image controls into BEV controls so BEV dragging is unconstrained."""
        refs = self._active_manual_control_refs()
        ground_points = []
        for ref in refs:
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is not None:
                ground_points.append(ego_point)

        if len(ground_points) < 2:
            return -1

        self.manual_line_points = [
            {
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "z": 0.0,
            }
            for point in ground_points
        ]
        self.manual_camera_line_points = []
        self.manual_point_actions = [
            ("line", index) for index in range(len(self.manual_line_points))
        ]
        self.manual_line_points_dirty = True
        self.manual_camera_line_points_dirty = True
        return int(np.clip(selected_order, 0, len(self.manual_line_points) - 1))

    def _on_camera_right_down(self, event, cam_name: str):
        hit_ref = self._nearest_active_manual_control_at_display(
            cam_name, event.x, event.y
        )
        if hit_ref is None:
            return
        self.drag_state = {
            "type": "manual_control",
            "camera": cam_name,
            "ref": hit_ref,
        }

    def _on_camera_right_drag(self, event, cam_name: str):
        if not self.drag_state or self.drag_state.get("type") != "manual_control":
            return
        if self.drag_state.get("camera") != cam_name:
            return
        ref = self.drag_state.get("ref", {})
        source_point = self._display_to_source_image_point(cam_name, event.x, event.y)
        if source_point is None:
            return

        if ref.get("type") == "camera_line":
            index = int(ref.get("index", -1))
            if not (0 <= index < len(self.manual_camera_line_points)):
                return
            source_cam = self.manual_camera_line_points[index].get("camera", "")
            if source_cam == cam_name:
                u, v = source_point
                meta = self.camera_display_meta.get(cam_name, {})
                source_w = max(float(meta.get("source_width", 1.0)), 1.0)
                source_h = max(float(meta.get("source_height", 1.0)), 1.0)
                self.manual_camera_line_points[index] = {
                    "camera": cam_name,
                    "u": round(float(np.clip(u, 0.0, source_w - 1.0)), 2),
                    "v": round(float(np.clip(v, 0.0, source_h - 1.0)), 2),
                }
                self.manual_camera_line_points_dirty = True
                self._update_display()
                return

        ego_point = self._camera_image_point_to_ego_ground(
            cam_name, source_point[0], source_point[1]
        )
        if ego_point is None:
            return
        if self._set_manual_control_ref_from_ego(ref, ego_point):
            self._update_display()

    def _on_canvas_right_down(self, event):
        hit_ref = self._nearest_active_manual_control_at_canvas(event.x, event.y)
        if hit_ref is None:
            return
        if hit_ref.get("type") == "camera_line":
            converted_index = self._convert_active_camera_controls_to_bev_controls(
                int(hit_ref.get("order", 0))
            )
            if converted_index < 0:
                return
            hit_ref = {"type": "line", "index": converted_index}
        self.drag_state = {
            "type": "manual_control",
            "ref": hit_ref,
        }

    def _on_canvas_right_drag(self, event):
        if not self.drag_state or self.drag_state.get("type") != "manual_control":
            return
        wx, wy = self._canvas_to_world(event.x, event.y)
        if self._set_manual_control_ref_from_ego(
            self.drag_state.get("ref", {}),
            (wx, wy, 0.0),
        ):
            self._update_display()

    def _on_canvas_left_drag(self, event):
        if self.drag_state and self.drag_state.get("type") == "saved_traj_keyframe":
            self._drag_saved_trajectory_keyframe(event.x, event.y)
            return
        if not self.drag_state or self.drag_state.get("type") != "cluster_endpoint":
            return
        base_traj = self.drag_state.get("base_traj")
        if base_traj is None:
            return
        wx, wy = self._canvas_to_world(event.x, event.y)
        candidate = _cluster_drag_candidate(
            base_traj,
            (wx, wy),
            initial_speed_mps=self._estimate_t0_speed_mps(),
        )
        if candidate is None:
            return
        self.cluster_preview_traj = candidate.astype(np.float32)
        self.cluster_preview_is_edited = True
        self._update_display()

    def _on_left_release(self, _event):
        if self.drag_state and self.drag_state.get("type") in {"cluster_endpoint", "saved_traj_keyframe"}:
            self.drag_state = None

    def _on_right_release(self, _event):
        self.drag_state = None

    def _on_camera_click(self, event, cam_name: str):
        if self.draw_line_enabled:
            self._add_manual_camera_line_point(cam_name, event.x, event.y)
            return

    def _on_canvas_click(self, event):
        """Handle canvas click to select trajectory."""
        if self.draw_line_enabled:
            self._add_manual_line_point(event.x, event.y)
            return

        if getattr(self, "traj_geom_edit_active", False):
            if self._begin_saved_trajectory_keyframe_drag(event.x, event.y):
                return

        if self._nearest_cluster_endpoint_at_canvas(event.x, event.y):
            record = self.cluster_preview_record
            if record is None:
                return
            self.drag_state = {
                "type": "cluster_endpoint",
                "base_traj": np.asarray(record["trajectory"], dtype=np.float64).copy(),
            }
            return

        for i, traj in enumerate(self.trajectories):
            if self._is_traj_pending_deleted(i):
                continue
            if self._is_gt_trajectory(traj, i):
                continue

            x_coords = traj["x"]
            y_coords = traj["y"]
            if len(x_coords) > 0:
                cx = sum(x_coords) / len(x_coords)
                cy = sum(y_coords) / len(y_coords)
                canvas_x, canvas_y = self._world_to_canvas(cx, cy)
                dist = ((event.x - canvas_x)**2 + (event.y - canvas_y)**2)**0.5
                if dist < 50:
                    self.current_traj_idx = i
                    self._update_display()
                    return
