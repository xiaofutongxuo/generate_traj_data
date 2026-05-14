"""DrawCameraMixin for the enhanced trajectory GUI."""

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

class DrawCameraMixin:

    def _draw_camera_images(self):
        """Draw camera images with trajectory projection."""
        if self.conv_data is None:
            for cam in self.cameras:
                self.camera_labels[cam].config(image="")
            return
        
        try:
            frames = self.conv_data["image_frames"]  # [6, T, C, H, W]
            
            for cam_idx, cam in enumerate(self.cameras):
                cam_frame_rgb = self._get_camera_base_image(frames, cam_idx, cam)
                if cam_frame_rgb is None:
                    continue
                
                # Project trajectories onto this camera
                if self.calibration and cam in self.calibration:
                    cam_frame_rgb = self._draw_trajectory_projection(cam_frame_rgb, cam)
                cam_frame_rgb = self._draw_manual_camera_points(cam_frame_rgb, cam)
                
                # Resize for display
                h, w = cam_frame_rgb.shape[:2]
                new_h = self._camera_display_height(cam)
                new_w = int(new_h * w / h)
                self.camera_display_meta[cam] = {
                    "source_width": w,
                    "source_height": h,
                    "display_width": new_w,
                    "display_height": new_h,
                }
                cam_frame_rgb = cv2.resize(cam_frame_rgb, (new_w, new_h))
                
                # Convert to PhotoImage
                image = Image.fromarray(cam_frame_rgb)
                self.camera_labels[cam].image = ImageTk.PhotoImage(image)
                self.camera_labels[cam].config(image=self.camera_labels[cam].image)
                
        except Exception as e:
            print(f"Warning: Could not display images: {e}")
            self.camera_display_meta = {}
            for cam in self.cameras:
                self.camera_labels[cam].config(image="")

    def _get_camera_base_image(self, frames, cam_idx: int, cam: str):
        """Return the cached RGB base image for a camera at calibration size."""
        if cam in self.camera_base_images:
            return self.camera_base_images[cam].copy()

        cam_frame = frames[cam_idx, 0]  # [C, H, W] or [H, W, C]
        cam_frame_rgb = load_image_from_frame(cam_frame)

        if self.calibration and cam in self.calibration:
            calib = self.calibration[cam]
            if (
                cam_frame_rgb.shape[1] != calib.image_width
                or cam_frame_rgb.shape[0] != calib.image_height
            ):
                cam_frame_rgb = cv2.resize(
                    cam_frame_rgb, (calib.image_width, calib.image_height)
                )

        self.camera_base_images[cam] = cam_frame_rgb
        return cam_frame_rgb.copy()

    def _camera_display_height(self, cam_name):
        """Return display height in pixels, giving FC the most screen space."""
        if len(self.cameras) <= 3:
            return 620 if cam_name == "FC" else 270
        return 460 if cam_name == "FC" else 190

    def _draw_manual_camera_points(self, img, cam_name):
        """Draw editable Bezier control handles before display resizing."""
        img = img.copy()
        height, width = img.shape[:2]

        line_points = []
        for ref in self._active_manual_control_refs():
            ego_point = self._manual_control_ref_to_ego(ref)
            if ego_point is None:
                continue
            image_point = self._ego_ground_to_source_image_point(cam_name, ego_point)
            if image_point is None:
                continue
            u, v = image_point
            if 0 <= u < width and 0 <= v < height:
                line_points.append((int(round(u)), int(round(v))))

        for idx, center in enumerate(line_points):
            cv2.circle(img, center, 7, (0, 212, 255), -1)
            cv2.circle(img, center, 7, (0, 0, 0), 2)
            cv2.putText(
                img,
                f"L{idx + 1}",
                (center[0] + 10, center[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 212, 255),
                2,
                cv2.LINE_AA,
            )

        return img

    def _draw_manual_stop_markers_on_image(self, img, cam_name):
        """Project stop markers onto a camera image."""
        if not self.manual_stop_points:
            return img
        if self.calibration is None or cam_name not in self.calibration:
            return img

        base_traj = self._base_manual_bezier_trajectory()
        if base_traj is None or len(base_traj) < 2:
            return img

        calib = self.calibration[cam_name]
        out = img.copy()
        height, width = out.shape[:2]

        for stop in self.manual_stop_points:
            ego_point = _point_at_path_fraction(base_traj, float(stop.get("fraction", 0.0)))
            if ego_point is None:
                continue

            bev_point = ego_to_bev_points(np.asarray(ego_point, dtype=np.float32).reshape(1, 3))
            u, v, z = calib.project_bev_to_image(bev_point)
            visible = calib.is_point_visible(u, v, z)
            if not bool(visible[0]):
                continue

            center = (int(round(u[0])), int(round(v[0])))
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                continue

            cv2.circle(out, center, 6, (255, 92, 92), -1)
            cv2.circle(out, center, 6, (255, 255, 255), 1)

        return out

    def _draw_generated_stop_markers_on_image(self, img, cam_name):
        """Project detected generated-trajectory stop markers onto FC."""
        if cam_name != "FC" or not self.trajectories:
            return img

        out = img.copy()
        height, width = out.shape[:2]
        for i, traj in enumerate(self.trajectories):
            if self._is_traj_pending_deleted(i):
                continue
            speed = self._trajectory_speed_profile(traj)
            stop_segments = _detect_stop_segments(speed)
            if not stop_segments:
                continue
            points_xyz = np.column_stack([traj["x"], traj["y"], traj["z"]])
            if len(points_xyz) == 0:
                continue
            is_selected = i == self.current_traj_idx
            is_kept = self.trajectory_states.get(i, True)
            fill = (255, 32, 32) if is_kept else (138, 32, 32)
            for segment in stop_segments:
                marker_idx = min(int(segment["end"]), len(points_xyz) - 1)
                image_point = self._ego_ground_to_source_image_point(cam_name, points_xyz[marker_idx])
                if image_point is None:
                    continue
                u, v = image_point
                center = (int(round(u)), int(round(v)))
                if not (0 <= center[0] < width and 0 <= center[1] < height):
                    continue
                radius = 7 if is_selected else 6
                cv2.circle(out, center, radius, fill, -1)
                cv2.circle(out, center, radius, (255, 255, 255), 2 if is_selected else 1)
        return out

    def _draw_gt_stop_markers_on_image(self, img, cam_name):
        """Project detected GT stop markers onto FC."""
        if cam_name != "FC":
            return img
        gt = self._get_gt_future_xyz()
        if gt is None or len(gt) == 0:
            return img

        speed = _smoothed_gt_speed_profile_from_xyz(gt)
        stop_segments = _detect_stop_segments(speed)
        if not stop_segments:
            return img

        out = img.copy()
        height, width = out.shape[:2]
        for segment in stop_segments:
            marker_idx = min(int(segment["end"]), len(gt) - 1)
            image_point = self._ego_ground_to_source_image_point(cam_name, gt[marker_idx])
            if image_point is None:
                continue
            u, v = image_point
            center = (int(round(u)), int(round(v)))
            if not (0 <= center[0] < width and 0 <= center[1] < height):
                continue
            cv2.circle(out, center, 6, (255, 32, 32), -1)
            cv2.circle(out, center, 6, (255, 255, 255), 1)
        return out

    def _draw_speed_hover_marker_on_image(self, img, cam_name, source: str):
        """Project the speed-window hover frame onto FC."""
        if (
            cam_name != "FC"
            or self.speed_hover_frame_idx is None
            or self.speed_hover_source != source
        ):
            return img

        points_xyz = self._trajectory_points_for_speed_source(source)
        if points_xyz is None or len(points_xyz) == 0:
            return img

        frame_idx = int(np.clip(self.speed_hover_frame_idx, 0, len(points_xyz) - 1))
        frame_label = self._speed_frame_label_for_source(source, frame_idx)
        image_point = self._ego_ground_to_source_image_point(cam_name, points_xyz[frame_idx])
        if image_point is None:
            return img

        out = img.copy()
        height, width = out.shape[:2]
        u, v = image_point
        center = (int(round(u)), int(round(v)))
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            return out

        radius = 12
        cv2.circle(out, center, radius, HOVER_FRAME_COLOR_RGB, -1)
        cv2.circle(out, center, radius, (0, 0, 0), 2)
        cv2.putText(
            out,
            f"F{frame_label}",
            (center[0] + radius + 5, center[1] + radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            HOVER_FRAME_COLOR_RGB,
            2,
            cv2.LINE_AA,
        )
        return out

    def _draw_trajectory_projection(self, img, cam_name):
        """Draw trajectory projection onto camera image."""
        if self.calibration is None or cam_name not in self.calibration:
            return img
        
        calib = self.calibration[cam_name]

        gt = self._get_gt_future_xyz()
        if gt is not None:
            img = draw_trajectory_on_image(
                img,
                gt,
                calib,
                color=GT_COLOR_RGB,
                thickness=4,
                draw_points=True,
                point_radius=4,
                coordinate_frame="ego",
            )

        manual_preview = self._build_manual_bezier_trajectory()
        if manual_preview is not None:
            img = draw_trajectory_on_image(
                img,
                manual_preview,
                calib,
                color=MANUAL_TRAJ_COLOR_RGB,
                thickness=4,
                draw_points=True,
                point_radius=4,
                coordinate_frame="ego",
            )
            img = self._draw_manual_stop_markers_on_image(img, cam_name)

        if self.cluster_preview_traj is not None:
            img = draw_trajectory_on_image(
                img,
                self.cluster_preview_traj,
                calib,
                color=CLUSTER_TRAJ_COLOR_RGB,
                thickness=5,
                draw_points=True,
                point_radius=5,
                coordinate_frame="ego",
            )
        
        # Draw each trajectory
        for i, traj in enumerate(self.trajectories):
            if self._is_traj_pending_deleted(i):
                continue
            if self._is_gt_trajectory(traj, i):
                continue

            is_selected = (i == self.current_traj_idx)
            is_kept = self.trajectory_states.get(i, True)
            style = self._trajectory_draw_style(i, is_selected, is_kept)
            
            # Get trajectory points in ego frame
            points_xyz = np.column_stack([
                traj["x"], traj["y"], traj["z"]
            ])
            
            img = draw_trajectory_on_image(
                img,
                points_xyz,
                calib,
                color=style["rgb"],
                thickness=style["camera_width"],
                draw_points=style["draw_points"],
                point_radius=style["point_radius"],
                coordinate_frame="ego",
            )

        img = self._draw_gt_stop_markers_on_image(img, cam_name)
        img = self._draw_generated_stop_markers_on_image(img, cam_name)
        img = self._draw_speed_hover_marker_on_image(img, cam_name, "history")
        img = self._draw_speed_hover_marker_on_image(img, cam_name, "pred")
        img = self._draw_speed_hover_marker_on_image(img, cam_name, "gt")
        
        return img

    def _get_default_extrinsics(self, cam_name):
        """Get default camera extrinsics based on typical vehicle camera setup.
        
        Ego frame: x=forward, y=left, z=up
        Camera frame: x=right, y=down, z=forward
        """
        # Camera mounting positions relative to ego center
        # [x_forward, y_left, z_up]
        cam_positions = {
            'FC': (0.0,  0.0,  1.5),   # Front Center
            'FL': (0.3,  0.5,  1.5),   # Front Left
            'FR': (0.3, -0.5,  1.5),   # Front Right
            'RC': (0.0,  0.0,  1.2),   # Rear Center
            'RL': (-0.3, 0.5,  1.5),   # Rear Left
            'RR': (-0.3,-0.5,  1.5),   # Rear Right
        }
        
        if cam_name not in cam_positions:
            return None
        
        tx, ty, tz = cam_positions[cam_name]
        
        # Rotation from ego to camera
        # ego_x=forward -> camera_z=forward
        # ego_y=left -> camera_x=-left
        # ego_z=up -> camera_y=-up
        # So camera = [[0,0,1], [0,-1,0], [1,0,0]] @ ego
        R_cam_ego = np.array([
            [ 0,  0, 1],
            [ 0, -1, 0],
            [ 1,  0, 0]
        ])
        
        return {
            'rotation': R_cam_ego.flatten().tolist(),
            'translation': [tx, ty, tz]
        }
