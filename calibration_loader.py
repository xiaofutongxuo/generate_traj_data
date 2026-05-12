# SPDX-License-Identifier: Apache-2.0
"""Load camera calibration parameters for trajectory projection.

This module loads the camera extrinsics and intrinsics from the calibration
directory and provides utilities for projecting 3D trajectories onto 2D images.
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


CAMERA_NAMES = ["FC", "FC_FAR", "FL", "FR", "RC", "RL", "RR"]


class CameraCalibration:
    """Container for camera intrinsic and extrinsic parameters."""

    def __init__(
        self,
        camera_name: str,
        fx: float, fy: float,
        cx: float, cy: float,
        distortion_coeffs: list[float],
        T_bev_to_camera: np.ndarray,
        image_width: int = 1920,
        image_height: int = 1080,
        distortion_model: str = "opencv_rational_8",
    ):
        """Initialize camera calibration.

        Args:
            camera_name: Name of the camera
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point coordinates
            distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3, k4, k5, k6]
            T_bev_to_camera: 3x4 transformation matrix from BEV to camera
            image_width: Image width in pixels
            image_height: Image height in pixels
            distortion_model: Distortion model name
        """
        self.camera_name = camera_name
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion_coeffs = np.array(distortion_coeffs)
        if self.distortion_coeffs.size < 8:
            self.distortion_coeffs = np.pad(
                self.distortion_coeffs, (0, 8 - self.distortion_coeffs.size)
            )
        self.T_bev_to_camera = T_bev_to_camera
        self.image_width = image_width
        self.image_height = image_height
        self.distortion_model = distortion_model

    def project_bev_to_image(
        self,
        bev_points: np.ndarray,
        depth_culling: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project BEV points to image coordinates.

        Args:
            bev_points: BEV points in shape (N, 3) with columns [x_right, y_forward, z_up]
            depth_culling: Whether to cull points with Z <= 0

        Returns:
            Tuple of (u, v, z) coordinates in image space, where invalid points
            are marked with -1 in u and v, and 0 in z
        """
        ones = np.ones((bev_points.shape[0], 1))
        hom_bev = np.hstack([bev_points, ones])

        cam_points = self.T_bev_to_camera @ hom_bev.T

        X, Y, Z = cam_points[0], cam_points[1], cam_points[2]

        if depth_culling:
            valid = Z > 0
        else:
            valid = np.ones(Z.shape, dtype=bool)

        u = np.full(bev_points.shape[0], -1.0)
        v = np.full(bev_points.shape[0], -1.0)
        z = np.zeros(bev_points.shape[0])

        x = np.where(valid, X / Z, 0.0)
        y = np.where(valid, Y / Z, 0.0)

        if self.distortion_model == "opencv_rational_8":
            k1, k2, p1, p2, k3, k4, k5, k6 = self.distortion_coeffs[:8]

            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
            denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
            radial = numerator / np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
            tangential_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            tangential_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

            x_distorted = x * radial + tangential_x
            y_distorted = y * radial + tangential_y

            u_valid = self.fx * x_distorted + self.cx
            v_valid = self.fy * y_distorted + self.cy

            u[valid] = u_valid[valid]
            v[valid] = v_valid[valid]
            z[valid] = Z[valid]
        else:
            u[valid] = self.fx * x[valid] + self.cx
            v[valid] = self.fy * y[valid] + self.cy
            z[valid] = Z[valid]

        return u, v, z

    def is_point_visible(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Check if projected points are within image bounds.

        Args:
            u: X coordinates in image space
            v: Y coordinates in image space
            z: Depth values

        Returns:
            Boolean array indicating visibility
        """
        valid_depth = z > 0
        in_bounds = (
            (u >= 0) & (u < self.image_width) &
            (v >= 0) & (v < self.image_height)
        )
        return valid_depth & in_bounds


def load_calibration_for_segment(
    calibration_dir: str,
    dataset_name: str,
    segment_name: str,
) -> dict[str, CameraCalibration]:
    """Load calibration for a specific segment.

    Args:
        calibration_dir: Directory containing calibration JSONL files
        dataset_name: Dataset name (e.g., 'data_26_3_24_1')
        segment_name: Segment name (e.g., '2026-03-24-12-06-59')

    Returns:
        Dictionary mapping camera names to CameraCalibration objects
    """
    calib_dataset_name = dataset_name
    calib_file = Path(calibration_dir) / f"{calib_dataset_name}_roll_only_3d_raw_distorted_extrinsics.jsonl"
    if not calib_file.exists() and dataset_name.endswith("_converted"):
        calib_dataset_name = dataset_name.removesuffix("_converted")
        calib_file = Path(calibration_dir) / f"{calib_dataset_name}_roll_only_3d_raw_distorted_extrinsics.jsonl"
    manifest_file = Path(calibration_dir) / "manifest.json"

    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_file}")

    camera_models = {}
    if manifest_file.exists():
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
            camera_models = manifest.get("camera_models", {})

    calibrations = {}

    with open(calib_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("dataset") != calib_dataset_name:
                continue
            if record.get("segment") != segment_name:
                continue

            cameras = record.get("cameras", {})
            for cam_name, cam_data in cameras.items():
                T_matrix = np.array(cam_data["T_bev_m_to_camera_m_3x4"]).reshape(3, 4)

                if cam_name in camera_models:
                    cam_model = camera_models[cam_name]
                    fx = cam_model["fx"]
                    fy = cam_model["fy"]
                    cx = cam_model["cx"]
                    cy = cam_model["cy"]
                    dist_coeffs = cam_model.get("distortion_coefficients_k1_k2_p1_p2_k3_k4_k5_k6", [])
                    img_width = cam_model.get("raw_image_width", 1920)
                    img_height = cam_model.get("raw_image_height", 1080)
                    dist_model = cam_model.get("distortion_model", "opencv_rational_8")
                else:
                    fx = 1000.0
                    fy = 1000.0
                    cx = 960.0
                    cy = 540.0
                    dist_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    img_width = 1920
                    img_height = 1080
                    dist_model = "opencv_rational_8"

                calibrations[cam_name] = CameraCalibration(
                    camera_name=cam_name,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    distortion_coeffs=dist_coeffs,
                    T_bev_to_camera=T_matrix,
                    image_width=img_width,
                    image_height=img_height,
                    distortion_model=dist_model,
                )

            return calibrations

    raise ValueError(f"Segment {segment_name} not found in {calib_file}")


def load_all_calibrations(calibration_dir: str) -> dict[str, dict[str, CameraCalibration]]:
    """Load all calibrations from the calibration directory.

    Args:
        calibration_dir: Directory containing calibration JSONL files

    Returns:
        Dictionary mapping (dataset_name, segment_name) to camera calibrations
    """
    calib_dir = Path(calibration_dir)
    calibrations = {}

    for calib_file in calib_dir.glob("*_roll_only_3d_raw_distorted_extrinsics.jsonl"):
        dataset_name = calib_file.stem.replace("_roll_only_3d_raw_distorted_extrinsics", "")

        manifest_file = calib_dir / "manifest.json"
        camera_models = {}
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                camera_models = manifest.get("camera_models", {})

        with open(calib_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                segment_name = record.get("segment", "")
                timestamp = record.get("timestamp", 0)

                cameras = record.get("cameras", {})
                seg_calibrations = {}

                for cam_name, cam_data in cameras.items():
                    T_matrix = np.array(cam_data["T_bev_m_to_camera_m_3x4"]).reshape(3, 4)

                    if cam_name in camera_models:
                        cam_model = camera_models[cam_name]
                        fx = cam_model["fx"]
                        fy = cam_model["fy"]
                        cx = cam_model["cx"]
                        cy = cam_model["cy"]
                        dist_coeffs = cam_model.get("distortion_coefficients_k1_k2_p1_p2_k3_k4_k5_k6", [])
                        img_width = cam_model.get("raw_image_width", 1920)
                        img_height = cam_model.get("raw_image_height", 1080)
                        dist_model = cam_model.get("distortion_model", "opencv_rational_8")
                    else:
                        fx = 1000.0
                        fy = 1000.0
                        cx = 960.0
                        cy = 540.0
                        dist_coeffs = [0.0] * 8
                        img_width = 1920
                        img_height = 1080
                        dist_model = "opencv_rational_8"

                    seg_calibrations[cam_name] = CameraCalibration(
                        camera_name=cam_name,
                        fx=fx, fy=fy, cx=cx, cy=cy,
                        distortion_coeffs=dist_coeffs,
                        T_bev_to_camera=T_matrix,
                        image_width=img_width,
                        image_height=img_height,
                        distortion_model=dist_model,
                    )

                key = (dataset_name, segment_name)
                if key not in calibrations:
                    calibrations[key] = {}
                calibrations[key][timestamp] = seg_calibrations

    return calibrations


def get_calibration_at_timestamp(
    calibrations: dict[str, dict[str, CameraCalibration]],
    dataset_name: str,
    segment_name: str,
    timestamp_us: int,
) -> dict[str, CameraCalibration]:
    """Get the closest calibration for a given timestamp.

    Args:
        calibrations: Output from load_all_calibrations
        dataset_name: Dataset name
        segment_name: Segment name
        timestamp_us: Timestamp in microseconds

    Returns:
        Dictionary of camera calibrations
    """
    key = (dataset_name, segment_name)
    if key not in calibrations:
        raise ValueError(f"No calibrations found for {key}")

    timestamp_dict = calibrations[key]
    if not timestamp_dict:
        raise ValueError(f"Empty calibrations for {key}")

    timestamps = sorted(timestamp_dict.keys())
    closest_ts = min(timestamps, key=lambda x: abs(x - timestamp_us))

    return timestamp_dict[closest_ts]
