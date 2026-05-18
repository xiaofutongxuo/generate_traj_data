# SPDX-License-Identifier: Apache-2.0
"""Visualization utilities for trajectory projection onto images.

This module provides functions to render trajectories onto camera images
for inspection and debugging.
"""

import colorsys
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

if os.environ.get("GENERATE_TRAJ_USE_TORCH", "1").lower() in {"0", "false", "no"}:
    torch = None
else:
    try:
        import torch
    except ModuleNotFoundError:
        torch = None


def ego_to_bev_points(points_xyz: np.ndarray) -> np.ndarray:
    """Convert ego-local [x_forward, y_left, z_up] points to BEV calibration axes."""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    bev = np.empty_like(points_xyz)
    bev[:, 0] = -points_xyz[:, 1]
    bev[:, 1] = points_xyz[:, 0]
    bev[:, 2] = points_xyz[:, 2]
    return bev


def generate_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate n distinct RGB colors for visualization.

    Args:
        n: Number of colors to generate

    Returns:
        List of (R, G, B) tuples
    """
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def draw_trajectory_on_image(
    image: np.ndarray,
    trajectory_xyz: np.ndarray,
    camera_calib,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    draw_points: bool = True,
    point_radius: int = 4,
    visible_only: bool = True,
    coordinate_frame: str = "ego",
) -> np.ndarray:
    """Draw a 3D trajectory onto an image using camera calibration.

    Args:
        image: Image array in RGB format [H, W, 3]
        trajectory_xyz: Trajectory points [N, 3]. By default these are ego-local
            [x_forward, y_left, z_up] in meters. Pass coordinate_frame="bev" for
            calibration BEV axes [x_right, y_forward, z_up].
        camera_calib: CameraCalibration object
        color: RGB color for the trajectory
        thickness: Line thickness for trajectory drawing
        draw_points: Whether to draw individual points
        point_radius: Radius for point circles
        visible_only: Only draw points that are visible in the image
        coordinate_frame: "ego" or "bev"

    Returns:
        Image with trajectory drawn
    """
    img = image.copy()

    if len(trajectory_xyz) < 2:
        return img

    if coordinate_frame == "ego":
        points_for_projection = ego_to_bev_points(trajectory_xyz)
    elif coordinate_frame == "bev":
        points_for_projection = np.asarray(trajectory_xyz)
    else:
        raise ValueError(f"Unsupported coordinate_frame: {coordinate_frame}")

    u, v, z = camera_calib.project_bev_to_image(points_for_projection)

    if visible_only:
        visible = camera_calib.is_point_visible(u, v, z)
    else:
        visible = z > 0

    valid_indices = np.where(visible)[0]
    if len(valid_indices) < 2:
        return img

    for i in range(len(valid_indices) - 1):
        idx1 = valid_indices[i]
        idx2 = valid_indices[i + 1]

        pt1 = (int(round(u[idx1])), int(round(v[idx1])))
        pt2 = (int(round(u[idx2])), int(round(v[idx2])))

        cv2.line(img, pt1, pt2, color, thickness)

    if draw_points:
        for idx in valid_indices:
            center = (int(round(u[idx])), int(round(v[idx])))
            cv2.circle(img, center, point_radius, color, -1)

    return img


def draw_multiple_trajectories_on_image(
    image: np.ndarray,
    trajectories: list[np.ndarray],
    camera_calib,
    colors: Optional[list[tuple[int, int, int]]] = None,
    thickness: int = 3,
    draw_points: bool = True,
    point_radius: int = 4,
    draw_legend: bool = True,
    legend_position: tuple[int, int] = (20, 30),
    legend_font_scale: float = 0.6,
    legend_thickness: int = 2,
) -> np.ndarray:
    """Draw multiple trajectories onto an image.

    Args:
        image: Image array in RGB format [H, W, 3]
        trajectories: List of trajectory arrays, each [N, 3] in BEV coords
        camera_calib: CameraCalibration object
        colors: List of RGB colors. If None, generates distinct colors.
        thickness: Line thickness for trajectories
        draw_points: Whether to draw individual points
        point_radius: Radius for point circles
        draw_legend: Whether to draw a legend
        legend_position: (x, y) position for legend
        legend_font_scale: Font scale for legend text
        legend_thickness: Text thickness for legend

    Returns:
        Image with all trajectories drawn
    """
    img = image.copy()

    if colors is None:
        colors = generate_distinct_colors(len(trajectories))

    for traj, color in zip(trajectories, colors):
        img = draw_trajectory_on_image(
            img, traj, camera_calib,
            color=color, thickness=thickness,
            draw_points=draw_points, point_radius=point_radius,
        )

    if draw_legend:
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (traj, color) in enumerate(zip(trajectories, colors)):
            text = f"Traj {i+1}"
            pos = (legend_position[0], legend_position[1] + i * 25)
            cv2.putText(img, text, pos, font, legend_font_scale, color, legend_thickness)

    return img


def create_trajectory_grid_visualization(
    trajectories: list[np.ndarray],
    gt_trajectory: Optional[np.ndarray] = None,
    colors: Optional[list[tuple[int, int, int]]] = None,
    grid_size: tuple[int, int] = (2, 4),
    image_shape: tuple[int, int] = (320, 512),
) -> np.ndarray:
    """Create a grid visualization of trajectories in bird's eye view.

    Args:
        trajectories: List of predicted trajectories [N, 3]
        gt_trajectory: Ground truth trajectory [N, 3]
        colors: List of colors for trajectories
        grid_size: (rows, cols) for the grid
        image_shape: (height, width) for each cell

    Returns:
        Grid visualization image
    """
    rows, cols = grid_size
    grid = np.ones((rows * image_shape[0], cols * image_shape[1], 3), dtype=np.uint8) * 255

    if colors is None:
        colors = generate_distinct_colors(len(trajectories))

    scale_x = image_shape[1] / 100.0
    scale_y = image_shape[0] / 100.0

    center_x = image_shape[1] / 2
    center_y = image_shape[0] / 2

    def draw_traj_on_bev(canvas, traj, color, thickness=2):
        if len(traj) < 2:
            return canvas

        for i in range(len(traj) - 1):
            x1 = int(traj[i, 0] * scale_x + center_x)
            y1 = int(-traj[i, 1] * scale_y + center_y)
            x2 = int(traj[i + 1, 0] * scale_x + center_x)
            y2 = int(-traj[i + 1, 1] * scale_y + center_y)

            if 0 <= x1 < image_shape[1] and 0 <= y1 < image_shape[0]:
                if 0 <= x2 < image_shape[1] and 0 <= y2 < image_shape[0]:
                    cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

        for i in range(len(traj)):
            x = int(traj[i, 0] * scale_x + center_x)
            y = int(-traj[i, 1] * scale_y + center_y)
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                cv2.circle(canvas, (x, y), 3, color, -1)

        return canvas

    num_cells = min(len(trajectories), rows * cols)
    for i in range(num_cells):
        row = i // cols
        col = i % cols
        cell_canvas = np.ones(image_shape + (3,), dtype=np.uint8) * 255

        if gt_trajectory is not None:
            cell_canvas = draw_traj_on_bev(cell_canvas, gt_trajectory, (200, 200, 200), 1)

        cell_canvas = draw_traj_on_bev(cell_canvas, trajectories[i], colors[i % len(colors)])

        y_start = row * image_shape[0]
        x_start = col * image_shape[1]
        grid[y_start:y_start + image_shape[0], x_start:x_start + image_shape[1]] = cell_canvas

    return grid


def load_image_from_frame(frame_tensor) -> np.ndarray:
    """Convert a frame tensor to numpy image.

    Args:
        frame_tensor: Frame tensor in [C, H, W] format, values 0-255

    Returns:
        Image in [H, W, 3] RGB format
    """
    if torch is not None and torch.is_tensor(frame_tensor):
        img = frame_tensor.detach().cpu().numpy()
    else:
        img = np.asarray(frame_tensor)

    while img.ndim > 3:
        img = img[0]

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    elif img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
        img = img.transpose(1, 2, 0)

    if img.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got shape {img.shape}")

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] > 3:
        img = img[:, :, :3]

    if np.issubdtype(img.dtype, np.floating):
        if img.size and float(np.nanmax(img)) <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def visualize_sample(
    image: np.ndarray,
    predicted_trajectories: list[np.ndarray],
    gt_trajectory: np.ndarray,
    camera_calib,
    save_path: Optional[Path] = None,
    show_timestamps: bool = False,
) -> np.ndarray:
    """Create a complete visualization for a sample.

    Args:
        image: Input image [H, W, 3]
        predicted_trajectories: List of predicted trajectories
        gt_trajectory: Ground truth trajectory
        camera_calib: CameraCalibration object
        save_path: Optional path to save the visualization
        show_timestamps: Whether to show frame info

    Returns:
        Visualization image
    """
    colors = generate_distinct_colors(len(predicted_trajectories) + 1)

    vis_img = draw_multiple_trajectories_on_image(
        image,
        predicted_trajectories,
        camera_calib,
        colors=colors[:-1],
        thickness=3,
        draw_points=True,
        draw_legend=True,
    )

    if gt_trajectory is not None:
        vis_img = draw_trajectory_on_image(
            vis_img,
            gt_trajectory,
            camera_calib,
            color=colors[-1],
            thickness=3,
            draw_points=True,
        )

    if show_timestamps:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"GT trajectory shown in {colors[-1]}"
        cv2.putText(vis_img, text, (20, vis_img.shape[0] - 20),
                    font, 0.5, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(str(save_path), vis_img[:, :, ::-1])

    return vis_img
