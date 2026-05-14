"""Projection helpers kept for compatibility with the original GUI module."""

from __future__ import annotations

import numpy as np

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def project_3d_to_2d(points_xyz, pose_xyz, pose_quat, intrinsics, extrinsics, image_size=(1920, 1280)):
    """Project 3D world points to 2D image coordinates.
    
    Args:
        points_xyz: (N, 3) array of 3D points in ego frame [x=forward, y=left, z=up]
        pose_xyz: (3,) ego vehicle position (unused for local points)
        pose_quat: (4,) ego vehicle quaternion (unused for local points)
        intrinsics: camera intrinsics dict with fx, fy, cx, cy
        extrinsics: camera extrinsics dict with 'T_bev_to_camera' (3x4 matrix)
        image_size: (width, height) tuple
    
    Returns:
        (N, 2) array of 2D pixel coordinates
    """
    import numpy as np
    
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']
    img_w, img_h = image_size
    
    # Scale intrinsics to actual image size
    scale_x = img_w / intrinsics.get('image_width', img_w)
    scale_y = img_h / intrinsics.get('image_height', img_h)
    fx = fx * scale_x
    fy = fy * scale_y
    cx = cx * scale_x
    cy = cy * scale_y
    
    # Get transformation matrix
    T = extrinsics['T_bev_to_camera']
    R = T[:, :3]
    t = T[:, 3]
    
    # Convert ego frame to BEV frame
    # Ego: x=forward, y=left
    # BEV: x=east, y=north
    # Forward (ego_x) = North (bev_y), Left (ego_y) = West (-bev_x)
    # So: bev_x = -ego_y, bev_y = ego_x
    x_ego, y_ego, z_ego = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    x_bev = -y_ego
    y_bev = x_ego
    z_bev = z_ego
    p_bev = np.stack([x_bev, y_bev, z_bev], axis=1)
    
    # Transform BEV to camera frame
    p_cam = (R @ p_bev.T + t.reshape(3, 1)).T  # (N, 3)
    
    # Project to image
    u_coords = []
    v_coords = []
    valid = []
    
    for i in range(len(p_cam)):
        x, y, z = p_cam[i]
        if z > 0.1:  # Point is in front of camera
            u = fx * x / z + cx
            v = fy * y / z + cy
            # Check if within image bounds
            if 0 <= u < img_w and 0 <= v < img_h:
                u_coords.append(u)
                v_coords.append(v)
                valid.append(True)
            else:
                u_coords.append(-1)
                v_coords.append(-1)
                valid.append(False)
        else:
            u_coords.append(-1)
            v_coords.append(-1)
            valid.append(False)
    
    return np.array(list(zip(u_coords, v_coords))), np.array(valid)

__all__ = [name for name in globals() if (name.startswith("_") and not name.startswith("__")) or name in {"quaternion_to_rotation_matrix", "project_3d_to_2d"}]
