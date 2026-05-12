# SPDX-License-Identifier: Apache-2.0
"""VLM Trajectory Generation with Alpamayo 1.5.

This package generates trajectory predictions using the Alpamayo 1.5 Vision Language
Model and projects them onto camera images for visualization.
"""

__version__ = "0.1.0"

from .config import Config, ModelConfig, DataConfig, InferenceConfig, OutputConfig
from .model_loader import load_alpamayo_model, get_processor, build_inference_inputs
from .data_loader import (
    get_dataset_names,
    get_clip_stems_from_dataset,
    load_data,
    to_device,
    get_t0_candidates,
)
from .calibration_loader import (
    CameraCalibration,
    load_calibration_for_segment,
    load_all_calibrations,
    get_calibration_at_timestamp,
)
from .visualization import (
    draw_trajectory_on_image,
    draw_multiple_trajectories_on_image,
    create_trajectory_grid_visualization,
    visualize_sample,
    load_image_from_frame,
    generate_distinct_colors,
)

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "InferenceConfig",
    "OutputConfig",
    "load_alpamayo_model",
    "get_processor",
    "build_inference_inputs",
    "get_dataset_names",
    "get_clip_stems_from_dataset",
    "load_data",
    "to_device",
    "get_t0_candidates",
    "CameraCalibration",
    "load_calibration_for_segment",
    "load_all_calibrations",
    "get_calibration_at_timestamp",
    "draw_trajectory_on_image",
    "draw_multiple_trajectories_on_image",
    "create_trajectory_grid_visualization",
    "visualize_sample",
    "load_image_from_frame",
    "generate_distinct_colors",
]
