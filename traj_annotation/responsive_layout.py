"""Responsive layout helpers for the trajectory annotation GUI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponsiveLayout:
    window_width: int
    window_height: int
    min_width: int
    min_height: int
    visual_scale: float
    geometry: str
    start_maximized: bool
    bev_canvas_width: int
    bev_canvas_height: int
    speed_canvas_width: int
    speed_canvas_height: int
    bev_forward_scale: float
    bev_lateral_scale: float
    bev_origin_bottom_margin: int
    camera_fc_height: int
    camera_aux_height: int
    camera_fc_height_many: int
    camera_aux_height_many: int
    right_panel_width: int
    trajectory_listbox_width: int
    dataset_combo_width: int
    clip_combo_width: int
    t0_combo_width: int
    scene_combo_width: int


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(float(value)))))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, float(value))))


def compute_responsive_layout(screen_width: int, screen_height: int) -> ResponsiveLayout:
    """Return size settings that fit common laptop, desktop, and Windows screens."""
    screen_width = max(800, int(screen_width or 1900))
    screen_height = max(600, int(screen_height or 1150))

    margin_w = 32 if screen_width < 1600 else 48
    margin_h = 72 if screen_height < 1000 else 64
    window_width = min(clamp_int(screen_width - margin_w, 760, 1900), screen_width)
    window_height = min(clamp_int(screen_height - margin_h, 560, 1150), screen_height)
    min_width = min(900, window_width)
    min_height = min(620, window_height)

    width_scale = clamp_float(window_width / 1900.0, 0.74, 1.0)
    height_scale = clamp_float(window_height / 1150.0, 0.60, 1.0)
    visual_scale = clamp_float(min(width_scale, height_scale), 0.60, 1.0)

    bev_canvas_width = clamp_int(560 * visual_scale, 400, 560)
    bev_canvas_height = clamp_int(700 * visual_scale, 420, 700)
    speed_canvas_width = bev_canvas_width
    speed_canvas_height = clamp_int(180 * visual_scale, 115, 180)
    bev_forward_scale = 6.2 * bev_canvas_height / 700.0
    bev_lateral_scale = 10.0 * bev_canvas_width / 560.0
    bev_origin_bottom_margin = clamp_int(65 * visual_scale, 40, 65)

    camera_fc_height = clamp_int(720 * visual_scale, 430, 720)
    camera_aux_height = clamp_int(300 * visual_scale, 180, 300)
    camera_fc_height_many = clamp_int(600 * visual_scale, 360, 600)
    camera_aux_height_many = clamp_int(240 * visual_scale, 145, 240)

    right_panel_width = clamp_int(window_width * 0.24, 340, 460)
    trajectory_listbox_width = clamp_int(right_panel_width / 8.5, 40, 56)

    compact = window_width < 1500
    dataset_combo_width = 24 if compact else 30
    clip_combo_width = 20 if compact else 24
    t0_combo_width = 18 if compact else 22
    scene_combo_width = 15 if compact else 18

    start_maximized = screen_width <= 1920 or screen_height <= 1200

    return ResponsiveLayout(
        window_width=window_width,
        window_height=window_height,
        min_width=min_width,
        min_height=min_height,
        visual_scale=visual_scale,
        geometry=f"{window_width}x{window_height}+0+0",
        start_maximized=start_maximized,
        bev_canvas_width=bev_canvas_width,
        bev_canvas_height=bev_canvas_height,
        speed_canvas_width=speed_canvas_width,
        speed_canvas_height=speed_canvas_height,
        bev_forward_scale=bev_forward_scale,
        bev_lateral_scale=bev_lateral_scale,
        bev_origin_bottom_margin=bev_origin_bottom_margin,
        camera_fc_height=camera_fc_height,
        camera_aux_height=camera_aux_height,
        camera_fc_height_many=camera_fc_height_many,
        camera_aux_height_many=camera_aux_height_many,
        right_panel_width=right_panel_width,
        trajectory_listbox_width=trajectory_listbox_width,
        dataset_combo_width=dataset_combo_width,
        clip_combo_width=clip_combo_width,
        t0_combo_width=t0_combo_width,
        scene_combo_width=scene_combo_width,
    )
