"""Command-line entrypoint for the enhanced trajectory GUI."""

from __future__ import annotations

import argparse

from .environment import setup_environment
setup_environment()

from .viewer import TrajectoryViewerEnhanced


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced GUI for viewing VLM-generated trajectories"
    )
    parser.add_argument(
        "--data_root", type=str, default="/home/tsingyu/train_data",
        help="Root directory for training data",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory containing generated trajectories",
    )
    parser.add_argument(
        "--calibration_dir", type=str,
        default="/home/tsingyu/yzb/triplane_tokenization/cailibration",
        help="Directory containing calibration files",
    )
    parser.add_argument(
        "--cameras", type=str, default="RL,FC,RR",
        help="Comma-separated list of cameras to display",
    )
    parser.add_argument(
        "--start_index", type=int, default=None,
        help="1-based sample index to show at startup",
    )
    parser.add_argument(
        "--start_dataset", type=str, default="",
        help="Dataset name to show at startup, e.g. data_26_3_24_1_converted",
    )
    parser.add_argument(
        "--start_clip", type=str, default="",
        help="Clip stem to show at startup, e.g. 2026-03-24-12-06-59",
    )
    parser.add_argument(
        "--start_t0", type=int, default=None,
        help="Exact t0_us timestamp to show at startup",
    )
    parser.add_argument(
        "--no_restore_last", action="store_true",
        help="Do not restore the last viewed sample from the previous GUI session",
    )
    parser.add_argument(
        "--gt_only", action="store_true",
        help="Show GT samples directly from data_root and do not load generated VLA trajectories",
    )
    parser.add_argument(
        "--gt_stride_frames", type=int, default=3,
        help="Frame stride for GT-only t0 sampling",
    )
    parser.add_argument(
        "--index_mode", type=str, default="video_frames",
        choices=["generated", "video_frames", "merged"],
        help=(
            "Sample index source: generated uses output parquet t0 values; "
            "video_frames uses every master video timestamp; merged uses video "
            "frames plus any generated t0 not present in the master timestamps."
        ),
    )
    parser.add_argument(
        "--frame_stride", type=int, default=1,
        help="Frame stride for video_frames/merged sample indexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cameras = [c.strip() for c in args.cameras.split(",")]
    TrajectoryViewerEnhanced(
        data_root=args.data_root,
        output_dir=args.output_dir,
        calibration_dir=args.calibration_dir,
        cameras=cameras,
        start_index=args.start_index,
        start_dataset=args.start_dataset,
        start_clip=args.start_clip,
        start_t0=args.start_t0,
        restore_last=not args.no_restore_last,
        gt_only=args.gt_only,
        gt_stride_frames=args.gt_stride_frames,
        index_mode=args.index_mode,
        frame_stride=args.frame_stride,
    )


__all__ = ["parse_args", "main"]
