#!/usr/bin/env python3
"""Export repaired 5-8 GT future xy trajectories with the existing logic."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from kmeans_cluster import export_future_xy_txt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_ROOT = Path("/home/ubuntu/Public/train_data")
DEFAULT_DATASET = "data_26_5_8_converted"
DEFAULT_OUTPUT_TXT = SCRIPT_DIR / "future_trajectories_5_8_xy.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output_txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--history_steps", type=int, default=16)
    parser.add_argument("--t0_stride", type=int, default=1)
    parser.add_argument("--min_speed_mps", type=float, default=0.0)
    parser.add_argument("--min_forward_acc_mps2", type=float, default=-6.0)
    parser.add_argument("--max_forward_acc_mps2", type=float, default=2.0)
    parser.add_argument("--max_step_speed_mps", type=float, default=15.0)
    parser.add_argument("--max_step_acc_mps2", type=float, default=0.0)
    parser.add_argument("--allow_backward", action="store_true")
    return parser.parse_args(argv)


def export_dataset(args: argparse.Namespace):
    source_dir = args.train_data_root / args.dataset
    if not source_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {source_dir}")

    with tempfile.TemporaryDirectory(prefix="export_5_8_") as tmp_name:
        tmp_root = Path(tmp_name)
        (tmp_root / args.dataset).symlink_to(source_dir, target_is_directory=True)
        return export_future_xy_txt(
            tmp_root,
            args.output_txt,
            args.steps,
            args.history_steps,
            args.t0_stride,
            args.min_speed_mps,
            args.min_forward_acc_mps2,
            args.max_forward_acc_mps2,
            args.max_step_speed_mps,
            args.max_step_acc_mps2,
            args.allow_backward,
        )


def main() -> None:
    args = parse_args()
    stats = export_dataset(args)
    print(f"Exported {stats.kept} trajectories to {args.output_txt}")
    print(
        "Filtered export stats: "
        f"skipped_slow={stats.skipped_slow}, "
        f"skipped_backward={stats.skipped_backward}, "
        f"skipped_acc={stats.skipped_acc}, "
        f"skipped_jump={stats.skipped_jump}, "
        f"skipped_short_or_bad={stats.skipped_short_or_bad}"
    )
    print(
        "Filtered original forward acceleration: "
        f"min={stats.min_forward_acc_mps2:.6f} m/s^2, "
        f"max={stats.max_forward_acc_mps2:.6f} m/s^2"
    )


if __name__ == "__main__":
    main()
