#!/usr/bin/env python3
"""Visualize one generated GT + acc/curvature candidate set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "output"
DEFAULT_IMAGE = SCRIPT_DIR / "acc_curvature_smoke_visualization.png"


@dataclass(frozen=True)
class SmokeSample:
    dataset: str
    clip: str
    t0_us: int
    rows: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--clip", type=str, default="")
    parser.add_argument("--t0_us", type=int, default=0)
    return parser.parse_args(argv)


def load_first_sample(
    output_dir: Path,
    dataset: str = "",
    clip: str = "",
    t0_us: int = 0,
) -> SmokeSample:
    parquet_files = sorted(output_dir.glob("*/*.egomotion.parquet"))
    if dataset:
        parquet_files = [path for path in parquet_files if path.parent.name == dataset]
    if clip:
        parquet_files = [path for path in parquet_files if path.name == f"{clip}.egomotion.parquet"]
    if not parquet_files:
        raise FileNotFoundError(f"No output parquet files found under {output_dir}")

    for path in parquet_files:
        df = pd.read_parquet(path)
        if df.empty or "t0_us" not in df.columns:
            continue
        chosen_t0 = int(t0_us) if int(t0_us) else int(df["t0_us"].iloc[0])
        rows = df[df["t0_us"].astype("int64") == chosen_t0].sort_values("sample_idx")
        if rows.empty:
            continue
        return SmokeSample(
            dataset=path.parent.name,
            clip=path.name.replace(".egomotion.parquet", ""),
            t0_us=chosen_t0,
            rows=rows,
        )
    raise ValueError(f"No matching t0 found under {output_dir}")


def _as_xy(row: pd.Series, prefix: str = "") -> np.ndarray:
    x = np.asarray(row[f"{prefix}x"], dtype=np.float64)
    y = np.asarray(row[f"{prefix}y"], dtype=np.float64)
    return np.column_stack([x, y])


def _plot_xy(ax, xy: np.ndarray, label: str, color: str, linewidth: float, linestyle: str = "-") -> None:
    plot_xy = np.column_stack([-xy[:, 1], xy[:, 0]])
    ax.plot(plot_xy[:, 0], plot_xy[:, 1], label=label, color=color, linewidth=linewidth, linestyle=linestyle)
    ax.scatter(plot_xy[-1, 0], plot_xy[-1, 1], color=color, marker="x", s=45)


def draw_sample(sample: SmokeSample, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    palette = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]

    first = sample.rows.iloc[0]
    if "history_x" in sample.rows.columns and isinstance(first.get("history_x"), (list, tuple, np.ndarray)):
        hist_xy = _as_xy(first, prefix="history_")
        _plot_xy(ax, hist_xy, "history", "#777777", 2.0, "--")

    for idx, (_, row) in enumerate(sample.rows.iterrows()):
        source = str(row.get("source", "traj"))
        sample_idx = int(row.get("sample_idx", idx))
        xy = _as_xy(row)
        if source == "gt" or sample_idx == 0:
            _plot_xy(ax, xy, "GT", "#ffd84d", 3.0)
        else:
            color = palette[(sample_idx - 1) % len(palette)]
            _plot_xy(ax, xy, f"{source} #{sample_idx}", color, 1.8, "--")

    ax.scatter([0.0], [0.0], color="black", s=30, label="ego t0")
    ax.set_title(f"{sample.dataset}/{sample.clip}\nt0={sample.t0_us}")
    ax.set_xlabel("right (m)")
    ax.set_ylabel("forward (m)")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(image_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample = load_first_sample(args.output_dir, args.dataset, args.clip, args.t0_us)
    draw_sample(sample, args.image)
    print(f"Wrote visualization: {args.image}")
    print(f"Visualized {len(sample.rows)} trajectories for {sample.dataset}/{sample.clip} t0={sample.t0_us}")


if __name__ == "__main__":
    main()
