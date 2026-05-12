#!/usr/bin/env python3
"""GUI tool for visualizing and filtering VLM-generated trajectories.

Usage:
    python trajectory_gui.py --data_root /path/to/train_data
                            --output_dir /path/to/output

Controls:
    Left/Right Arrow: Navigate between samples
    Up/Down Arrow: Navigate between trajectories in current sample
    Delete/Backspace: Mark selected trajectory for deletion
    Ctrl+S: Save filtered results
    Q: Quit
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import get_dataset_names, get_clip_stems_from_dataset, load_data, get_t0_candidates
from calibration_loader import load_calibration_for_segment
from visualization import load_image_from_frame


# Camera name to index mapping
CAMERA_IDX_TO_NAME = {0: "FL", 1: "FC", 2: "FR", 3: "RL", 4: "RC", 5: "RR"}


def _row_t0_us(row) -> int:
    """Return t0 for new parquet rows, with fallback for older outputs."""
    if "t0_us" in row and not pd.isna(row["t0_us"]):
        return int(row["t0_us"])
    timestamps = row["timestamp"]
    if len(timestamps) == 0:
        return 0
    return int(timestamps[0]) - 100_000


class TrajectoryViewer:
    """GUI for viewing and filtering VLM-generated trajectories."""

    def __init__(
        self,
        data_root: str,
        output_dir: str,
        calibration_dir: str,
        cameras: list[str] = None,
    ):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.calibration_dir = Path(calibration_dir)
        self.cameras = cameras or ["RL", "FC", "RR"]

        # Load dataset info
        self.datasets = get_dataset_names(str(self.data_root))
        self.samples = []  # List of (dataset_name, clip_stem, t0_us) tuples

        for dataset_name in self.datasets:
            dataset_output = self.output_dir / dataset_name
            if not dataset_output.exists():
                continue

            # Get clips from output directory
            for parquet_file in dataset_output.glob("*.egomotion.parquet"):
                # clip_stem should NOT include .egomotion extension
                clip_stem = str(parquet_file.name).replace('.egomotion.parquet', '')
                # Load t0_us from parquet
                try:
                    df = pd.read_parquet(parquet_file)
                    if len(df) > 0:
                        if "t0_us" in df.columns:
                            for t0_us in sorted(df["t0_us"].dropna().astype("int64").unique()):
                                self.samples.append((dataset_name, clip_stem, int(t0_us)))
                        else:
                            self.samples.append((dataset_name, clip_stem, _row_t0_us(df.iloc[0])))
                except Exception as e:
                    print(f"Warning: Could not load {parquet_file}: {e}")

        self.samples.sort(key=lambda x: (x[0], x[1]))

        if not self.samples:
            raise ValueError(f"No trajectory files found in {self.output_dir}")

        self.current_idx = 0
        self.trajectories = []  # Current loaded trajectories
        self.trajectory_states = {}  # trajectory_idx -> kept (True/False)
        self.current_traj_idx = 0
        self.image = None
        self.image_tk = None

        # Load first sample
        self._load_sample(0)

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Trajectory Viewer")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2b2b2b")

        self._create_widgets()
        self._update_display()

        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self._prev_sample())
        self.root.bind("<Right>", lambda e: self._next_sample())
        self.root.bind("<Up>", lambda e: self._prev_traj())
        self.root.bind("<Down>", lambda e: self._next_traj())
        self.root.bind("<Delete>", lambda e: self._delete_traj())
        self.root.bind("<BackSpace>", lambda e: self._delete_traj())
        self.root.bind("<Control-s>", lambda e: self._save_results())
        self.root.bind("<q>", lambda e: self.root.quit())

        self.root.mainloop()

    def _base_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        return self.output_dir / dataset_name / f"{clip_stem}.egomotion.parquet"

    def _filtered_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        return self.output_dir / dataset_name / "filtered" / f"{clip_stem}.egomotion.parquet"

    def _active_traj_file(self, dataset_name: str, clip_stem: str) -> Path:
        filtered_file = self._filtered_traj_file(dataset_name, clip_stem)
        if filtered_file.exists():
            return filtered_file
        return self._base_traj_file(dataset_name, clip_stem)

    def _load_sample(self, idx: int):
        """Load a sample by index."""
        if idx < 0 or idx >= len(self.samples):
            return

        self.current_idx = idx
        dataset_name, clip_stem, t0_us = self.samples[idx]

        # Load trajectories from parquet
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        if "t0_us" in df.columns:
            df = df[df["t0_us"].astype("int64") == int(t0_us)]

        self.trajectories = []
        for _, row in df.iterrows():
            self.trajectories.append({
                "timestamp": row["timestamp"],
                "x": row["x"],
                "y": row["y"],
                "z": row["z"],
                "qx": row["qx"],
                "qy": row["qy"],
                "qz": row["qz"],
                "qw": row["qw"],
                "vx": row["vx"],
                "vy": row["vy"],
                "vz": row["vz"],
                "curvature": row["curvature"],
            })

        # Initialize states (all kept by default)
        self.trajectory_states = {i: True for i in range(len(self.trajectories))}
        self.current_traj_idx = 0

        # Load original data for visualization
        try:
            self.conv_data = load_data(
                str(self.data_root),
                clip_stem,
                dataset_name,
                t0_us=t0_us,
                cameras=self.cameras,
            )
        except Exception as e:
            print(f"Warning: Could not load image data: {e}")
            self.conv_data = None

    def _create_widgets(self):
        """Create GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg="#2b2b2b")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Title
        title_frame = tk.Frame(main_frame, bg="#2b2b2b")
        title_frame.pack(fill=tk.X)

        self.title_label = tk.Label(
            title_frame,
            text="Trajectory Viewer",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#2b2b2b",
        )
        self.title_label.pack(side=tk.LEFT)

        self.nav_label = tk.Label(
            title_frame,
            text="",
            font=("Arial", 12),
            fg="#888888",
            bg="#2b2b2b",
        )
        self.nav_label.pack(side=tk.RIGHT)

        # Content area
        content_frame = tk.Frame(main_frame, bg="#2b2b2b")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Left panel - Trajectory visualization
        left_frame = tk.Frame(content_frame, bg="#1e1e1e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Trajectory canvas
        self.traj_canvas = tk.Canvas(
            left_frame,
            width=700,
            height=700,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.traj_canvas.pack(padx=5, pady=5)
        self.traj_canvas.bind("<Button-1>", self._on_canvas_click)

        # Right panel - Image and trajectory list
        right_frame = tk.Frame(content_frame, bg="#2b2b2b")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Camera image
        self.image_label = tk.Label(right_frame, bg="#1e1e1e")
        self.image_label.pack(pady=(0, 10))

        # Trajectory list
        list_frame = tk.Frame(right_frame, bg="#2b2b2b")
        list_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            list_frame,
            text="Trajectories:",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#2b2b2b",
        ).pack(anchor=tk.W)

        # Scrollable list
        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.traj_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=list_scroll.set,
            font=("Courier", 10),
            bg="#1e1e1e",
            fg="white",
            selectbackground="#444444",
            selectforeground="white",
            height=15,
        )
        self.traj_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.traj_listbox.yview)

        self.traj_listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Status bar
        status_frame = tk.Frame(main_frame, bg="#2b2b2b")
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_label = tk.Label(
            status_frame,
            text="",
            font=("Arial", 10),
            fg="#888888",
            bg="#2b2b2b",
        )
        self.status_label.pack(side=tk.LEFT)

        # Button frame
        btn_frame = tk.Frame(status_frame, bg="#2b2b2b")
        btn_frame.pack(side=tk.RIGHT)

        tk.Button(
            btn_frame,
            text="Delete (Del)",
            command=self._delete_traj,
            bg="#c0392b",
            fg="white",
            padx=10,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Keep",
            command=self._keep_traj,
            bg="#27ae60",
            fg="white",
            padx=10,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame,
            text="Save (Ctrl+S)",
            command=self._save_results,
            bg="#2980b9",
            fg="white",
            padx=10,
        ).pack(side=tk.LEFT, padx=5)

        # Help text
        help_frame = tk.Frame(main_frame, bg="#2b2b2b")
        help_frame.pack(fill=tk.X, pady=(10, 0))

        help_text = "Controls: ←/→ Navigate samples | ↑/↓ Navigate trajectories | Del Remove | Ctrl+S Save | Q Quit"
        tk.Label(
            help_frame,
            text=help_text,
            font=("Arial", 9),
            fg="#666666",
            bg="#2b2b2b",
        ).pack()

    def _on_canvas_click(self, event):
        """Handle canvas click to select trajectory."""
        # Find clicked trajectory
        for i, traj in enumerate(self.trajectories):
            x_coords = traj["x"]
            y_coords = traj["y"]

            # Simple proximity check with center of trajectory
            if len(x_coords) > 0:
                cx = sum(x_coords) / len(x_coords)
                cy = sum(y_coords) / len(y_coords)

                # Convert to canvas coordinates
                canvas_x, canvas_y = self._world_to_canvas(cx, cy)
                dist = ((event.x - canvas_x)**2 + (event.y - canvas_y)**2)**0.5

                if dist < 50:
                    self.current_traj_idx = i
                    self._update_display()
                    return

    def _world_to_canvas(self, wx, wy, scale=15, offset=(350, 350)):
        """Convert world coordinates to canvas coordinates."""
        return wx * scale + offset[0], -wy * scale + offset[1]

    def _canvas_to_world(self, cx, cy, scale=15, offset=(350, 350)):
        """Convert canvas coordinates to world coordinates."""
        return (cx - offset[0]) / scale, -(cy - offset[1]) / scale

    def _draw_trajectories(self):
        """Draw all trajectories on the canvas."""
        self.traj_canvas.delete("all")

        if not self.trajectories:
            return

        # Colors for trajectories
        colors = [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#9b59b6",  # Purple
            "#f39c12",  # Orange
            "#1abc9c",  # Teal
            "#e91e63",  # Pink
        ]

        for i, traj in enumerate(self.trajectories):
            x_coords = traj["x"]
            y_coords = traj["y"]

            if len(x_coords) < 2:
                continue

            # Determine color and style
            is_selected = (i == self.current_traj_idx)
            is_kept = self.trajectory_states.get(i, True)

            if not is_kept:
                color = "#555555"
                width = 1
                dash = (5, 5)
            elif is_selected:
                color = "#2ecc71"  # Green for selected
                width = 4
                dash = None
            else:
                color = colors[i % len(colors)]
                width = 2
                dash = None

            # Draw trajectory
            points = []
            for x, y in zip(x_coords, y_coords):
                px, py = self._world_to_canvas(x, y)
                points.extend([px, py])

            if len(points) >= 4:
                self.traj_canvas.create_line(
                    *points,
                    fill=color,
                    width=width,
                    dash=dash,
                    smooth=True,
                )

            # Draw start point (t0)
            if len(x_coords) > 0:
                px, py = self._world_to_canvas(x_coords[0], y_coords[0])
                r = 6 if is_selected else 4
                self.traj_canvas.create_oval(
                    px - r, py - r, px + r, py + r,
                    fill=color,
                    outline="white" if is_selected else "black",
                    width=2 if is_selected else 1,
                )

            # Draw trajectory ID
            if len(x_coords) > 5:
                px, py = self._world_to_canvas(x_coords[5], y_coords[5])
                traj_id = f"T{i}"
                self.traj_canvas.create_text(
                    px, py,
                    text=traj_id,
                    fill=color,
                    font=("Arial", 10, "bold"),
                )

        # Draw origin marker
        px, py = self._world_to_canvas(0, 0)
        self.traj_canvas.create_oval(
            px - 4, py - 4, px + 4, py + 4,
            fill="#2ecc71",
            outline="white",
        )

        # Draw direction arrow
        arrow_len = 10
        self.traj_canvas.create_line(
            px, py, px, py - arrow_len,
            fill="#2ecc71",
            width=3,
            arrow=tk.LAST,
        )

    def _draw_image(self):
        """Draw the camera image."""
        if self.conv_data is None:
            self.image_label.config(image="")
            return

        # Get first camera frame
        try:
            frames = self.conv_data["image_frames"]
            if len(frames) > 1:  # Use second camera (FC) for display
                frame_idx = 1
            else:
                frame_idx = 0

            frame = frames[frame_idx, 0]  # [C, H, W]
            frame = load_image_from_frame(frame)

            # Resize for display
            h, w = frame.shape[:2]
            new_h, new_w = 650, int(650 * w / h)
            frame = cv2.resize(frame, (new_w, new_h))

            # Convert to PhotoImage
            image = Image.fromarray(frame)
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image_tk)
        except Exception as e:
            print(f"Warning: Could not display image: {e}")
            self.image_label.config(image="")

    def _update_display(self):
        """Update the display."""
        # Update title
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        self.title_label.config(
            text=f"Dataset: {dataset_name} | Clip: {clip_stem}"
        )
        self.nav_label.config(
            text=f"{self.current_idx + 1} / {len(self.samples)}"
        )

        # Update status
        kept = sum(1 for v in self.trajectory_states.values() if v)
        removed = len(self.trajectories) - kept
        self.status_label.config(
            text=f"Trajectories: {len(self.trajectories)} | Kept: {kept} | Removed: {removed}"
        )

        # Update listbox
        self.traj_listbox.delete(0, tk.END)
        for i, traj in enumerate(self.trajectories):
            is_kept = self.trajectory_states.get(i, True)
            status = "✓" if is_kept else "✗"
            x_start = traj["x"][0] if len(traj["x"]) > 0 else 0
            y_start = traj["y"][0] if len(traj["y"]) > 0 else 0
            x_end = traj["x"][-1] if len(traj["x"]) > 0 else 0
            y_end = traj["y"][-1] if len(traj["y"]) > 0 else 0

            text = f"[{status}] T{i}: ({x_start:.1f},{y_start:.1f}) -> ({x_end:.1f},{y_end:.1f})"
            if is_kept:
                self.traj_listbox.insert(tk.END, text)
            else:
                self.traj_listbox.insert(tk.END, text)

        # Select current trajectory in list
        if self.current_traj_idx < len(self.trajectories):
            self.traj_listbox.selection_clear(0, tk.END)
            self.traj_listbox.selection_set(self.current_traj_idx)
            self.traj_listbox.see(self.current_traj_idx)

        # Draw
        self._draw_trajectories()
        self._draw_image()

    def _on_list_select(self, event):
        """Handle list selection."""
        selection = self.traj_listbox.curselection()
        if selection:
            self.current_traj_idx = selection[0]
            self._update_display()

    def _prev_sample(self):
        """Go to previous sample."""
        if self.current_idx > 0:
            self._load_sample(self.current_idx - 1)
            self._update_display()

    def _next_sample(self):
        """Go to next sample."""
        if self.current_idx < len(self.samples) - 1:
            self._load_sample(self.current_idx + 1)
            self._update_display()

    def _prev_traj(self):
        """Go to previous trajectory."""
        if self.current_traj_idx > 0:
            self.current_traj_idx -= 1
            self._update_display()

    def _next_traj(self):
        """Go to next trajectory."""
        if self.current_traj_idx < len(self.trajectories) - 1:
            self.current_traj_idx += 1
            self._update_display()

    def _delete_traj(self):
        """Mark current trajectory for deletion."""
        if 0 <= self.current_traj_idx < len(self.trajectories):
            self.trajectory_states[self.current_traj_idx] = False
            self._update_display()

    def _keep_traj(self):
        """Keep current trajectory."""
        if 0 <= self.current_traj_idx < len(self.trajectories):
            self.trajectory_states[self.current_traj_idx] = True
            self._update_display()

    def _save_results(self):
        """Save filtered trajectories."""
        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)

        # Load the currently active parquet so repeated saves keep prior deletions.
        df = pd.read_parquet(traj_file)

        # Mark deleted trajectories for the current t0 only. Other t0 frames in
        # the same clip parquet are preserved.
        if "t0_us" in df.columns:
            current_indices = df.index[df["t0_us"].astype("int64") == int(t0_us)].tolist()
            drop_indices = [
                row_idx
                for local_idx, row_idx in enumerate(current_indices)
                if not self.trajectory_states.get(local_idx, True)
            ]
            df_filtered = df.drop(index=drop_indices).reset_index(drop=True)
        else:
            kept_mask = [self.trajectory_states.get(i, True) for i in range(len(df))]
            df_filtered = df[kept_mask].reset_index(drop=True)

        # Save to filtered directory
        filtered_file = self._filtered_traj_file(dataset_name, clip_stem)
        filtered_dir = filtered_file.parent
        filtered_dir.mkdir(parents=True, exist_ok=True)
        df_filtered.to_parquet(filtered_file, index=False)

        self._load_sample(self.current_idx)
        self._update_display()

        messagebox.showinfo(
            "Saved",
            f"Saved {len(df_filtered)} trajectories to {filtered_file}\n"
            f"(Removed {len(df) - len(df_filtered)} trajectories)"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="GUI for viewing and filtering VLM-generated trajectories"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/tsingyu/train_data",
        help="Root directory for training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory containing generated trajectories",
    )
    parser.add_argument(
        "--calibration_dir",
        type=str,
        default="/home/tsingyu/yzb/triplane_tokenization/cailibration",
        help="Directory containing calibration files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    viewer = TrajectoryViewer(
        data_root=args.data_root,
        output_dir=args.output_dir,
        calibration_dir=args.calibration_dir,
    )
