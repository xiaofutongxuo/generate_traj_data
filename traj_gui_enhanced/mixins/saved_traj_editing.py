"""Saved pseudo-GT trajectory geometry editing mixin."""

from __future__ import annotations

import numpy as np

from ..tk_compat import messagebox
from ..dynamics import (
    deform_trajectory_by_keyframe_drag,
    editable_trajectory_keyframes,
    optimize_pseudo_gt_trajectory,
    trajectory_components_from_xyz,
)


class SavedTrajectoryEditingMixin:
    """Direct BEV geometry editing for already-saved non-GT trajectories."""

    def _selected_saved_traj_xyz(self) -> np.ndarray | None:
        if not (0 <= int(getattr(self, "current_traj_idx", -1)) < len(self.trajectories)):
            return None
        traj = self.trajectories[int(self.current_traj_idx)]
        try:
            return np.column_stack([
                np.asarray(traj["x"], dtype=np.float64),
                np.asarray(traj["y"], dtype=np.float64),
                np.asarray(traj["z"], dtype=np.float64),
            ])
        except Exception:
            return None

    def _apply_components_to_saved_traj(self, traj_idx: int, components: dict[str, np.ndarray]) -> None:
        traj = self.trajectories[int(traj_idx)]
        for key, values in components.items():
            traj[key] = np.asarray(values, dtype=np.float64)

    def _reset_saved_trajectory_edit_state(self) -> None:
        self.traj_geom_edit_active = False
        self.traj_geom_edit_dirty = False
        self.traj_geom_edit_traj_idx = None
        self.traj_geom_edit_original_traj = None
        self.traj_geom_edit_original_xyz = None

    def _start_saved_trajectory_edit(self, redraw: bool = True) -> bool:
        if getattr(self, "traj_geom_edit_active", False):
            if redraw:
                messagebox.showwarning(
                    "Trajectory Edit Active",
                    "Save or cancel the current trajectory geometry edit before starting another edit.",
                )
            return False
        if getattr(self, "gt_only", False):
            messagebox.showwarning("GT Only", "GT-only mode does not edit output pseudo-GT rows.")
            return False
        if getattr(self, "speed_edit_active", False):
            messagebox.showwarning("Speed Edit Active", "Save or discard the speed edit first.")
            return False
        if not (0 <= int(getattr(self, "current_traj_idx", -1)) < len(self.trajectories)):
            messagebox.showwarning("No Trajectory", "Select a pseudo-GT trajectory before editing.")
            return False
        traj_idx = int(self.current_traj_idx)
        traj = self.trajectories[traj_idx]
        if self._is_traj_pending_deleted(traj_idx):
            return False
        if self._is_gt_trajectory(traj, traj_idx):
            messagebox.showwarning("Keep GT", "Source-data GT cannot be edited as output pseudo-GT.")
            return False

        xyz = self._selected_saved_traj_xyz()
        if xyz is None or len(xyz) < 2:
            messagebox.showwarning("Edit Failed", "Selected trajectory has no editable xyz points.")
            return False

        self.traj_geom_edit_active = True
        self.traj_geom_edit_dirty = False
        self.traj_geom_edit_traj_idx = traj_idx
        self.traj_geom_edit_original_xyz = xyz.copy()
        self.traj_geom_edit_original_traj = {
            key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
            for key, value in traj.items()
        }
        if hasattr(self, "_update_draw_cursor"):
            self._update_draw_cursor()
        if redraw and hasattr(self, "_update_display"):
            self._update_display()
        return True

    def _apply_saved_trajectory_keyframe_drag(
        self,
        frame_idx: int,
        target_xy: tuple[float, float],
        base_xyz: np.ndarray | None = None,
    ) -> bool:
        if not getattr(self, "traj_geom_edit_active", False):
            return False
        traj_idx = self.traj_geom_edit_traj_idx
        if traj_idx is None or not (0 <= int(traj_idx) < len(self.trajectories)):
            return False
        source_xyz = base_xyz if base_xyz is not None else self._selected_saved_traj_xyz()
        if source_xyz is None:
            return False
        current_xyz = np.asarray(source_xyz, dtype=np.float64)

        candidate = deform_trajectory_by_keyframe_drag(
            current_xyz,
            frame_idx=int(frame_idx),
            target_xy=(float(target_xy[0]), float(target_xy[1])),
        )
        optimized = optimize_pseudo_gt_trajectory(candidate)
        components = trajectory_components_from_xyz(optimized.xyz)
        self._apply_components_to_saved_traj(int(traj_idx), components)
        self.traj_geom_edit_dirty = True
        if hasattr(self, "_refresh_trajectory_smoothness"):
            self._refresh_trajectory_smoothness()
        return True

    def _restore_saved_trajectory_edit(self, redraw: bool = True) -> None:
        if (
            not getattr(self, "traj_geom_edit_active", False)
            or self.traj_geom_edit_traj_idx is None
            or self.traj_geom_edit_original_xyz is None
        ):
            return
        components = trajectory_components_from_xyz(self.traj_geom_edit_original_xyz)
        self._apply_components_to_saved_traj(int(self.traj_geom_edit_traj_idx), components)
        self.traj_geom_edit_dirty = True
        if redraw and hasattr(self, "_update_display"):
            self._update_display()

    def _cancel_saved_trajectory_edit(self, redraw: bool = True) -> None:
        if (
            getattr(self, "traj_geom_edit_active", False)
            and self.traj_geom_edit_original_traj is not None
            and self.traj_geom_edit_traj_idx is not None
            and 0 <= int(self.traj_geom_edit_traj_idx) < len(self.trajectories)
        ):
            self.trajectories[int(self.traj_geom_edit_traj_idx)] = {
                key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                for key, value in self.traj_geom_edit_original_traj.items()
            }
        self._reset_saved_trajectory_edit_state()
        if hasattr(self, "_update_draw_cursor"):
            self._update_draw_cursor()
        if redraw and hasattr(self, "_update_display"):
            self._update_display()

    def _save_saved_trajectory_edit(self) -> None:
        if not getattr(self, "traj_geom_edit_active", False) or self.traj_geom_edit_traj_idx is None:
            return
        if not getattr(self, "traj_geom_edit_dirty", False):
            self._cancel_saved_trajectory_edit(redraw=True)
            return
        traj_idx = int(self.traj_geom_edit_traj_idx)
        if not self._write_selected_trajectory_to_parquet(traj_idx):
            return
        self._reset_saved_trajectory_edit_state()
        if hasattr(self, "_update_draw_cursor"):
            self._update_draw_cursor()
        if hasattr(self, "_load_sample"):
            self._load_sample(self.current_idx)
        if hasattr(self, "_update_display"):
            self._update_display()

    def _saved_trajectory_edit_keyframes(self) -> list[int]:
        xyz = self._selected_saved_traj_xyz()
        if xyz is None:
            return []
        return editable_trajectory_keyframes(len(xyz), interval=8)

    def _nearest_saved_trajectory_keyframe_at_canvas(self, canvas_x: float, canvas_y: float):
        if not getattr(self, "traj_geom_edit_active", False):
            return None
        xyz = self._selected_saved_traj_xyz()
        if xyz is None:
            return None
        hit_frame = None
        hit_distance = None
        for frame_idx in self._saved_trajectory_edit_keyframes():
            px, py = self._world_to_canvas(float(xyz[frame_idx, 0]), float(xyz[frame_idx, 1]))
            distance = ((float(canvas_x) - px) ** 2 + (float(canvas_y) - py) ** 2) ** 0.5
            if distance <= 15 and (hit_distance is None or distance < hit_distance):
                hit_frame = int(frame_idx)
                hit_distance = distance
        return hit_frame

    def _begin_saved_trajectory_keyframe_drag(self, canvas_x: float, canvas_y: float) -> bool:
        frame_idx = self._nearest_saved_trajectory_keyframe_at_canvas(canvas_x, canvas_y)
        if frame_idx is None:
            return False
        self.drag_state = {
            "type": "saved_traj_keyframe",
            "traj_idx": int(self.traj_geom_edit_traj_idx),
            "frame_idx": int(frame_idx),
            "base_xyz": self._selected_saved_traj_xyz(),
        }
        return True

    def _drag_saved_trajectory_keyframe(self, canvas_x: float, canvas_y: float) -> bool:
        if not self.drag_state or self.drag_state.get("type") != "saved_traj_keyframe":
            return False
        if not getattr(self, "traj_geom_edit_active", False):
            return False
        if int(self.drag_state.get("traj_idx", -1)) != int(self.traj_geom_edit_traj_idx):
            return False
        wx, wy = self._canvas_to_world(canvas_x, canvas_y)
        ok = self._apply_saved_trajectory_keyframe_drag(
            int(self.drag_state.get("frame_idx", -1)),
            (float(wx), float(wy)),
            base_xyz=self.drag_state.get("base_xyz"),
        )
        if ok and hasattr(self, "_update_display"):
            self._update_display()
        return ok

    def _draw_saved_trajectory_edit_handles(self) -> None:
        if not getattr(self, "traj_geom_edit_active", False):
            return
        xyz = self._selected_saved_traj_xyz()
        if xyz is None:
            return
        for frame_idx in self._saved_trajectory_edit_keyframes():
            px, py = self._world_to_canvas(float(xyz[frame_idx, 0]), float(xyz[frame_idx, 1]))
            radius = 6 if frame_idx != len(xyz) - 1 else 8
            self.traj_canvas.create_rectangle(
                px - radius,
                py - radius,
                px + radius,
                py + radius,
                fill="#f7f7f7",
                outline="#111111",
                width=2,
            )
            if frame_idx == len(xyz) - 1:
                self.traj_canvas.create_text(
                    px,
                    py - 15,
                    text="END",
                    fill="#f7f7f7",
                    font=("Arial", 8, "bold"),
                )
