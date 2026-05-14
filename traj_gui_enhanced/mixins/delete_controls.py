"""DeleteControlsMixin for staged trajectory deletion and undo."""

from __future__ import annotations

from copy import deepcopy

from ..tk_compat import messagebox
from ..trajectory_identity import (
    TrajectoryKey,
    is_deletable_trajectory_record,
    is_gt_trajectory_record,
    trajectory_key_from_record,
)


class DeleteControlsMixin:

    def _reset_pending_delete_state(self) -> None:
        self.pending_deleted_traj_keys: set[TrajectoryKey] = set()
        self.pending_delete_stack: list[tuple[TrajectoryKey, int, dict | None]] = []
        self.traj_listbox_to_traj_idx: list[int] = []
        self.pending_manual_points_delete = False

    def _manual_points_snapshot_for_pending_delete(self) -> dict:
        return {
            "manual_line_points": deepcopy(getattr(self, "manual_line_points", [])),
            "manual_camera_line_points": deepcopy(getattr(self, "manual_camera_line_points", [])),
            "manual_stop_points": deepcopy(getattr(self, "manual_stop_points", [])),
            "manual_point_actions": deepcopy(getattr(self, "manual_point_actions", [])),
            "manual_line_points_dirty": bool(getattr(self, "manual_line_points_dirty", False)),
            "manual_camera_line_points_dirty": bool(getattr(self, "manual_camera_line_points_dirty", False)),
            "manual_stop_points_dirty": bool(getattr(self, "manual_stop_points_dirty", False)),
        }

    def _clear_manual_points_for_pending_delete(self) -> dict | None:
        if not hasattr(self, "_selected_trajectory_matches_manual_curve"):
            return None
        if not self._selected_trajectory_matches_manual_curve():
            return None
        snapshot = self._manual_points_snapshot_for_pending_delete()
        self.manual_line_points = []
        self.manual_camera_line_points = []
        self.manual_stop_points = []
        self.manual_point_actions = []
        self.manual_line_points_dirty = True
        self.manual_camera_line_points_dirty = True
        self.manual_stop_points_dirty = True
        self.pending_manual_points_delete = True
        return snapshot

    def _restore_manual_points_snapshot(self, snapshot: dict | None) -> None:
        if not snapshot:
            return
        self.manual_line_points = deepcopy(snapshot["manual_line_points"])
        self.manual_camera_line_points = deepcopy(snapshot["manual_camera_line_points"])
        self.manual_stop_points = deepcopy(snapshot["manual_stop_points"])
        self.manual_point_actions = deepcopy(snapshot["manual_point_actions"])
        self.manual_line_points_dirty = bool(snapshot["manual_line_points_dirty"])
        self.manual_camera_line_points_dirty = bool(snapshot["manual_camera_line_points_dirty"])
        self.manual_stop_points_dirty = bool(snapshot["manual_stop_points_dirty"])
        self.pending_manual_points_delete = any(
            pending_snapshot is not None
            for _key, _traj_idx, pending_snapshot in self.pending_delete_stack
        )

    def _trajectory_key_for_index(self, traj_idx: int) -> TrajectoryKey:
        if 0 <= int(traj_idx) < len(self.trajectories):
            return trajectory_key_from_record(
                self.trajectories[int(traj_idx)],
                fallback_index=int(traj_idx),
                fallback_t0_us=self._current_sample_t0_us(),
            )
        return self._current_sample_t0_us(), int(traj_idx)

    def _current_sample_t0_us(self) -> int | None:
        if not hasattr(self, "samples") or not (0 <= int(self.current_idx) < len(self.samples)):
            return None
        return int(self.samples[int(self.current_idx)][2])

    def _is_gt_trajectory(self, traj: dict, fallback_index: int = -1) -> bool:
        return is_gt_trajectory_record(traj, fallback_index)

    def _is_traj_deletable(self, traj_idx: int) -> bool:
        if not (0 <= int(traj_idx) < len(self.trajectories)):
            return False
        return is_deletable_trajectory_record(self.trajectories[int(traj_idx)], int(traj_idx))

    def _is_traj_pending_deleted(self, traj_idx: int) -> bool:
        if not hasattr(self, "pending_deleted_traj_keys"):
            self._reset_pending_delete_state()
        return self._trajectory_key_for_index(int(traj_idx)) in self.pending_deleted_traj_keys

    def _pending_delete_count(self) -> int:
        if not hasattr(self, "pending_deleted_traj_keys"):
            self._reset_pending_delete_state()
        return len(self.pending_deleted_traj_keys)

    def _visible_trajectory_indices(self) -> list[int]:
        if not hasattr(self, "pending_deleted_traj_keys"):
            self._reset_pending_delete_state()
        return [
            idx
            for idx in range(len(self.trajectories))
            if not self._is_traj_pending_deleted(idx)
        ]

    def _visible_editable_trajectory_indices(self) -> list[int]:
        return [
            idx
            for idx in self._visible_trajectory_indices()
            if not self._is_gt_trajectory(self.trajectories[idx], idx)
        ]

    def _select_nearest_visible_trajectory(self, preferred_idx: int | None = None) -> int:
        visible = self._visible_trajectory_indices()
        if not visible:
            self.current_traj_idx = 0
            return 0

        if preferred_idx is None:
            preferred_idx = int(getattr(self, "current_traj_idx", 0))
        preferred_idx = int(preferred_idx)
        if preferred_idx in visible:
            self.current_traj_idx = preferred_idx
            return preferred_idx

        editable = [
            idx for idx in visible
            if idx >= preferred_idx and not self._is_gt_trajectory(self.trajectories[idx], idx)
        ]
        if not editable:
            editable = [
                idx for idx in reversed(visible)
                if idx <= preferred_idx and not self._is_gt_trajectory(self.trajectories[idx], idx)
            ]
        if not editable:
            editable = visible
        self.current_traj_idx = int(editable[0])
        return self.current_traj_idx

    def _stage_delete_traj_idx(self, traj_idx: int) -> bool:
        if getattr(self, "traj_geom_edit_active", False):
            return False
        traj_idx = int(traj_idx)
        if not self._is_traj_deletable(traj_idx):
            return False
        if not hasattr(self, "pending_deleted_traj_keys"):
            self._reset_pending_delete_state()

        key = self._trajectory_key_for_index(traj_idx)
        if key in self.pending_deleted_traj_keys:
            return False
        manual_snapshot = self._clear_manual_points_for_pending_delete()
        self.pending_deleted_traj_keys.add(key)
        self.pending_delete_stack.append((key, traj_idx, manual_snapshot))
        if hasattr(self, "trajectory_states"):
            self.trajectory_states[traj_idx] = False
        self._select_nearest_visible_trajectory(traj_idx + 1)
        return True

    def _delete_traj(self):
        if not (0 <= int(self.current_traj_idx) < len(self.trajectories)):
            return
        if getattr(self, "traj_geom_edit_active", False):
            messagebox.showwarning(
                "Trajectory Edit Active",
                "Save or cancel the current trajectory geometry edit before deleting trajectories.",
            )
            return
        traj_idx = int(self.current_traj_idx)
        if not self._stage_delete_traj_idx(traj_idx):
            messagebox.showwarning(
                "Keep GT",
                "The GT row stays in the parquet. Delete generated/manual/cluster trajectories only.",
            )
            return
        self._update_display()

    def _undo_delete_traj(self, redraw: bool = True) -> bool:
        if not hasattr(self, "pending_delete_stack"):
            self._reset_pending_delete_state()
        while self.pending_delete_stack:
            key, traj_idx, manual_snapshot = self.pending_delete_stack.pop()
            if key not in self.pending_deleted_traj_keys:
                continue
            self.pending_deleted_traj_keys.remove(key)
            self._restore_manual_points_snapshot(manual_snapshot)
            if hasattr(self, "trajectory_states") and 0 <= int(traj_idx) < len(self.trajectory_states):
                self.trajectory_states[int(traj_idx)] = True
            self.current_traj_idx = int(traj_idx)
            if redraw and hasattr(self, "_update_display"):
                self._update_display()
            return True
        return False

    def _keep_traj(self):
        self._undo_delete_traj(redraw=True)

    def _persist_pending_manual_point_deletes(self) -> None:
        if bool(getattr(self, "pending_manual_points_delete", False)) and hasattr(self, "_persist_current_manual_points"):
            self._persist_current_manual_points()

    def _traj_index_from_listbox_row(self, listbox_row: int) -> int | None:
        if not hasattr(self, "traj_listbox_to_traj_idx"):
            return None
        row = int(listbox_row)
        if not (0 <= row < len(self.traj_listbox_to_traj_idx)):
            return None
        return int(self.traj_listbox_to_traj_idx[row])
