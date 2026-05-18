"""ClusterControlsMixin for the enhanced trajectory GUI."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from ..tk_compat import Image, ImageTk, messagebox, tk, ttk

from ..environment import setup_environment
setup_environment()

from data_loader import get_dataset_names, get_clip_stems_from_dataset, load_data, get_t0_candidates
from calibration_loader import load_calibration_for_segment
from visualization import draw_trajectory_on_image, ego_to_bev_points, load_image_from_frame

from ..constants import *
from ..math_utils import *
from ..speed_utils import *
from ..projection_utils import *
from ..cluster_utils import *
from ..save_audit import apply_gui_edit_metadata, write_text_file_with_audit, write_parquet_with_audit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ClusterControlsMixin:

    def _load_cluster_center_library(self) -> dict[str, list[dict]]:
        """Load editable cluster center library from four category txt files."""
        kmeans_dir = PROJECT_ROOT / "k_means"
        library = {category: [] for category in CLUSTER_CATEGORY_ORDER}

        def _append_record(category: str, center_id: int, count: int, traj: np.ndarray, source: str):
            is_bezier_added = int(center_id) in self.bezier_cluster_center_ids.get(category, set())
            final_right = -float(traj[-1, 1])
            label_prefix = "B" if is_bezier_added else "C"
            label = (
                f"{label_prefix}{int(center_id):02d} n={int(count)} "
                f"right={final_right:.1f} fwd={float(traj[-1, 0]):.1f}"
            )
            library[category].append({
                "id": int(center_id),
                "label": label,
                "trajectory": traj.astype(np.float32),
                "source": source,
                "count": int(count),
                "category": category,
                "is_bezier_added": is_bezier_added,
            })

        def _parse_category_file(path: Path, category: str):
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts[0] != "CENTER" or len(parts) < 4:
                        continue
                    try:
                        center_id = int(parts[1])
                        count = int(parts[2])
                        xy = np.asarray(parts[3:], dtype=np.float64).reshape(-1, 2)
                    except ValueError:
                        continue
                    traj = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
                    _append_record(category, center_id, count, traj, path.name)

        category_files_exist = False
        for category, filename in CLUSTER_CATEGORY_FILES.items():
            path = kmeans_dir / filename
            if path.exists():
                category_files_exist = True
            _parse_category_file(path, category)
        if category_files_exist:
            for records in library.values():
                records.sort(key=lambda item: int(item["id"]))
            return library

        def _parse_result_file(path: Path, mode: str):
            if not path.exists():
                return
            headers = {}
            centers = {}
            counts = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        payload = line[1:].strip()
                        if "=" in payload:
                            key, value = payload.split("=", 1)
                            headers[key.strip()] = value.strip()
                        continue
                    parts = line.split()
                    if parts[0] != "CENTER":
                        continue
                    cluster_id = int(parts[1])
                    counts[cluster_id] = int(parts[2])
                    xy = np.asarray(parts[3:], dtype=np.float64).reshape(-1, 2)
                    traj = np.column_stack([xy, np.zeros(len(xy), dtype=np.float64)])
                    centers[cluster_id] = traj

            straight_clusters = int(headers.get("straight_clusters", "0") or 0)
            for cluster_id, traj in sorted(centers.items()):
                final_right = -float(traj[-1, 1])
                if mode == "stop":
                    category = "stop"
                elif straight_clusters and cluster_id < straight_clusters:
                    category = "straight"
                else:
                    category = "right" if final_right >= 0.0 else "left"
                _append_record(category, cluster_id, counts.get(cluster_id, 0), traj, path.name)

        _parse_result_file(kmeans_dir / "kmeans_results.txt", mode="main")
        _parse_result_file(kmeans_dir / "kmeans_stop_results.txt", mode="stop")
        for category, records in library.items():
            if records:
                self._write_cluster_category_file(category, records)
        return library

    def _cluster_category_file(self, category: str) -> Path:
        filename = CLUSTER_CATEGORY_FILES[category]
        return PROJECT_ROOT / "k_means" / filename

    def _bezier_cluster_center_meta_file(self) -> Path:
        return PROJECT_ROOT / "k_means" / "bezier_centers.json"

    def _load_bezier_cluster_center_ids(self) -> dict[str, set[int]]:
        """Load ids for centers created by Save Bezier Center."""
        ids = {category: set() for category in CLUSTER_CATEGORY_FILES}
        path = self._bezier_cluster_center_meta_file()
        if not path.exists():
            return ids

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load Bezier center metadata {path}: {e}")
            return ids

        raw_centers = data.get("centers", {}) if isinstance(data, dict) else {}
        for category, values in raw_centers.items():
            if category not in ids or not isinstance(values, list):
                continue
            for value in values:
                try:
                    ids[category].add(int(value))
                except (TypeError, ValueError):
                    continue
        return ids

    def _write_bezier_cluster_center_ids(self) -> None:
        """Persist ids for centers created by Save Bezier Center."""
        path = self._bezier_cluster_center_meta_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "centers": {
                category: sorted(int(value) for value in values)
                for category, values in sorted(self.bezier_cluster_center_ids.items())
                if values
            },
        }
        content = json.dumps(payload, indent=2) + "\n"
        write_text_file_with_audit(
            path,
            content,
            output_dir=self.output_dir,
            operation="write_bezier_cluster_center_ids",
            affected_rows=sum(len(values) for values in self.bezier_cluster_center_ids.values()),
            metadata={"categories": sorted(payload["centers"])},
            backup_group="k_means/bezier_centers",
        )

    def _write_cluster_category_file(self, category: str, records: list[dict]) -> None:
        path = self._cluster_category_file(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# cluster_centers_xy_v1",
            f"# category={category}",
            "# ego-local xy: x=forward, y=left; plot right=-y",
            "# columns: CENTER center_id count x0 y0 ... x63 y63",
        ]
        for record in sorted(records, key=lambda item: int(item["id"])):
            traj = np.asarray(record["trajectory"], dtype=np.float64)
            values = " ".join(f"{v:.6f}" for v in traj[:, :2].reshape(-1))
            lines.append(f"CENTER {int(record['id'])} {int(record.get('count', 1))} {values}")
        content = "\n".join(lines) + "\n"
        write_text_file_with_audit(
            path,
            content,
            output_dir=self.output_dir,
            operation="write_cluster_category_file",
            affected_rows=len(records),
            metadata={"category": category},
            backup_group=f"k_means/{path.stem}",
        )

    def _cluster_records_for_current_category(self) -> list[dict]:
        if self.cluster_category_var is None:
            return []
        return self.cluster_center_library.get(self.cluster_category_var.get(), [])

    def _refresh_cluster_choice_values(self):
        if self.cluster_choice_combo is None or self.cluster_choice_var is None:
            return
        values = [record["label"] for record in self._cluster_records_for_current_category()]
        self.cluster_choice_combo["values"] = values
        if values:
            if self.cluster_choice_var.get() not in values:
                self.cluster_choice_var.set(values[0])
        else:
            self.cluster_choice_var.set("")

    def _on_cluster_category_selected(self, _event=None):
        self._refresh_cluster_choice_values()
        self._preview_selected_cluster_center(show_warning=False)

    def _on_cluster_choice_selected(self, _event=None):
        self._preview_selected_cluster_center(show_warning=False)

    def _cycle_selected_cluster_center(self, direction: int):
        records = self._cluster_records_for_current_category()
        if not records or self.cluster_choice_var is None:
            self.cluster_preview_record = None
            self.cluster_preview_traj = None
            self._update_display()
            return

        labels = [record["label"] for record in records]
        current = self.cluster_choice_var.get()
        if current in labels:
            current_idx = labels.index(current)
        else:
            current_idx = 0
        next_idx = (current_idx + direction) % len(labels)
        self.cluster_choice_var.set(labels[next_idx])
        self._preview_selected_cluster_center(show_warning=False)

    def _selected_cluster_record(self) -> Optional[dict]:
        label = self.cluster_choice_var.get() if self.cluster_choice_var is not None else ""
        for record in self._cluster_records_for_current_category():
            if record["label"] == label:
                return record
        return None

    def _preview_selected_cluster_center(self, show_warning: bool = True):
        record = self._selected_cluster_record()
        if record is None:
            self.cluster_preview_record = None
            self.cluster_preview_traj = None
            self.cluster_preview_is_edited = False
            if show_warning:
                messagebox.showwarning("No Cluster", "No cluster center is available for this category.")
            self._update_display()
            return
        self.cluster_preview_record = record
        raw_traj = np.asarray(record["trajectory"], dtype=np.float32)
        smoothed_traj = _prepare_cluster_preview_trajectory(
            raw_traj,
            initial_speed_mps=self._estimate_t0_speed_mps(),
        )
        if smoothed_traj is None:
            self.cluster_preview_traj = None
            self.cluster_preview_is_edited = False
            if show_warning:
                messagebox.showwarning(
                    "Cluster Unavailable",
                    "This cluster center cannot be fit to the current t0 speed and acceleration limits.",
                )
            self._update_display()
            return
        self.cluster_preview_traj = smoothed_traj.astype(np.float32)
        self.cluster_preview_is_edited = False
        self._update_display()

    def _current_cluster_preview_label(self) -> str:
        if self.cluster_preview_traj is None or len(self.cluster_preview_traj) == 0:
            return "Cluster"
        cluster_id = "Cluster"
        count = None
        if self.cluster_preview_record is not None:
            cluster_id = f"C{int(self.cluster_preview_record.get('id', 0)):02d}"
            count = self.cluster_preview_record.get("count")
        final_forward = float(self.cluster_preview_traj[-1, 0])
        final_right = -float(self.cluster_preview_traj[-1, 1])
        suffix = " edited" if self.cluster_preview_is_edited else ""
        if count is None:
            return f"{cluster_id}{suffix} right={final_right:.1f} fwd={final_forward:.1f}"
        return f"{cluster_id}{suffix} n={int(count)} right={final_right:.1f} fwd={final_forward:.1f}"

    def _hide_cluster_preview(self) -> None:
        """Hide the current unsaved cluster preview without touching saved trajectories."""
        self.cluster_preview_record = None
        self.cluster_preview_traj = None
        self.cluster_preview_is_edited = False
        self._update_display()

    def _save_selected_cluster_center_trajectory(self):
        if self.gt_only:
            messagebox.showwarning(
                "GT Only",
                "GT-only mode does not load or append generated trajectory parquet files.",
            )
            return
        if self.cluster_preview_traj is None:
            self._preview_selected_cluster_center()
        if self.cluster_preview_traj is None:
            return

        dataset_name, clip_stem, t0_us = self.samples[self.current_idx]
        traj_file = self._active_traj_file(dataset_name, clip_stem)
        df = pd.read_parquet(traj_file)
        if "source" not in df.columns:
            df["source"] = ""
        smoothed_preview = _prepare_cluster_preview_trajectory(
            self.cluster_preview_traj,
            initial_speed_mps=self._estimate_t0_speed_mps(),
        )
        if smoothed_preview is None:
            messagebox.showwarning(
                "Save Failed",
                "The dragged cluster trajectory cannot be fit within the t0 speed and acceleration limits.",
            )
            return
        self.cluster_preview_traj = smoothed_preview.astype(np.float32)
        row = self._manual_trajectory_to_row(
            self.cluster_preview_traj,
            df,
            source="cluster_center",
        )

        for column in df.columns:
            if column not in row:
                row[column] = None
        new_row = pd.DataFrame([{column: row[column] for column in df.columns}])
        df_appended = pd.concat([df, new_row], ignore_index=True)
        new_row_idx = df_appended.index[-1]
        df_appended = apply_gui_edit_metadata(
            df_appended,
            row_indices=[new_row_idx],
            operation="append_cluster_center",
        )

        write_parquet_with_audit(
            traj_file,
            df_appended,
            output_dir=self.output_dir,
            operation="append_cluster_center",
            dataset_name=dataset_name,
            clip_stem=clip_stem,
            t0_us=int(t0_us),
            affected_rows=1,
            metadata={
                "sample_idx": int(row["sample_idx"]),
                "cluster_id": int(self.cluster_preview_record.get("id", -1))
                if self.cluster_preview_record is not None
                else None,
            },
        )

        self._load_sample(self.current_idx)
        self._update_display()

        cluster_label = (
            self._current_cluster_preview_label()
            if self.cluster_preview_traj is not None
            else ""
        )
        messagebox.showinfo(
            "Saved Cluster Center",
            (
                f"Appended {cluster_label} as sample_idx={row['sample_idx']} "
                f"for t0={int(t0_us)} to {traj_file}"
            ),
        )

    def _delete_current_bezier_cluster_center(self):
        """Delete the currently displayed center only if it was added from a Bezier curve."""
        record = self.cluster_preview_record
        if record is None and self.cluster_preview_traj is None:
            record = self._selected_cluster_record()
        category = str(record.get("category", "")) if record is not None else ""
        if record is None or category not in CLUSTER_CATEGORY_FILES:
            messagebox.showwarning("No Center", "Display a Bezier-added center before deleting.")
            return

        center_id = int(record.get("id", -1))
        if not bool(record.get("is_bezier_added", False)):
            messagebox.showwarning(
                "Cannot Delete",
                "Only the currently displayed center created by Save Bezier Center can be deleted.",
            )
            return

        label = str(record.get("label", f"C{center_id:02d}"))
        if not messagebox.askyesno(
            "Delete Bezier Center",
            f"Delete the currently displayed {category}/{label} from {self._cluster_category_file(category)}?",
        ):
            return

        records = [
            item for item in self.cluster_center_library.get(category, [])
            if int(item.get("id", -1)) != center_id
        ]
        self.cluster_center_library[category] = records
        self.bezier_cluster_center_ids.setdefault(category, set()).discard(center_id)
        self._write_cluster_category_file(category, records)
        self._write_bezier_cluster_center_ids()

        self.cluster_preview_record = None
        self.cluster_preview_traj = None
        self.cluster_preview_is_edited = False
        self._refresh_cluster_choice_values()
        self._preview_selected_cluster_center(show_warning=False)
        self._update_display()

        messagebox.showinfo(
            "Deleted Bezier Center",
            f"Deleted {category}/{label}.",
        )
