"""Audited parquet writes for enhanced GUI save operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import uuid

import pandas as pd


EDIT_LOG_FILENAME = "edit_log.jsonl"
BACKUP_DIRNAME = ".backups"
GUI_METADATA_COLUMNS = ("edit_version", "edited_by_gui", "edit_time", "edit_operation")


@dataclass(frozen=True)
class SaveAuditRecord:
    """Result of one audited parquet write."""

    traj_file: Path
    backup_file: Path | None
    log_file: Path
    operation: str
    edit_time: str
    rows_before: int
    rows_after: int
    affected_rows: int


@dataclass(frozen=True)
class FileAuditRecord:
    """Result of one audited non-parquet file write or restore."""

    target_file: Path
    backup_file: Path | None
    log_file: Path
    operation: str
    edit_time: str
    bytes_before: int
    bytes_after: int
    affected_rows: int


def utc_edit_time() -> str:
    """Return an ISO-8601 UTC timestamp for GUI edit metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip())
    return slug.strip("._") or "save"


def _int_or_zero(value) -> int:
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def apply_gui_edit_metadata(
    df: pd.DataFrame,
    row_indices,
    operation: str,
    edit_time: str | None = None,
) -> pd.DataFrame:
    """Return a copy with GUI edit metadata applied to selected rows."""
    updated = df.copy()
    for column in GUI_METADATA_COLUMNS:
        if column not in updated.columns:
            updated[column] = pd.NA

    timestamp = edit_time or utc_edit_time()
    for row_idx in row_indices:
        if row_idx not in updated.index:
            continue
        current_version = _int_or_zero(updated.at[row_idx, "edit_version"])
        updated.at[row_idx, "edit_version"] = current_version + 1
        updated.at[row_idx, "edited_by_gui"] = True
        updated.at[row_idx, "edit_time"] = timestamp
        updated.at[row_idx, "edit_operation"] = str(operation)
    return updated


def _backup_path_for_write(
    output_dir: Path,
    dataset_name: str,
    clip_stem: str,
    operation: str,
    edit_time: str,
) -> Path:
    timestamp_part = _safe_slug(edit_time.replace(":", "").replace("+", ""))
    operation_part = _safe_slug(operation)
    filename = f"{timestamp_part}-{operation_part}-{uuid.uuid4().hex[:8]}.egomotion.parquet"
    return output_dir / BACKUP_DIRNAME / str(dataset_name) / str(clip_stem) / filename


def _relative_or_absolute(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _backup_group_parts(backup_group: str) -> list[str]:
    return [_safe_slug(part) for part in Path(str(backup_group)).parts if part not in {"", "."}]


def _default_backup_group_for_file(target: Path, root: Path) -> str:
    try:
        relative = target.relative_to(root)
        return str(relative.with_suffix(""))
    except ValueError:
        return _safe_slug(target.stem)


def _backup_path_for_file_write(
    output_dir: Path,
    target: Path,
    operation: str,
    edit_time: str,
    backup_group: str | None = None,
) -> Path:
    group = backup_group or _default_backup_group_for_file(target, output_dir)
    timestamp_part = _safe_slug(edit_time.replace(":", "").replace("+", ""))
    operation_part = _safe_slug(operation)
    suffix = "".join(target.suffixes) or ".bak"
    filename = f"{timestamp_part}-{operation_part}-{uuid.uuid4().hex[:8]}{suffix}"
    return output_dir / BACKUP_DIRNAME / "files" / Path(*_backup_group_parts(group)) / filename


def _append_audit_log(output_dir: Path, log_row: dict) -> Path:
    log_file = output_dir / EDIT_LOG_FILENAME
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_row, ensure_ascii=False, sort_keys=True) + "\n")
    return log_file


def read_audit_log_rows(output_dir: str | Path, limit: int | None = None) -> list[dict]:
    """Read JSONL audit rows, skipping malformed historical lines."""
    log_file = Path(output_dir) / EDIT_LOG_FILENAME
    if not log_file.exists():
        return []
    rows: list[dict] = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    if limit is None or limit <= 0:
        return rows
    return rows[-int(limit):]


def _path_from_log_value(value: str | None, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def latest_backup_for_target(
    output_dir: str | Path,
    target_file: str | Path,
    *,
    dataset_name: str | None = None,
    clip_stem: str | None = None,
) -> tuple[Path, dict] | None:
    """Return the latest existing backup path and log row for a target file."""
    root = Path(output_dir)
    target = Path(target_file)
    target_abs = target.resolve()
    target_rel = _relative_or_absolute(target, root)
    for row in reversed(read_audit_log_rows(root)):
        if dataset_name is not None and str(row.get("dataset_name", "")) != str(dataset_name):
            continue
        if clip_stem is not None and str(row.get("clip_stem", "")) != str(clip_stem):
            continue
        row_target = row.get("traj_file") or row.get("target_file")
        row_target_path = _path_from_log_value(row_target, root)
        if row_target_path is None:
            continue
        try:
            matches_target = row_target_path.resolve() == target_abs
        except FileNotFoundError:
            matches_target = _relative_or_absolute(row_target_path, root) == target_rel
        if not matches_target and row_target != target_rel:
            continue
        backup_path = _path_from_log_value(row.get("backup_file"), root)
        if backup_path is not None and backup_path.exists():
            return backup_path, row
    return None


def write_text_file_with_audit(
    target_file: str | Path,
    content: str,
    *,
    output_dir: str | Path,
    operation: str,
    dataset_name: str | None = None,
    clip_stem: str | None = None,
    t0_us: int | None = None,
    affected_rows: int = 0,
    edit_time: str | None = None,
    metadata: dict | None = None,
    backup_group: str | None = None,
) -> FileAuditRecord:
    """Backup an existing text file, atomically replace it, and append an audit row."""
    target = Path(target_file)
    root = Path(output_dir)
    timestamp = edit_time or utc_edit_time()
    bytes_before = target.stat().st_size if target.exists() else 0
    backup_file: Path | None = None

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup_file = _backup_path_for_file_write(
            root,
            target=target,
            operation=operation,
            edit_time=timestamp,
            backup_group=backup_group,
        )
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_file)

    tmp_file = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp_file.write_text(content, encoding="utf-8")
        tmp_file.replace(target)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

    bytes_after = target.stat().st_size if target.exists() else 0
    log_row = {
        "edit_time": timestamp,
        "operation": str(operation),
        "file_kind": "text",
        "dataset_name": str(dataset_name) if dataset_name is not None else "",
        "clip_stem": str(clip_stem) if clip_stem is not None else "",
        "t0_us": int(t0_us) if t0_us is not None else None,
        "target_file": _relative_or_absolute(target, root),
        "backup_file": _relative_or_absolute(backup_file, root),
        "bytes_before": int(bytes_before),
        "bytes_after": int(bytes_after),
        "affected_rows": int(affected_rows),
    }
    if metadata:
        log_row["metadata"] = metadata
    log_file = _append_audit_log(root, log_row)

    return FileAuditRecord(
        target_file=target,
        backup_file=backup_file,
        log_file=log_file,
        operation=str(operation),
        edit_time=timestamp,
        bytes_before=int(bytes_before),
        bytes_after=int(bytes_after),
        affected_rows=int(affected_rows),
    )


def restore_file_from_backup_with_audit(
    target_file: str | Path,
    source_backup_file: str | Path,
    *,
    output_dir: str | Path,
    operation: str = "restore_backup",
    dataset_name: str | None = None,
    clip_stem: str | None = None,
    t0_us: int | None = None,
    affected_rows: int = 0,
    edit_time: str | None = None,
    metadata: dict | None = None,
    backup_group: str | None = None,
) -> FileAuditRecord:
    """Restore a backup over the active file, preserving the pre-restore active file."""
    target = Path(target_file)
    source = Path(source_backup_file)
    if not source.exists():
        raise FileNotFoundError(source)

    root = Path(output_dir)
    timestamp = edit_time or utc_edit_time()
    bytes_before = target.stat().st_size if target.exists() else 0
    backup_file: Path | None = None

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup_file = _backup_path_for_file_write(
            root,
            target=target,
            operation=operation,
            edit_time=timestamp,
            backup_group=backup_group,
        )
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_file)

    tmp_file = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.restore.tmp")
    try:
        shutil.copy2(source, tmp_file)
        tmp_file.replace(target)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

    bytes_after = target.stat().st_size if target.exists() else 0
    log_metadata = dict(metadata or {})
    log_metadata["restored_from_backup"] = _relative_or_absolute(source, root)
    log_row = {
        "edit_time": timestamp,
        "operation": str(operation),
        "file_kind": "restore",
        "dataset_name": str(dataset_name) if dataset_name is not None else "",
        "clip_stem": str(clip_stem) if clip_stem is not None else "",
        "t0_us": int(t0_us) if t0_us is not None else None,
        "target_file": _relative_or_absolute(target, root),
        "backup_file": _relative_or_absolute(backup_file, root),
        "bytes_before": int(bytes_before),
        "bytes_after": int(bytes_after),
        "affected_rows": int(affected_rows),
        "metadata": log_metadata,
    }
    log_file = _append_audit_log(root, log_row)

    return FileAuditRecord(
        target_file=target,
        backup_file=backup_file,
        log_file=log_file,
        operation=str(operation),
        edit_time=timestamp,
        bytes_before=int(bytes_before),
        bytes_after=int(bytes_after),
        affected_rows=int(affected_rows),
    )


def write_parquet_with_audit(
    traj_file: str | Path,
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    operation: str,
    dataset_name: str,
    clip_stem: str,
    t0_us: int | None = None,
    affected_rows: int = 0,
    edit_time: str | None = None,
    metadata: dict | None = None,
) -> SaveAuditRecord:
    """Backup the current parquet, write the replacement, and append an edit log row."""
    target = Path(traj_file)
    root = Path(output_dir)
    timestamp = edit_time or utc_edit_time()
    rows_before = 0
    backup_file: Path | None = None

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            rows_before = int(len(pd.read_parquet(target)))
        except Exception:
            rows_before = 0
        backup_file = _backup_path_for_write(
            root,
            dataset_name=dataset_name,
            clip_stem=clip_stem,
            operation=operation,
            edit_time=timestamp,
        )
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_file)

    tmp_file = target.with_name(f"{target.name}.{uuid.uuid4().hex[:8]}.tmp.parquet")
    try:
        df.to_parquet(tmp_file, index=False)
        tmp_file.replace(target)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

    log_row = {
        "edit_time": timestamp,
        "operation": str(operation),
        "dataset_name": str(dataset_name),
        "clip_stem": str(clip_stem),
        "t0_us": int(t0_us) if t0_us is not None else None,
        "traj_file": _relative_or_absolute(target, root),
        "backup_file": _relative_or_absolute(backup_file, root),
        "rows_before": rows_before,
        "rows_after": int(len(df)),
        "affected_rows": int(affected_rows),
    }
    if metadata:
        log_row["metadata"] = metadata
    log_file = _append_audit_log(root, log_row)

    return SaveAuditRecord(
        traj_file=target,
        backup_file=backup_file,
        log_file=log_file,
        operation=str(operation),
        edit_time=timestamp,
        rows_before=rows_before,
        rows_after=int(len(df)),
        affected_rows=int(affected_rows),
    )


__all__ = [
    "BACKUP_DIRNAME",
    "EDIT_LOG_FILENAME",
    "FileAuditRecord",
    "GUI_METADATA_COLUMNS",
    "SaveAuditRecord",
    "apply_gui_edit_metadata",
    "latest_backup_for_target",
    "read_audit_log_rows",
    "restore_file_from_backup_with_audit",
    "utc_edit_time",
    "write_text_file_with_audit",
    "write_parquet_with_audit",
]
