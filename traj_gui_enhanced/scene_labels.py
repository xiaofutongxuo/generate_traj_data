"""Scene-label loading helpers for the enhanced trajectory GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SampleKey = tuple[str, str, int]


def _load_dataset_scene_labels(data_root: str | Path, dataset_name: str) -> dict[tuple[str, int], str]:
    """Load one dataset's scene labels keyed by clip and timestamp."""
    label_file = Path(data_root) / str(dataset_name) / "scene_labels.json"
    if not label_file.exists():
        return {}
    try:
        data = json.loads(label_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: Could not load scene labels {label_file}: {exc}")
        return {}

    result: dict[tuple[str, int], str] = {}
    for clip_record in data.get("clips", []) if isinstance(data, dict) else []:
        if not isinstance(clip_record, dict):
            continue
        clip_stem = str(clip_record.get("clip") or clip_record.get("segment") or "").strip()
        if not clip_stem:
            continue
        for point in clip_record.get("points", []):
            if not isinstance(point, dict):
                continue
            label = str(point.get("scenario_type") or "").strip()
            if not label:
                continue
            try:
                timestamp = int(point["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            result[(clip_stem, timestamp)] = label
    return result


def build_scene_label_index(
    data_root: str | Path,
    samples: Iterable[SampleKey],
) -> tuple[dict[SampleKey, str], dict[str, list[str]]]:
    """Build scene labels for visible samples and sorted scene choices per dataset."""
    samples_by_dataset: dict[str, list[tuple[str, int]]] = {}
    for dataset_name, clip_stem, t0_us in samples:
        samples_by_dataset.setdefault(str(dataset_name), []).append((str(clip_stem), int(t0_us)))

    scene_by_sample: dict[SampleKey, str] = {}
    scenes_by_dataset: dict[str, set[str]] = {}
    for dataset_name, dataset_samples in samples_by_dataset.items():
        labels = _load_dataset_scene_labels(data_root, dataset_name)
        if not labels:
            continue
        for clip_stem, t0_us in dataset_samples:
            scene = labels.get((clip_stem, int(t0_us)))
            if not scene:
                continue
            key = (dataset_name, clip_stem, int(t0_us))
            scene_by_sample[key] = scene
            scenes_by_dataset.setdefault(dataset_name, set()).add(scene)

    return (
        scene_by_sample,
        {dataset_name: sorted(scenes) for dataset_name, scenes in scenes_by_dataset.items()},
    )


__all__ = ["SampleKey", "build_scene_label_index"]
