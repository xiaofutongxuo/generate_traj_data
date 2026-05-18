# Project Structure Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate Alpamayo inference from the GUI annotation tool so annotators can use a clearly named GUI-only trajectory diversity workflow.

**Architecture:** Shared data/calibration/frame/dynamics utilities live in `traj_core/`; GUI annotation code lives in `traj_annotation/`; Alpamayo inference lives in `traj_inference/`. The GUI entrypoint is `trajectory_annotator.py`; the inference entrypoint remains root `run_inference.py` as a thin wrapper around `traj_inference.runner`.

**Tech Stack:** Python packages, Tkinter GUI, pandas/pyarrow parquet IO, existing `unittest` regression suite.

---

### Task 1: Structure Boundary Tests

**Files:**
- Create: `tests/test_project_structure.py`

- [x] **Step 1: Write failing tests**

Add tests that import `trajectory_annotator`, `traj_annotation.cli`, `traj_core.data_loader`, `traj_core.dynamics`, and `traj_inference.runner`. Assert `trajectory_gui_enhanced.py` is no longer present as the annotator entrypoint.

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
./.venv/bin/python -m unittest tests.test_project_structure
```

Expected before migration: fail because the new packages and entrypoint do not exist.

### Task 2: Move Shared Core Modules

**Files:**
- Move: `data_loader.py` -> `traj_core/data_loader.py`
- Move: `calibration_loader.py` -> `traj_core/calibration_loader.py`
- Move: `frame_index.py` -> `traj_core/frame_index.py`
- Move: `visualization.py` -> `traj_core/visualization.py`
- Move: `traj_gui_enhanced/dynamics/` -> `traj_core/dynamics/`
- Move: `traj_gui_enhanced/trajectory_identity.py` -> `traj_core/trajectory_identity.py`
- Move: `traj_gui_enhanced/constants.py`, `math_utils.py`, `speed_utils.py`, `cluster_utils.py` -> `traj_core/`
- Create: `traj_core/__init__.py`

- [x] **Step 1: Move files**
- [x] **Step 2: Update imports from top-level modules and `traj_gui_enhanced.*` to `traj_core.*`**
- [x] **Step 3: Run structure tests**

### Task 3: Move Annotation GUI

**Files:**
- Move: `traj_gui_enhanced/` -> `traj_annotation/`
- Create: `trajectory_annotator.py`
- Delete: `trajectory_gui_enhanced.py`

- [x] **Step 1: Move package**
- [x] **Step 2: Update imports from `traj_gui_enhanced` to `traj_annotation`**
- [x] **Step 3: Update GUI imports of core utilities**
- [x] **Step 4: Run GUI helper tests**

### Task 4: Move Inference Code

**Files:**
- Move: `config.py` -> `traj_inference/config.py`
- Move: `model_loader.py` -> `traj_inference/model_loader.py`
- Move: `run_inference.py` -> `traj_inference/runner.py`
- Create: `traj_inference/__init__.py`
- Create: root `run_inference.py`

- [x] **Step 1: Move inference implementation**
- [x] **Step 2: Make root `run_inference.py` call `traj_inference.runner.main`**
- [x] **Step 3: Update imports in `traj_inference/runner.py`**
- [x] **Step 4: Keep `run_inference.py --help` import-light when torch/model loader are absent**

### Task 5: Update Tests And Docs

**Files:**
- Modify: `tests/test_gui_helpers.py`
- Modify: `README.md`
- Modify: `docs/trajectory_gui_feature_status_and_todo.md`
- Modify: `docs/trajectory_gui_optimization_log.md`
- Modify: `docs/trajectory_gui_user_manual.html`

- [x] **Step 1: Update tests to new packages**
- [x] **Step 2: Update README commands to `trajectory_annotator.py`**
- [x] **Step 3: Update TODO/current status docs to describe `traj_core`, `traj_annotation`, `traj_inference`**
- [x] **Step 4: Append optimization log entry**

### Task 6: Final Verification

**Files:**
- Test: full repo

- [x] **Step 1: Run unit tests**

```bash
./.venv/bin/python -m unittest discover -s tests
```

- [x] **Step 2: Run compile check**

```bash
./.venv/bin/python -m py_compile $(find traj_core traj_annotation traj_inference -name '*.py' -print) run_inference.py trajectory_annotator.py trajectory_gui.py trajectory_viewer.py
```

- [x] **Step 3: Run annotator help**

```bash
./.venv/bin/python trajectory_annotator.py --help
```

- [x] **Step 4: Run inference help without model loading**

```bash
./.venv/bin/python run_inference.py --help
```

- [x] **Step 5: Run diff check**

```bash
git diff --check
```
