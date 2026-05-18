# Windows GUI Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trajectory annotation GUI usable on Windows for browsing, manual expansion, editing, deletion, save audit, and backup restore, without supporting Windows model inference.

**Architecture:** Keep the existing Tk GUI and data/parquet logic unchanged where possible. Add cross-platform GUI defaults and platform-aware runtime setup, and keep Linux-only Alpamayo inference dependencies out of GUI import paths.

**Tech Stack:** Python standard library, Tkinter, pathlib, pandas/pyarrow, OpenCV, Pillow, existing `unittest` suite.

---

> Current structure note: after the project split, the GUI package is `traj_annotation/`, shared helpers live in `traj_core/`, and the GUI entrypoint is `trajectory_annotator.py`.

### Task 1: Cross-Platform GUI Defaults

**Files:**
- Modify: `traj_annotation/cli.py`
- Test: `tests/test_gui_helpers.py`

- [x] **Step 1: Write failing tests**

Add tests that import `traj_annotation.cli` and assert default GUI paths come from environment variables first, otherwise from relative cross-platform paths instead of `/home/...`.

- [x] **Step 2: Verify tests fail**

Run:

```bash
./.venv/bin/python -m unittest tests.test_gui_helpers.GuiHelperTests.test_gui_cli_defaults_are_cross_platform tests.test_gui_helpers.GuiHelperTests.test_gui_cli_defaults_use_environment_variables
```

- [x] **Step 3: Implement defaults**

Add helper functions in `traj_annotation/cli.py`:

```python
def _env_first(*names: str, default: str) -> str:
    ...

def default_data_root() -> str:
    ...

def default_output_dir() -> str:
    ...

def default_calibration_dir() -> str:
    ...
```

Use these helpers as argparse defaults.

- [x] **Step 4: Verify tests pass**

### Task 2: Windows Runtime Environment Setup

**Files:**
- Modify: `traj_annotation/environment.py`
- Test: `tests/test_gui_helpers.py`

- [x] **Step 1: Write failing tests**

Add tests that simulate Windows platform and missing Tkinter, then assert `setup_environment(platform_name="win32")` does not set `DISPLAY` and does not invoke Linux `.runtime/tk` fallback.

- [x] **Step 2: Verify tests fail**
- [x] **Step 3: Implement platform-aware setup**
- [x] **Step 4: Verify tests pass**

### Task 3: Remove GUI Import Linux Path Side Effect

**Files:**
- Modify: `traj_core/data_loader.py`
- Test: `tests/test_gui_helpers.py`

- [x] **Step 1: Write failing test**

Add a test that asserts `traj_core.data_loader` no longer prepends `/home/tsingyu/lxh/alpamayo_1.5/src` unconditionally.

- [x] **Step 2: Verify test fails**
- [x] **Step 3: Implement minimal cleanup**
- [x] **Step 4: Verify test passes**

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/trajectory_gui_feature_status_and_todo.md`
- Modify: `docs/trajectory_gui_optimization_log.md`

- [x] **Step 1: Update docs**

Document Windows GUI-only scope, PowerShell startup example, environment variables, dependency notes, and unsupported Windows model inference.

- [x] **Step 2: Verify docs do not contradict current behavior**

### Task 5: Final Verification

**Files:**
- Test: full repo

- [x] **Step 1: Run full unit tests**

```bash
./.venv/bin/python -m unittest discover -s tests
```

- [x] **Step 2: Run compile check**

```bash
./.venv/bin/python -m py_compile $(find traj_core traj_annotation traj_inference -name '*.py' -print) run_inference.py trajectory_annotator.py trajectory_gui.py trajectory_viewer.py
```

- [x] **Step 3: Run CLI help**

```bash
./.venv/bin/python trajectory_annotator.py --help
```

- [x] **Step 4: Run whitespace check**

```bash
git diff --check
```
