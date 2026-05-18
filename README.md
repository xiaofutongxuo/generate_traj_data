# Trajectory Generation and Annotation Tools

本项目包含两条互相分离的工作流：

- `traj_inference/`：基于 Alpamayo 1.5 生成多候选未来轨迹。
- `traj_annotation/`：面向标注人员的 GUI 工具，用于检查、清洗、手工补充、编辑和删除伪 GT 轨迹。

共享的数据读取、标定、视频帧索引、轨迹动力学和 parquet 字段工具放在 `traj_core/`。当前代码仓库位置是：

```bash
/home/ubuntu/Public/hzq/generate_traj_data
```

## Current Machine Paths

当前机器上建议显式使用下面这些路径，避免依赖旧默认值：

```bash
PROJECT_ROOT=/home/ubuntu/Public/hzq/generate_traj_data
PYTHON=/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python
MODEL_PATH=/home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B

# run_inference.py / traj_inference 使用的数据缓存和兜底标定目录
TRAIN_DATA_ROOT=/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted
CALIBRATION_DIR=/home/ubuntu/Public/yzb/triplane_tokenization/cailibration

# trajectory_annotator.py / traj_annotation 读取原始可视化/GT 数据的位置
GUI_DATA_ROOT=/home/ubuntu/Public/train_data
RAW_TRAIN_DATA_ROOT=/home/ubuntu/Public/train_data

# 当前使用的轨迹输出目录
OUTPUT_DIR=/home/ubuntu/Public/yzb/generate_traj_data/output
VIS_OUTPUT_DIR=/home/ubuntu/Public/hzq/generate_traj_data/visualizations
```

注意：项目代码在 `hzq/generate_traj_data`，但当前可用的数据缓存仍在
`yzb/triplane_tokenization` 下；标注 GUI 的原始数据目录是
`/home/ubuntu/Public/train_data`。相机标定优先从
`/home/ubuntu/Public/train_data/{dataset}_converted/calibration/` 读取，旧的
`CALIBRATION_DIR` 仅作为 JSONL 兜底目录。

## Quick Start

### Generate Expert-Only Trajectories

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
  --model_path /home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B \
  --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --output_dir /home/ubuntu/Public/yzb/generate_traj_data/output \
  --vis_output_dir /home/ubuntu/Public/hzq/generate_traj_data/visualizations \
  --t0_source video_frames \
  --frame_stride 1 \
  --batch_size 6 \
  --num_traj_samples 6 \
  --expert
```

默认 `--t0_source video_frames --frame_stride 1` 会按主视频时间戳逐帧生成，视频约 10Hz，
即 0.1s 一个 `t0_us`。video-frame 模式会要求每个 `t0_us` 当前帧和后续 6.4s 的真实 egomotion 连续可用；相邻
egomotion 缺口不超过 0.3s 时会插值补小范围漏帧，超过 0.3s 视为断点。clip 尾部如果能
按 timestamp 接上下一个 clip，会继续使用后续 clip 的 egomotion；最后一段或不连续尾部会被过滤。

`--expert` 模式默认按 1 条轨迹一个内部 chunk 采样，再拼回
`--num_traj_samples 6`，可降低 24GB RTX 3090 上的 OOM 风险。只有在显存更宽裕时才建议设置
`EXPERT_SAMPLE_CHUNK_SIZE` 来提高内部 chunk 大小。

### Open Trajectory Annotator

给标注同事使用时，优先阅读 `docs/GUI操作手册.html`；需要编辑文档时改
`docs/GUI操作手册.md` 后再重新导出 HTML。

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python trajectory_annotator.py \
  --data_root /home/ubuntu/Public/train_data \
  --output_dir /home/ubuntu/Public/yzb/generate_traj_data/output \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --no_restore_last
```

GUI 会打开 Tk 窗口。本机如果 `$DISPLAY` 未设置，标注 GUI 会尝试自动使用本地
`:1` 桌面显示；远程或 headless session 需要 VNC、桌面会话或 SSH X11 forwarding。

### Windows GUI-Only Usage

Windows first-version support is for the trajectory annotator GUI workflow only: browsing existing samples,
manual Bezier expansion, cluster-center expansion, trajectory editing/deletion, parquet writes, and
save audit/backup restore. Windows local Alpamayo model inference is not covered.

Install Python 3.10 or 3.11 from python.org with Tk support enabled, then create a virtual
environment from PowerShell:

```powershell
cd D:\path\to\generate_traj_data
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Launch the GUI with explicit Windows paths:

```powershell
.\.venv\Scripts\python.exe trajectory_annotator.py `
  --data_root D:\traj\train_data `
  --output_dir D:\traj\output `
  --calibration_dir D:\traj\calibration `
  --no_restore_last
```

Alternatively, set GUI-specific environment variables and omit the path flags:

```powershell
$env:GENERATE_TRAJ_GUI_DATA_ROOT = "D:\traj\train_data"
$env:GENERATE_TRAJ_GUI_OUTPUT_DIR = "D:\traj\output"
$env:GENERATE_TRAJ_GUI_CALIBRATION_DIR = "D:\traj\calibration"
.\.venv\Scripts\python.exe trajectory_annotator.py --no_restore_last
```

The GUI CLI defaults are cross-platform relative paths (`train_data`, `output`, `calibration`) when
these variables are not set. On Windows, runtime setup does not touch X11 `DISPLAY` or the
Linux-only `.runtime/tk` fallback.

GUI 默认 `--index_mode video_frames --frame_stride 1`，会按
`data-timestamps/{clip}.timestamps.parquet` 中的主视频帧逐帧建索引。状态栏会显示当前
clip 覆盖率，例如 `Video t0: N | Generated t0: M | Missing: K`，并标出当前帧是否已有
generated 轨迹。只想看 output 中已生成的 t0 时，可以传 `--index_mode generated`。
逐帧索引同样会过滤当前帧或未来 6.4s 不完整、跨大缺口的数据，避免 GUI 显示外推出来的 GT future。

GUI 默认会把最后浏览位置保存到：

```text
/home/ubuntu/Public/yzb/generate_traj_data/output/.trajectory_gui_state.json
```

常用启动定位参数：

```bash
--start_index 37
--start_dataset data_26_3_24_1_converted --start_clip 2026-03-24-12-06-59
--start_dataset data_26_3_24_1_converted --start_clip 2026-03-24-12-06-59 --start_t0 1774325223236390
--no_restore_last
```

## Project Structure

```text
generate_traj_data/
├── trajectory_annotator.py        # GUI annotation entrypoint
├── run_inference.py               # Thin inference compatibility entrypoint
├── traj_core/                     # Shared data, calibration, frame, dynamics utilities
│   ├── data_loader.py             # Raw/data-cache loading and GT helpers
│   ├── calibration_loader.py      # Camera calibration loading
│   ├── frame_index.py             # Shared video-frame t0 indexing helpers
│   ├── visualization.py           # Trajectory projection visualization
│   ├── constants.py               # Shared trajectory thresholds and colors
│   ├── math_utils.py              # Bezier, resampling, trajectory math
│   ├── speed_utils.py             # Speed profile, stop, smoothing helpers
│   ├── cluster_utils.py           # Cluster preview and diagnostics helpers
│   ├── trajectory_identity.py     # Source normalization and delete keys
│   └── dynamics/                  # Pseudo-GT dynamics diagnostics and optimization
├── traj_annotation/               # GUI trajectory diversity annotation tool
│   ├── cli.py                     # Annotator CLI
│   ├── viewer.py                  # TrajectoryViewerEnhanced composition
│   ├── environment.py             # Runtime DISPLAY / torch-light setup
│   ├── save_audit.py              # GUI save backup, log and restore helpers
│   ├── projection_utils.py        # GUI projection compatibility helpers
│   └── mixins/                    # GUI feature areas split by responsibility
├── traj_inference/                # Alpamayo model inference workflow
│   ├── config.py                  # Inference configuration defaults
│   ├── model_loader.py            # Alpamayo 1.5 model loading utilities
│   └── runner.py                  # Main trajectory generation implementation
├── tests/                         # unittest helper coverage
├── docs/                          # GUI status, TODO and user docs
├── output/                        # Generated parquet files and GUI state
└── README.md
```

## Setup

使用已有 Alpamayo runtime 运行推理和 GUI：

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python --version
```

本地 `.venv` 用于轻量 helper 测试；完整模型推理请使用
`/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python`.

## Inference Usage

Quick Start 命令覆盖了当前常用推理方式。需要缩小范围时加 `--datasets` 或
`--max_samples`；需要接入微调权重时加 `--traj_checkpoint` 或 `--lora_adapter`。

| Argument | Recommended value / default | Description |
|----------|-----------------------------|-------------|
| `--model_path` | `/home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B` | Alpamayo 1.5 model path |
| `--traj_checkpoint` | None | Trajectory finetune checkpoint |
| `--lora_adapter` | None | LoRA adapter |
| `--train_data_root` | `/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted` | Inference data cache root |
| `--calibration_dir` | `/home/ubuntu/Public/yzb/triplane_tokenization/cailibration` | Legacy JSONL fallback calibration directory; GUI prefers per-dataset XML under `--data_root` |
| `--output_dir` | `/home/ubuntu/Public/yzb/generate_traj_data/output` | Generated trajectory parquet output |
| `--vis_output_dir` | `/home/ubuntu/Public/hzq/generate_traj_data/visualizations` | Visualization image output |
| `--datasets` | Empty means all | Comma-separated dataset names |
| `--num_traj_samples` | 6 | Number of sampled trajectories per t0 |
| `--num_vis_samples` | 10 | Number of samples to visualize |
| `--vis_camera` | `FC` | Camera for saved visualization images |
| `--t0_source` | `video_frames` | Uses every master video timestamp |
| `--frame_stride` | 1 | Video-frame stride when `--t0_source video_frames` is used |
| `--max_samples` | 0 | Max t0 samples to process, 0 means all |
| `--batch_size` | 2 | Valid t0 frames per model batch |
| `--temperature` | 0.6 | Sampling temperature |
| `--top_p` | 0.98 | Top-p sampling |
| `--seed` | 42 | Random seed |
| `--expert` | False | Use expert branch without VLM CoT |

## Data Layout

### GUI Raw Data Root

`trajectory_annotator.py --data_root` expects raw converted data, currently:

```text
/home/ubuntu/Public/train_data/
├── data_26_3_24_1_converted/
│   ├── data-egomotion/
│   │   └── {segment}.egomotion.npy or {segment}.egomotion.parquet
│   ├── data-timestamps/
│   │   ├── {segment}.timestamps.npy
│   │   └── {segment}_fovs_*.timestamps.npy
│   └── mp4-converted/
│       └── {segment}_fovs_{CAM}.mp4
└── ...
```

### Scene Labels

标注 GUI 会在启动时尝试读取每个原始数据集目录下的场景类别标注：

```text
/home/ubuntu/Public/train_data/
└── {dataset_name}/
    └── scene_labels.json
```

当前支持的 JSON 结构为：

```json
{
  "clips": [
    {
      "clip": "2026-03-24-12-06-59",
      "points": [
        {
          "timestamp": 1774325223236390,
          "scenario_type": "straight"
        }
      ]
    }
  ]
}
```

`clip` 也可以写作 `segment`。GUI 会按 `(clip, timestamp)` 与当前可浏览样本的
`(clip_stem, t0_us)` 精确匹配，并把匹配到的 `scenario_type` 放入顶部 `Scene`
下拉框。选择某个场景后，左右切换样本会限制在当前 dataset 内同一场景类别的样本中；
选择 `None` 则恢复普通顺序导航。该功能只影响 GUI 浏览筛选，不会改变推理生成或
output parquet 内容。

### Inference Data Cache

`run_inference.py --train_data_root` currently points to:

```text
/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted/
```

### Calibration Files

GUI 会优先读取每个原始数据集目录下的本地 XML 标定：

```text
/home/ubuntu/Public/train_data/
├── data_26_5_14_1_converted/
│   ├── Vision_calibration.tar.gz
│   └── calibration/
│       ├── fc120/cameraIntrinsic.xml
│       ├── fc120/cameraExtrinsic.xml
│       ├── fc30/cameraIntrinsic.xml
│       ├── fc30/cameraExtrinsic.xml
│       ├── rl/cameraIntrinsic.xml
│       ├── rl/cameraExtrinsic.xml
│       └── ...
└── ...
```

`fc120` 会映射为 GUI 的 `FC`，`fc30` 会映射为 `FC_FAR`，
`fl/fr/rc/rl/rr` 分别映射为对应的大写相机名。旧的集中式 JSONL 目录仍作为兜底：
`run_inference.py` 如果 `--train_data_root` 指向推理缓存，也会尝试
`RAW_TRAIN_DATA_ROOT`（默认 `/home/ubuntu/Public/train_data`）读取本地 XML 标定。

```text
/home/ubuntu/Public/yzb/triplane_tokenization/cailibration/
├── manifest.json
├── data_26_3_24_1_roll_only_3d_raw_distorted_extrinsics.jsonl
├── data_26_3_24_2_roll_only_3d_raw_distorted_extrinsics.jsonl
└── ...
```

## Output

### Generated Trajectories

`run_inference.py` writes parquet files grouped by dataset and clip:

```text
output/
├── cot.jsonl
├── .trajectory_gui_state.json
├── edit_log.jsonl
├── .backups/
└── {dataset_name}/
    └── {clip_stem}.egomotion.parquet
```

Each parquet row is one generated or augmented pseudo-GT trajectory sample for one `t0_us`.
Real GT is loaded from `/home/ubuntu/Public/train_data` and is not written to output parquet.
Important columns include:

```text
t0_us, sample_idx, source, timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz,
curvature, edit_version, edited_by_gui, edit_time, edit_operation
```

Common `source` values are `vla`, `manual_bezier`, `cluster_center`, and legacy
`rule_cluster`. Legacy output rows with `source=gt` are ignored by the GUI's pseudo-GT
list and coverage counts; they are not treated as source-data GT.

Trajectory Annotator writes to active parquet in place, but every GUI parquet write first backs up the
previous file under `output/.backups/{dataset}/{clip}/` and appends one JSON row to
`output/edit_log.jsonl`. GUI-created or GUI-edited rows receive `edit_version`,
`edited_by_gui`, `edit_time`, and `edit_operation`; older rows without these columns remain
readable.

The same audit log is used for GUI sidecar edits. `output/manual_points.json` is backed up under
`output/.backups/files/manual_points/`; cluster-center library edits are backed up under
`output/.backups/files/k_means/`. In the GUI, `View Log` shows recent audit rows and
`Restore Backup` restores the latest backup for the current clip parquet after first backing up
the current active file.

Before pseudo-GT rows are written, VLA output, manual Bezier rows, cluster-center rows, and
saved speed edits are passed through `traj_core.dynamics` for conservative speed,
acceleration, jerk, and curvature checks. The same module recomputes `vx/vy/vz`,
`qx/qy/qz/qw`, and `curvature` from the optimized xyz points.

`cot.jsonl` is a sidecar file containing generated Chain-of-Thought text when CoT output exists.

### Visualizations

Saved visualization images are written under `visualizations/`. The trajectory annotator projects
predicted trajectories and GT trajectories onto the configured cameras using the calibration files.

### Coordinate Convention

输出 parquet 中的 `x/y/z` 使用 ego-local 坐标：`x` 为前向、`y` 为左向、`z` 为上向。
投影到相机时，代码会在内部转换到标定使用的 BEV 轴系：右向/前向/上向。

## Trajectory Annotator Workflow

Use `trajectory_annotator.py` for review and manual augmentation:

- Select dataset, clip, and `t0_us` from the top toolbar, or use the start arguments above.
- If `scene_labels.json` exists under the current dataset, use the `Scene` dropdown to filter
  sample navigation by matched `scenario_type`; `None` disables the scene filter.
- The default GUI index follows every 10Hz video frame. Frames without generated rows remain browsable,
  with an empty trajectory list and `Current: no generated` in the status bar. A video frame is kept
  only when the current frame and next 6.4s of source egomotion are continuously covered; small
  egomotion gaps up to 0.3s are interpolated, while larger gaps or final tails without enough future
  are filtered out.
- Delete generated trajectories with `Delete` / `Backspace`; `Ctrl+S` or `Save` writes the active
  parquet file in place after removing discarded rows.
- Use `View Log` to inspect recent GUI save operations, and `Restore Backup` to restore the latest
  backup for the current clip parquet.
- Use `Edit Traj` on a selected pseudo-GT row to show BEV keyframe handles. Drag handles to reshape
  the saved trajectory, then use `Save Edit`, `Cancel Edit`, or `Restore Edit`.
- Enable `Draw Bezier`, add control points on BEV or the `FC` camera view, drag points to adjust,
  and inspect the cyan preview in BEV and camera projection.
- `Save Curve Traj` appends the manual Bezier trajectory to the current clip parquet.
- Manual controls and stop markers are stored in `output/manual_points.json`; GUI writes are backed
  up and logged.
- The speed panels can inspect predicted pseudo-GT speed profiles and source-data GT speed profiles.
  GT speed is computed from source GT xyz differences and lightly smoothed for display to reduce
  raw-position noise spikes. History trajectory display is also lightly denoised with the current
  frame fixed, and the green history speed curve uses that display-smoothed history. Pseudo-GT edits
  write to output parquet; source-data GT is displayed for reference and is not written to output parquet.
- Cluster-center preview and Bezier-center saving use files under `k_means/`; GUI edits to those
  files are backed up and logged.

## Validation

Lightweight checks after code or README changes:

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m py_compile traj_core/*.py traj_core/dynamics/*.py traj_inference/*.py run_inference.py trajectory_annotator.py traj_annotation/*.py traj_annotation/mixins/*.py
./.venv/bin/python trajectory_annotator.py --help
./.venv/bin/python run_inference.py --help
```

For manual GUI smoke testing, use the "Open Trajectory Annotator" command above and confirm sample switching,
trajectory selection, BEV/FC projection, speed hover, delete/keep, manual Bezier save, and GT speed
editing still work as expected.
