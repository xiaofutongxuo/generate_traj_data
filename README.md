# VLM Trajectory Generation with Alpamayo 1.5

本项目用于基于 Alpamayo 1.5 生成多候选未来轨迹，并通过增强版 GUI
检查、清洗、手工补充和修正轨迹。当前代码仓库位置已经迁移到：

```bash
/home/ubuntu/Public/hzq/generate_traj_data
```

## Current Machine Paths

当前机器上建议显式使用下面这些路径，避免依赖旧默认值：

```bash
PROJECT_ROOT=/home/ubuntu/Public/hzq/generate_traj_data
PYTHON=/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python
MODEL_PATH=/home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B

# run_inference.py 使用的数据缓存和标定目录
TRAIN_DATA_ROOT=/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted
CALIBRATION_DIR=/home/ubuntu/Public/yzb/triplane_tokenization/cailibration

# trajectory_gui_enhanced.py 读取原始可视化/GT 数据的位置
GUI_DATA_ROOT=/home/ubuntu/Public/train_data

# 当前 hzq 仓库下的输出目录
OUTPUT_DIR=/home/ubuntu/Public/hzq/generate_traj_data/output
VIS_OUTPUT_DIR=/home/ubuntu/Public/hzq/generate_traj_data/visualizations
```

注意：项目代码在 `hzq/generate_traj_data`，但当前可用的数据缓存和标定文件仍在
`yzb/triplane_tokenization` 下；增强 GUI 的原始数据目录是
`/home/ubuntu/Public/train_data`。

## Quick Start

### Generate Expert-Only Trajectories

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
  --model_path /home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B \
  --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --output_dir /home/ubuntu/Public/hzq/generate_traj_data/output \
  --vis_output_dir /home/ubuntu/Public/hzq/generate_traj_data/visualizations \
  --t0_source video_frames \
  --frame_stride 1 \
  --batch_size 6 \
  --num_traj_samples 6 \
  --expert
```

默认 `--t0_source video_frames --frame_stride 1` 会按主视频时间戳逐帧生成，视频约 10Hz，
即 0.1s 一个 `t0_us`。旧的速度候选抽样模式仍可用：显式传
`--t0_source speed_candidates --candidate_stride N`。

`--expert` 模式默认按 1 条轨迹一个内部 chunk 采样，再拼回
`--num_traj_samples 6`，可降低 24GB RTX 3090 上的 OOM 风险。只有在显存更宽裕时才建议设置
`EXPERT_SAMPLE_CHUNK_SIZE` 来提高内部 chunk 大小。

### Open Enhanced GUI

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python trajectory_gui_enhanced.py \
  --data_root /home/ubuntu/Public/train_data \
  --output_dir /home/ubuntu/Public/hzq/generate_traj_data/output \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --no_restore_last
```

GUI 会打开 Tk 窗口。本机如果 `$DISPLAY` 未设置，增强 GUI 会尝试自动使用本地
`:1` 桌面显示；远程或 headless session 需要 VNC、桌面会话或 SSH X11 forwarding。

GUI 默认 `--index_mode video_frames --frame_stride 1`，会按
`data-timestamps/{clip}.timestamps.parquet` 中的主视频帧逐帧建索引。状态栏会显示当前
clip 覆盖率，例如 `Video t0: N | Generated t0: M | Missing: K`，并标出当前帧是否已有
generated 轨迹。只想看 output 中已生成的 t0 时，可以传 `--index_mode generated`。

GUI 默认会把最后浏览位置保存到：

```text
/home/ubuntu/Public/hzq/generate_traj_data/output/.trajectory_gui_state.json
```

常用启动定位参数：

```bash
--start_index 37
--start_dataset data_26_3_24_1_converted --start_clip 2026-03-24-12-06-59
--start_dataset data_26_3_24_1_converted --start_clip 2026-03-24-12-06-59 --start_t0 1774325223236390
--no_restore_last
```

只查看 GT 样本，不加载生成轨迹：

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python trajectory_gui_enhanced.py \
  --data_root /home/ubuntu/Public/train_data \
  --output_dir /home/ubuntu/Public/hzq/generate_traj_data/output \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --gt_only \
  --index_mode video_frames \
  --frame_stride 1
```

旧的稀疏 GT-only 浏览方式仍可用：`--gt_only --index_mode generated --gt_stride_frames 3`。

## Project Structure

```text
generate_traj_data/
├── config.py                      # Inference configuration defaults
├── model_loader.py                # Alpamayo 1.5 model loading utilities
├── data_loader.py                 # Raw/data-cache loading and GT helpers
├── calibration_loader.py          # Camera calibration loading
├── visualization.py               # Trajectory projection visualization
├── run_inference.py               # Main trajectory generation script
├── frame_index.py                  # Shared video-frame t0 indexing helpers
├── trajectory_gui.py              # Legacy GUI, kept unchanged
├── trajectory_gui_enhanced.py     # Backward-compatible enhanced GUI entrypoint
├── traj_gui_enhanced/             # Refactored enhanced GUI package
│   ├── cli.py                     # Enhanced GUI CLI
│   ├── viewer.py                  # TrajectoryViewerEnhanced composition
│   ├── constants.py               # Colors, camera maps, thresholds
│   ├── environment.py             # Runtime DISPLAY / torch-light setup
│   ├── math_utils.py              # Bezier, resampling, trajectory math
│   ├── speed_utils.py             # Speed profile, stop, smoothing helpers
│   ├── projection_utils.py        # Camera projection helpers
│   ├── cluster_utils.py           # Cluster preview and diagnostics helpers
│   └── mixins/                    # GUI feature areas split by responsibility
├── tests/                         # unittest helper coverage
├── docs/
│   └── trajectory_gui_optimization_log.md
├── output/                        # Generated parquet files and GUI state
└── README.md
```

`trajectory_gui_enhanced.py` 现在只是兼容入口，继续支持历史命令：

```bash
python trajectory_gui_enhanced.py ...
```

新的内部实现入口是：

```python
from traj_gui_enhanced.viewer import TrajectoryViewerEnhanced
from traj_gui_enhanced.cli import parse_args, main
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU for inference
- Existing Alpamayo runtime:

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python --version
```

Main runtime dependencies are expected to be available in that environment:

```bash
pip install torch numpy opencv-python scipy einops transformers
pip install matplotlib pandas tqdm scikit-learn pyarrow
```

The local `.venv` is useful for lightweight helper tests, but full model inference should use
`/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python`.

## Inference Usage

### Basic Inference

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
  --model_path /home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B \
  --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --output_dir /home/ubuntu/Public/hzq/generate_traj_data/output \
  --vis_output_dir /home/ubuntu/Public/hzq/generate_traj_data/visualizations \
  --num_traj_samples 6
```

### With Optional Checkpoints

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
  --model_path /home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B \
  --traj_checkpoint /path/to/traj_checkpoint.pt \
  --lora_adapter /path/to/lora/adapter \
  --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --output_dir /home/ubuntu/Public/hzq/generate_traj_data/output \
  --vis_output_dir /home/ubuntu/Public/hzq/generate_traj_data/visualizations
```

### Select Specific Datasets

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
  --datasets data_26_3_24_1_converted,data_26_3_24_2_converted \
  --model_path /home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B \
  --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --output_dir /home/ubuntu/Public/hzq/generate_traj_data/output
```

### Common Options

建议在当前机器上显式传入路径参数。

| Argument | Recommended value / default | Description |
|----------|-----------------------------|-------------|
| `--model_path` | `/home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B` | Alpamayo 1.5 model path |
| `--traj_checkpoint` | None | Trajectory finetune checkpoint |
| `--lora_adapter` | None | LoRA adapter |
| `--train_data_root` | `/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted` | Inference data cache root |
| `--calibration_dir` | `/home/ubuntu/Public/yzb/triplane_tokenization/cailibration` | Calibration directory |
| `--output_dir` | `/home/ubuntu/Public/hzq/generate_traj_data/output` | Generated trajectory parquet output |
| `--vis_output_dir` | `/home/ubuntu/Public/hzq/generate_traj_data/visualizations` | Visualization image output |
| `--datasets` | Empty means all | Comma-separated dataset names |
| `--num_traj_samples` | 6 | Number of sampled trajectories per t0 |
| `--num_vis_samples` | 10 | Number of samples to visualize |
| `--vis_camera` | `FC` | Camera for saved visualization images |
| `--t0_source` | `video_frames` | `video_frames` uses every master video timestamp; `speed_candidates` uses the legacy sparse speed-filtered candidates |
| `--frame_stride` | 1 | Video-frame stride when `--t0_source video_frames` is used |
| `--min_speed_mps` | 2.0 | Minimum speed for legacy `speed_candidates` mode |
| `--max_samples` | 0 | Max t0 samples to process, 0 means all |
| `--batch_size` | 2 | Valid t0 frames per model batch |
| `--candidate_stride` | 3 | Process every Nth legacy speed candidate; ignored by `video_frames` mode |
| `--no_speed_filter` | False | Disable min-speed filtering in legacy `speed_candidates` mode |
| `--temperature` | 0.6 | Sampling temperature |
| `--top_p` | 0.98 | Top-p sampling |
| `--seed` | 42 | Random seed |
| `--expert` | False | Use expert branch without VLM CoT |

## Data Layout

### GUI Raw Data Root

`trajectory_gui_enhanced.py --data_root` expects raw converted data, currently:

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

### Inference Data Cache

`run_inference.py --train_data_root` currently points to:

```text
/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted/
```

### Calibration Directory

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
└── {dataset_name}/
    └── {clip_stem}.egomotion.parquet
```

Each parquet row is one trajectory sample for one `t0_us`. Important columns include:

```text
t0_us, sample_idx, timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz, curvature
```

`cot.jsonl` is a sidecar file containing generated Chain-of-Thought text when CoT output exists.

### Visualizations

Saved visualization images are written under `visualizations/`. The enhanced GUI projects
predicted trajectories and GT trajectories onto the configured cameras using the calibration files.

## Enhanced GUI Cleaning Workflow

Use `trajectory_gui_enhanced.py` for review and manual augmentation:

- Select dataset, clip, and `t0_us` from the top toolbar, or use the start arguments above.
- The default GUI index follows every 10Hz video frame. Frames without generated rows remain browsable,
  with an empty trajectory list and `Current: no generated` in the status bar.
- Delete generated trajectories with `Delete` / `Backspace`; `Ctrl+S` or `Save` writes the active
  parquet file in place after removing discarded rows.
- Enable `Draw Bezier`, add control points on BEV or the `FC` camera view, drag points to adjust,
  and inspect the cyan preview in BEV and camera projection.
- `Save Curve Traj` appends the manual Bezier trajectory to the current clip parquet.
- Manual controls and stop markers are stored in `output/manual_points.json`.
- The speed panels can inspect and edit predicted or GT speed profiles, then write the updated
  trajectory back to the active parquet file.
- Cluster-center preview and Bezier-center saving use files under `k_means/`.

## Camera Models

| Camera | Description |
|--------|-------------|
| `FC` | Front Center |
| `FC_FAR` | Front Center Far |
| `FL` | Front Left |
| `FR` | Front Right |
| `RL` | Rear Left |
| `RC` | Rear Center |
| `RR` | Rear Right |

## Validation

Lightweight checks after code or README changes:

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m py_compile frame_index.py data_loader.py run_inference.py trajectory_gui_enhanced.py traj_gui_enhanced/*.py traj_gui_enhanced/mixins/*.py
./.venv/bin/python trajectory_gui_enhanced.py --help
```

For manual GUI smoke testing, use the "Open Enhanced GUI" command above and confirm sample switching,
trajectory selection, BEV/FC projection, speed hover, delete/keep, manual Bezier save, and GT speed
editing still work as expected.

## Notes

- Trajectories are in BEV coordinates: `[x_right, y_forward, z_up]` in meters.
- Projection uses the raw distorted camera model with `opencv_rational_8` distortion.
- The enhanced GUI implementation has been split into `traj_gui_enhanced/`; the root
  `trajectory_gui_enhanced.py` file is intentionally a thin compatibility wrapper.
