# VLM Trajectory Generation with Alpamayo 1.5

#  真实alpamayo轨迹
cd /home/ubuntu/Public/yzb/generate_traj_data
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py --candidate_stride 1 --batch_size 6 --expert

# 可视化 



/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python trajectory_gui_enhanced.py \
  --data_root /home/ubuntu/Public/train_data \
  --output_dir /home/ubuntu/Public/yzb/generate_traj_data/output \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --no_restore_last


Note: in `--expert` mode, the code samples one trajectory at a time by default
and concatenates the chunks back to `--num_traj_samples 6`, which avoids OOM on
a 24GB RTX 3090. Set `EXPERT_SAMPLE_CHUNK_SIZE` only if you have more free GPU
memory and want to sample more trajectories per internal chunk.

The visualization command opens a Tk GUI. On this workstation it auto-detects
the local `:1` desktop display if `$DISPLAY` is unset; on a remote/headless
session use a desktop session, VNC, or SSH X11 forwarding.

The GUI remembers the last viewed sample in
`/home/ubuntu/Public/yzb/generate_traj_data/output/.trajectory_gui_state.json`
and resumes there by default. You can also use the Dataset / Clip / t0 selectors
and the Jump button in the top toolbar to move to a specific sample. To manually
choose the starting data from the command line, append one of these options:

```bash
--start_index 37
--start_dataset data_26_3_24_1_converted --start_clip 2026-03-24-12-06-59
--start_dataset data_26_3_24_1_converted --start_clip 2026-03-24-12-06-59 --start_t0 1774325223236390
--no_restore_last
```

Generate trajectory predictions using the Alpamayo 1.5 Vision Language Model and visualize them projected onto camera images.

## Project Structure

```
generate_traj_data/
├── config.py              # Configuration classes
├── model_loader.py        # Alpamayo 1.5 model loading utilities
├── data_loader.py         # Data loading from train_data directory
├── calibration_loader.py  # Camera calibration parameter loading
├── visualization.py       # Trajectory projection and visualization
├── run_inference.py       # Main inference script
└── README.md             # This file
```

## Setup

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Dependencies

The following packages are required (some are included in the alpamayo_1.5 environment):

```bash
pip install torch numpy opencv-python scipy einops transformers
pip install matplotlib pandas tqdm
pip install scikit-learn
```

Use the existing Alpamayo uv virtual environment:

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python --version
```

## Usage

### Basic Usage

```bash
cd /home/ubuntu/Public/yzb/generate_traj_data

/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
    --model_path /home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B \
    --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
    --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
    --output_dir ./output \
    --vis_output_dir ./visualizations \
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
    --output_dir ./output \
    --vis_output_dir ./visualizations
```

### Select Specific Datasets

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python run_inference.py \
    --datasets data_26_3_24_1_converted,data_26_3_24_2_converted \
    --train_data_root /home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted \
    --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration
```

### Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `/home/ubuntu/Public/lxh/models/Alpamayo-1.5-10B` | Path to Alpamayo 1.5 model |
| `--traj_checkpoint` | None | Path to trajectory finetune checkpoint |
| `--lora_adapter` | None | Path to LoRA adapter |
| `--train_data_root` | `/home/ubuntu/Public/yzb/triplane_tokenization/data_cache/alpamayo_extracted` | Root directory for training data |
| `--calibration_dir` | `/home/ubuntu/Public/yzb/triplane_tokenization/cailibration` | Directory containing calibration files |
| `--output_dir` | `./output` | Output directory for trajectories |
| `--vis_output_dir` | `./visualizations` | Output directory for visualizations |
| `--datasets` | "" | Comma-separated dataset names (empty = all) |
| `--num_traj_samples` | 6 | Number of trajectory samples per input |
| `--num_vis_samples` | 10 | Number of samples to visualize |
| `--vis_camera` | `FC` | Camera for visualization |
| `--min_speed_mps` | 2.0 | Minimum speed for t0 candidates |
| `--max_samples` | 0 | Max samples to process (0 = all) |
| `--temperature` | 0.6 | Sampling temperature |
| `--top_p` | 0.98 | Top-p sampling |
| `--seed` | 42 | Random seed |

## Data Format

### Expected Training Data Structure

```
train_data/
├── data_26_3_24_1_converted/
│   ├── data-egomotion/
│   │   └── {segment}.egomotion.npy
│   ├── data-timestamps/
│   │   ├── {segment}.timestamps.npy
│   │   └── {segment}_fovs_*.timestamps.npy
│   └── mp4-converted/
│       └── {segment}_fovs_{CAM}.mp4
├── data_26_3_24_2_converted/
│   └── ...
```

### Expected Calibration Structure

```
cailibration/
├── manifest.json
├── data_26_3_24_1_roll_only_3d_raw_distorted_extrinsics.jsonl
├── data_26_3_24_2_roll_only_3d_raw_distorted_extrinsics.jsonl
└── ...
```

## Output

### Trajectory Results

Generated trajectories are saved to `output/trajectories.json`:

```json
{
  "config": {
    "model_path": "...",
    "num_traj_samples": 6,
    "temperature": 0.6,
    "top_p": 0.98
  },
  "results": [
    {
      "clip_id": "2026-03-24-12-06-59",
      "dataset_name": "data_26_3_24_1_converted",
      "t0_us": 1234567890,
      "num_samples": 6,
      "trajectory_length": 64,
      "predicted_trajectories": [[[[x, y, z], ...]]],
      "gt_trajectory": [[[[x, y, z], ...]]],
      "t0_heading": {...}
    }
  ]
}
```

### Visualizations

Visualization images show the predicted trajectories projected onto camera images:

- Predicted trajectories: Different colors for each sample
- Ground truth trajectory: Gray/thin line
- Both are projected using the camera calibration parameters

### Enhanced GUI Data Cleaning

Use `trajectory_gui_enhanced.py` for high-quality cleaning and manual multimodal
trajectory augmentation:

- Delete generated trajectories with `Delete` / `Backspace`, then save filtered
  results with `Ctrl+S` or the `Save` button. Filtered parquet files are written
  under `output/{dataset}/filtered/`.
- Enable `Draw Bezier`, left-click any number of control/waypoint points on the
  `FC` camera image, right-drag existing points to adjust them, and inspect the
  cyan preview in both BEV and camera projection. The smooth curve passes through
  each clicked point.
- Click `Save Curve Traj` to convert the FC control points into a smooth 64-step
  future trajectory (`z=0`) and append it to the filtered parquet for the current
  `t0_us`. The curve starts from the ego origin and uses the current local heading
  and estimated t0 speed from ego history for a smoother initial segment.

## Camera Models

The system supports these camera models:

| Camera | Description |
|--------|-------------|
| FC | Front Center |
| FC_FAR | Front Center Far |
| FL | Front Left |
| FR | Front Right |
| RL | Rear Left |
| RC | Rear Center |
| RR | Rear Right |

## Notes

- Trajectories are in BEV coordinates: `[x_right, y_forward, z_up]` in meters
- Projection uses the raw distorted model with `opencv_rational_8` distortion
- The VLM generates multiple trajectory samples per input using diffusion
- Each sample can be inspected via the visualization output
