# Trajectory Annotator

本项目当前主要用于轨迹多样化标注：打开 GUI，检查已有伪 GT 轨迹，删除不合理轨迹，补充 Bezier 或 cluster center 轨迹，并把结果写回 `output`。

当前唯一推荐的 GUI 入口是：

```bash
python trajectory_annotator.py
```

不要再使用旧入口 `trajectory_gui_enhanced.py` 或旧包名 `traj_gui_enhanced/`。当前 GUI 代码在 `traj_annotation/`，共享工具在 `traj_core/`。

完整标注说明见：

- `docs/GUI操作手册.md`
- `docs/GUI操作手册.html`

## 1. 项目结构

```text
generate_traj_data/
├── trajectory_annotator.py        # GUI 标注入口
├── run_inference.py               # VLA/Alpamayo 推理入口，可选，不给标注人员使用
├── requirements.txt               # GUI-only 依赖，不包含模型推理依赖
├── traj_annotation/               # GUI 界面、交互、保存、审计
├── traj_core/                     # 数据读取、标定、帧索引、动力学、object 读取
├── traj_inference/                # Alpamayo 推理逻辑
├── k_means/                       # cluster center 轨迹库
├── tests/                         # 单元测试
├── docs/
└── output/                        # GUI 写回的伪 GT、日志、备份
```

## 2. 标注需要的数据

启动 GUI 前至少需要：

```text
train_data/
└── {dataset}_converted/
    ├── data-egomotion/
    │   └── {clip}.egomotion.parquet
    ├── data-timestamps/
    │   ├── {clip}.timestamps.parquet
    │   └── {clip}_fovs_{CAM}.timestamps.parquet
    ├── mp4-converted/
    │   └── {clip}_fovs_{CAM}.mp4
    ├── calibration/
    └── data-objects/              # 可选，感知目标，仅辅助参考
        └── {clip}.objects.parquet
```

还需要一个 `output/` 目录，用来存放和写回伪 GT：

```text
output/
├── {dataset}_converted/
│   └── {clip}.egomotion.parquet
├── manual_points.json             # GUI 自动生成
├── edit_log.jsonl                 # GUI 自动生成
├── .trajectory_gui_state.json     # GUI 自动生成
└── .backups/                      # GUI 自动生成
```

说明：

- `train_data` 是源数据和真实 GT，GUI 不会修改它。
- `output` 是标注结果目录，GUI 会写回这里的 parquet。
- `calibration/` 优先从 `train_data/{dataset}/calibration/` 读取；`--calibration_dir` 只是旧 JSONL 标定的兜底目录。
- `data-objects/` 来自原始 CSD 的感知结果，可能漏检，也可能误检，只能辅助判断碰撞风险，不能当成真值。

## 3. 安装 GUI 环境

推荐 Python 3.10 或 3.11。

### Conda

```bash
cd /path/to/generate_traj_data

conda create -n traj_gui python=3.10 -y
conda activate traj_gui
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import tkinter, cv2, pandas, pyarrow, scipy; from PIL import Image; print('GUI env OK')"
```

如果 `tkinter` 报错：

```bash
conda install -c conda-forge tk -y
```

### Windows venv

安装 Python 时请勾选 Tcl/Tk。

```powershell
cd D:\path\to\generate_traj_data

py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux venv

```bash
cd /path/to/generate_traj_data

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Ubuntu 如果缺少 Tk：

```bash
sudo apt install python3-tk
```

## 4. 启动 GUI

### 当前机器示例

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

/home/ubuntu/Public/hzq/generate_traj_data/.venv/bin/python trajectory_annotator.py \
  --data_root /home/ubuntu/Public/train_data \
  --output_dir /home/ubuntu/Public/yzb/generate_traj_data/output \
  --index_mode video_frames \
  --frame_stride 5 \
  --cameras RL,FC,RR \
  --no_restore_last
```

### Windows 示例

```powershell
cd D:\path\to\generate_traj_data

.\.venv\Scripts\python.exe trajectory_annotator.py `
  --data_root D:\traj\train_data `
  --output_dir D:\traj\output `
  --index_mode video_frames `
  --frame_stride 5 `
  --cameras RL,FC,RR `
  --no_restore_last
```

### 常用定位参数

```bash
--start_dataset data_26_3_24_2_converted
--start_clip 2026-03-24-12-33-20
--start_t0 1774326806736412
--no_restore_last
```

只看已有 `output` 中的样本：

```bash
--index_mode generated
```

只看真实 GT，不加载伪 GT：

```bash
--gt_only
```

只显示前视相机：

```bash
--cameras FC
```

如果 `train_data/{dataset}/calibration/` 不存在，才需要额外加 `--calibration_dir /path/to/old_calibration` 指向旧 JSONL 标定目录。

## 5. GUI 功能摘要

- 顶部选择 Dataset、Clip、t0、Scene。
- 左侧 BEV 显示历史轨迹、GT future、伪 GT、Bezier/cluster 预览。
- 中间相机图显示当前帧和轨迹投影。
- 右侧轨迹列表显示当前 t0 下可编辑的伪 GT。
- 底部速度图用于检查速度突变、停车和不平滑轨迹。
- `Draw Bezier` 可手动画一条新轨迹并保存。
- `Cluster Centers` 可选择常见路线，预览后保存。
- `Edit Traj` 可编辑已保存的非 GT 轨迹。
- `Save` 会写回 active parquet，写入前自动备份，写入后记录 `edit_log.jsonl`。
- `Restore Backup` 可恢复当前 clip 最近一次 parquet 备份。

### 交通参与者显示

如果提供了 `data-objects/`，勾选 BEV 顶部的 `交通参与者` 后，GUI 会在 BEV 和 FC 前视图中用点显示感知目标位置。

注意：

- 这些点来自原始感知结果，不是真值。
- `bev_object` 也会漏检和误检。
- 雨天、护栏、路边设施、远处车辆、侧方目标都可能导致点位不准。
- 这个功能只作为碰撞风险辅助参考，最终仍要结合相机画面和轨迹本身判断。

## 6. 输出和备份

GUI 保存后主要影响：

```text
output/{dataset}/{clip}.egomotion.parquet
output/manual_points.json
output/edit_log.jsonl
output/.backups/
```

建议：

- 标注前备份一份原始 `output`。
- 多人不要同时写同一个 `output` 目录。
- 不要删除 `.backups/` 和 `edit_log.jsonl`。

## 7. 可选：模型推理

标注人员不需要运行模型推理。模型推理仍通过：

```bash
python run_inference.py --help
```

推理依赖不在 `requirements.txt` 里。需要使用 Alpamayo/VLA 推理时，请使用专门的模型环境和 `traj_inference/` 相关配置。

## 8. 验证

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data

./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m py_compile \
  trajectory_annotator.py run_inference.py \
  traj_core/*.py traj_core/dynamics/*.py \
  traj_annotation/*.py traj_annotation/mixins/*.py \
  traj_inference/*.py
./.venv/bin/python trajectory_annotator.py --help
```

## 9. 常见问题

- GUI 没样本：检查 `--data_root`、`data-timestamps/`、`data-egomotion/`，以及当前 t0 后是否有 6.4s 连续 future。
- 图像不显示：检查 `mp4-converted/`、相机 timestamp 和 `--cameras`。
- 投影不对：优先检查 `train_data/{dataset}/calibration/`。
- 没有交通参与者点：检查 `data-objects/{clip}.objects.parquet` 和 `交通参与者` 复选框。
- 手动关闭窗口后输入异常：当前窗口右上角 X 已接入清理流程；如果仍复现，记录启动命令和关闭方式再排查。
