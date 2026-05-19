# GUI 轨迹多样化标注操作手册

本文档给只负责标注的同事使用。目标很简单：把工程和数据放到一台机器上，打开 GUI，检查已有伪 GT 轨迹，删除不好的轨迹，补充新的轨迹，最后保存结果。

这里不讲模型推理，也不需要在本机跑 VLA。标注人员只需要用 `trajectory_annotator.py`。

## 1. 这个工具是做什么的

这个 GUI 用来做轨迹多样化标注，主要能做这些事：

- 看每个时刻的相机画面、BEV 俯视图、真实 GT 未来轨迹和已有伪 GT 轨迹。
- 删除明显不合理的伪 GT 轨迹。
- 手动画一条 Bezier 曲线，保存为新的扩充轨迹。
- 从 cluster center 轨迹库里选择一条轨迹，预览、拖动终点、保存为新的扩充轨迹。
- 对已经保存的非 GT 轨迹做简单几何编辑或速度曲线优化。
- 保存时自动备份旧文件，并写入操作日志，方便后续追溯。

请记住一句话：

```text
train_data 里是真实数据和真实 GT；
output 里是要标注、要扩充、会被 GUI 修改的伪 GT。
```

## 2. 标注前需要准备什么

把工程从 GitHub 下载到新机器后，至少需要准备下面几类东西。

### 2.1 代码工程

也就是这个仓库本身，例如：

```text
generate_traj_data/
├── trajectory_annotator.py
├── requirements.txt
├── traj_annotation/
├── traj_core/
├── k_means/
└── docs/
```

其中 `k_means/` 里的 `stop.txt`、`straight.txt`、`left.txt`、`right.txt`、`s_curve.txt` 会被 GUI 用来加载 cluster center 轨迹。如果需要使用 cluster center 扩充功能，请确保这些文件也在工程里。

### 2.2 原始数据目录：`--data_root`

这是 GUI 读图片、读视频帧时间戳、读真实 GT 的地方。目录结构一般长这样：

```text
train_data/
└── data_xxx_converted/
    ├── data-egomotion/
    │   └── {clip}.egomotion.parquet
    ├── data-timestamps/
    │   ├── {clip}.timestamps.parquet
    │   └── {clip}_fovs_{CAM}.timestamps.parquet
    ├── data-objects/
    │   └── {clip}.objects.parquet
    ├── mp4-converted/
    │   └── {clip}_fovs_{CAM}.mp4
    ├── Vision_calibration.tar.gz
    └── calibration/
        ├── fc120/
        │   ├── cameraIntrinsic.xml
        │   └── cameraExtrinsic.xml
        ├── rl/
        ├── rr/
        └── ...
```

默认 GUI 会显示 `RL,FC,RR` 三个相机，所以至少要有这些相机对应的 mp4 和 timestamps 文件。如果启动时只想看前视相机，可以加：

```bash
--cameras FC
```

这样就只需要 FC 对应的视频和时间戳。

`data-objects/` 是交通参与者数据目录。没有它也能标注；有它的话，GUI 会在 BEV 和 FC 前视画面里用点标出其他车、行人等交通参与者的位置，更方便判断扩充轨迹有没有碰撞风险。

### 2.3 标注输出目录：`--output_dir`

这是 GUI 读取已有伪 GT、写入标注结果的地方。通常应该提前放好模型生成出来的 parquet：

```text
output/
└── {dataset_name}/
    └── {clip}.egomotion.parquet
```

GUI 保存后还会在这个目录下生成或更新：

```text
output/
├── manual_points.json
├── edit_log.jsonl
├── .trajectory_gui_state.json
├── .backups/
└── {dataset_name}/
    └── {clip}.egomotion.parquet
```

说明：

- `{dataset_name}` 必须和 `train_data/` 下面的数据集目录名一致。
- `{clip}` 必须和 `data-egomotion/`、`data-timestamps/`、`mp4-converted/` 里的 clip 名一致。
- GUI 会直接写回 active parquet，但写之前会自动备份到 `.backups/`。
- `edit_log.jsonl` 是操作日志，可以用来查看谁在什么时候保存了什么。

### 2.4 标定目录：`--calibration_dir`

优先推荐把标定文件放在 `train_data/{dataset}/calibration/` 里。这样 `--calibration_dir` 只是兜底路径。

如果 `train_data/{dataset}/calibration/` 不存在，就需要额外提供旧格式的集中式标定目录，例如：

```text
calibration/
├── manifest.json
├── data_xxx_roll_only_3d_raw_distorted_extrinsics.jsonl
└── ...
```

简单理解：

- 有 `train_data/{dataset}/calibration/`：基本够用。
- 没有本地 calibration：必须给 `--calibration_dir` 指向旧 JSONL 标定目录。

### 2.5 可选：交通参与者数据

如果需要在 BEV 里看交通参与者，需要提前从原始 `.csd` 提取：

```text
train_data/{dataset_name}/data-objects/{clip}.objects.parquet
```

这个文件一般由 `data_processed_tool` 生成，不需要标注人员自己手写。里面每一行是某一帧的一个交通参与者，包含位置、朝向、长宽高、速度等信息。

简单理解：

- 没有 `data-objects/`：GUI 正常打开，只是不显示交通参与者。
- 有 `data-objects/`：BEV 顶部勾选 `交通参与者` 后，BEV 和 FC 前视画面会同时显示交通参与者的位置点；取消勾选后两边一起隐藏。
- 标注同事不需要拿原始 `.csd`，只需要拿转换好的 parquet。

### 2.6 可选：场景类别文件

如果想按场景类别筛选样本，可以在数据集目录下放：

```text
train_data/{dataset_name}/scene_labels.json
```

格式示例：

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

没有这个文件也能正常标注，只是顶部 `Scene` 筛选功能不会显示可用类别。

## 3. 新机器安装依赖

推荐使用 Python 3.10 或 3.11。

可以用 conda，也可以不用 conda。两种方式选一种即可，不要混着装。

### 3.1 方式一：conda 环境

如果机器上已经有 Anaconda 或 Miniconda，推荐用这种方式。Windows、Linux 都可以用。

打开终端或 Anaconda Prompt，进入工程目录：

```bash
cd /path/to/generate_traj_data
```

创建环境：

```bash
conda create -n traj_gui python=3.10 -y
conda activate traj_gui
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

装完后可以简单检查一下：

```bash
python -c "import tkinter; import cv2; import pandas; import pyarrow; import scipy; from PIL import Image; print('GUI env OK')"
```

如果这里报 `tkinter` 相关错误，可以尝试：

```bash
conda install -c conda-forge tk -y
```

注意：

- conda 只是用来管理 Python 环境，不代表要安装模型推理依赖。
- 不需要安装 `torch`、`transformers`、`peft`、`accelerate`。
- 后续启动 GUI 前，先执行 `conda activate traj_gui`。

### 3.2 方式二：不用 conda，Windows venv

先安装 Python。安装时请勾选 Tcl/Tk，因为 GUI 依赖 tkinter。

然后打开 PowerShell：

```powershell
cd D:\path\to\generate_traj_data
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.3 方式三：不用 conda，Linux venv

```bash
cd /path/to/generate_traj_data
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

如果 Linux 上打开 GUI 报 tkinter 相关错误，通常是系统没有装 Tk。Ubuntu 可让维护人员安装：

```bash
sudo apt install python3-tk
```

## 4. 启动 GUI

### 4.1 conda 环境启动示例

如果第 3 节使用的是 conda，先激活环境，再启动：

```bash
cd /path/to/generate_traj_data
conda activate traj_gui

python trajectory_annotator.py \
  --data_root /path/to/train_data \
  --output_dir /path/to/output \
  --calibration_dir /path/to/calibration \
  --no_restore_last
```

Windows 上路径可以写成这样：

```powershell
cd D:\path\to\generate_traj_data
conda activate traj_gui

python trajectory_annotator.py `
  --data_root D:\traj\train_data `
  --output_dir D:\traj\output `
  --calibration_dir D:\traj\calibration `
  --no_restore_last
```

### 4.2 Windows venv 启动示例

```powershell
cd D:\path\to\generate_traj_data

.\.venv\Scripts\python.exe trajectory_annotator.py `
  --data_root D:\traj\train_data `
  --output_dir D:\traj\output `
  --calibration_dir D:\traj\calibration `
  --no_restore_last
```

如果标定已经在 `D:\traj\train_data\{dataset}\calibration\` 里面，`--calibration_dir` 仍然可以传一个空的兜底目录，例如 `D:\traj\calibration`。

### 4.3 Linux venv 启动示例

```bash
cd /path/to/generate_traj_data

./.venv/bin/python trajectory_annotator.py \
  --data_root /data/train_data \
  --output_dir /data/output \
  --calibration_dir /data/calibration \
  --no_restore_last
```

### 4.4 常用启动参数

只看某个数据集：

```bash
--start_dataset data_xxx_converted
```

打开某个 clip：

```bash
--start_dataset data_xxx_converted --start_clip 2026-03-24-12-06-59
```

打开某个具体时刻：

```bash
--start_dataset data_xxx_converted --start_clip 2026-03-24-12-06-59 --start_t0 1774325223236390
```

只看已有 output 里的样本，不按视频帧逐帧浏览：

```bash
--index_mode generated
```

只显示一个相机：

```bash
--cameras FC
```

## 5. 打开后先看哪里

GUI 打开后可以按这个顺序看：

1. 顶部工具栏：选择 Dataset、Clip、t0。
2. 左侧 BEV：看车身坐标系下的历史轨迹、真实 GT future、伪 GT 轨迹和手工/cluster 预览。
3. 中间相机图：看当前帧的真实画面，以及轨迹投影是否合理。
4. 右侧 Trajectories：看当前 t0 下已有的伪 GT 轨迹列表。
5. 底部速度图：看当前轨迹速度是否突变、是否停车、是否不平滑。
6. 状态栏：看当前 clip 覆盖情况，例如有多少视频帧、有多少已经生成伪 GT、有多少缺失。

颜色和列表状态不用死记，简单判断即可：

- 真实 GT 是参考，不会写回 output。
- 右侧列表里的非 GT 轨迹是可以删除、编辑、保存的伪 GT。
- 如果某条轨迹速度不平滑，列表里会有异常标记，BEV 上也更容易看出问题。
- 如果提供了 `data-objects/`，左侧 BEV 顶部可以勾选 `交通参与者`，BEV 和 FC 前视画面会同时显示其他交通参与者的位置点。

## 6. 推荐标注流程

建议每个样本按这个流程走：

1. 选中一个 Dataset、Clip、t0。
2. 看相机图和 BEV，确认这个时刻是否正常；如果有交通参与者显示，顺便看一下是否有明显碰撞风险。
3. 看右侧已有伪 GT 轨迹。
4. 明显不合理的轨迹先标记删除。
5. 如果轨迹种类不够，选择一种方式补轨迹：
   - 用 Draw Bezier 手动画；
   - 或用 Cluster Centers 选一个中心轨迹。
6. 看速度图，确认没有明显速度突变。
7. 点击保存按钮写回 output。
8. 切到下一个 t0。

## 7. 删除不好的伪 GT

操作方式：

1. 在右侧 `Trajectories` 列表里选中一条伪 GT。
2. 点击删除相关按钮，将它标记为待删除。
3. 如果点错了，可以撤销待删除状态。
4. 确认无误后点击保存，GUI 会把删除结果写回 parquet。

注意：

- 删除的是 output 里的伪 GT，不会删除 train_data 里的真实 GT。
- 保存前只是标记，保存后才真正写回文件。
- 保存时会自动备份旧 parquet。

## 8. 手动画 Bezier 轨迹

适合场景：已有轨迹不够，或者想补一条更合理的路线。

基本步骤：

1. 点击 `Draw Bezier`。
2. 在 BEV 或 FC 相机图上点控制点。
3. 可以拖动控制点调整曲线。
4. 如果这条轨迹最后要停车，点击 `Add Final Stop`，再设置 `Stop Time(s)`。
5. 看 BEV、相机投影和速度图。
6. 确认合理后点击 `Save Curve Traj`。

保存后：

- 新轨迹会写入当前 clip 的 parquet。
- `source` 会是 `manual_bezier`。
- 手工控制点会记录到 `output/manual_points.json`。
- 写入前会做保守的动力学优化，尽量避免速度和加速度太离谱。

## 9. 用 Cluster Center 扩充轨迹

适合场景：想快速补一条常见路线，例如直行、左转、右转、停车、S 曲线。

基本步骤：

1. 在 `Cluster Centers` 区域选择类别：`stop`、`straight`、`left`、`right`、`s_curve`。
2. 选择某个 center。
3. GUI 会在 BEV 和相机图中显示预览轨迹。
4. 如果终点不合适，可以拖动 cluster endpoint。
5. 看速度图是否合理。
6. 确认后点击 `Confirm Save`。

保存后：

- 新轨迹会写入当前 clip 的 parquet。
- `source` 会是 `cluster_center`。
- 写入前会做速度、加速度、曲率等检查和优化。

几个容易混淆的点：

- `Hide` 只是隐藏当前还没保存的预览，不会删除文件。
- `Delete Current Bezier Center` 删除的是 cluster center 库里由 `Save Bezier Center` 创建的 center，不是已经保存进 output 的轨迹。
- 已经保存进 output 的 cluster 轨迹，要在右侧 Trajectories 列表里删除。

## 10. 编辑已经保存的轨迹

如果某条非 GT 轨迹大体方向对，但局部不太好，可以尝试编辑。

常见操作：

- `Edit Traj`：进入几何编辑模式。
- 拖动 BEV 上的关键点或终点：调整轨迹形状。
- `Save Edit`：接受修改并写回 parquet。
- `Cancel Edit`：放弃这次修改。
- `Restore Edit`：回到进入编辑前的轨迹。

注意：

- 只能编辑非 GT 伪 GT。
- 保存编辑也会写日志和备份。
- 如果轨迹被动力学检查判定太不合理，保存前可能会被优化或限制。

## 11. 速度曲线和停车点

底部速度图主要用来发现轨迹是否突然加速、突然减速、速度抖动，或者停车是否符合预期。

简单看法：

- 曲线很尖、突然跳起来，通常不太好。
- 轨迹显示停车时，BEV 和 FC 图里会出现红点；这是停车段提示，不是交通参与者。
- 如果有 `优化速度曲线` 按钮，可以对当前伪 GT 做一次速度平滑和重采样。

速度优化一般流程：

1. 选中一条非 GT 轨迹。
2. 点击 `优化速度曲线`。
3. 看优化后的轨迹和速度图。
4. 合适就接受，不合适就取消。

真实 GT 速度只作为参考，不会被写入 output。

## 12. 场景筛选

如果提供了 `scene_labels.json`，顶部会出现可用场景类别。

用法：

- 选择 `straight`、`left`、`right` 等类别后，样本切换会限制在同类场景里。
- 选择 `None` 后，恢复普通顺序浏览。

这个功能只影响浏览顺序，不会修改 output。

## 13. 保存、备份和日志

GUI 的保存策略是：

```text
直接写回 output 里的 active parquet；
写回前自动备份；
写回后记录 edit_log.jsonl。
```

常见保存产物：

```text
output/
├── edit_log.jsonl
├── manual_points.json
├── .trajectory_gui_state.json
├── .backups/
└── {dataset_name}/
    └── {clip}.egomotion.parquet
```

界面里：

- `View Log`：查看最近保存记录。
- `Restore Backup`：恢复当前 clip 最近一次 parquet 备份。

建议：

- 标注前先备份一份原始 output。
- 多人同时标注时，不要多人同时写同一个 output 目录。
- 每个人最好负责不同 dataset 或不同 output 副本，最后再统一合并。

## 14. 新机器上最小可用清单

把下面这些准备好，就可以开始标注：

```text
1. generate_traj_data 工程代码
2. train_data 原始数据目录
3. output 伪 GT parquet 目录
4. calibration 标定目录，或 train_data/{dataset}/calibration/
5. 可选：train_data/{dataset}/data-objects/ 交通参与者 parquet
6. Python 3.10/3.11 + requirements.txt 里的 GUI 依赖
```

启动时只需要传：

```bash
python trajectory_annotator.py \
  --data_root /path/to/train_data \
  --output_dir /path/to/output \
  --calibration_dir /path/to/calibration \
  --no_restore_last
```

## 15. 常见问题

### 15.1 GUI 打开后没有样本

先检查：

- `--data_root` 是否指向了真正的 train_data 根目录。
- 数据集目录下是否有 `data-timestamps/{clip}.timestamps.parquet`。
- 数据集目录下是否有 `data-egomotion/{clip}.egomotion.parquet`。
- 如果使用默认 `video_frames` 模式，当前帧后面 6.4 秒是否有连续 egomotion。尾部不够长的帧会被过滤。

如果只想看已有 output，可以尝试：

```bash
--index_mode generated
```

### 15.2 图像不显示

检查：

- `mp4-converted/` 下面是否有对应相机视频。
- `data-timestamps/` 下面是否有对应相机 timestamp。
- 启动参数 `--cameras` 是否包含了不存在的相机。

例如只看 FC：

```bash
--cameras FC
```

### 15.3 相机投影明显不对

优先检查标定：

- `train_data/{dataset}/calibration/` 是否完整。
- 如果没有本地 XML 标定，`--calibration_dir` 是否指向旧 JSONL 标定目录。
- dataset 名和标定文件名是否能对应上。

### 15.4 BEV 或 FC 里没有交通参与者

先检查：

- 数据集目录下是否有 `data-objects/{clip}.objects.parquet`。
- 当前 `t0` 附近是否有对应的 object 时间戳。
- 左侧 BEV 顶部的 `交通参与者` 是否被勾选。这个开关会同时控制 BEV 和 FC 前视画面里的交通参与者位置点。

没有 `data-objects/` 不影响轨迹标注，只是少了碰撞风险辅助参考。

### 15.5 保存失败

检查：

- `--output_dir` 是否有写权限。
- 对应 parquet 是否正在被别的程序占用。
- Windows 上路径是否过长或包含奇怪字符。
- 多个人是否同时在写同一个 output。

### 15.6 不小心删错或保存错

可以先看：

- `output/edit_log.jsonl`
- GUI 里的 `View Log`
- GUI 里的 `Restore Backup`
- `output/.backups/`

恢复前建议先复制一份当前 output，避免二次误操作。

### 15.7 屏幕比较小，按钮或面板显示不全

新版 GUI 会根据屏幕大小自动缩放 BEV、速度曲线、相机图和右侧列表。Windows 小屏上会尽量自动最大化窗口。

如果仍然看不全：

- 先把窗口最大化。
- 用窗口右侧和底部滚动条查看被挡住的区域。
- 鼠标滚轮可以上下滚动；按住 Shift 再滚动可以左右滚动。
- 临时只看一个相机时，可以启动时加 `--cameras FC`，界面会更宽松。

## 16. 不要做的事

- 不要在标注机器上跑 VLA 推理；这套 GUI-only 环境没有准备模型推理依赖。
- 不要手动改 train_data 里的真实 GT。
- 不要多人同时写同一个 output 目录。
- 不要把 `output/.backups/` 和 `edit_log.jsonl` 当垃圾删掉，它们是追溯和恢复用的。
- 不要把旧的 `source=gt` output 行当真实 GT；真实 GT 只从 train_data 读取。

## 17. 一句话版流程

```text
准备 train_data、output、calibration；
有条件的话准备 data-objects；
安装 requirements.txt；
运行 trajectory_annotator.py；
逐帧检查已有伪 GT；
删掉坏轨迹，补充 Bezier 或 cluster 轨迹；
看速度图和相机投影；
保存；
继续下一个样本。
```
