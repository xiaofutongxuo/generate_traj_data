# Trajectory GUI 功能状态与维护 TODO

创建日期：2026-05-13  
适用项目：`/home/ubuntu/Public/hzq/generate_traj_data`

本文档用于快速了解当前项目已实现功能、已知问题、潜在风险和后续 TODO。历史逐次优化记录见
`docs/trajectory_gui_optimization_log.md`；本文件更偏向后续维护入口和需求清单。

## 1. 当前代码入口与运行路径

当前推荐项目根目录：

```bash
cd /home/ubuntu/Public/hzq/generate_traj_data
```

增强 GUI 兼容入口仍然是：

```bash
/home/ubuntu/Public/lxh/alpamayo_1.5/ar1_venv/bin/python trajectory_gui_enhanced.py \
  --data_root /home/ubuntu/Public/train_data \
  --output_dir /home/ubuntu/Public/yzb/generate_traj_data/output \
  --calibration_dir /home/ubuntu/Public/yzb/triplane_tokenization/cailibration \
  --no_restore_last
```

当前代码实现已经拆分到 `traj_gui_enhanced/` 包中：

```text
trajectory_gui_enhanced.py       # 兼容入口，仅 re-export 和调用 cli.main()
traj_gui_enhanced/cli.py         # GUI CLI 参数
traj_gui_enhanced/viewer.py      # TrajectoryViewerEnhanced 主类和 mixin 组合
frame_index.py                   # 视频帧 t0 索引与覆盖率统计工具
traj_gui_enhanced/dynamics/      # 伪 GT 动力学诊断、优化和 parquet 字段重算
traj_gui_enhanced/*_utils.py     # 轨迹、速度、投影、cluster 工具函数
traj_gui_enhanced/mixins/        # GUI 功能区：I/O、布局、导航、手工编辑、速度、绘制等
```

注意：GUI 在远程 X11 forwarding 下可能比较慢；如果 `$DISPLAY` 为空，当前环境设置会自动使用本机
`DISPLAY=:1`，窗口会出现在服务器桌面/VNC，而不是本地 VS Code Remote SSH 终端。

## 2. 当前已实现功能

### 2.1 轨迹生成与输出

- `run_inference.py` 支持 Alpamayo 1.5 模型加载、批量样本处理、expert-only 轨迹生成、CoT 保存和可视化图片保存。
- 输出按 dataset 和 clip 拆分为 parquet：

```text
output/{dataset_name}/{clip_stem}.egomotion.parquet
```

- 每一行是一条未来轨迹样本，output 语义上只保存生成/扩充出来的伪 GT，不保存原始真实 GT。常见列包括：

```text
t0_us, sample_idx, source, timestamp, qx, qy, qz, qw, x, y, z, vx, vy, vz, curvature
```

- `source` 用于区分伪 GT 来源：VLA 生成轨迹写入 `vla`，手工 Bezier 写入 `manual_bezier`，cluster center 写入 `cluster_center` 或历史 `rule_cluster`。
- 真实 GT 只从 `/home/ubuntu/Public/train_data` 原始数据集读取，用于 GUI 显示、对比和诊断。
- 历史 output 中如果存在 `source=gt` 行，会被视为旧数据污染；GUI 默认不把它计入伪 GT 覆盖率，也不放入右侧可编辑轨迹列表。
- VLA 输出、manual Bezier、cluster center 和已选中伪 GT 的保存写回会经过 `traj_gui_enhanced/dynamics/` 做速度、加速度、jerk、曲率诊断和保守优化，并统一重算 `vx/vy/vz/qx/qy/qz/qw/curvature`。
- CoT 文本保存到：

```text
output/cot.jsonl
```

### 2.2 GUI 样本索引与导航

- GUI 默认 `--index_mode video_frames --frame_stride 1`，从 `data_root/{dataset}/data-timestamps/{clip}.timestamps.parquet` 读取 10Hz 主视频帧时间戳，逐帧作为可浏览 `t0_us`。
- 可浏览 `t0_us` 会再经过真实 egomotion 连续性过滤：当前帧和未来 6.4s 必须能从原始数据覆盖；相邻 egomotion 时间戳缺口不超过 `0.3s` 时允许线性插值补小范围漏帧，超过 `0.3s` 视为断点。
- 如果当前 clip 尾部不够 6.4s future，但后续 clip 的时间戳连续，GUI 和推理会自动接上下一个 clip 的 egomotion；如果已经是最后一段或后续时间不连续，则该尾部 `t0_us` 会被丢弃，避免显示/生成外推 GT。
- generated 模式可通过 `--index_mode generated` 启用，它只从 `output/{dataset}/*.egomotion.parquet` 读取已有 `t0_us`，只显示已经生成到 output 中的样本。
- merged 模式可通过 `--index_mode merged` 启用，它以视频帧为主，并补入 output 中存在但不在主视频时间戳中的生成 t0。
- GT-only 模式默认也会跟随 `video_frames` 索引逐帧浏览；如果需要旧的稀疏 GT-only 行为，可使用 `--gt_only --index_mode generated --gt_stride_frames N`。
- 状态栏会显示当前 clip 的覆盖率，例如 `Video t0: N | Generated t0: M | Missing: K | Current: generated/no generated`。
- 支持 Dataset / Clip / t0 下拉选择和 Jump。
- 支持启动定位参数：`--start_index`、`--start_dataset`、`--start_clip`、`--start_t0`。
- 支持保存和恢复最后浏览位置：`output/.trajectory_gui_state.json`。

### 2.3 BEV 与相机投影显示

- 左侧 BEV 画布显示历史轨迹、GT future、生成/扩充轨迹、手工 Bezier 预览、cluster center 预览和停车点。
- 中间区域显示相机图像，默认 `RL,FC,RR`，并支持切换投影相机。
- 相机标定优先从 `data_root/{dataset}_converted/calibration/` 读取本地 XML（由 `Vision_calibration.tar.gz` 解压得到），旧的 `--calibration_dir` JSONL 目录仅作为兜底。
- 自动停车点、GT 停车点、人工停车点会在 BEV 和 FC 图像中显示红点。
- 速度图 hover 时可在速度图、BEV、FC 中同步高亮同一未来帧。

### 2.4 轨迹列表、删除、保留和保存

- 右侧 `Trajectories` 列表显示当前 `t0_us` 下 output 中的伪 GT 轨迹行。
- GT 从原始数据集读取并单独显示，不再作为 output 轨迹行参与删除/保留。
- VLA 生成轨迹、manual 轨迹、cluster center 追加轨迹可以被标记为删除。
- `Save (Ctrl+S)` 当前会直接写回 active parquet，删除被标记的行。
- 当前真实行为是写回原 parquet 文件；不是写到 `filtered/` 子目录。
- 已保存的非 GT 伪 GT 轨迹支持 `Edit Traj` 几何编辑第一版：进入编辑后在 BEV 上显示关键帧手柄，左键拖动手柄会对整条轨迹做局部形变，再经过 dynamics 优化；`Save Edit` 写回原 parquet 行，`Cancel Edit` 放弃修改，`Restore Edit` 恢复进入编辑时的原始轨迹。

### 2.5 速度曲线与停车检测

- `Diversity Speed Profile` 在同一面板中叠加历史速度和当前选中生成/扩充轨迹未来速度，横轴约为 `-15..63`，绿色线为 history，当前帧 `0` 用垂直虚线标识。
- `GT Speed Profile` 在同一面板中叠加历史速度和当前 GT future 速度，横轴约为 `-15..63`，绿色线为 history，当前帧 `0` 用垂直虚线标识。
- GT future 速度由原始 GT `xyz` 差分得到后，会做轻量显示平滑，降低原始位置噪声在速度曲线中的尖刺；该平滑只影响速度显示、GT 停车检测和伪 GT 平滑性参考，不修改真实 GT 坐标。
- History 轨迹在 GUI 显示层会做轻量去噪平滑，并固定第一帧和当前帧不动；绿色 history 速度线、BEV history 轨迹、hover 高亮和手工 Bezier 当前速度估计使用平滑后的 history，但不会修改 `ego_history_xyz` 原始数据。
- 录制开头如果真实历史帧不足，GUI 只显示真实存在的历史点；不足 2 个历史点时不绘制 history 速度/轨迹，避免把外推数据误画成真实历史。
- 速度图 hover 到负帧时会在 BEV 历史轨迹上同步高亮对应历史点；hover 到非负帧时沿用当前未来轨迹/GT 高亮。
- 停车检测阈值当前为 `speed < 0.1 m/s` 且连续至少 5 帧。
- 速度不平滑的生成/扩充轨迹会在列表中标为 `[×]`，并用红色轨迹显示。
- 速度平滑诊断主要基于加速度、jerk，并允许与 GT 速度接近的轻微波动。

### 2.6 速度优化与写回

- 生成/扩充轨迹支持按钮式 `优化速度曲线`：
  - 平滑速度曲线；
  - 沿原路径按速度分布重采样点密度；
  - 再进行加速度限制重采样；
  - 接受后写回当前 parquet。
- GT 速度面板仍可用于查看原始数据集 GT 的速度趋势和诊断。
- 真实 GT 不再写入 output parquet；GT 优化/停车保存入口当前会提示 GT 来自源数据，output 只保存伪 GT。
- `AUTO_OPTIMIZE_GT_ON_LOAD = False`，加载样本时不会再自动优化并写回 GT 行。
- 新增通用 `dynamics` 子包，当前已用于：
  - `run_inference.py` 保存 VLA 生成轨迹前的伪 GT 优化；
  - `Save Curve Traj` / cluster `Confirm Save` 追加轨迹前的伪 GT 优化；
  - 已保存非 GT 轨迹速度编辑写回 parquet 前的伪 GT 优化。
- 已新增 `SavedTrajectoryEditingMixin`，支持对已保存非 GT 轨迹做 BEV 关键帧/终点拖拽编辑，并通过 `Save Edit` / `Cancel Edit` / `Restore Edit` 完成接受、取消、恢复原状流程。

### 2.7 手工 Bezier 与停车点编辑

- `Draw Bezier` 支持在 BEV 或 FC 图像中添加控制点。
- 支持右键拖拽控制点，BEV 和 FC 的同一条手工曲线会同步变化。
- 支持 `Add Final Stop`，并可通过 `Stop Time(s)` 设置最终停车时长。
- 手工控制点、图像控制点和停车点保存到：

```text
output/manual_points.json
```

- `Save Curve Traj` 会将手工 Bezier 轨迹追加到当前 clip parquet 中。
- 新保存的手工 Bezier 轨迹会写入 `source=manual_bezier`，用于后续删除/识别。
- 手工 Bezier 轨迹构建时会执行加速度限制重采样，尽量保证车辆动力学合理。

### 2.8 Cluster center 扩充轨迹

- Cluster center 从 `k_means/*.txt` 加载，支持 `stop / straight / left / right / s_curve` 分类。
- GUI 支持下拉选择、`+/-` 快捷切换、BEV/FC 预览。
- 支持拖拽 cluster endpoint，拖拽候选会进行加速度、曲率和位移幅度限制。
- `Confirm Save` 会把当前 cluster center 预览追加到 active parquet。
- 新保存的 cluster center 轨迹会写入 `source=cluster_center`，用于后续删除/识别。
- `Save Bezier Center` 可将当前手工 Bezier 保存为新的 cluster center。
- `Delete Current Bezier Center` 只允许删除由 `Save Bezier Center` 创建的 center。

### 2.9 GT 修复与质量诊断

- `data_loader.py` 中实现了基于速度/加速度限制的 GT future 修复。
- GUI 中有 `Repair GT` / `Restore GT` 按钮。
- 状态栏显示 GT 加速度范围、最大步长、异常数量等质量诊断。

### 2.10 测试与验证

- 当前有 `tests/test_gui_helpers.py`，使用标准库 `unittest`。
- 覆盖内容包括入口 wrapper、Tk import、cluster 路径、颜色、速度来源、历史速度叠加、轨迹身份/删除 key、pending delete 撤销、停车检测、速度平滑、加速度限制、重采样端点保持、伪 GT 动力学诊断/优化和保存前 parquet 写回优化。

常用验证命令：

```bash
./.venv/bin/python -m unittest discover -s tests
./.venv/bin/python -m py_compile frame_index.py data_loader.py run_inference.py trajectory_gui_enhanced.py traj_gui_enhanced/*.py traj_gui_enhanced/mixins/*.py
./.venv/bin/python trajectory_gui_enhanced.py --help
```

## 3. 当前潜在问题与维护风险

### 3.1 旧稀疏索引模式仍需谨慎使用

TODO-001 已将默认推理和 GUI 索引改为 10Hz 视频帧逐帧模式。但旧模式仍保留：

- `run_inference.py --t0_source speed_candidates` 会走旧的 `data_loader.get_t0_candidates()`，内部仍是每 30 帧扫描一次候选，并可叠加 `--candidate_stride`。
- `trajectory_gui_enhanced.py --index_mode generated` 只显示 output parquet 中已经存在的 `t0_us`。

因此如果需要“每个视频帧都生成并检查”，应使用默认 `--t0_source video_frames --frame_stride 1` 和默认 GUI `--index_mode video_frames --frame_stride 1`。

### 3.2 文档与真实行为有历史偏差

旧优化日志中曾提到 `filtered/` 和 `gt_speed_edits.json`，但当前代码真实行为是：

- 删除/保留保存：直接写回 active parquet。
- 伪 GT 速度编辑：直接写回 active parquet。
- GT 编辑：当前不再写回 output parquet，真实 GT 只从原始数据集读取。
- 未看到当前代码写入 `gt_speed_edits.json`。

后续维护时应以当前代码为准，并逐步修正历史日志或在日志中标注“历史行为已变更”。

### 3.3 远程 GUI 操作体验风险

- `$DISPLAY` 为空时会自动设置为 `:1`，容易把窗口开到服务器本机桌面，而不是用户 X forwarding 会话。
- X11 forwarding 下速度图 hover 会触发 BEV、速度图和相机图像全量重绘，可能导致交互很慢，表现为“能显示但不好操作”。

### 3.4 CLI 默认路径仍有旧路径

`traj_gui_enhanced/cli.py` 中 `--data_root` 和 `--calibration_dir` 默认值仍是 `/home/tsingyu/...`。README 已建议显式传参；当前标定实际优先跟随 `--data_root` 下的数据集本地 XML，`--calibration_dir` 只是旧 JSONL 兜底目录。长期应改为环境变量或项目配置。

### 3.5 GT 与 output 语义风险

当前已将 `AUTO_OPTIMIZE_GT_ON_LOAD` 设为 `False`，并停止在加载样本时把 GT 自动写回 output parquet。

后续仍需注意历史 output：

- 旧文件中可能存在 `source=gt` 行，这是历史 GUI 写入的 GT 副本，不是原始真实 GT。
- 当前 GUI 会忽略这些 legacy GT 行，并只从原始数据集读取真实 GT。
- 如需清理磁盘上的历史 `source=gt` 行，应单独做迁移/备份工具，不建议在 GUI 加载时静默删除。

### 3.6 仍缺少 GUI 行为级测试

当前测试主要覆盖纯函数。以下关键行为还没有自动化保护：

- parquet 行追加、删除、写回；
- GT 修复/恢复；
- manual_points.json 读写；
- cluster center 文件写入和删除；
- GUI 事件绑定和点击/拖拽流程。

### 3.7 Windows 支持不足

当前代码和文档大量依赖 Linux 绝对路径、X11 DISPLAY、Linux `.runtime/tk` 结构和 shell 命令。核心 Python/Tk 逻辑理论上可迁移，但目前没有 Windows 启动、依赖、路径和 GUI 行为验证。

## 4. 新需求 TODO List

### P0 TODO-001 修复可视化漏帧/样本覆盖问题（已实现，待进一步实测检查）

需求来源：用户反馈“当前可视化似乎过滤掉了很多帧，没有将完整数据所有帧都可视化出来”。

当前状态：

- “完整数据所有帧”已按用户确认定义为主视频时间戳中的所有视频帧，视频约 10Hz，即 0.1s 一帧。
- 新增 `frame_index.py`，集中读取 `data-timestamps/{clip}.timestamps.parquet` 并生成逐帧 `t0_us`。
- `run_inference.py` 默认 `--t0_source video_frames --frame_stride 1`，不会再套旧的 `candidate_stride`，因此默认逐视频帧生成扩充轨迹。
- 旧稀疏候选模式仍可通过 `--t0_source speed_candidates` 使用，方便需要快速抽样时回退。
- GUI 默认 `--index_mode video_frames --frame_stride 1`，即使某个有效视频帧还没有生成轨迹，也能显示图像、真实 history、GT 和状态栏覆盖率。
- 如果某个视频帧没有 output 轨迹，右侧轨迹列表为空，状态栏显示 `Current: no generated`。
- `load_data()` 新增 history/future valid mask；录制开头真实历史帧不足时，GUI 不绘制无效历史速度/轨迹。
- 新增全数据集 egomotion 连续性读取与 `filter_t0s_with_full_future()`：跨 clip 时按全局 timestamp 排序；小范围漏帧（相邻 egomotion 缺口 `<=0.3s`）允许插值补齐；较大缺口视为视频不连续，当前帧或相关未来窗口会被过滤掉。
- 靠近 clip 结尾时，如果后续 clip 时间连续，会接上下一个 clip 的 egomotion；如果后续不存在或不连续，未来 6.4s 不完整的尾部 t0 会被丢弃，不再使用外推出来的 GT future。
- `filter_t0s_with_full_future()` 已改为向量化计算，避免 GUI 启动时逐 t0 反复排序全量 egomotion timestamp；本机实测 302 个 clip、约 17.2 万个候选 t0 的全量索引过滤约 0.9s。
- 代码级验证已通过，但还需要进一步用真实推理输出和 GUI 手工浏览检查确认：当前实现没有跑全量模型推理，也没有逐帧打开完整 clip 做人工验收。

后续检查项：

- 对一个 clip 统计 `data-timestamps` 视频帧数、output 中 unique `t0_us` 数、GUI samples 数，状态栏能解释三者关系。
- 默认全帧模式下，GUI 能访问主视频时间戳中未来 6.4s 连续覆盖的每一个 t0；clip 最末尾且无法接续的数据会被过滤。
- 录制开头没有足够 history 时，history 速度/轨迹不会显示外推出来的历史段。
- 人工检查原始数据中少量漏帧场景，确认 26/27/29/30 这类小缺口能用前后帧插值补齐；大于 `0.3s` 的断点不会被强行连接。
- 用小 clip 或 `--max_samples` 先跑一次 `run_inference.py --t0_source video_frames --frame_stride 1`，确认 output 覆盖率按预期增加。
- 打开 GUI 手工翻看连续帧，确认没有明显漏帧、跳帧或 t0 顺序异常。

相关文件：

- `data_loader.py`
- `frame_index.py`
- `run_inference.py`
- `traj_gui_enhanced/mixins/sample_io.py`
- `traj_gui_enhanced/cli.py`
- `README.md`

### P1 TODO-002 增加历史轨迹速度显示（已完成）

需求来源：当前左下角只显示 GT 和扩充轨迹的未来 64 帧速度，需要增加历史轨迹速度。

当前状态：

- `Diversity Speed Profile` 和 `GT Speed Profile` 已在原面板内叠加历史速度。
- history 使用绿色线显示，future 保持各自原有颜色。
- x 轴从历史负帧延伸到未来帧，16 帧 history 对应 `-15..0`，64 帧 future 对应 `0..63`。
- 当前帧 `0` 使用垂直虚线标识。
- 速度图 hover 到负帧时会高亮 BEV 历史轨迹点；hover 到非负帧时高亮当前未来轨迹/GT。

已实现方向：

- 从 `conv_data["ego_history_xyz"]` 相邻历史点差分计算 history speed，避免误用相对原点差分。
- 复用现有两个速度面板，不新增独立 history 面板。
- history 与 future 使用同一 y 轴，便于观察 t0 前后的速度趋势。

验收建议：

- 每个样本都能显示历史速度。
- 历史速度与 BEV 历史轨迹点一一对应。
- 没有历史数据时不影响未来速度图。

相关文件：

- `traj_gui_enhanced/mixins/speed_controls.py`
- `traj_gui_enhanced/mixins/draw_speed.py`
- `traj_gui_enhanced/mixins/draw_bev.py`
- `traj_gui_enhanced/mixins/widget_layout.py`

### P1 TODO-003 增加扩充轨迹删除功能（已完成）

需求来源：需要增加一个扩充过后的轨迹删除功能。

当前状态：

- Delete/Backspace 会将当前非 GT 轨迹加入 pending delete，并立即从 GUI 列表、BEV 和相机投影中隐藏。
- `Undo Delete` 会恢复最近一次 pending delete 的轨迹可视化。
- `Confirm Save (Ctrl+S)` 才会根据 `(t0_us, sample_idx)` 从 active parquet 中真正删除 pending delete 对应行。
- 如果 pending delete 的轨迹匹配当前手工 Bezier 控制点，控制点会先在内存中隐藏；撤销删除会恢复，确认保存才会同步写回 `manual_points.json`。
- 切换样本前如果还有 pending delete，会提示先保存或撤销，避免误以为已经写入。
- GT 行受保护。
- 后续追加的 manual Bezier 和 cluster center 轨迹分别写入 `source=manual_bezier`、`source=cluster_center`。
- `Delete Current Bezier Center` 删除的是 cluster center 库中的 Bezier center，不是已经追加到 output parquet 的扩充轨迹。

已实现方向：

- 新增 `traj_gui_enhanced/trajectory_identity.py`，集中维护 source 规范化、GT 保护、轨迹 key 和 parquet 删除过滤。
- 新增 `traj_gui_enhanced/mixins/delete_controls.py`，集中维护 pending delete、撤销删除和列表行到真实轨迹 index 的映射。
- 删除后的轨迹直接从 GUI 中消失，保存前仍保留在原 parquet 中。
- 对匹配当前手工 Bezier 曲线的删除，会 staged 清空 manual 控制点，确认保存后同步写回 `manual_points.json`。

验收建议：

- 能明确删除已经保存到 output parquet 的 manual/cluster 扩充轨迹。
- 不能误删 GT。
- 删除后重新打开同一 t0，扩充轨迹不再出现。

相关文件：

- `traj_gui_enhanced/mixins/navigation.py`
- `traj_gui_enhanced/mixins/sample_io.py`
- `traj_gui_enhanced/mixins/manual_editing.py`
- `traj_gui_enhanced/mixins/cluster_controls.py`
- `traj_gui_enhanced/mixins/delete_controls.py`
- `traj_gui_enhanced/trajectory_identity.py`

### P1 TODO-004 人工修改伪 GT 后进行加速度/曲率优化并保存（代码闭环已完成，建议真实 GUI 验收）

需求来源：在可视化工具中修改伪 GT/扩充轨迹后，需要对整条轨迹进行加速度和曲率优化，限制修改幅度，防止车辆动力学不合理，并保存更新 output 中的扩充轨迹。

当前状态：

- 数据身份语义已先行整理：output 只保存伪 GT；VLA 新输出会写入 `source=vla`；legacy `source=gt` output 行会被 GUI 忽略；空 `source` 的旧 output 行按伪 GT 处理，避免把 VLA 的 `sample_idx=0` 误判为 GT。
- 手工 Bezier 新建轨迹会走加速度限制重采样。
- cluster endpoint 拖拽会进行加速度、曲率、位移幅度限制。
- 生成/扩充轨迹的速度优化会沿原路径平滑速度并重采样。
- 已新增 `traj_gui_enhanced/dynamics/` 子包，提供 `DynamicsLimits`、`diagnose_trajectory_dynamics()`、`optimize_pseudo_gt_trajectory()` 和 `trajectory_components_from_xyz()`。
- VLA 输出、manual Bezier、cluster center 追加轨迹、已保存非 GT 速度编辑写回 parquet 前，都会走通用 dynamics 优化并重算 parquet 轨迹字段。
- 已实现直接编辑“已经保存到 output 的扩充轨迹”的第一版 GUI 工作流：`Edit Traj` 进入模式，BEV 关键帧手柄左键拖动，拖动后局部形变 + dynamics 优化，`Save Edit` 写回原 parquet 行，`Cancel Edit` 放弃，`Restore Edit` 回到进入编辑时的原始轨迹。
- 已补齐编辑状态保护：几何编辑未保存/取消前，不能删除轨迹、不能启动速度编辑、不能切换样本或切换到其他轨迹，也不能开启第二个几何编辑。
- 仍建议进一步真实 GUI 验收：手柄间隔、拖拽影响半径、优化强度、按钮布局和提示文案是否符合实际清洗效率。

建议实现方向：

- 为已保存的扩充轨迹增加编辑模式：
  - 选择一条 manual/cluster 扩充轨迹；
  - 允许拖拽关键点、终点或控制点；（第一版已支持 BEV 关键帧/终点手柄）
  - 修改后构造候选轨迹；
  - 对候选轨迹进行加速度、曲率、最大速度、最大单步位移限制；
  - 重新计算 `vx/vy/vz/qx/qy/qz/qw/curvature`；
  - 保存时写回对应 parquet 行。
- 优化策略当前已封装到 `traj_gui_enhanced/dynamics/`，内部复用 `_acceleration_limited_resample_path()` 与 `_smooth_curvature_preserving_ends()`；后续 GUI 几何编辑应直接调用该模块，避免再复制一套动力学逻辑。
- UI 需要有明确的 `接受/取消/恢复原状` 流程，避免误写 output。

验收建议：

- 人工拖动后轨迹不会出现明显尖跳、反向、过大加速度或曲率突变。
- 保存后重新打开同一样本，修改后的伪 GT 仍在。
- 超过动力学限制的修改会被拒绝或自动收敛到可行范围，并给出提示。
- 真实 GUI 中连续拖动 endpoint/中间关键帧时，轨迹形变符合预期，不会因为手柄太密/太稀导致效率低。

相关文件：

- `traj_gui_enhanced/mixins/manual_events.py`
- `traj_gui_enhanced/mixins/manual_editing.py`
- `traj_gui_enhanced/mixins/saved_traj_editing.py`
- `traj_gui_enhanced/mixins/sample_io.py`
- `traj_gui_enhanced/mixins/speed_controls.py`
- `traj_gui_enhanced/dynamics/`
- `traj_gui_enhanced/cluster_utils.py`
- `traj_gui_enhanced/speed_utils.py`
- `traj_gui_enhanced/math_utils.py`

### P2 TODO-005 增加历史/未来速度统一诊断视图

这是 TODO-002 的延伸项。

建议把 history speed、GT future speed、selected pseudo-GT future speed 放到统一诊断视图中，帮助判断 t0 附近速度是否连续。例如：

- 历史最后一帧速度；
- 未来第一帧速度；
- 速度突变量；
- 是否超过加速度阈值；
- 是否建议优化。

### P2 TODO-006 梳理保存行为和数据版本

当前多个操作会直接写回 active parquet。建议建立更清晰的数据版本策略：

- 明确哪些操作是立即写回，哪些操作只是 GUI 内临时状态。
- 增加写回前备份或 undo 文件，例如 `.bak` 或操作日志。
- 在 parquet 中增加 `source`、`edit_version`、`edited_by_gui`、`edit_time` 等元数据列。
- README 和优化日志同步更新真实保存行为。

### P3 TODO-007 Windows 系统支持

需求来源：希望工具支持 Windows 使用。

当前状态：

- 当前默认路径、X11 DISPLAY、`.runtime/tk` fallback、README 命令都偏 Linux。
- 核心 GUI 使用 Tk，理论上可跨平台；但数据路径、依赖安装、视频读取和性能需要单独验证。

建议实现方向：

- 所有默认路径改为环境变量或 config 文件，不硬编码 Linux 用户路径。
- 使用 `pathlib` 保持路径兼容；避免 shell-only 命令进入 Python 逻辑。
- 提供 Windows README 小节：
  - Python 版本；
  - 依赖安装；
  - 数据目录映射；
  - GUI 启动命令；
  - OpenCV 视频读取注意事项。
- 在 Windows 上至少验证：
  - `python trajectory_gui_enhanced.py --help`
  - import `TrajectoryViewerEnhanced`
  - 打开一个最小样本 GUI
  - parquet 读写

验收建议：

- Windows 本机能打开 GUI 并完成样本切换、轨迹选择、保存/取消类操作。
- Linux 现有命令不回归。

## 5. 建议近期维护顺序

1. TODO-001 代码已实现，仍需进一步实测检查：完整帧定义为所有主视频帧；推理和 GUI 默认逐帧，但会过滤未来 6.4s 不连续或不足的数据。
2. TODO-002 已完成：历史速度已叠加到现有两个速度面板中。
3. TODO-003 已完成：扩充轨迹 staged delete、撤销和确认保存已经可用。
4. TODO-004 代码闭环已完成：基础 dynamics、保存前优化、已保存轨迹 BEV 关键帧编辑、接受/取消/恢复原状和编辑状态保护已经接入；下一步主要是真实 GUI 验收和手感调参。
5. 最后做 TODO-007：Windows 支持属于平台化工作，应在核心数据和保存语义稳定后再推进。

## 6. 待确认问题

- “完整数据所有帧”已确认指所有主视频帧；默认不要求完整 history，开头无历史帧时不显示 history；但需要完整 future，若跨 clip 连续则接续，若尾部或大缺口导致 future 不足则过滤。
- 已明确：扩充/伪 GT 包括 VLA 生成轨迹、manual Bezier、cluster center 等所有非真实 GT output 行。
- 伪 GT 编辑目前已覆盖速度曲线保存前优化和 BEV 关键帧/终点几何编辑；仍需根据真实 GUI 试用调整交互细节。
- 保存策略是否允许继续直接覆盖 output parquet，还是需要引入备份/版本目录？
