# kmeans 轨迹数据流程

这个目录保存 5-8 轨迹导出、500 类加速度/曲率聚类、以及目标数据集 GT+多样化轨迹生成的对接入口。

## 当前规则

- 从 `data_26_5_8_converted` 获取真实未来 64 步轨迹。
- 未来 xy txt 仍使用 `kmeans_cluster.export_future_xy_txt()`：按当前逻辑做速度积分修复、前向加速度过滤、跳变过滤。
- 聚类和目标匹配的动态特征使用“位置反推速度/加速度 + parquet 原生曲率”的组合：
  - 从未来 xy 位置反推逐帧速度；
  - 速度平滑 1 次；
  - 对平滑后的速度差分得到原始加速度；
  - 原始加速度再平滑 2 次作为聚类加速度特征；
  - 曲率从 `train_data/<dataset>/data-egomotion/*.parquet` 的 `curvature` 未来窗口读取，再平滑 1 次；
  - `curvature_min_speed_mps=0.0`，默认不按低速阈值清零曲率。
- 注意曲率符号约定：
  - parquet 原生 `curvature` 与本目录未来 xy 的 ego-local rollout 坐标符号相反；
  - 动态特征匹配阶段必须保持 parquet 原生曲率符号，因为 train/target 同源；
  - 只有在用 `integrate_acc_curvature*()` 积分生成 endpoint 或最终候选 xy 时，才通过 `ego_rollout_curvature_from_native()` 取反；
  - 不要把取反后的曲率写回 feature，也不要把 parquet 原生曲率直接送进 ego-local rollout。
- 5-8 聚类中心数固定为 500。
- 后续 GT 匹配沿用同一套显式平滑参数：速度 1 次、原始加速度 2 次、曲率 1 次。
- 目标数据集默认 `t0_stride=1`，每一帧都生成。
- target future 从每个 clip 的 `t0_idx=0` 开始导出：
  - 如果这是 dataset 的第一个 clip，开头缺少完整 16 帧 history，则这些帧不导出；
  - 如果不是第一个 clip，开头缺少的 history 从时间上连续的上一个 clip 末尾补齐，并照常做多样化扩充。
- 目标数据集输出使用 `generate_target_diverse_gt.py` 的加速度/曲率聚类匹配逻辑，每个时间戳保存 GT，并最多保存 4 条聚类多样化轨迹。
- 输出 parquet 的 future 和 history 动态量也从位置反推和平滑，不再直接使用原生速度记录。

## 多样化匹配规则

`generate_target_diverse_gt.py` 使用 `cluster_acc_curvature_top2.py` 里的加速度/曲率匹配思路，不走旧的规则扰动类别逻辑。

1. 先导出目标 GT
   - 从目标数据集导出 `target_future_trajectories_xy.txt`。
   - 导出时沿用当前 GT 处理逻辑：速度积分修复 xy、前向加速度过滤、step speed/second-diff acceleration 跳变过滤、可选 backward 过滤。
   - 输出 parquet 中 `source=gt` 的行是该时间戳的优化/修复后 GT。

2. 训练 5-8 的 500 类动态聚类
   - 读取 `future_trajectories_5_8_xy.txt`。
   - 对每条 5-8 轨迹从未来 xy 位置反推速度和加速度，从同 clip 的 parquet 未来窗口读取原生曲率。
   - 按“位置反推速度 -> 速度平滑 1 次 -> 差分 -> 原始加速度平滑 2 次；parquet 原生曲率 -> 曲率平滑 1 次”的规则构造特征。
   - KMeans 训练 500 个加速度/曲率中心，并保留每个 cluster 的成员。

3. 目标 GT 动态特征
   - 对目标 GT 使用同一套“位置反推速度/加速度 + parquet 原生曲率”规则。
   - 使用 5-8 的 feature mean/std 做标准化，与 500 个 cluster center 计算 dynamic distance。

4. 先选 dynamic top cluster
   - 先按 dynamic distance 选择 top 2 cluster，不因为 endpoint 约束提前跳到更远 cluster。
   - 每个时间戳最多从这 2 个 cluster 中各选 2 个 member，因此最多输出 `2 * 2 = 4` 条 `source=acc_curvature_cluster` 候选。

5. 再在 top cluster 内选 member
   - 对 top cluster 内 member 用其加速度/曲率 profile，以当前 GT 的 t0 平均速度 rollout endpoint；rollout 前会把 parquet 原生曲率取反为 ego-local xy 曲率。
   - 在 GT 终点航向坐标系下检查 endpoint box：
     - 默认横向误差 `<= 2.0m`；
     - 默认纵向误差允许候选短 `5.0m`、长 `4.0m`，即 `-5.0m <= longitudinal <= 4.0m`。
   - 对 endpoint-feasible member 最小化匹配成本：
     - `feature_distance + member_endpoint_weight * endpoint_distance`
     - `member_endpoint_weight` 默认 `0.15`
   - 如果某个 top cluster 没有 endpoint-feasible member，该 cluster 不输出候选，但不会自动跳到第 3 个 cluster。

6. 输出摘要
   - `acc_curvature_diverse_gt_summary.csv` 记录每条候选的 cluster、member、feature distance、dynamic cluster distance、endpoint distance、endpoint lateral/longitudinal error、平均 xy 点距。

## 加速策略

当前批量扩充已经做了两处向量化：

- `integrate_acc_curvature_endpoints_batch()`
  - 对全部 5-8 cluster member 的加速度/曲率 profile 按当前 GT 的 t0 速度批量积分，只计算 endpoint。
  - endpoint 积分使用取反后的 ego-local 曲率，避免左转/右转方向被翻转。
  - 替代旧逻辑里“每个 cluster member 调一次 `integrate_acc_curvature()`”的 Python 内层循环。
- `select_top_clusters_then_endpoint_members_fast()`
  - 先固定 dynamic top 2 cluster。
  - 再对 top cluster 内所有 member 的 endpoint box 和匹配成本做 numpy 批量计算。

`t0_stride=1` 会显著增加目标 GT 数量；当前 6 个默认目标数据集会导出约 6 万级别的 target rows。完整运行还要加上目标 GT 导出、特征构建、候选 rollout、history 拼接和 parquet 写入时间。

后续还可以继续加速：

- speed-bin endpoint 缓存：默认按 `0.05 m/s` 的 `gt_speed_mps` 速度桶复用全部 member endpoint。这样会极大减少重复 batch rollout；候选最终 rollout 仍使用精确 t0 speed。可用 `--endpoint_speed_bin_mps 0` 关闭缓存，回到完全精确 endpoint 筛选。
- 分块写 parquet：当前先收集所有输出行再写文件，数据集很大时内存占用会升高；可改为按 clip 或按若干 t0 分块写入。
- 并行 dataset：不同 dataset/clip 之间可拆进程并行，但要避免多个进程同时写同一个 output parquet。

## 历史轨迹格式

`train_data` 的 `data-egomotion/*.egomotion.parquet` 没有单独的 `history_*` 字段；历史轨迹是在读取某个 `t0_us` 时，从同一个逐帧 egomotion 表里取 `t0` 当前帧和之前 15 帧得到的。`generate_traj_data/data_loader.py` 中的 `ego_history_xyz/ego_history_rot` 也是这样构造的。

为了让生成的 output parquet 自包含 16 帧历史，每个样本行会保存两套同构字段：

- future 64 帧：`timestamp/qx/qy/qz/qw/x/y/z/vx/vy/vz/ax/ay/az/curvature`
- history 16 帧：同一组字段加 `history_` 前缀，例如 `history_timestamp/history_x/history_vx/history_ax/history_curvature`

history 在非首个 clip 的开头会从时间上连续的上一个 clip 拼接，不再依赖 GUI 临时回读。dataset 第一个 clip 的开头没有可复用 history 时不导出，因此 GUI 也不会显示这些无 history 的 output 样本。future 和 history 的动态字段集合与 `train_data` 的 egomotion schema 对齐；加前缀只是为了避免和 future 数组列冲突。`history_vx/history_ax/history_curvature` 由 `history_x/y/z` 反推并平滑得到，避免原生速度记录中的跳变传入速度面板。

## 文件说明

- `export_5_8_future_txt.py`
  - 导出 `data_26_5_8_converted` 的未来轨迹 txt。
  - 默认输出：`future_trajectories_5_8_xy.txt`。
- `future_trajectories_5_8_xy.txt`
  - 已导出的 5-8 未来 64 步 ego-local xy 轨迹。
- `cluster_5_8_500.py`
  - 从 5-8 txt 读取轨迹，使用位置反推加速度和 parquet 原生曲率特征做 500 类 K-Means。
  - 默认输出中心：`acc_curvature_500_centers.txt`。
  - 默认输出摘要：`acc_curvature_500_summary.json`。
- `acc_curvature_500_centers.txt`
  - 500 个聚类中心文件。
  - `CENTER` 行保存中心的加速度+曲率特征；`CENTER_MEDOID_XY` 行保存该中心 medoid 的 xy 轨迹，便于可视化或回查。
- `generate_target_diverse_gt.py`
  - 读取目标数据集，例如 `data_26_3_24_1_converted` 到 `data_26_3_25_2_converted`。
  - 调用加速度/曲率聚类匹配逻辑：先 dynamic top 2 cluster，再在每簇内做 endpoint 约束和成本最小化。
  - 默认写入 `generate_traj_data/output`。
- `cluster_acc_curvature_top2.py`
  - 保留的实验/可视化脚本；默认参数已同步为从位置反推速度/加速度、读取 parquet 原生曲率，并分别 1/2/1 次平滑。
## 常用命令

在仓库根目录运行，使用项目虚拟环境：

```bash
./generate_traj_data/.venv/bin/python generate_traj_data/kmeans/export_5_8_future_txt.py
```

```bash
./generate_traj_data/.venv/bin/python generate_traj_data/kmeans/cluster_5_8_500.py
```

```bash
./generate_traj_data/.venv/bin/python generate_traj_data/kmeans/generate_target_diverse_gt.py
```

目标数据集可显式指定：

```bash
./generate_traj_data/.venv/bin/python generate_traj_data/kmeans/generate_target_diverse_gt.py \
  --datasets data_26_3_24_1_converted,data_26_3_24_2_converted,data_26_3_24_3_converted,data_26_3_25_1_converted,data_26_3_25_2_converted
```

快速冒烟测试可加 `--limit`：

```bash
./generate_traj_data/.venv/bin/python generate_traj_data/kmeans/generate_target_diverse_gt.py --limit 20
```

## 输出位置

- 5-8 txt：`generate_traj_data/kmeans/future_trajectories_5_8_xy.txt`
- 500 类中心：`generate_traj_data/kmeans/acc_curvature_500_centers.txt`
- 500 类摘要：`generate_traj_data/kmeans/acc_curvature_500_summary.json`
- 目标 GT+多样化候选 parquet：`generate_traj_data/output/<dataset>/<clip>.egomotion.parquet`
- 目标摘要：`generate_traj_data/output/acc_curvature_diverse_gt_summary.csv`

## 验证

```bash
cd generate_traj_data/kmeans
../.venv/bin/python -m unittest test_cluster_acc_curvature_top2 test_pipeline_wrappers
```
