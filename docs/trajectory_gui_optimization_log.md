# Trajectory GUI Optimization Log

本文档记录轨迹 GUI/标注工具的每次优化，按修改序号追踪需求、改动位置、验证方式和后续注意事项。

> 路径说明：031 之前的条目保留当时真实文件名，例如 `trajectory_gui_enhanced.py` 和
> `traj_gui_enhanced/`；031 之后当前入口为 `trajectory_annotator.py`，GUI 包为
> `traj_annotation/`，共享工具为 `traj_core/`，推理包为 `traj_inference/`。

## 001 停车点显式标识与悬浮提示

- 日期：2026-05-11
- 需求：若轨迹中含有停车动作，在对应位置标记为红色大点；鼠标悬浮于点上时显示停车持续时间或帧数。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 初始新增停车判定参数：`STOP_SPEED_THRESHOLD_MPS = 0.3`、`STOP_MIN_FRAMES = 5`、`STOP_MARKER_RADIUS_PX = 10`。当前阈值已在 008 调整为 `0.1`，红点尺寸已在 003 调整为 `5`。
  - 新增 `_speed_profile_from_trajectory()`，优先使用轨迹 `vx/vy/vz` 计算逐帧速度，速度字段不可用时回退到 `x/y/z` 位置差分。
  - 新增 `_detect_stop_segments()`，识别连续低速帧段，并输出起止帧、帧数、持续时间和平均速度。
  - 新增 `_draw_generated_stop_markers()`，在 BEV 视角为检测到的停车段绘制红色大点及简短 STOP 标签。
  - 新增 hover 命中区域与 tooltip：`_on_traj_canvas_motion()`、`_show_stop_tooltip()`、`_hide_stop_tooltip()`。
- 当前行为：
  - 任意轨迹中连续至少 5 帧速度小于当前 `STOP_SPEED_THRESHOLD_MPS` 时，会被识别为停车段。
  - 停车点标记在该停车段的末帧位置。
  - 悬浮提示显示轨迹编号、停车持续时间、帧数、帧范围、平均速度。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。
  - 已用模拟速度序列验证 `_detect_stop_segments()` 可识别连续停车帧段。
  - GUI 运行后可在 BEV 画布上查看红色停车点，鼠标悬浮红点应出现 tooltip。
- 注意事项：
  - 停车阈值和最小帧数目前是经验值，如后续出现误检/漏检，可按数据特征调整。

## 002 BEV 左侧速度窗口

- 日期：2026-05-11
- 需求：在可视化界面左侧 BEV 视角下添加速度窗口，记录对应轨迹速度变化；横轴为时间或帧数，纵轴为速度。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 在 BEV canvas 下方新增 `Speed Profile` 画布。
  - 新增 `_selected_speed_profile_source()`，根据当前选中的轨迹获取速度曲线；无生成轨迹时回退显示 GT future 的速度曲线。
  - 新增 `_draw_speed_profile()`，绘制当前轨迹速度-帧数曲线。
  - 在速度窗口中用红色背景带标出检测到的停车帧段，并绘制当前停车阈值线。
  - 在 `_update_display()` 中同步刷新速度窗口，使其随轨迹切换、样本切换、删除/保留状态变化而更新。
- 当前行为：
  - 横轴为 frame index，纵轴为 speed，单位 m/s。
  - 曲线颜色与当前选中轨迹在 BEV 中的颜色保持一致。
  - 停车段与 001 的停车识别逻辑共用同一套阈值。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。
  - GUI 运行后切换右侧轨迹列表或方向键切换轨迹，速度窗口应同步变化。
- 注意事项：
  - 当前速度窗口显示当前选中轨迹，不叠加所有轨迹，避免左侧区域过于拥挤。

## 003 停车红点尺寸收敛

- 日期：2026-05-11
- 需求：停车红点只需要通过红色突出，不应过大影响轨迹定位准确性。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 将 `STOP_MARKER_RADIUS_PX` 从 10 调整为 5。
  - 生成轨迹停车点在 BEV 中不再绘制常驻 `STOP` 文本，仅保留小红点。
  - 手工停车点也同步改为小红点，并保留 BEV hover 提示停车时长。
  - FC 中的手工停车点从大圆点加文字改为小红点。
- 当前行为：
  - 停车位置主要通过小红点定位，减少遮挡。
  - 详细停车信息通过悬浮提示查看。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 004 停车点同步显示到 FC 视角

- 日期：2026-05-11
- 需求：BEV 和 FC 视角中的轨迹修改/标记应同步，停车红点不能只显示在 BEV 中。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 新增 `_draw_generated_stop_markers_on_image()`，将自动识别出的生成轨迹停车段投影到 FC 图像。
  - FC 红点使用与 BEV 相同的停车段检测结果，标记同一帧位置。
  - 红点绘制发生在 FC 图像 resize 前，使用原始标定坐标投影，避免显示缩放导致位置偏差。
- 当前行为：
  - 生成轨迹停车点在 BEV 和 FC 中同步显示。
  - 手工停车点仍沿用已有相机投影链路，并调整为小红点。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。
- 注意事项：
  - 当前新增的自动停车红点只投影到 FC，避免其它相机视图过度拥挤。

## 005 速度窗口 hover 与 BEV/FC 同帧高亮

- 日期：2026-05-11
- 需求：速度窗口中的曲线可滑动选择，并与 BEV 和 FC 中的轨迹点对应；同一点使用相同颜色或方式标识。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 新增 `speed_hover_frame_idx` 状态，记录速度窗口当前悬浮选中的帧。
  - 新增 `_draw_speed_hover_marker_on_bev()`，在 BEV 中用亮绿色标记当前帧轨迹点。
  - 新增 `_draw_speed_hover_marker_on_image()`，在 FC 中用同一亮绿色标记当前帧轨迹点。
  - 速度窗口中同步绘制亮绿色竖线、曲线点和帧号/速度文本。
- 当前行为：
  - 在速度窗口内移动鼠标时，速度窗口、BEV、FC 三处同步显示同一帧。
  - 离开速度窗口后，同步高亮消失。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 006 速度窗口矩形范围内按竖直方向选中

- 日期：2026-05-11
- 需求：速度窗口中的选中逻辑应以整个坐标系矩形框为悬停范围；同一竖直方向上的点定位为同一帧。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 新增 `_speed_plot_geometry()`，统一维护速度窗口坐标框边界。
  - 新增 `_on_speed_canvas_motion()`，仅判断鼠标是否位于坐标框矩形内，帧号只由横向位置映射。
  - 鼠标在图框内同一 x 位置上下移动时，保持同一帧选中。
- 当前行为：
  - 悬停范围是速度图坐标框内部完整矩形，而不是曲线附近的小范围。
  - 横向位置决定 frame index，纵向位置不影响帧号。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 007 新增真值轨迹速度窗口

- 日期：2026-05-11
- 需求：明黄色轨迹为真值轨迹，需要在多样化轨迹速度窗口下方新增专门的真值轨迹速度窗口。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 将原速度窗口标题改为 `Diversity Speed Profile`。
  - 新增 `GT Speed Profile` 画布，固定显示当前样本明黄色 GT future 的速度曲线。
  - 新增 `_gt_speed_profile_source()` 与 `_draw_gt_speed_profile()`，GT 速度窗口使用与多样化速度窗口相同的绘图和停车段标注逻辑。
- 当前行为：
  - 上方速度窗口追踪当前选中的多样化轨迹；编辑人工 Bezier/停车点时优先追踪人工预览轨迹。
  - 下方速度窗口追踪 GT 轨迹。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 008 停车检测阈值调整为 0.1 m/s

- 日期：2026-05-11
- 需求：停车点检测改为速度小于 0.1，而不是更宽松的阈值。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 将 `STOP_SPEED_THRESHOLD_MPS` 调整为 `0.1`。
  - 多样化轨迹、GT 轨迹和速度窗口中的停车段检测共用该阈值。
- 当前行为：
  - 连续至少 5 帧速度 `< 0.1 m/s` 会被识别为停车段。
  - 速度窗口中的停车阈值线同步显示为 `stop < 0.1m/s`。
- 验证方式：
  - 已用模拟速度序列验证 `_detect_stop_segments()` 在新阈值下工作。
  - 已通过项目 venv 的 Python 静态编译检查。

## 009 停车红点同步覆盖 BEV 和 FC

- 日期：2026-05-11
- 需求：检测到的停车点直接用红色小点在 BEV 和 FC 两个窗口的对应轨迹位置标识。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - GT 轨迹在 BEV 中新增停车红点标识和 hover 信息。
  - 新增 `_draw_gt_stop_markers_on_image()`，将 GT 停车点投影到 FC。
  - 多样化轨迹继续使用 `_draw_generated_stop_markers()` 与 `_draw_generated_stop_markers_on_image()` 同步 BEV/FC 红点。
- 当前行为：
  - 自动检测出的 GT 与多样化轨迹停车点均在 BEV 和 FC 中显示为红色小点。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 010 人工停车点标识与速度同步

- 日期：2026-05-11
- 需求：人工添加停车点也要同样标识；添加停车点后轨迹速度变化需要同步到相应速度窗口。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 手工停车点在 BEV 和 FC 中统一显示为红色小点。
  - 多样化速度窗口在人工 Bezier/停车点编辑过程中优先显示手工预览轨迹速度。
  - 手工预览轨迹速度来自 `_build_manual_bezier_trajectory()`，因此新增停车点后的减速/驻停变化会进入速度曲线。
- 当前行为：
  - 添加人工停车点后，BEV/FC 立即显示红点。
  - 多样化速度窗口立即切换为人工预览轨迹速度，并同步显示停车段。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 011 轨迹列表速度平滑性诊断

- 日期：2026-05-11
- 需求：右侧 `Trajectories` 列表需要检查每条多样化轨迹的速度曲线；若速度不够平滑，在对应轨迹前标识 `[×]`，鼠标悬停叉号显示建议删除原因。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 新增 `_speed_smoothness_diagnostics()`，基于速度曲线的一阶加速度和二阶 jerk 做平滑性初筛。
  - 新增 `_refresh_trajectory_smoothness()`，刷新列表前逐条诊断多样化轨迹。
  - `Trajectories` 列表中速度不平滑的轨迹前显示 `[×]`，并以浅红色显示该行。
  - 新增列表 hover tooltip，鼠标悬停在 `[×]` 区域显示 `建议删除原因：速度不够平滑`。
- 当前行为：
  - 该标记仅作为建议删除提示，不会自动删除轨迹。
  - 当前唯一原因是 `速度不够平滑`。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 012 多样化轨迹速度曲线编辑与点密度同步

已删除：右键进入调整、左键拖拽速度曲线、再次右键保存/恢复的手动编辑入口已删除。  
保留：接受后的速度变化仍会通过轨迹重采样与点密度保持一致，新的入口见 016。

- 日期：2026-05-11
- 需求：选中多样化轨迹后，可在速度窗口右键选择是否调整；通过左键拖拽修改速度曲线；再次右键可选择保存或恢复；速度修改必须和轨迹点密度严格绑定。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 在 `Diversity Speed Profile` 上新增右键菜单：`是，进行调整`、`保存更改`、`不保存，恢复原状`。
  - 新增速度编辑状态：`speed_edit_active`、`speed_edit_original_traj`、`speed_edit_speed` 等。
  - 左键拖拽速度曲线时更新编辑速度，并沿原轨迹几何路径按速度分布重新采样轨迹点。
  - 重新采样后反算 `vx/vy/vz`、`qz/qw`、`curvature`，保证速度曲线和 BEV/FC 中点密度一致。
  - 保存时将当前选中轨迹写入 `filtered` parquet；不保存则恢复进入编辑前的轨迹。
- 当前行为：
  - 速度编辑仅作用于当前选中的多样化轨迹，不作用于 GT。
  - 编辑期间不能切换到其它轨迹，需先保存或恢复。
  - 编辑期间不能切换样本或保存过滤结果，需先保存或恢复速度编辑。
  - 为保持原轨迹终点，编辑速度会作为沿原路径的相对密度分布；保存后速度显示来自新点位反算结果。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。
  - 已用模拟轨迹验证速度重采样函数会保持路径终点并改变点密度分布。

## 013 列表诊断标识合并与平滑性判定放宽

- 日期：2026-05-11
- 需求：`Trajectories` 列表已有 `[√]` 标识；若建议删除，应直接改为 `[×]`，不要同时存在。速度平滑性检测不应过严，若与 GT 速度曲线差别不大可先保留。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 列表行首改为单一状态：正常保留显示 `[✓]`，建议删除显示 `[×]`。
  - `_speed_smoothness_diagnostics()` 新增 GT 速度参考输入，计算与 GT 的平均速度差。
  - 放宽加速度和 jerk 阈值；与 GT 平均差异较小时，除非出现极端尖刺，否则不标记为建议删除。
- 当前行为：
  - `[×]` 表示建议删除，hover 原因仍为 `速度不够平滑`。
  - 接近 GT 的速度曲线会被保留，即使存在轻微波动。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 014 速度编辑菜单文案与局部拖拽

已删除：速度窗口右键菜单和局部拖拽调整入口已删除。  
保留：局部/平滑化思想迁移到 016 的自动优化流程中。

- 日期：2026-05-11
- 需求：右键菜单按钮从 `是，进行调整` 改为 `进行调整`；拖拽一个速度点时只牵动周围一小部分并保持平滑，不要牵动整体范围。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 右键菜单文案改为 `进行调整`。
  - 新增 `SPEED_EDIT_LOCAL_RADIUS_FRAMES`，速度拖拽采用局部余弦权重，只影响鼠标所在帧附近的小范围。
  - 移除原先跨拖拽区间的大范围线性联动，降低整体曲线被误改的风险。
- 当前行为：
  - 单点拖拽会形成局部平滑修改。
  - 修改后仍会沿原轨迹路径重采样点位并反算速度字段，保持点密度与速度绑定。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 015 Trajectories 列表稳定显示与红色曲线区分

- 日期：2026-05-11
- 需求：拉宽 `Trajectories` 列框，从左至右稳定显示行首 `[√]`/`[×]`；建议删除的 `[×]` 轨迹曲线需要用红色区分；速度编辑保存后应及时刷新判断。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 固定右侧列表区域宽度，并设置 listbox `width=56`、左对齐，确保行首状态可见。
  - 列表状态符号统一为 `[√]` 和 `[×]`，不再使用其它勾号字符。
  - `_trajectory_draw_style()` 读取平滑性诊断结果，建议删除的轨迹在 BEV/FC 中以红色曲线显示。
  - 保存速度编辑后沿用既有 `_load_sample()` + `_update_display()` 流程，重新加载并刷新平滑性诊断。
- 当前行为：
  - 切换样本再切回时，列表会重新计算并稳定显示 `[√]`/`[×]`。
  - `[×]` 行为浅红色，且对应轨迹曲线也为红色。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 016 按钮式速度优化与 GT 停车编辑

- 日期：2026-05-11
- 需求：恢复右侧 CoT 显示；删除多样化轨迹速度窗口右键拖拽调整，改为按钮式自动平滑优化；GT 速度窗口增加自动优化和停车添加功能，并提供明确的保存/取消/撤回流程。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 修复右侧栏布局，将 `Trajectories` 列表放入独立容器，避免其占满右栏导致 CoT/Cluster 区域不可见。
  - `Diversity Speed Profile` 表头右侧新增 `优化速度曲线` 按钮。
  - 多样化轨迹优化后在速度窗口下方显示 `接受`、`取消`；接受后写入 `filtered` parquet，取消则恢复原轨迹。
  - `GT Speed Profile` 表头右侧新增 `优化速度曲线` 和 `停车添加`。
  - GT 优化后显示 `接受`、`取消`；GT 停车添加时显示 `保存`、`取消`、`撤回`。
  - GT 修改持久化到 `output_dir/gt_speed_edits.json`，并在当前样本再次打开时优先显示已保存的 GT 编辑结果。
- 当前行为：
  - 多样化轨迹速度优化会自动平滑毛刺/抖动，并同步重采样轨迹点密度。
  - GT 停车添加时，在 GT 速度图中点击某一帧，会使该帧及其后续轨迹点保持停车状态；撤回会恢复到点击前并允许重新选择。
  - CoT 区域恢复在右侧栏中显示。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 017 速度窗口确认按钮归属位置修正

- 日期：2026-05-11
- 需求：多样化轨迹速度优化后的 `接受`/`取消` 应显示在多样化速度窗口下方，不应显示在 GT 速度窗口下方，避免混淆。
- 改动文件：`trajectory_gui_enhanced.py`
- 主要改动：
  - 保存 GT 速度窗口表头 frame 引用。
  - `_pack_pred_speed_actions()` 显示多样化确认按钮时，固定插入到 GT 表头之前。
- 当前行为：
  - 多样化轨迹优化的 `接受`/`取消` 位于 `Diversity Speed Profile` 下方。
  - GT 优化/停车添加的确认按钮仍位于 `GT Speed Profile` 下方。
- 验证方式：
  - 已通过项目 venv 的 Python 静态编译检查。

## 018 Enhanced GUI 架构拆分

- 日期：2026-05-12
- 需求：`trajectory_gui_enhanced.py` 接近 6000 行，后续开发维护困难，需要按功能拆分为更清晰的包结构，同时保持现有命令和行为不变。
- 改动文件：`trajectory_gui_enhanced.py`、`traj_gui_enhanced/`、`README.md`
- 主要改动：
  - 新增 `traj_gui_enhanced/` 包，将常量、运行环境设置、纯轨迹/速度/投影/聚类工具函数拆入独立模块。
  - 将 `TrajectoryViewerEnhanced` 拆为主 `viewer.py` 和多个 `mixins/` 功能模块，覆盖样本 I/O、界面布局、导航、手工编辑、聚类中心、GT 控制、速度控制、BEV 绘制、相机绘制和速度图绘制。
  - `trajectory_gui_enhanced.py` 改为兼容入口，继续支持原有 `python trajectory_gui_enhanced.py ...` 启动方式，并 re-export `TrajectoryViewerEnhanced`、`parse_args`、`main`。
  - 新增 `tests/test_gui_helpers.py`，用标准库 `unittest` 覆盖关键纯函数，避免新增 pytest 依赖。
- 当前行为：
  - 本次为零行为变更重构；CLI 参数、GUI 文案、快捷键、parquet schema 和保存逻辑保持不变。
  - 为支持无 tkinter 的最小环境执行 `--help` 和 import 检查，Tk/ImageTk 改为 GUI 实例化时懒加载。
  - 若系统 Python 未安装 `tkinter`，会优先尝试使用项目本地 `.runtime/tk` 中的 Python 3.10 Tk runtime。
- 验证方式：
  - 已通过项目 venv 的 `unittest` helper 测试。
  - 已通过项目 venv 的 Python 静态编译检查。
  - 已验证 `trajectory_gui_enhanced.py --help` 和新旧 import 入口。
  - 已用 `timeout` 启动真实 GUI，确认进入 Tk mainloop 后由 timeout 正常终止。

## 019 现有速度面板叠加历史速度

- 日期：2026-05-13
- 需求：不要新增独立 history 速度窗口；在现有 `Diversity Speed Profile` 和 `GT Speed Profile` 中加入历史轨迹速度，横轴从历史负帧延伸到未来帧，例如 `-15..63`，并在当前帧 `0` 画垂直虚线。
- 改动文件：`traj_gui_enhanced/speed_utils.py`、`traj_gui_enhanced/constants.py`、`traj_gui_enhanced/mixins/speed_controls.py`、`traj_gui_enhanced/mixins/draw_speed.py`、`traj_gui_enhanced/mixins/draw_bev.py`、`traj_gui_enhanced/mixins/draw_camera.py`、`tests/test_gui_helpers.py`
- 主要改动：
  - 新增 `_history_speed_profile_from_xyz()`，使用 history 相邻点差分计算历史速度，避免把 history 点相对原点的距离误当速度。
  - 两个现有速度图都叠加绿色 history 曲线；future 曲线保持原有颜色。
  - 速度图横轴改为 history 负帧到 future 正帧，并在 `F0` 位置绘制垂直虚线。
  - hover 到负帧时映射到 history source，并在 BEV 历史轨迹上高亮对应历史点；hover 到非负帧时保持原有 pred/GT 高亮。
- 当前行为：
  - history 16 帧显示为 `F-15..F0`，future 64 帧显示为 `F0..F63`。
  - `Diversity Speed Profile` 用绿色 history + 当前选中生成/扩充轨迹未来速度。
  - `GT Speed Profile` 用绿色 history + GT future 速度。
- 验证方式：
  - 已通过项目 venv 的 `unittest` helper 测试。
  - 已通过项目 venv 的 Python 静态编译检查。

## 020 扩充轨迹 staged delete 与撤销

- 日期：2026-05-13
- 需求：删除扩充轨迹时，先只从 GUI 可视化中消失，原始 parquet 数据保留；如果误删，可通过撤销删除恢复；只有确认保存后才真正从 output parquet 中删除相关数据。
- 改动文件：`traj_gui_enhanced/trajectory_identity.py`、`traj_gui_enhanced/mixins/delete_controls.py`、`traj_gui_enhanced/viewer.py`、`traj_gui_enhanced/mixins/navigation.py`、`traj_gui_enhanced/mixins/widget_layout.py`、`traj_gui_enhanced/mixins/sample_io.py`、`traj_gui_enhanced/mixins/draw_bev.py`、`traj_gui_enhanced/mixins/draw_camera.py`、`traj_gui_enhanced/mixins/manual_events.py`、`traj_gui_enhanced/mixins/manual_editing.py`、`traj_gui_enhanced/mixins/cluster_controls.py`、`traj_gui_enhanced/mixins/speed_controls.py`、`tests/test_gui_helpers.py`
- 主要改动：
  - 新增 `trajectory_identity.py`，集中维护 source 规范化、GT 保护、`(t0_us, sample_idx)` 轨迹 key 和 parquet 行删除过滤逻辑。
  - 新增 `DeleteControlsMixin`，集中维护 pending delete 集合、撤销栈、可见轨迹列表和 listbox 行到真实轨迹 index 的映射。
  - `Delete/Backspace` 现在会将当前非 GT 轨迹加入 pending delete，并立即从右侧列表、BEV 和相机投影中隐藏。
  - 新增 `Undo Delete` 按钮，恢复最近一次 pending delete 的轨迹。
  - `Confirm Save (Ctrl+S)` 根据 pending delete key 真正从 active parquet 中删除对应行；保存前切换样本会提示先保存或撤销。
  - 若删除轨迹匹配当前手工 Bezier 控制点，控制点会先在内存中 staged 隐藏；撤销删除恢复，确认保存后同步写回 `manual_points.json`。
  - `Save Curve Traj` 追加行写入 `source=manual_bezier`；`Confirm Save` cluster center 追加行写入 `source=cluster_center`。
- 当前行为：
  - GT 行仍受保护，不能删除。
  - 删除后的轨迹在保存前只是 GUI 内隐藏，撤销删除可恢复显示。
  - 保存后重新打开同一 t0，已确认删除的轨迹不再出现。
  - 匹配当前手工 Bezier 曲线的删除会在确认保存后同步清理 `manual_points.json` 中对应控制点。
- 验证方式：
  - 已通过项目 venv 的 `unittest` helper 测试。
  - 已通过项目 venv 的 Python 静态编译检查。

## 021 视频帧逐帧 t0 索引与 GUI 覆盖率

- 日期：2026-05-13
- 需求：完整数据所有帧已明确为所有 10Hz 视频帧，不能再被旧的 30 帧候选扫描和 `candidate_stride` 漏掉；GUI 也需要能浏览每个视频帧，并标出哪些帧还没有生成轨迹。
- 改动文件：`frame_index.py`、`run_inference.py`、`data_loader.py`、`traj_gui_enhanced/cli.py`、`traj_gui_enhanced/viewer.py`、`traj_gui_enhanced/mixins/sample_io.py`、`traj_gui_enhanced/mixins/widget_layout.py`、`traj_gui_enhanced/mixins/speed_controls.py`、`traj_gui_enhanced/mixins/draw_bev.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 新增 `frame_index.py`，统一从 `data-timestamps/{clip}.timestamps.parquet` 读取主视频帧时间戳，并生成逐帧 `t0_us`。
  - `run_inference.py` 默认改为 `--t0_source video_frames --frame_stride 1`，按视频帧逐帧生成；旧稀疏速度候选模式保留为 `--t0_source speed_candidates`。
  - GUI 默认改为 `--index_mode video_frames --frame_stride 1`，不再只依赖 output parquet 中已有的 `t0_us`。
  - GUI 状态栏新增 clip 覆盖率：`Video t0: N | Generated t0: M | Missing: K | Current: generated/no generated`。
  - 如果当前视频帧没有生成轨迹，GUI 仍显示图像、真实 history、GT，右侧轨迹列表为空。
  - `load_data()` 新增 history/future valid mask；录制开头真实历史帧不足时，history 速度和 BEV 历史轨迹不绘制外推出来的历史段。
  - 后续修正：跨 clip egomotion 拼接不能只看 timestamp 连续；如果相邻 clip 边界的空间跳变对应速度超过默认 `80m/s`，会切断连续段，避免 clip 开头历史速度出现百/千 m/s 的异常值。
  - 后续修正：dataset 级 egomotion 连续段改为缓存并向量化按 clip/t0 定位，避免 GUI 启动时重复扫描全量 egomotion；clip 开头的速度面板改为优先使用 `vx/vy/vz` 历史速度，位置 history 不跨坐标跳变，但速度 history 不再消失。
- 当前行为：
  - 默认推理和默认 GUI 都按主视频帧逐帧工作。
  - `--candidate_stride` 只对旧 `speed_candidates` 模式生效。
  - `--index_mode generated` 可回到只浏览已生成 output t0 的旧 GUI 行为。
- 后续注意：
  - 当前只完成代码级接入和轻量验证，还需要用真实推理输出进一步检查。
  - 建议先用小 clip 或 `--max_samples` 运行 `run_inference.py --t0_source video_frames --frame_stride 1`，再打开 GUI 连续翻帧，确认 output 覆盖率、t0 顺序、无轨迹帧显示和开头无 history 的显示都符合预期。
- 验证方式：
  - 已通过新增 helper 测试，覆盖视频帧索引、GUI video-frame sample index、history valid mask。
  - 已通过项目 venv 的 `unittest` helper 测试。

## 022 output GT / 伪 GT 数据语义整理

- 日期：2026-05-14
- 需求：真实 GT 只应来自原始数据集 `/home/ubuntu/Public/train_data`；output parquet 应只保存 VLA 生成、manual Bezier、cluster center 等伪 GT，避免把 VLA 的 `sample_idx=0` 误判为 GT。
- 改动文件：`run_inference.py`、`traj_gui_enhanced/trajectory_identity.py`、`traj_gui_enhanced/mixins/sample_io.py`、`traj_gui_enhanced/mixins/gt_controls.py`、`traj_gui_enhanced/constants.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - `run_inference.py` 新写入的 VLA 轨迹行增加 `source=vla`。
  - GT 判断规则改为只认显式 `source=gt`，不再把 `source` 为空且 `sample_idx=0` 的 output 行自动当作 GT。
  - GUI 读取 output 时过滤 legacy `source=gt` 行；这些行不进入右侧伪 GT 轨迹列表，也不计入 generated 覆盖率。
  - 空 `source` 的旧 output 行按伪 GT 处理，用于兼容早期未写 source 的 VLA 输出。
  - `AUTO_OPTIMIZE_GT_ON_LOAD` 改为 `False`，并停止在加载样本时自动把 GT 副本写回 output。
  - `_write_gt_trajectory_to_parquet()` 改为提示真实 GT 来自原始数据集，不再写入 output parquet。
- 当前行为：
  - 真实 GT 用 `load_data()` 从原始数据集读取并显示。
  - output parquet 是伪 GT 数据容器，后续动力学约束优化将面向所有非 `source=gt` output 行。
  - 已存在于磁盘上的 legacy `source=gt` 行不会被本次代码自动删除；若需要物理清理，应另做带备份的迁移工具。
- 验证方式：
  - 已新增并通过身份规则测试：空 source + `sample_idx=0` 不再是 GT，`source=vla` 可删除/可编辑。
  - 已新增并通过 output 覆盖率测试：`source=gt` 行不计入 generated t0。

## 023 伪 GT dynamics 基础模块与保存前优化

- 日期：2026-05-14
- 需求：VLA 生成轨迹、manual Bezier、cluster center 以及人工修改后的非 GT 轨迹都可能存在速度、加速度或曲率不合理的问题，需要先建立独立的动力学约束模块，并在伪 GT 写入 output 前统一优化和重算字段。
- 改动文件：`traj_gui_enhanced/dynamics/`、`run_inference.py`、`traj_gui_enhanced/mixins/sample_io.py`、`traj_gui_enhanced/mixins/manual_editing.py`、`traj_gui_enhanced/mixins/speed_controls.py`、`tests/test_gui_helpers.py`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 新增 `traj_gui_enhanced/dynamics/limits.py`，集中定义伪 GT 速度、单步位移、加速度、jerk、曲率等约束阈值。
  - 新增 `metrics.py` / `diagnostics.py` / `optimizer.py`，提供轨迹动力学指标、违规诊断、保守平滑与加速度限制重采样。
  - 新增 `trajectory_components_from_xyz()`，统一从 xyz 重算 `vx/vy/vz/qx/qy/qz/qw/curvature`，避免 GUI 和推理脚本各自维护一套字段生成逻辑。
  - `run_inference.py` 保存 VLA 轨迹前会先调用 `optimize_pseudo_gt_trajectory()`，并写入优化后的伪 GT 字段。
  - `Save Curve Traj` 和 cluster `Confirm Save` 追加轨迹前会走同一套 dynamics 优化。
  - 已保存非 GT 轨迹通过速度编辑保存回 parquet 前，会再次进行 dynamics 优化并更新内存和 parquet 中的对应行。
- 当前行为：
  - 真实 GT 仍只从原始数据集读取，dynamics 保存前优化只面向非 GT output 行。
  - 若轨迹无法完全满足全部阈值，优化器会返回当前评分最好的保守结果；后续 GUI 直接几何编辑时还需要补充更明确的提示/接受/取消流程。
  - 本次尚未实现“直接拖拽已保存轨迹几何点/终点并写回原行”的完整交互，只完成底层模块和已有保存路径接入。
- 验证方式：
  - 已新增并通过 dynamics 单元测试：异常轨迹可被标记为速度/步长/加速度/曲率违规；尖刺轨迹优化后曲率下降且端点保持。
  - 已新增并通过保存路径测试：非 GT 轨迹写回 parquet 前会先优化，保存后的曲率低于保存前。

## 024 已保存伪 GT 的 BEV 关键帧几何编辑第一版

- 日期：2026-05-14
- 需求：在 GUI 中不仅能优化速度曲线，还要能直接修改已经保存到 output 的非 GT 伪 GT 轨迹几何形状；修改后整条轨迹需要经过动力学优化，并能接受、取消或恢复原状。
- 改动文件：`traj_gui_enhanced/dynamics/editing.py`、`traj_gui_enhanced/mixins/saved_traj_editing.py`、`traj_gui_enhanced/viewer.py`、`traj_gui_enhanced/mixins/manual_events.py`、`traj_gui_enhanced/mixins/draw_bev.py`、`traj_gui_enhanced/mixins/widget_layout.py`、`traj_gui_enhanced/mixins/navigation.py`、`traj_gui_enhanced/mixins/sample_io.py`、`tests/test_gui_helpers.py`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 新增 `editable_trajectory_keyframes()`，默认每 8 帧取一个 BEV 编辑手柄，并包含最后一帧 endpoint。
  - 新增 `deform_trajectory_by_keyframe_drag()`，拖动某个关键帧时按帧距离做局部平滑形变，同时锁住第一帧，避免起点突然跳动。
  - 新增 `SavedTrajectoryEditingMixin`，维护已保存非 GT 轨迹的几何编辑状态、原始轨迹备份、dirty 状态、保存/取消/恢复逻辑。
  - GUI 新增 `Edit Traj`、`Save Edit`、`Cancel Edit`、`Restore Edit` 按钮。
  - 进入 `Edit Traj` 后，BEV 上会给当前选中非 GT 轨迹绘制关键帧手柄；左键拖动手柄会局部形变轨迹，并立即调用 dynamics 优化和字段重算。
  - `Save Edit` 写回当前 parquet 原行；`Cancel Edit` 放弃内存修改；`Restore Edit` 将当前编辑恢复到进入编辑时的原始轨迹。
  - 切换样本、切换轨迹或 `Ctrl+S` 全局保存时，如果存在未处理的几何编辑，会提示先保存或取消。
- 当前行为：
  - 真实 GT 仍不可作为 output 伪 GT 编辑。
  - 编辑入口只面向当前选中且未 pending delete 的非 GT output 轨迹。
  - 这是第一版 BEV 几何编辑，后续需要真实 GUI 操作检查手柄间隔、影响半径、优化强度和按钮布局。
- 验证方式：
  - 已新增并通过 keyframe/局部形变纯函数测试。
  - 已新增并通过 mixin 状态测试：进入编辑、拖动 endpoint、重算字段、取消恢复原始轨迹。

## 025 伪 GT 几何编辑状态保护与保存闭环

- 日期：2026-05-14
- 需求：TODO-004 的几何编辑已经接入后，还需要防止未保存编辑状态与删除、速度编辑、切换样本/轨迹等操作交叉，避免内存状态和 parquet 写回混乱。
- 改动文件：`traj_gui_enhanced/mixins/delete_controls.py`、`traj_gui_enhanced/mixins/saved_traj_editing.py`、`traj_gui_enhanced/mixins/speed_controls.py`、`tests/test_gui_helpers.py`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 几何编辑 active 时，`Delete/Backspace` 和 staged delete 会被阻止。
  - 几何编辑 active 时，不能启动伪 GT 速度编辑。
  - 几何编辑 active 时，不能再次启动另一个 `Edit Traj` 覆盖当前编辑状态。
  - 已有导航和全局保存保护继续保留：切换样本、切换轨迹、`Ctrl+S` 前会提示先保存或取消几何编辑。
  - 补充 `Save Edit` 闭环测试，确认保存会调用 parquet 写回路径、清理编辑状态并重新加载当前样本。
- 当前行为：
  - TODO-004 的代码闭环已经完成：非 GT 轨迹可编辑、编辑后走 dynamics 优化、可保存写回原 parquet 行、可取消和恢复原状，并有未保存状态保护。
  - 后续主要是用真实 GUI 做手感验收和参数调优。
- 验证方式：
  - 已新增并通过删除保护、二次编辑保护、速度编辑保护、`Save Edit` 写回状态测试。

## 026 逐帧索引 future 连续性过滤启动卡顿修复

- 日期：2026-05-14
- 问题：TODO-001 后续补充的“当前帧 + 未来 6.4s egomotion 连续性过滤”功能正确，但初版实现对每个 `t0_us` 都重复排序全 dataset timestamp，并逐点循环判断覆盖，导致 GUI 默认 `video_frames` 模式启动前长时间卡在样本索引构建。
- 改动文件：`data_loader.py`、`tests/test_gui_helpers.py`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 新增 `_sorted_unique_timestamps()` 和 `_coverage_mask_for_sorted_timestamps()`，把 timestamp 排序去重和覆盖判断拆开。
  - `filter_t0s_with_full_future()` 改为一次性构造 `t0 + [0..64] * 0.1s` 的矩阵，并用 `np.searchsorted()` 向量化判断所有当前帧和 future 帧覆盖情况。
  - 保留原有语义：小缺口 `<=0.3s` 插值，大缺口和最后尾部 future 不足时过滤。
  - 增加大索引性能回归测试，防止后续又退回逐 t0 慢循环。
- 验证方式：
  - `test_full_future_filter_large_index_is_vectorized` 初版失败于约 2.96s，向量化后同组测试 0.06s 通过。
  - 本机实测 `/home/ubuntu/Public/train_data` 全量 302 个 clip、约 17.2 万个候选 t0 的连续性过滤约 0.87s。
  - 用真实 GUI 命令短启动，已能输出 `Video-frame sample index: 170962 samples ...` 并进入 Tk 主循环。

## 027 History 轨迹显示层平滑去噪

- 日期：2026-05-14
- 需求：此前 GT future 速度已经做显示平滑，但 history 轨迹和 history 速度直接使用原始 `ego_history_xyz` 差分，原始 egomotion 噪声会造成绿色 history 线和当前速度估计出现尖刺。
- 改动文件：`traj_gui_enhanced/speed_utils.py`、`traj_gui_enhanced/mixins/speed_controls.py`、`traj_gui_enhanced/mixins/manual_editing.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`、`docs/trajectory_gui_user_manual.html`
- 主要改动：
  - 新增 `_smooth_history_xyz_for_display()`：对 history 点做轻量 binomial 平滑，保持第一帧和当前帧不动。
  - 新增 `_smoothed_history_speed_profile_from_xyz()`：先使用平滑后的 history 点计算速度，再做轻量显示平滑。
  - GUI 的绿色 history 速度线、BEV history 轨迹、hover 高亮点都使用显示层平滑后的 history。
  - 手工 Bezier 当前速度估计使用有效 history mask 和显示层平滑后的尾部点，降低原始 history 噪声对手工轨迹初速度的影响。
  - 原始 `ego_history_xyz` 不写回、不覆盖，仍保留为 source data。
- 验证方式：
  - 新增测试覆盖 history 平滑保持当前帧、降低中间噪声、降低 history 差分速度尖刺，并确认 GUI 取 display-smoothed history 时不会修改 `conv_data` 中的原始 history。

## 028 GUI parquet 写回备份、日志和元数据

- 日期：2026-05-18
- 需求：TODO-007 希望梳理保存行为和数据版本，避免多个 GUI 操作直接覆盖 active parquet 后难以追踪或恢复。
- 改动文件：`traj_gui_enhanced/save_audit.py`、`traj_gui_enhanced/mixins/sample_io.py`、`traj_gui_enhanced/mixins/manual_editing.py`、`traj_gui_enhanced/mixins/cluster_controls.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 新增 `save_audit.py`，集中提供 `apply_gui_edit_metadata()` 和 `write_parquet_with_audit()`。
  - GUI parquet 写回前会将原文件备份到 `output/.backups/{dataset}/{clip}/`。
  - GUI parquet 写回后会追加一行到 `output/edit_log.jsonl`，记录操作类型、样本身份、备份路径、写入前后行数和影响行数。
  - 新增/编辑的 GUI 伪 GT 行写入 `edit_version`、`edited_by_gui`、`edit_time`、`edit_operation`。
  - 已接入选中轨迹编辑保存、pending delete 确认保存、manual Bezier 追加和 cluster center 追加。
- 当前行为：
  - GUI 仍覆盖 active parquet，保持原读取路径不变。
  - 第一版只覆盖 GUI parquet 写回；`manual_points.json`、cluster center 库文件和 GUI 恢复入口在后续 029 中补齐。
- 验证方式：
  - 新增测试覆盖 metadata 递增、审计写回创建备份与 JSONL 日志、选中轨迹保存审计、删除确认保存审计。

## 029 TODO-007 保存审计闭环完善

- 日期：2026-05-18
- 需求：继续完善 TODO-007，使保存行为和数据版本不只覆盖 parquet 写回，也覆盖 GUI sidecar 文件，并提供 GUI 恢复入口。
- 改动文件：`traj_gui_enhanced/save_audit.py`、`traj_gui_enhanced/mixins/sample_io.py`、`traj_gui_enhanced/mixins/cluster_controls.py`、`traj_gui_enhanced/mixins/widget_layout.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - `save_audit.py` 新增通用文本文件审计写入、审计日志读取、最近备份查找和从备份恢复 active 文件的能力。
  - `manual_points.json` 写入接入 `save_manual_points` 日志，旧文件备份到 `output/.backups/files/manual_points/`。
  - cluster center 分类库和 Bezier center 元数据写入分别接入 `write_cluster_category_file`、`write_bezier_cluster_center_ids` 日志，旧文件备份到 `output/.backups/files/k_means/`。
  - GUI 全局操作区新增 `View Log` 和 `Restore Backup`；恢复当前 clip parquet 时会先备份当前 active parquet，并记录 `restore_parquet_backup`。
- 当前行为：
  - GUI 仍覆盖 active 文件，但 parquet、manual controls 和 cluster center 库的 GUI 写入都已有写前备份和 JSONL 日志。
  - 离线批处理输出仍不纳入 GUI 审计策略。
- 验证方式：
  - 新增测试覆盖文本文件审计、备份恢复、manual points 写入审计、cluster center 文件写入审计和当前 clip parquet 恢复入口。

## 030 TODO-008 Windows GUI 第一版支持

- 日期：2026-05-18
- 需求：工具需要扩展到 Windows 上使用；本阶段用户明确不要求 Windows 本机模型推理，只需要 Windows 上使用增强 GUI 并进行轨迹扩充。
- 改动文件：`traj_gui_enhanced/cli.py`、`traj_gui_enhanced/environment.py`、`data_loader.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - GUI CLI 默认路径改为跨平台相对路径：`train_data`、`output`、`calibration`。
  - GUI CLI 新增环境变量覆盖：`GENERATE_TRAJ_GUI_DATA_ROOT`、`GENERATE_TRAJ_GUI_OUTPUT_DIR`、`GENERATE_TRAJ_GUI_CALIBRATION_DIR`，并兼容 `TRAIN_DATA_ROOT`、`OUTPUT_DIR`、`CALIBRATION_DIR`。
  - `environment.py` 支持 Windows 平台分支：Windows 下不设置 X11 `DISPLAY`，不加载 Linux `.runtime/tk` fallback，并使用 `os.pathsep` 拼接 path-like 环境变量。
  - `data_loader.py` 移除无条件插入 `/home/tsingyu/lxh/alpamayo_1.5/src` 的副作用，仅在 `ALPAMAYO_SRC` 或 repo-local `alpamayo_1.5/src` 存在时才加入 `sys.path`。
  - README 新增 Windows GUI-only PowerShell 安装和启动示例。
- 当前行为：
  - Windows 支持范围仅为增强 GUI 浏览、扩充、编辑、删除、保存和审计恢复已有数据/轨迹。
  - Windows 本机 `run_inference.py` 模型推理仍不在本阶段支持范围内。
- 验证方式：
  - 新增测试覆盖跨平台 GUI 默认路径、环境变量覆盖、Windows 环境初始化跳过 Linux fallback、以及 `data_loader` 不再污染 GUI import 的 Linux `sys.path`。

## 031 推理与 GUI 标注目录结构拆分

- 日期：2026-05-18
- 需求：模型推理和 GUI 轨迹扩充标注需要分离，方便把 GUI 工具提供给标注人员做轨迹多样化标注，同时保持现有功能一致。
- 改动文件：`trajectory_annotator.py`、`run_inference.py`、`traj_core/`、`traj_annotation/`、`traj_inference/`、`tests/test_project_structure.py`、`tests/test_gui_helpers.py`、`README.md`、`docs/trajectory_gui_feature_status_and_todo.md`
- 主要改动：
  - 新增 `trajectory_annotator.py` 作为 GUI 标注工具入口；旧 `trajectory_gui_enhanced.py` 入口移除。
  - 共享模块迁入 `traj_core/`：数据读取、标定、视频帧索引、可视化投影、轨迹常量、速度/几何工具、cluster 工具、伪 GT 动力学和轨迹身份/删除 key。
  - GUI 标注模块迁入 `traj_annotation/`：CLI、viewer、mixins、Tk 环境、保存审计、scene labels 和 GUI 专用投影兼容工具。
  - Alpamayo 推理模块迁入 `traj_inference/`：配置、模型加载和推理 runner；根目录 `run_inference.py` 保留为薄入口。
  - `traj_inference.runner` 调整为 import/`--help` 不强制导入 torch/model loader，避免 GUI-only 环境因为推理依赖缺失而无法检查结构。
- 当前行为：
  - GUI 标注人员只需要使用 `trajectory_annotator.py` 和 `traj_annotation/`。
  - 模型推理仍通过 `run_inference.py` 启动，内部转调 `traj_inference.runner`。
  - 原有 GUI 浏览、轨迹扩充、删除、编辑、保存审计逻辑保持一致。
- 验证方式：
  - 新增 `tests/test_project_structure.py` 覆盖新入口和三层包 import。
  - 更新 `tests/test_gui_helpers.py` 到新包路径。
