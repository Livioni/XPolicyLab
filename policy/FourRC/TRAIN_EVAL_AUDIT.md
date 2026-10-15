# 4RC Stage2 训练 / RoboTwin 评测核查（2026-09-21）

> 当前评测只接受 `FiLM + componentwise v2`、3072 维 concat、带腕部输入的 Stage2 权重。
> 本文是旧版 checkpoint 的历史实验记录；当前接口与运行命令见 [README.md](README.md)。

> 历史记录：本文指标对应当时的“默认初始姿态 + 重建 query 传播”实现。后续已按用户要求改为每个窗口使用首帧实测关节 FK 初始化 query，并按 waypoint 归档观测；当前行为见 [README](README.md)。 2026-09-25 已适配上游 `f053eed`：Stage2 使用离散步索引且训练包含首帧重复填充，本文关于物理时间条件和训练不含静态首窗口的描述仅适用于旧版。

本次完整阅读 `train_4rc_stage2.py`、`configs/train/4rc-stage2-action.py`，继续追踪 dataset、TCP 标签生成、query encoder、action policy / loss、官方滑窗推理和 RoboTwin 执行路径。实测 checkpoint 为 `4RC/checkpoints/RoboTwin-Stage2/180000`。

**结论：没有发现 left / right 标记反转，也没有发现相机→基座→执行末端的转换错误。保持当前映射。** 此次 `click_bell / seed=100000` 的失败直接表现为右臂预测和实际轨迹偏离铃铛：400 步重放中，指尖 TCP 与铃铛接触点的最小水平距离仍为 **12.69 cm**。这不能由毫米级控制跟踪误差解释。尚未通过消融确定模型产生错误落点的根本原因，不能据此宣称整体策略已修复。

## 1. 左右臂证据链

| 环节 | left / 第 0 臂 | right / 第 1 臂 |
| --- | --- | --- |
| `robot_state.npy` | 列 0:6；夹爪列 6 | 列 7:13；夹爪列 13 |
| 标签生成 FK | `fl_joint1..6` → `fl_link6` | `fr_joint1..6` → `fr_link6` |
| 训练标签 | `TCP_third/left_state.npy` | `TCP_third/right_state.npy` |
| Dataset / Action DiT | 第 0 臂；展平向量前 10 维 | 第 1 臂；展平向量后 10 维 |
| 适配器返回 | `left_ee_pose` / `left_ee_joint_state` | `right_ee_pose` / `right_ee_joint_state` |
| RoboTwin 执行向量 | `[0:7]` pose、`[7]` gripper | `[8:15]` pose、`[15]` gripper |
| 初始第三视角 query，未 padding | `(222.583, 100.716)` | `(96.590, 101.019)` |

第三相机面对机器人，**机器人左臂在图像右侧**。不能按图像从左到右排序 query 或动作。query encoder 中的 `offsets.flip(-1)` 仅交换局部坐标的 x/y 顺序，不交换手臂。图像增强仅模糊、颜色扰动、灰度；没有水平翻转。

- [训练标签生成](../../../convert_robotwin_tcp.py)：`tcp_pose_series`。
- [训练 dataset](../../../../4RC/arc/datasets/robotwin_action.py)：`_read_states`；[图像读取](../../../../4RC/arc/datasets/robotwin.py)：`__getitem__`。
- [Action 编解码](../../../../4RC/arc/models/arc/arc_action.py)：`normalize_actions` / `denormalize_actions`。
- [评测转换](geometry.py)：`convert_actions`；[最终动作拼接](../../../scripts/eval_policy_xpolicylab.py)：`xpolicylab_action_to_robotwin`。

用真实 clean 左臂样本和 random 右臂样本，共 8 个窗口，重新从关节做 FK 生成标签，与已存标签最大绝对差均为 0。正常预测的双臂平均位置 ADE 为 **2.16–13.43 mm**；强行交换输出臂后误差约 **0.39–0.53 m**。

![左右臂与实际轨迹](../../../eval_result/FourRC_audit/arm_mapping_and_rollout.png)

## 2. 训练全过程与当前推理的对应

1. **配置 / 恢复**：显式配置优先；resume 默认读取保存的 `config.json`。新训练严格加载 Stage1；恢复严格加载 Stage2 / optimizer / scheduler，并对齐累计 global step。推理严格加载 Stage2 完整权重和 buffer。
2. **数据**：clean / random 权重 0.33 / 0.66，按 episode 哈希划分训练与验证；每个窗口为连续真实 8 帧 + 下一帧开始的 16 帧。短未来重复末状态但通过 mask 排除监督。语言从同 episode 的指令中抽取。
3. **输入**：第三视角 RGB 320×240；reflect padding `(1,1,6,6)` → 322×252，归一化 `[-1,1]`；K、query 同步平移 `(1,6)`。首帧 query 为 `[left,right]` GT 像素投影。当前适配器的图像、K、query 在所测窗口中与训练 dataloader **逐元素完全相同**。
4. **恢复 / 条件**：Stage1 分支恢复历史 TCP 和几何；按计划在整条 GT 历史投影与 recovered 历史投影之间选取，采样局部 TCP 特征，再加入臂身份、物理时间和冻结 T5 的语言特征。GT TCP 投影只决定特征采样位置，不能把它误解为直接将 GT 未来动作喂入模型。
5. **动作监督 / 损失**：未来位置、旋转均变换到最后历史帧的 OpenCV 相机系；每臂 `[xyz, rotation6d(前两列), 2*grip-1]` 共 10 维。位置按 checkpoint 的每臂 mean/std 归一化；flow matching 预测速度场。总目标包含 geometry、TCP、action 三项，各自权重按配置。几何 GT 深度和外参用于监督，不是部署必须输入的历史深度。
6. **优化**：backbone、geometry、motion/query/TCP、history、action、language projection 分组学习率；T5 冻结；BF16、梯度累积 / 裁剪、AdamW、warmup + cosine。新训练仅设置 action normalization，不覆盖 Stage1 TCP normalization。
7. **验证 / 保存**：分别评测 recovered、teacher-forced、shuffled-instruction，独立随机生成器；保存完整训练状态及配置。不能把 teacher-forced 指标当作当前闭环部署结果。
8. **部署**：同一 reconstruct → make_condition → sample_condition 路径，8 步采样；位置反归一化、rotation6d 转 SO(3)、gripper score `>=0` 为开。同窗口 / 同 seed 下，适配器调用的官方 API 与训练验证 recovered 路径的动作位置差 **为 0**。部署仍预测 16、执行前 8，然后收集 8 个新观测。

## 3. 坐标与控制实测

训练标签是指尖内接触面中点，局部偏移为 `[0.147335, -0.0000015, -0.000713515] m`，不是 RoboTwin 自带 TCP getter 的 0.12 m 偏移。

```text
T_base_tcp = inverse(T_world_base) @ inverse(T_camera_world) @ T_camera_tcp
p_base_ee = p_base_tcp - R_base_tcp @ fingertip_offset
T_world_ee = T_world_base @ T_base_ee
```

位置和旋转一起变换。最后转回 world ee 是 RoboTwin `take_action(ee)` 的接口要求；其内部负责世界→规划器基座和 IK。四元数顺序为 **wxyz**。

对现有日志 `predictions_1789977166941832752_3289969.jsonl` 的 seed 100000 重放，执行前 8 帧 / 窗口、共 400 个目标：

| 指标 | left | right |
| --- | ---: | ---: |
| 路径规划成功 | 400/400 | 400/400 |
| 执行后 EE 平均位置误差 | 0.497 mm | 1.247 mm |
| 最大位置误差 | 1.298 mm | 2.848 mm |

重放任务仍失败。实际动的是语言要求的右臂；左臂保持初始位置附近。铃铛接触点 world xyz 为 `[0.072695, -0.110309, 0.766790] m`。右臂 TCP 最接近铃铛的水平距离为 0.126899 m，出现在第 376 步。这里只把 TCP 距离作为偏离目标的诊断指标；任务成功判据仍是原始接触检测。

## 4. 确认存在的训练 / 闭环差异

| 差异 | 判断 |
| --- | --- |
| 首窗口 8 张真实图 vs 重复首帧 8 次 | 已有 bootstrap 约定；训练不包含这样的静态首窗口。可能导致启动迟滞，未完成独立因果消融。 |
| 训练每窗初始 query 来自 GT；部署跨窗传播恢复 query | 存在误差累积风险。当前 6 个抽查窗口恢复位置误差约 0.27–5.90 mm，初始 query 误差不超过约 1.55 px，没有发现严重漂移。 |
| 训练 GT / recovered 历史混合；部署全 recovered | 按所存 500000 步配置和当前 schedule 计算，180000 步时 GT 选择概率约 82%，不是 50%。但本次 GT 条件离线替换只小幅改变预测，错误落点仍存在，不能将其认定为此次主因。 |
| 模型标称 15 Hz；执行每个 waypoint 的完整轨迹 | 400 个目标的模型时间为 26.667 s，规划路径长度对应约 120.908 s 仿真时间（非推理耗时）。实际并非 15 Hz 控制。 |
| 数据采集 `save_freq=15` 被记录为 `frequency=15` | `save_freq` 在采集代码中是物理步间隔；仿真步长 1/250 s，通常每 0.06 s 采一帧，阶段边界还额外采帧。metadata 中 15 Hz 并非严格物理采样率。训练据 metadata 编码时间，当前推理也一致使用 15，不能只修改评测频率就声称修复。 |
| 当前配置文件与 checkpoint 配置不同 | 当前 `lr_backbone=1e-6`；180000 保存的是 `1e-5`。visualize/checkpoint 周期也不同。模型架构、历史/未来长度、采样步数一致；部署应以 checkpoint 配置为准。 |

闭环 query 分析使用保存的有损第三视角视频和同 seed 重放的实际机械臂位置，属于近似诊断；不是重新统计成功率。离线 8 窗口来自两个按当前哈希规则属于训练分区的 episode，**不能作为泛化性能结论**。

下一步有价值的实验是：同一固定种子集做控制时序 / bootstrap 的独立消融，以及在未参与训练的专家历史上测未来 TCP 误差与落点。当前证据不支持先交换手臂、改四元数顺序、强行平移轨迹或把 GT 历史加入正式评测。

## 5. 本次改动与复现材料

- 保留模型、坐标转换、执行 8 帧、第三视角输入和视频输出行为。
- 增加 `test_distinct_arms_reach_correct_robotwin_action_slots`：左右不同位置、旋转、夹爪和时间轨迹，经相机→基座→world ee→真实动作拼接后逐臂验证，避免对称测试漏掉交换错误。
- 适配器测试结果：`10 passed`；格式检查通过。
- 所有实测 JSON、日志、诊断脚本和图保存在 [eval_result/FourRC_audit](../../../eval_result/FourRC_audit/)。主要结果：`offline_training_comparison.json`、`replay_execution.json`、`closedloop_recovery_comparison.json`、`scene.json`。

从 RoboTwin 根目录复现本次固定案例（脚本记录了本机路径和 checkpoint；更换案例需相应修改）：

```bash
conda run -n 4rc python -m pytest XPolicyLab/policy/FourRC/tests -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n 4rc python eval_result/FourRC_audit/offline_audit.py
conda activate RoboTwin
python eval_result/FourRC_audit/replay_audit.py eval_result/FourRC_predictions/predictions_1789977166941832752_3289969.jsonl
python eval_result/FourRC_audit/scene_audit.py
conda activate 4rc
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python eval_result/FourRC_audit/closedloop_audit.py
python eval_result/FourRC_audit/audit_plot.py
```
