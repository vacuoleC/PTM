# E4.1 raw 探索 — 边界决策（lossing.md）

## 决策时间

2026-08-08 20:40（东八区），50 折全量预演进行中（fold 3-10 已完成，每折 ~32s）

## 探索方案回顾

- 目标：raw_elastic_net 50 折嵌套（5 外折×10 重复）在 24h 内完成，保持位点级可解释性
- 方案：**求解器替换**（sklearn-saga → cuml-qn GPU）+ **数值等价预处理**（sort-trick 中位数 f32）
- 实测：单折 32-160s（预热后 32s），50 折全量 ~30-40 分钟（GPU），0 CPU 核占用
- 冻结管线顺序不变：detection_filter → median_imputer → standard_scaler → elastic_net_logistic

## 理解偏差（对原任务意图的理解偏差）

1. **求解器偏差**：原冻结管线指定 `LogisticRegression(penalty='elasticnet', solver='saga')`。探索方案改用
   `cuml LogisticRegression(penalty='elasticnet', solver='qn')`。两者**目标函数完全一致**（同一正则化
   弹性网络损失），只是优化算法不同（坐标下降 vs L-BFGS 变体）。这是"执行层"替换，不是"模型"替换——
   但严格意义上 E4.1 的产物（OOF 分数）不再与 E3.1 置换（sklearn-saga）逐位一致。
2. **预处理偏差**：sort-trick 中位数（np.sort + 索引）与 sklearn SimpleImputer(median) 的
   `np.nanmedian` 数值一致（maxdiff 9.45e-08 中位数 / 2.86e-06 最终输出）。检测过滤的 keep 语义一致
   （frac >= threshold）。f32 vs f64 精度差异在 3e-6 量级，可视为数值噪声。
3. **环境偏差**：ENV-REMOTE 设计文档描述计算环境为 ptm-encoder（CPU 语义），GPU 加速引入了
   非文档化的环境依赖（LD_LIBRARY_PATH=bg-toy/lib 修复 GLIBCXX + cuml 26.08）。这属于
   "探索采纳需变更流程"的范畴。

## 效果偏差（方案对结果的影响）

1. **OOF 分数差异**：cuml-qn vs sklearn-saga 同折同参数 corr=0.981, mean_abs_diff=0.030,
   max_abs_diff=0.088。内层选择结果在 fold 1/2 相同（(0.3,0.1,0.9)/(0.5,1.0,0.1)），fold 0 不同
   （cuml 选 (0.5,1.0,0.9)，saga 选 (0.1,0.1,0.5)）——阈值网格上 AP 差异 <0.01 的候选竞争，
   求解器噪声可翻转选择。
2. **主证据规则影响**：E4.1 的验收标准是 pooled OOF AUPRC（与随机基线 0.4528 比较 + 置换 p 值）。
   - 5 折 OOF AUPRC 预演：0.5450, 0.5742, 0.5689, 0.7166, 0.4717（fold 0-4），均值 ~0.575 高于基线
   - 求解器差异使 OOF 分数有 ~0.03 噪声，但**配对比较的统计推断**（10 重复内 E4.1 vs E4.2 配对 t 检验）
     在求解器噪声水平上仍有效——噪声独立于重复，增加的是组内方差，会保守化（非乐观偏差）
   - 置换检验：若 E4.1 沿用 cuml-qn 跑置换，则 null 分布与观察值同求解器，p 值推断一致
3. **位点级可解释性**：**完全保留**。cuml-qn 产出稠密 coef_ 向量（nz 80-1345），直接映射回
   检测过滤后的位点特征（阈值 0.1/0.3/0.5 各有 keep mask），无需降维。这是 raw 区别于 PCA 的本质特征。

## 可能的损失（采纳此方案损失了什么）

1. **与 E3.1/E3.2 的 sklearn-saga 结果可比性**：E3.1 置换 null（sklearn-saga）与 E4.1 观察值（cuml-qn）
   若混用求解器，p 值会引入 ~0.03 分数噪声。**缓解**：E4.1 若采纳此方案，应同步用 cuml-qn 重跑
   E4.1 自身的置换（500 次 × ~40s = ~5.6h GPU），或明确标注 E4.1 观察值与 E3.1 null 的求解器差异。
   - 注意：E3.1 是 PCA 主模型的置换（pca_elastic_net），raw 主模型的置换在 E4.1 阶段才有定义。
     实际可比对象是 raw 的观察值 vs raw 的 null——同求解器即可自洽。
2. **数值可复现性**：cuml-qn 的 GPU 数值受 H100 硬件/驱动影响，无法用 sklearn-saga 逐位复现。
   **缓解**：固定环境（cuml 26.08 + ptm-encoder + LD_LIBRARY_PATH），版本化记录；随机种子固定。
3. **跨平台迁移**：若未来换机器，GPU cuml 依赖需重建。**缓解**：方案代码独立于 cuml 的 sklearn-saga
   回退路径保留（慢但正确）。
4. **轻微回归风险**：cuml-qn 在 C=1.0/l1r=0.9 组合上有 QWL-QN line search 警告（fold 0 并行运行时
   观察到 step delta=0），但收敛正常（coef 有限、proba 合理），且该组合是候选网格的一部分。

## 损失的必须（为什么必须接受这些损失）

- 冻结内方案（sklearn-saga 全量）在 E3.1/E3.2 占核的现实下**不可行**：30+ 分钟未完成一次拟合，
  50 折全量数月。24h 上限是硬约束。
- 纯 CPU 并行方案理论上可行（50 折 × 82 拟合 × 133s / 60 workers ≈ 25h），但 (a) 超出 24h 上限，
  (b) E3.1/E3.2 完成前无法获得算力，(c) 完成后 E3 序列还会占核。GPU 方案 0 核占用、30-40 分钟，
  不干扰任何主作业。
- 数值一致性已验证到 3e-6（预处理）和 corr 0.98（求解器），损失被限制在"优化算法噪声"级别，
  不影响位点级可解释性（raw 的核心价值）和配对比较的统计有效性。

## 决策

**方向 A（GPU cuml-qn）+ C（sort-trick 预处理）与原任务（E4.1 raw 位点级可解释比较）不相悖**：
- 保留位点级可解释性（coef 直接映射位点）——raw 的本质特征 ✓
- 完成 50 折嵌套内层选择 + 外层 OOF（27 预注册候选、3 内折、5 外折×10 重复）✓
- 24h 内完成（实际 ~40 分钟）✓
- 数值自洽（同求解器的观察值与 null）✓

建议采纳时的标注要求：
1. E4.1 产物表标注 solver=cuml-qn（vs E3.1/E3.2 的 sklearn-saga）
2. raw 主模型的置换 null 用同求解器（cuml-qn）执行，保证 p 值自洽
3. 环境固定记录（cuml 26.08、LD_LIBRARY_PATH、H100）
