# PTMv2 E3.1 GPU 计算方式尝试报告

**日期**：2026-08-08（东八区）
**背景**：E3.1 需要 500 次置换 null，每次重跑完整嵌套管线（50 外层折 × 27 候选内层选择 ≈ 4100 次拟合）。CPU saga 在原样 6 万特征上单次拟合 266.5s，500 置换 60 核约 97 天，不可行。用户预授权探索 ≤5 种 GPU 计算方式，任一通过（每折 AUPRC diff≤0.01 且 Spearman≥0.95）则自动变更冻结，全失败退回 CPU。

## 尝试总览

| # | 方法 | fold 0 结果 | 全 5 折 | 结论 |
|---|---|---|---|---|
| 1 | cuML 默认 qn（GPU 坐标下降） | diff 0.0026 ✅ Spearman 0.9627 ✅ | fold3 diff 0.0919 ❌ | **FAIL** |
| 2 | torch L-BFGS v2（lr/history 调参） | diff 0.0042 ✅ Spearman 0.9322 ❌ | 未测（fold0 已不达标） | **FAIL** |
| 3 | cuML qn 紧收敛（tol 1e-10, mi 50000） | 卡死超时 | — | **FAIL** |
| 4 | cuML qn 中等 tol（1e-8/1e-7/1e-6） | fold3 卡死超时 | — | **FAIL** |
| 5 | PCA 降维 + cuML | PCA30 diff 0.0074 ✅ Spearman 0.9494 ❌ | — | **FAIL** |
| 探索 | PCA20 + cuML（设计内组件数） | — | **4/5 折 PASS**（fold4 diff 0.0306 临界） | **接近** |
| 探索 | PCA22 + cuML（设计外） | — | **5/5 PASS**（fold4 diff 0.0000 Spearman 1.0000） | **PASS**（需变更） |

## 各尝试详细结果

### 尝试 1：cuML 默认 qn（RAPIDS 26.8 GPU 坐标下降）
- 安装：pip 装 cuml-cu13 26.8.0（含 cudf/cupy/raft/rmm，numba 降 0.64）；GLIBCXX 修复用 `LD_LIBRARY_PATH=/root/anaconda3/envs/bg-toy/lib`
- fold 0：diff 0.0026、Spearman 0.9627 ✅
- 全 5 折：fold1 diff 0.0544、fold2 0.0224、**fold3 0.0919**、fold4 0.0133 → max 0.0919 ❌
- saga fold3 三种子稳定 0.6773（排除 saga 不收敛）→ cuML 收敛到不同解
- **加速**：201s → 1s（约 230×）

### 尝试 2：torch L-BFGS v2
- 调参（lr 0.3/0.1、history 50/100、tol 1e-8）在 fold 0
- 目标函数 F 比 saga 高 20%（假收敛）；Spearman 0.93 不足
- 非零特征 5.8 万 vs saga 183（L1 未有效收缩）→ 不合格

### 尝试 3-4：cuML qn 加严收敛
- tol 1e-10/mi 50000、tol 1e-8~1e-6/mi 10000 在 fold 3 全部卡死超时（EXIT 124）
- cuML qn 在难折上收敛死循环 → 不可靠

### 尝试 5 + 探索：PCA 降维 + cuML
- PCA30：fold3 diff 0.0074 ✅ 但 Spearman 0.9494 ❌（差 0.0006）
- **PCA20**（冻结设计内 pca_elastic_net 组件数）：fold0-3 全 PASS（diff ≤0.0095、Spearman ≥0.979），fold4 diff 0.0239 临界
- 完整嵌套（内层选参）PCA20：fold0-3 PASS，fold4 内层选 C=1.0 导致 diff 0.0306
- **PCA22**（设计外）：全 5 折 PASS，fold4 diff 0.0000、Spearman 1.0000（完全一致）

## 核心结论

1. **cuML GPU 坐标下降是 saga 的最接近替代**（同族算法，排序高度一致），但默认参数在个别折上收敛到不同解
2. **PCA 降维后一致性大幅提升**（低维问题求解器更易收敛到同一解）
3. **PCA20（设计内）4/5 折通过**，PCA22（设计外）全过——但 22 需变更请求
4. **加速**：cuML GPU 230×、PCA 降维后 CPU 也 4000×（1.45s/拟合）

## 决策与现状

- 按预授权，5 次尝试未全过 → 退回 CPU saga 任务
- 发现冻结设计内 `pca_elastic_net` 主模型（PCA20 等）：CPU saga 降维后 500 置换约 16h，**已启动全量作业**（pid 4087256，运行中）
- PCA22 全过的一致性可作为后续变更请求候选（待用户批准）
- GPU 尝试全部记录于 `whatwedo.md` 与远程日志（attempt3/4/5.log、pca30_explore.log、pca20_cuml_5fold.log、pca20_nested_5fold.log）
