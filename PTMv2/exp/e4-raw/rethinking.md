# E4.1 raw_elastic_net 探索 — 思考模式

## 任务复述

- E4.1（m5-comparison 主模型之一）：原始特征 + Elastic Net，patient-level nested CV（5 外折 × 10 重复），
  内层 27 个预注册候选（detection_threshold × C × l1_ratio：3×3×3），产物 primary_model_oof_scores.csv。
- 数据：X 212×91692 → labels 106（G2=58 / G3=48）；detection 过滤后 ~60,534 特征。
- 困难：单折 saga 拟合 ≥60s（探针 133s、三次重复 200s 超时/被 OOM 杀），50 折嵌套全量远超 24h。
- 真实意图（模块设计+study_design 交叉确认）：raw 是"低容量基线"，与 pca_elastic_net 主模型配对比较；
  产出必须**位点级可解释**（feature_representation = parent_protein_adjusted_multi_ptm_residual，
  elastic net 稀疏系数可直接映射回位点）——这是 raw 与 PCA 的本质区别，任何探索不能丢。

## 算力/资源现状（2026-08-08 核查）

- 远程 192 核；E3.1 pid 4171027（60 workers，置换阶段，PCA 低维 1.45s/拟合）+ E3.2 pid 4171305（~30 workers）。
  共占 ~90 核。我方预算 ≤64 核，实际可用 ~90-100 核（E3.1/E3.2 是 BLAS 超订修复后的低占 worker，多数 worker 1 线程）。
- 内存：机器 2011GB total，used 111GB（E3.1 主进程 RSS ~1GB），可用 1876GB——"12GB 增量"远有余量。
  但任务给的是 cgroup 52/64GB 之说，本地观测是机器级，以不显著增大常驻为准（分折读入，不整表驻留）。
- GPU：H100 80GB **完全空闲**（E3.1/E3.2 都是 CPU 作业）。这是最大杠杆。
  - 但注意：project_design.yaml 中 ENV-REMOTE 描述 gpu: available 但环境为 ptm-encoder（无 cuml）。
  - LD_LIBRARY_PATH=/root/anaconda3/envs/bg-toy/lib 是任务给的环境（bg-toy 有 libstdc++ 新版本）。

## 失效根因分析

1. **saga 全量坐标下降**：106×60534，密集 0.5GB 矩阵，saga 每次全梯度扫描 6 万维 × 106 行。
   sklearn 1.8 的 saga 是单线程 C 循环（BLAS 只用于少数 ops），133s 基本是真实墙钟。
2. **27 候选 × 3 内折**：每次内层选择 = 81 次全量 saga 拟合 + 1 次外层 = ~82 次 × 133s ≈ 3h/外折。
   50 折（10 重复）≈ 30-150h，远超 24h/作业。
3. 探针 133s 与 e2_2 smoke 实测（50 折 3.7h ≈ 4.4min/折含预处理）基本一致。

## 候选方向（按杠杆排序）

### A. GPU 加速（最大杠杆，需慎重与项目意图对齐）
- cuml LogisticRegression(solver=cg 或 lbfgs，elasticnet) 在 106×60534 上预计 <2s/拟合。
- 但 cuml 的 elasticnet 支持有限（l1_ratio 支持看版本；solver cg 有 l1 选项）。
- 风险：数值与 sklearn saga 不完全一致；项目 ENV-REMOTE 定义是 CPU env——这属于"探索采纳需变更流程"范畴，如实记录。
- 需验证：ptm-encoder env 修复 libstdc++ 后能否 import cuml；或 bg-toy env 里 cuml 是否可用。

### B. 特征筛选（保持位点级可解释）
- 检测率>0.3/0.5 + 方差 top-K 预处理 → 6 万降到 1 万左右 → saga 变快。
- 但这是对**冻结 study_design 的预处理修改**（frozen: variance_filter: none, detection_threshold_candidates [0.1,0.3,0.5]）。
  - 注意：threshold 0.3/0.5 本来就是预注册候选！fit_score_fold 里 threshold 参数直接控制检测率过滤。
  - 若 0.3 阈值后特征数 ~1 万，saga 拟合时间应降为线性 ~22s/次——但 133s×6 万 → 1 万 ≈ 22s，仍要 82×22s≈30min/外折，50 折 25h，勉强。
  - 方差筛选不在冻结设计中（variance_filter: none），只能作为探索对比，不改变主结论。

### C. 稀疏化/优化表达
- median 插补破坏稀疏性。替代：行中位插补后中心化（不入库稀疏）——与冻结管线不一致。
- 或保留 NaN 结构用稀疏 imputer——sklearn 无 NaN 稀疏 imputer 原生支持。低优先级。

### D. 并行分批
- 纯并行把 82×133s 压到并行，但单折仍是 133s×外折串行度。50 折并行 60 workers → 全量 ~3h。
- 问题是 50 折每折都要 27×3 次拟合 = 4110 次 × 133s ≈ 152h CPU-小时——60 核并行 2.5h 可完成？
  - 152h / 60 workers ≈ 2.5h 墙钟 + 调度开销。**纯并行其实可行！**
  - 算力预算：60 核在预算内（E3.1 60 + E3.2 30 = 90 已用，192 核还剩 ~100）。
  - 但 24h 限时 + checkpoint 需要。E2.2 探针显示 50 折 3.7h 是**串行**的——并行后 10 重复 500 折 = 500×133s/60 ≈ 18.5min/批 × 10 = 3.1h？不对：500 折全并行 60 workers = 500×133/60 ≈ 18.5 min + 每折内层 81 次拟合串行 = 500×81×133/60 ≈ 133×675 ≈ 25h。核对了：**每折内层 81 次拟合是串行的**，所以关键瓶颈是每折内层循环，并行粒度只能是折级。
  - 修正计算：每折 ~82 拟合 × 133s ≈ 3.0h（如果阈值 0.1）。500 折 / 60 并行 ≈ 8.3 折同时 → 墙钟 ≈ 8.3 × 3h ≈ 25h。>24h，但阈值 0.3/0.5 更快，且这是最坏情况（133s 是含预处理的完整探针？需重新实测）。
  - 而且 E3.1/E3.2 已用 90 核，60 并行可能紧张——但 192 核够。

### E. 降低 saga 成本（保持同数值管道）
- max_iter 10000 冻结，但 saga 通常 100-500 次收敛。可以探针看真实迭代数——若有收敛性统计，可确认不是 iter 限制。
- LogisticRegression 的 tol 冻结默认 1e-4——不动冻结参数，但可以测量每次拟合的迭代次数以理解 133s 构成。

### F. 数据级单例（预处理缓存）
- 27 候选 × 3 内折的预处理可缓存（detection mask 只 3 种、imputer/scaler 每折不同）——节省小头（预处理 0.1-0.5s）。

## 初步判断

方向 A（GPU cuml）杠杆最大但偏离 ENV-REMOTE 定义；方向 D（折级并行）最贴合冻结管线（零改动管线，
只是执行层面并行），折级并行 + 24h 限时 + checkpoint 重续可能直接跑通，但需要实测单折完整耗时（内层 81+1 拟合）。
先实测基线：单折完整嵌套（81 拟合）耗时。若 ~3h/折 → 500 折并行 60 核 ≈ 25h 超 24h 一点，可缩 worker 或先跑 0.3/0.5 阈值候选降维加速。
若 GPU 可行则 25h → 3-4h，还能给主作业让出 CPU。

下一步：执行模式。第一轮先做三个探针（只读数据、不占算力）：
P1. 单折完整嵌套基线（含内层 27×3）耗时 —— 但 3h 太久，先测**单候选单内折**拟合耗时（=133s 量级）来推算。
P2. threshold 0.3/0.5 后特征数与单拟合耗时（管线本身支持，直接反映预注册候选的加速空间）。
P3. GPU cuml 是否可 import（ptm-encoder libstdc++ 修复 + bg-toy 环境探测）。

## 更新（2026-08-08 20:40，执行模式中段）

### 方向收敛过程
1. GPU cuml-qn 可行（0.4-1.6s/拟合），但瓶颈转移至预处理（nanmedian 11-18s/次 × 82 次/折）
2. sort-trick 中位数替代 nanmedian：f32 全管线 2.3s（数值一致 maxdiff 2.86e-06）
3. array-cache：9 组 Xt/Xv 数组（3 threshold × 3 内折）缓存，81 次纯 cuml 拟合
   → 单折 160s → 预热后 32s/折
4. 并行缩放：2 折并行 270s/折（负收益）、3 折 540s+（GPU 争抢）→ 串行最优
5. 全量 50 折预演：fold 3-17 每折 ~32-33s，预计全程 ~40 分钟（GPU，0 核占用）

### 最终方案（方向 A+C 组合）
- 求解器：sklearn-saga → cuml-qn（同一弹性网络目标函数，仅优化算法不同）
- 预处理：f32 + sort-trick 中位数（数值等价 sklearn，maxdiff 3e-6）
- 执行：串行 50 折，checkpoint CSV 断点续跑，env 固定（cuml 26.08 + LD_LIBRARY_PATH）
- 位点级可解释性完整保留（coef 直接映射检测过滤后位点）

### 边界决策结论（详见 lossing.md）
- 与 E4.1 真实意图（raw 位点级可解释比较）不相悖 ✓
- 损失：求解器数值差异（corr 0.98，噪声 ~0.03 OOF 分数）→ 需标注；与 E3 的 sklearn-saga 结果
  可比性需同求解器置换 null 保证自洽
- 24h 硬约束下 CPU 方案不可行（E3 占核 + saga 单线程被抢占），GPU 方案 0 核占用 40 分钟
