# PTMv2 行动指南：小样本 PTM 的可建模性审计与低容量基线

## 一、v2 的问题与边界

v1 已经回答了两个问题：多 PTM 残差中存在稳定疾病信号；但当前 qPTM+CPTAC 联合多类 encoder 没有显示出优于公平 PCA 的稳定增益。v2 不再以“让深度模型获胜”为目标，而要回答：

> 在样本稀缺、特征极高维的定量 PTM 数据中，哪些表示和模型在折外具有可重复性；当前标签与数据量是否足以支持建模？

成功不等于非线性模型获胜。获得一个可靠的低容量基线、确认某标签当前不可稳定建模、或量化未来数据需求，都是有效结果。

## 二、文件与数据隔离

实际目录采用以下结构：

```text
D:\coding\PTM\
├─ PTMv1/                         # 本地只读归档，已被 Git 忽略
│  ├─ data/
│  ├─ outputs/
│  ├─ scr/
│  └─ ARCHIVE.md
└─ PTMv2/                         # 活跃 v2 工作区
   ├─ config/
   ├─ data/
   │  ├─ manifest.tsv             # 跟踪数据来源、相对路径、大小与 SHA256
   │  └─ processed/               # v2 新生成的中间数据；不覆盖 v1
   ├─ scr/
   ├─ outputs/
   │  ├─ figures/
   │  ├─ tables/
   │  └─ reports/
   ├─ tests/
   ├─ README.md
   └─ study_design.yaml
```

`PTMv1/data/` 是本地数据来源，但不是 v2 的 Git 依赖。不要假定 `data/raw/`、`CHECKSUM.sha256` 或 Linux 的 `ln -s` 已存在。第一份 v2 数据产物应是 `data/manifest.tsv`：记录本地源文件、哈希、样本数、特征数、生成脚本版本和用途。这样即使 v1 不上传，v2 的数据来源也可审计。

## 三、先冻结研究设计，再运行分析

先创建并提交 `study_design.yaml`；只有确定主标签后，才能创建 `v2_design_frozen` tag。不能一边比较结果，一边决定“LSCC G2 vs G3 还是另一个标签”。

建议的最小设计如下：

```yaml
study_name: "PTMv2 小样本可建模性审计"
version: "2.0"

data:
  source_manifest: "data/manifest.tsv"
  cohort: "LSCC"
  target_variable: "LSCC_G2_vs_G3"  # 冻结前可替换；冻结后不可改
  unit_of_split: "patient"
  require_patient_overlap_check: true
  feature_representation: "parent_protein_adjusted_multi_ptm_residual"

preprocessing:
  detection_rate_definition: "non_missing_fraction"  # ~isnan，不是 X > 0
  detection_thresholds: [0.1, 0.3, 0.5]
  imputation: "median_fit_on_training_fold"
  scaling: "standard_scaler_fit_on_training_fold"

evaluation:
  outer_splits: 5
  outer_repeats: 10
  inner_splits: 3
  primary_metric: "average_precision"
  permutation_count_primary: 500
  random_seed: 0

primary_models:
  - "raw_features + elastic_net_logistic"
  - "pca + elastic_net_logistic"

exploratory_models:
  - "truncated_svd + linear_svm_with_inner_fold_calibration"
  - "umap + elastic_net_logistic"

decision_rules:
  - "主模型必须在完整预处理与调参流程的置换 null 之外，才称为有预测证据。"
  - "模型比较报告配对重复 CV 差异及校正检验，不以单个最佳折作结论。"
  - "UMAP 与多视图模型均为探索性结果，不改变主结论。"
```

注意：`Elastic Net` 若使用 sklearn 的 `LogisticRegression`，需要 `penalty="elasticnet"`、`solver="saga"`、`C` 和 `l1_ratio`；不能写成不存在的 `alpha` 参数。`LinearSVC` 没有 `predict_proba`，若需要 AUPRC 以外的概率指标，校准必须在训练折内完成。

## 四、工作包 A：可建模性审计（先于模型比较）

目标是确认候选标签是否有足够的可重复预测证据，而不是寻找最高分。

1. **病人与标签审计**：确认每名病人只出现在一个训练/测试侧；报告类别数、类别比例、缺失率及其与标签的关系。
2. **特征测量审计**：检测率定义为非缺失比例。残差既有正值也有负值，因此不能以 `X > 0` 判断是否检测到。
3. **固定最小管线的置换 null**：先预定义一个低容量主管线，以全部训练折内的过滤、填补、缩放、模型拟合和调参组成完整流程；对标签置换 500 次，经验 p 值使用同一流程计算。
4. **学习曲线**：选择仍能维持内层分层验证的规模，例如 50%、70%、100%，并重复抽样。不能把 20%（约 21 个样本）塞进五折嵌套 CV。学习曲线平坦只表示“当前设计下未观察到增益”，不能直接断言标签噪声。

产物为 `outputs/reports/modelability_audit.md`、置换 null CSV 与学习曲线图。

## 五、工作包 B：严格的低容量主比较

所有处理必须被封装在训练折内的 sklearn Pipeline 中。推荐顺序：

```text
训练折检测率过滤 → 训练折中位数填补 → 训练折标准化
→ 可选 PCA/TruncatedSVD → 分类器
```

外层使用重复、分层、患者级分割；若一个病人有多个记录，使用 `StratifiedGroupKFold`。若每名病人只有一条记录，也必须显式断言病人 ID 没有重叠，而非口头假定。

内层仅在训练数据上选择有限的预注册候选：检测率阈值、PCA 维数、`C` 和 `l1_ratio`。候选网格必须小；小样本下宽网格本身就是过拟合来源。每次外层重复应保存一份完整折外预测，并在该重复内计算 pooled OOF AUPRC。

主报告包含：

- 每次重复的 OOF AUPRC 分布，而不是把五个相关折直接当作独立 95% CI；
- 主模型与次模型的配对 AUPRC 差及校正比较；
- 对主管线的经验置换 p 值；
- 选择出的阈值、维数和正则化参数的频率；
- 与随机基线比较。随机 AUPRC 是阳性类别比例；LSCC G2/G3 中约为 `48 / 106 = 0.453`，不是任意的 0.55 或 0.5。

## 六、UMAP、t-SNE 与多视图的边界

- **t-SNE 只用于探索性作图**，不能用作折外预测特征，也不能凭图上分群声明标签可泛化。
- **UMAP 是预注册的探索性候选**：每个训练折单独拟合后再转换测试折；不因为看到 PCA 的结果才决定是否运行。它不优于 PCA 时，如实报告为“当前数据不支持该流形假设”。
- **多视图不是普通对比学习**。磷酸化和乙酰化是不同特征空间；只有在同一患者真实具有两种测量时，才可测试低容量多视图方法，例如各模态独立降维后拼接或正则化 CCA。绝不把不同行业/不同条件的 qPTM 行拼成伪配对样本。

## 七、失败边界：只做可靠的诊断

小样本下，校准斜率、十等分校准曲线和少数“高置信错误”高度不稳定。它们可以作为描述性附录，不作为主结论。主诊断应更克制：

- 测试样本的缺失率是否明显高于训练折；
- 折外误差是否集中于某个测量批次、临床亚组或病人子集；
- 在训练折降维表示中，以收缩协方差估计的异常距离是否偏高；
- 各特征/维数/模型选择在重复间是否稳定。

不要将 PCA 后的异常距离直接解释为某个 PTM 位点造成，也不要将预测关联写成因果关系。

## 八、工作包 C/D 的开启条件

### C：低容量多视图比较

仅在以下条件同时满足时启动：真实患者级的磷酸化与乙酰化配对存在；单模态主管线通过可建模性审计；多视图方案在完全相同的外层分割下与单模态比较。

### D：数据路线图与深度 encoder 的重新开门条件

产物为 `outputs/reports/data_roadmap.md`，记录：当前失败边界、学习曲线支持的数据需求、所需真实配对条件，以及可能的数据来源。

只有存在计量尺度兼容的无标签预训练来源，或学习曲线显示新增样本有稳定收益时，才重新评估深度 encoder。训练损失下降、t-SNE 图分开或更强 GPU 都不是重新开门条件。

## 九、执行顺序

| 优先级 | 工作 | 完成标准 |
|---|---|---|
| P0 | 建立 v2 目录、数据 manifest 和 `study_design.yaml` | 路径、主标签和主比较全部冻结并提交 |
| P1 | 实现训练折内预处理 Pipeline 与患者级拆分测试 | 单元测试证明训练与测试侧不共享拟合状态 |
| P2 | 完成工作包 A | 生成可建模性审计、置换 null 与学习曲线 |
| P3 | 完成工作包 B | 生成重复 OOF 指标、配对比较和参数稳定性表 |
| P4 | 生成失败边界与数据路线图 | 明确继续、暂停或开启多视图的依据 |
| P5 | 条件满足时启动工作包 C | 只使用真实患者级配对多 PTM 数据 |

从 P0 开始。`study_design.yaml` 应由研究问题先行，而不是由模型结果倒推。
