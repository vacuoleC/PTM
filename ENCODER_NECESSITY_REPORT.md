# PTM Encoder 必要性评价报告

## 结论

**当前不值得继续投入“qPTM+CPTAC 联合、多类条件级冻结 encoder”的实现与大规模训练。**

这是策略路由，不是项目 no-go。Phase 0 已证明 CPTAC 多类 PTM 残差中存在稳定疾病信号；但目前可用数据与难任务均未给出 encoder 相对公平传统表示的增益证据。下一步应优先寻找 PTM-only 生物学任务或获得真正配对的多类 PTM 条件数据，而不是扩大当前 autoencoder。

## 已完成的证据

| 维度 | 结果 | 含义 |
|---|---:|---|
| Phase 0 肿瘤/正常 | LSCC/LUAD/UCEC XGBoost AUPRC 0.993/1.000/0.998 | PTM 信号存在，唯一 no-go 未触发。 |
| 难任务 | LSCC G2 vs G3，106 个肿瘤 | 肿瘤/正常已饱和，必须在困难任务判断表示价值。 |
| 多类增量 | 重复 CV 差 -0.0060；单侧 p=0.553 | 乙酰化在该任务无稳定增量。 |
| CPTAC-only encoder | encoder AUPRC 0.4891；PCA 0.4879；校正单侧 p=0.4909 | 64 维冻结 encoder 未优于公平 PCA。 |
| qPTM 条件丰富度 | 磷酸化 1,167 条件；乙酰化 82 条件 | 单修饰自监督来源充足，但不平衡。 |
| 共同位点 | CPTAC 磷酸化/乙酰化覆盖 79.0%/73.7% | 命名/位点空间可迁移。 |
| 配对多类条件 | 仅 2 个 | 无法诚实构造大规模配对多类 qPTM 条件矩阵。 |

## 设计与公平性

- 所有 encoder 训练仅使用无标签 CPTAC 残差；G2/G3 标签只进入每折下游逻辑回归头。
- PCA 同样在全部无标签样本上 transductive 拟合，避免 encoder 看见测试样本而 PCA 不能的偏差。
- 使用 10 次重复、5 折、按病人分组的 CV，并以 Nadeau--Bengio corrected resampled t 描述配对差异。
- qPTM 下载、校验、安全解压、条件审计和 CPTAC 位点重叠均可复现；原始数据不进 Git。

## 决策与下一步

1. 停止当前联合多类 qPTM+CPTAC autoencoder 方案；它会把不同实验的单修饰条件伪配对。
2. 保留并复用现有 CPTAC encoder 管线，作为以后真正配对数据或 PTM-only 任务的基线。
3. 优先进行 PTM-only 任务定义（如酶活性/药物反应的独立锚点）；若没有此类标签，不应把“癌症分级分类”包装成 encoder 成功指标。
4. 若将来获得配对的磷酸化+乙酰化条件矩阵，再重新运行冻结 encoder vs PCA 比较；必要时仅把 LoRA 作为诊断，不作为主要结论。

## 产物索引

- `outputs/lscc_grade_g2_vs_g3_increment_summary.csv`
- `outputs/lscc_grade_g2_vs_g3_encoder_necessity_summary.csv`
- `outputs/qptm_archive_audit.csv`
- `outputs/qptm_condition_audit.csv`
- `outputs/qptm_cptac_feature_overlap_audit.csv`

## 限制

本报告只覆盖肺与子宫 CPTAC 队列；不作泛癌声明。qPTM 的条件表不是患者表，且本轮没有外部临床预测验证或因果解释分析。
