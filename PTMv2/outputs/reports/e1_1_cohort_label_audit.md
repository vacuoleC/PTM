# E1.1 病人独立性、标签完整性与类别结构审计

## 为什么做

在任何模型训练前，验证冻结的 LSCC G2/G3 标签与特征矩阵是否按病人一一对应，避免样本重复、缺失标签或正常样本混入造成虚假的折外性能。

## 结果

- 特征矩阵含 212 个样本；其中 106 个带冻结分级标签。
- 标签患者缺失于矩阵：0；标签 sample_id 与 patient_id 不一致：0。
- 带标签的正常样本：0；缺失 target：0。
- 阳性类（G3）比例为 0.4528，这是随机分类器的 AUPRC 基线。

## 类别计数

|   target | class_name                | raw_label                    |   patients |
|---------:|:--------------------------|:-----------------------------|-----------:|
|        0 | moderately_differentiated | G2 Moderately differentiated |         58 |
|        1 | poorly_differentiated     | G3 Poorly differentiated     |         48 |

## 证据图

![冻结标签类别结构](D:/coding/PTM/PTMv2/outputs/figures/e1_1_cohort_label_distribution.svg)

该图只描述冻结标签的组成，不用于证明模型可泛化；后续 E2 将以患者级折分明确防止训练/测试重叠。
