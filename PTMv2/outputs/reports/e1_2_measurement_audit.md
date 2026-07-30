# E1.2 测量缺失与候选特征审计

## 为什么做

后续检测率过滤必须仅在训练折拟合；本审计只描述冻结标签子集的测量结构与预注册阈值的规模，不使用标签值选择特征。

## 模态摘要

| modification    |   features |   labelled_patients |   mean_detection_rate |   median_detection_rate |   feature_detection_q25 |   feature_detection_q75 |   all_missing_features |
|:----------------|-----------:|--------------------:|----------------------:|------------------------:|------------------------:|------------------------:|-----------------------:|
| phosphorylation |      72361 |                 106 |                0.3564 |                  0.2264 |                  0.0849 |                  0.6321 |                   7556 |
| acetylation     |      19331 |                 106 |                0.4107 |                  0.3208 |                  0.0943 |                  0.7264 |                   1759 |

## 固定阈值下的候选特征数

| modification    |   detection_threshold |   retained_features |   retained_percent |
|:----------------|----------------------:|--------------------:|-------------------:|
| phosphorylation |                  0.10 |               46700 |              64.54 |
| phosphorylation |                  0.30 |               31521 |              43.56 |
| phosphorylation |                  0.50 |               22910 |              31.66 |
| acetylation     |                  0.10 |               13834 |              71.56 |
| acetylation     |                  0.30 |                9996 |              51.71 |
| acetylation     |                  0.50 |                7447 |              38.52 |

## 证据图

![检测率阈值下的特征保留比例](D:/coding/PTM/PTMv2/outputs/figures/e1_2_detection_thresholds.svg)

该描述性审计不替代训练折内的过滤；E2 将把同一阈值候选封装在 Pipeline 中，避免测试数据影响特征可用性。
