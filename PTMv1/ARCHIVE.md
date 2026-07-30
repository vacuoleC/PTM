# PTMv1 归档说明

归档日期：2026-07-30

`PTMv1/` 保存第一阶段 PTM encoder 必要性研究的完整工作区：代码、配置、文档、正式输出、原始/缓存数据以及本地审计日志。该目录在归档后不再作为活跃开发根目录；后续新一轮工作应在仓库根目录下独立创建 `PTMv2/`。

## 阶段性结论

- 经母蛋白丰度回归校正的多 PTM 残差在 LSCC、LUAD、UCEC 的肿瘤/正常任务中存在稳定可学习信号；项目唯一的 Phase 0 no-go 未触发。
- LSCC G2 vs G3 困难任务中，磷酸化加乙酰化相对磷酸化的重复交叉验证平均 AUPRC 差为 -0.0060，未观察到稳定增量。
- CPTAC-only 冻结 autoencoder 相对公平 PCA 的平均 AUPRC 差为 +0.0012，校正单侧 p=0.4909；当前 encoder 未显示可检测优势。
- qPTM 与 CPTAC 的位点空间有较高重叠，但磷酸化与乙酰化真实共享的实验条件上下文仅 2 个，不能诚实构造联合配对多类 PTM 预训练矩阵。

因此，v1 的资源路由结论是：不继续扩大“qPTM+CPTAC 联合、配对多类条件 encoder”路线。该结论不是泛癌 PTM 无效、也不是所有 future encoder 方案的 no-go；后续应优先寻找 PTM-only 生物学锚点或真实配对的多类 PTM 数据。

## 重要入口

- `ENCODER_NECESSITY_REPORT.md`：正式结论、统计解释、数据集词典与限制。
- `.doc/PTM_STAGE_REPORT.md`：约 8 分钟的中文阶段汇报及六张可复现图。
- `README.md`：按项目树组织的函数级说明。
- `whatwedo.md`：第一阶段的完整工作记录。

## Git 与复现提示

本目录沿用原有相对路径结构，因此在 `PTMv1/` 内运行历史脚本时，需要以该目录作为工作目录。根目录的 `.git` 保留完整提交历史；本次归档通过 Git 重命名记录 v1 文件的新位置。原有的 `.log/`、qPTM 原始数据和 encoder checkpoint 仍按照 `PTMv1/.gitignore` 保持本地存储规则。
