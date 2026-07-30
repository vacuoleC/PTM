# PTM Encoder

定量 PTM 残差表征与 encoder 必要性评价项目。唯一配置入口是 `config.yml`；原始 qPTM 数据、运行日志和大模型 checkpoint 不进入 Git。

## 项目树与函数

```text
PTM/
├── config.yml                 参数、路径和实验设计
├── environment.yml            ptm-encoder 环境
├── ENCODER_NECESSITY_REPORT.md 最终决策报告
├── outputs/                   可版本化数值结果
└── scr/                       可复现实验脚本
```

### `scr/project_config.py`

- `load_config`：读取并验证唯一 YAML 配置。
- `configured_path` / `configured_template_path`：将配置路径解析为项目绝对路径。
- `get_cohort_class`：从配置取得 CPTAC 队列类。

### `scr/coverage_matrix.py`

- `split_sample_ids`：按正常后缀拆分肿瘤与正常样本。
- `cohort_coverage`：计算队列三模态交集覆盖。
- `main`：写出覆盖矩阵。

### `scr/build_matrix.py`

- `align_three_modalities`：取三模态样本交集。
- `simplify_feature_columns`：规范特征列层级。
- `report_missing_ptm_annotation` / `drop_unannotated_ptm`：诊断并移除缺少基因/位点注释的 PTM 列。
- `inspect_duplicates` / `collapse_duplicate_features`：检查并用中位数合并重复特征。
- `report_parent_protein_coverage` / `keep_parent_matched_sites`：审计并保留可匹配母蛋白的位点。
- `residual_for_one_site` / `stoich_resid`：逐位点回归 PTM 对母蛋白并取残差。
- `report_detection_rate`：汇报残差检测率。
- `add_modification_level`：添加修饰类型列层级。
- `make_tumor_labels`：从样本 ID 构造肿瘤/正常标签。
- `save_phase0_artifacts`：保存矩阵与标签；`main`：构建单队列 Phase 0 产物。

### `scr/transformers.py`

- `DetectionFilter.fit/transform`：按训练折检测率筛特征。
- `MedianImputer.fit/transform`：按训练折中位数填补缺失。

### `scr/run_floor.py`

- `load_phase0_data`：读残差、标签并构造病人分组。
- `make_linear_pipeline` / `make_xgboost_pipeline`：构造公平预处理+分类管线。
- `run_linear_floor` / `run_xgboost_floor`：运行肿瘤/正常 floor；`main`：执行二者。

### `scr/run_phase0_null.py`

- `fixed_group_splits`、`prepare_folds`、`score_one_fold`、`score_xgboost`：准备并评分固定病人折。
- `permute_labels`：生成标签置换；`benjamini_hochberg`：多重检验校正。
- `run_null_for_cohort`、`run_learning_curve`、`run_ucec_repeated_cv`：生成置换 null、学习曲线和 UCEC 重复 CV；`main`：汇总运行。

### `scr/audit_hard_task.py`

- `load_tumor_sample_ids`、`patient_ids_from_samples`、`load_aligned_clinical`：对齐肿瘤、病人与临床表。
- `single_clinical_column`、`summarize_candidate_labels`：规范并审计候选标签。
- `build_primary_labels`、`save_audit_and_labels`：生成 LSCC G2/G3 主任务标签与审计；`main`：执行。

### `scr/run_hard_task_ablation.py`

- `load_hard_task_data`：读困难任务矩阵、标签和病人组。
- `select_feature_set`：按修饰集合选列；`fixed_splits`：生成共享折。
- `score_feature_set`、`summarize_scores`、`run_ablation`：比较磷酸化和多类 PTM；`main`：执行。

### `scr/run_hard_task_increment.py`

- `report_progress`：长作业进度输出；`make_splits`、`score_feature_set`：重复 CV 基础。
- `repeated_scores`：计算配对重复 CV；`nadeau_bengio_summary`：校正统计。
- `acetylation_block`、`block_permutation_null`：构造乙酰化块置换 null；`main`：保存完整增量证据。

### `scr/run_encoder_necessity.py`

- `PretrainingData`：预训练数组与元数据容器；`MaskedPTMAutoencoder`：遮蔽重建网络。
- `encoder_config`、`resolve_device`、`set_random_seed`：读取设计、固定设备与随机性。
- `load_pretraining_data`：对齐三 CPTAC 队列共同位点并无标签标准化。
- `build_model`、`train_encoder`、`save_pretraining_artifacts`：训练并保存冻结 encoder。
- `load_frozen_encoder`、`transform_with_encoder`：加载 checkpoint 并生成表征。
- `make_splits`、`make_logistic_regression`、`metric_row`：下游公平 CV 基础。
- `corrected_comparison`：encoder 对 PCA 的校正配对统计。
- `evaluate_encoder`：比较 raw、transductive PCA 与冻结 encoder；`main`：按训练/评估阶段运行。

### `scr/audit_qptm_archive.py`

- `sha256_file`：流式归档校验；`safe_member_path`：拒绝路径穿越/符号链接。
- `audit_and_extract`：检查并安全解压 ZIP；`main`：保存归档审计摘要。

### `scr/audit_qptm_conditions.py`

- `audit_conditions`：分块统计人类修饰、条件、位点和配对多类条件数；`report_progress` 与 `main`：监督并保存结果。

### `scr/audit_qptm_cptac_overlap.py`

- `cptac_feature_keys`：提取 CPTAC 共同简单位点键。
- `main`：分块匹配 qPTM 与 CPTAC 位点并报告覆盖；`report_progress`：输出进度。

### `scr/monitor_remote_log.py`

- `valid_log_name`：限制监控对象在 `logs/` 内。
- `read_remote_log`：通过配置 SSH 读取尾部日志并过滤进度条噪声。
- `unseen_lines`、`append_terminal_log`：去重并追加本地审计日志。
- `monitor`：单次或循环监控；`main`：解析命令行。

### 其他辅助脚本

- `cptac_setup.configure_cptac`：配置 CPTAC 本地缓存。
- `load.py`、`focus.py`、`clinical_tital.py`：早期数据浏览/诊断脚本；不构成当前正式决策链。

## 运行顺序

1. `coverage_matrix.py` → `build_matrix.py` → `run_phase0_null.py`
2. `audit_hard_task.py` → `run_hard_task_ablation.py` → `run_hard_task_increment.py`
3. `run_encoder_necessity.py --stage all`
4. `audit_qptm_archive.py` → `audit_qptm_conditions.py` → `audit_qptm_cptac_overlap.py`

完整结论见 `ENCODER_NECESSITY_REPORT.md`。
