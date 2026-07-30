# E2.2 固定参数 OOF 冒烟运行包

此包只运行冻结的 E2.2 冒烟检查：对已经固定的 50 个患者级外层测试折，使用训练折内检测率过滤、训练折内中位数填补、训练折内标准化和固定 Elastic Net 参数（检测率阈值 0.1、`C=0.1`、`l1_ratio=0.5`）生成 OOF 概率。它**不**进行内层调参、置换检验、学习曲线或 encoder 训练。

## 前提

- 服务器项目目录是 `/data/PTM`；
- 已存在并可激活 `ptm-encoder` Conda 环境；
- `/data/PTM/PTMv1/outputs/` 中仍有 v1 的只读矩阵与 G2/G3 标签；
- 从 GitHub 拉取包含本 tar 包的提交。

## 运行

在服务器终端执行（每行单独执行）：

```bash
cd /data/PTM
git pull --ff-only origin main
tar -xzf PTMv2/releases/e2_2_remote_oof_bundle.tar.gz
conda activate ptm-encoder
test -f PTMv1/outputs/lscc_multi_ptm_resid.pkl.gz
test -f PTMv1/outputs/lscc_grade_g2_vs_g3_labels.csv
mkdir -p logs
nohup python PTMv2/scr/smoke_oof.py --config PTMv2/config/project.yaml > logs/e2_2_fixed_parameter_oof_smoke.log 2>&1 &
echo $! > logs/e2_2_fixed_parameter_oof_smoke.pid
```

监控进度：

```bash
tail -f logs/e2_2_fixed_parameter_oof_smoke.log
```

完成时，日志最后一行应为 `[E2.2] complete: wrote ...`，并且 CSV 应有 1,060 条预测记录（10 次重复 × 106 位病人）；因此 `wc -l` 应显示 1,061 行（含表头）：

```bash
wc -l PTMv2/outputs/tables/e2_2_fixed_parameter_oof_smoke.csv
sha256sum PTMv2/outputs/tables/e2_2_fixed_parameter_oof_smoke.csv
tail -n 5 logs/e2_2_fixed_parameter_oof_smoke.log
```

## 交付给 Codex

请将下列三项带回本地并作为附件交付：

1. `PTMv2/outputs/tables/e2_2_fixed_parameter_oof_smoke.csv`；
2. `logs/e2_2_fixed_parameter_oof_smoke.log`；
3. 上面三个核验命令的完整终端输出。

不要在服务器上修改 `whatwedo.md`、研究设计或其他代码，也不必提交运行产生的文件；我会在本地核验结果、补全工作账本并创建结果的原子 Git 提交。
