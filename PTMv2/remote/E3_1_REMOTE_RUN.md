# E3.1 远程置换 null 运行说明（sensecore）

## 目的
在远程 `sensecore` 上执行冻结主管线（raw Elastic Net）的 500 次完整置换 null。每次置换重跑训练折内处理、内层参数选择与外层 OOF 预测，得出经验 p 值。

## 前置（已由本地完成）
- `run_permutation_null.py` + 嵌套核心 + 单元测试（本地 4/4 通过）
- 本地高维 BLAS 崩溃（`0xc06d007f`）已确认——**完整评估必须远程执行**
- 运行包 `releases/e3_1_permutation_null_bundle.tar.gz`（SHA256 `c8ce64e3ab35c9424bd796d3e809fcdf999c721b6d5f0285225cd8c143784442`）

## 远程执行步骤

1. 确认远程环境与项目目录：
   ```bash
   ssh sensecore 'ls /data/PTM/PTMv2/scr/ | grep -E "run_permutation|nested"'
   ```
2. 将运行包解压到 `/data/PTM/PTMv2`（覆盖 scr/ 与配置，保留原始数据引用）：
   ```bash
   ssh sensecore 'cd /data/PTM/PTMv2 && tar xzf releases/e3_1_permutation_null_bundle.tar.gz'
   ```
3. 启动 500 次置换长任务（nohup + 时间戳 flush 日志）：
   ```bash
   ssh sensecore 'cd /data/PTM/PTMv2 && nohup /root/anaconda3/envs/ptm-encoder/bin/python -u scr/run_permutation_null.py --config config/project.yaml --n-permutations 500 > /data/PTM/logs/e3_1_permutation_null.log 2>&1 & echo $! > /data/PTM/logs/e3_1_permutation_null.pid; echo started'
   ```
4. 监控进度（每 ~10 分钟）：
   ```bash
   tail -3 /data/PTM/logs/e3_1_permutation_null.log
   # 期望输出: permutation N/500 done ... observed_auprc=... p_value=...
   ```

## 输出
- `/data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_smoke.csv`（500 次置换的 null AUPRC 表，文件名因脚本默认；完成后本地重命名为正式 `primary_model_permutation_null.csv`）
- 完整日志 `/data/PTM/logs/e3_1_permutation_null.log`

## 核验命令（回传前）
```bash
wc -l /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_smoke.csv
head -2 /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_smoke.csv
tail -2 /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_smoke.csv
```
