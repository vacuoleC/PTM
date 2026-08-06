# E3.1 远程置换 null 运行说明（sensecore）

## 目的
在远程 `sensecore` 上执行冻结主管线（raw Elastic Net）的 500 次完整置换 null。每次置换重跑训练折内处理、内层参数选择与外层 OOF 预测，得出经验 p 值。

## 前置（已由本地完成）
- `run_permutation_null.py`（多核并行 + checkpoint + 限时）+ 单元测试 5/5 通过
- 本地高维 BLAS 崩溃（`0xc06d007f`）——**完整评估必须远程执行**
- 远程：192 核 / 2TB 内存（Intel Xeon 8468V），`/root/anaconda3/envs/ptm-encoder/bin/python`

## 远程执行（setsid 彻底脱离 SSH）

1. 同步最新代码：
   ```bash
   ssh sensecore 'cd /data/PTM/PTMv2 && git fetch origin main && git checkout origin/main -- scr/ config/ releases/'
   ```
2. 启动 500 次置换（192 核并行，单进程 24h 上限内约 6-8 小时完成）：
   ```bash
   ssh sensecore 'cd /data/PTM/PTMv2 && setsid nohup /root/anaconda3/envs/ptm-encoder/bin/python -u scr/run_permutation_null.py --config config/project.yaml --n-permutations 500 --n-jobs 190 --checkpoint /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv --max-hours 24 > /data/PTM/logs/e3_1_permutation_null.log 2>&1 < /dev/null & echo $! > /data/PTM/logs/e3_1_permutation_null.pid; echo started'
   ```
3. 监控进度（每 ~10 分钟）：
   ```bash
   tail -2 /data/PTM/logs/e3_1_permutation_null.log
   wc -l /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv   # 期望 501 行（表头+500）
   ```

## 输出
- checkpoint：`/data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv`（`permutation,auprc`，每置换一行）
- 日志：`/data/PTM/logs/e3_1_permutation_null.log`
- 完成后本地重命名 checkpoint 为正式 `outputs/tables/primary_model_permutation_null.csv`

## 核验命令（回传前）
```bash
wc -l /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv
head -2 /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv
tail -2 /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv
grep "p_value" /data/PTM/logs/e3_1_permutation_null.log
```

## 接力恢复（若 24h 到达未完成）
同一命令重跑即可：`--resume` 已由 checkpoint 自动处理（`--checkpoint` 文件存在时跳过已完成置换）。继续启动直到 501 行。
