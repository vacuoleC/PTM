# E3.1 远程置换 null 运行说明（sensecore）

## 目的
在远程 `sensecore` 上执行冻结主管线（raw Elastic Net）的 500 次完整置换 null。每次置换重跑训练折内处理、内层参数选择与外层 OOF 预测，得出经验 p 值。

## 资源约束（老师指定）
- **本项目限用 64 核**（节点实际 192 核，但本项目最大 64 核）
- `--n-jobs` 定为 **60**（留 4 核给系统/SSH/监控/其他租户，避免满载抖动）
- 每个 worker 强制 `OPENBLAS_NUM_THREADS=1`（脚本内强制覆盖，非 setdefault），并行来自进程数而非线程数

## 前置（已由本地完成）
- `run_permutation_null.py`（多核并行 initializer 共享数据 + imap_unordered 优雅退出 + checkpoint + 限时）+ 单元测试 5/5 通过
- 本地高维 BLAS 崩溃（`0xc06d007f`）——**完整评估必须远程执行**
- 远程实测：单次 saga 拟合 266.5s（探针）；单置换估算约 9.6h

## 时间预期（60 并行）
- 单置换 CPU 约 9.6h；60 并行下 500 次 ≈ 500×9.6/60 ≈ **80h**
- 24h 作业上限 → 预计 **4 批接力**（checkpoint+resume 自动续跑，每批 24h 约 150 次）
- 观察值嵌套约 4h（每批开始时重跑，不计入置换）

## 远程执行（setsid 彻底脱离 SSH）

1. 同步最新代码：
   ```bash
   ssh sensecore 'cd /data/PTM/PTMv2 && git fetch origin main && git checkout origin/main -- scr/ config/ releases/'
   ```
2. 启动 500 次置换（60 并行）：
   ```bash
   ssh sensecore 'cd /data/PTM/PTMv2 && setsid nohup /root/anaconda3/envs/ptm-encoder/bin/python -u scr/run_permutation_null.py --config config/project.yaml --n-permutations 500 --n-jobs 60 --checkpoint /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv --max-hours 24 > /data/PTM/logs/e3_1_permutation_null.log 2>&1 < /dev/null & echo $! > /data/PTM/logs/e3_1_permutation_null.pid; echo started'
   ```
3. 监控进度（每 ~10 分钟）：
   ```bash
   tail -2 /data/PTM/logs/e3_1_permutation_null.log
   wc -l /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv   # 期望 501 行（表头+500）
   ```
4. 24h 到点接力：同一命令重跑（`--checkpoint` 存在时自动跳过已完成），直到 501 行。

## 输出
- checkpoint：`/data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv`（`permutation,auprc`）
- 日志：`/data/PTM/logs/e3_1_permutation_null.log`
- 完成后本地重命名 checkpoint 为正式 `outputs/tables/primary_model_permutation_null.csv`

## 核验命令（回传前）
```bash
wc -l /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv
head -2 /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv
tail -2 /data/PTM/PTMv2/outputs/tables/e3_1_permutation_null_checkpoint.csv
grep "p_value" /data/PTM/logs/e3_1_permutation_null.log
```
