# `.doc`：阶段汇报与文档快照

本目录集中保存本阶段的汇报产物，不改变项目的分析代码与正式 `outputs/`。

```text
.doc/
├─ PTM_STAGE_REPORT.md             # 约 8 分钟的中文讲稿型汇报
├─ figures/                        # 六张 300 dpi 中文论文风格 PNG
├─ scripts/
│  └─ generate_stage_report_figures.py
└─ reference/
   ├─ markdown/                    # 除根 README 与 .clinerules 外的 Markdown 快照
   ├─ configuration/               # 项目配置快照
   └─ audit-logs/                  # 预留给本地查看；.log 的追加式记录不复制入版本库
```

重新生成图片：

```powershell
D:\enviranment\ptm-encoder\python.exe .doc\scripts\generate_stage_report_figures.py
```

图表脚本只读取已经提交的 `outputs/*.csv`。当前 Windows 上该环境的 Matplotlib 在 PNG 写出阶段会异常退出，因此脚本使用 Pillow 绘制，不改变任何统计数据或分析结果。
