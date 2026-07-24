"""分块审计 qPTM 条件扰动表能否提供无标签预训练样本。"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from project_config import CONFIG, configured_path


def report_progress(message: str) -> None:
    """输出带时间戳的进度，供远端日志监控器读取。"""

    print(f"[{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')}] {message}", flush=True)


def audit_conditions() -> pd.DataFrame:
    """只用分块聚合统计 qPTM 的人类磷酸化/乙酰化条件与位点覆盖。"""

    settings = CONFIG["qptm"]
    columns = settings["columns"]
    required = list(columns.values())
    input_path = configured_path("qptm_data_file")
    statistics = defaultdict(lambda: {"events": 0, "sites": set(), "genes": set(), "conditions": set(), "samples": set(), "pmids": set(), "quantified": 0})
    total_rows = 0
    for chunk_number, chunk in enumerate(
        pd.read_csv(input_path, sep="\t", usecols=required, chunksize=settings["read_chunksize"], low_memory=False),
        start=1,
    ):
        total_rows += len(chunk)
        filtered = chunk.loc[
            (chunk[columns["organism"]] == settings["organism"])
            & chunk[columns["modification"]].isin(settings["modifications"])
        ].copy()
        for modification, group in filtered.groupby(columns["modification"], sort=False):
            summary = statistics[modification]
            summary["events"] += len(group)
            summary["sites"].update(zip(group[columns["protein"]].astype(str), group[columns["position"]].astype(str)))
            summary["genes"].update(group[columns["gene"]].dropna().astype(str))
            summary["conditions"].update(zip(group[columns["publication"]].astype(str), group[columns["sample"]].astype(str), group[columns["condition"]].astype(str)))
            summary["samples"].update(group[columns["sample"]].dropna().astype(str))
            summary["pmids"].update(group[columns["publication"]].dropna().astype(str))
            summary["quantified"] += pd.to_numeric(group[columns["peptide_log2_ratio"]], errors="coerce").notna().sum()
        if chunk_number % settings["progress_every_chunks"] == 0:
            report_progress(f"qPTM condition audit processed chunks={chunk_number}, rows={total_rows}")

    rows = []
    for modification in settings["modifications"]:
        summary = statistics[modification]
        rows.append({
            "organism": settings["organism"], "modification": modification,
            "events": summary["events"], "unique_sites": len(summary["sites"]),
            "unique_genes": len(summary["genes"]), "condition_contexts": len(summary["conditions"]),
            "unique_samples": len(summary["samples"]), "publications": len(summary["pmids"]),
            "peptide_log2_ratio_present": summary["quantified"],
            "peptide_log2_ratio_fraction": summary["quantified"] / summary["events"] if summary["events"] else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    """运行审计并保存可版本化的 qPTM 条件覆盖摘要。"""

    report_progress("qPTM condition audit started")
    result = audit_conditions()
    output_path = configured_path("qptm_condition_audit")
    result.to_csv(output_path, index=False)
    report_progress(f"qPTM condition audit completed; summary={output_path}")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
