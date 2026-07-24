"""审计 qPTM 条件扰动位点与 CPTAC encoder 特征的可对齐覆盖。"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import pandas as pd

from project_config import CONFIG, configured_path, configured_template_path


def report_progress(message: str) -> None:
    """输出带时间戳的分块进度。"""

    print(f"[{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')}] {message}", flush=True)


def cptac_feature_keys() -> dict[str, set[tuple[str, str, str]]]:
    """从三队列共同 encoder 特征中提取修饰、基因和单一位点位置键。"""

    shared = None
    for cohort in CONFIG["encoder"]["pretraining_cohorts"]:
        frame = pd.read_pickle(configured_template_path("residual_matrix_template", cohort=cohort.lower()))
        shared = frame.columns if shared is None else shared.intersection(frame.columns, sort=False)
    keys = defaultdict(set)
    for modification, gene, site in shared:
        pattern = CONFIG["qptm"]["cptac_site_patterns"].get(modification)
        match = re.fullmatch(pattern, str(site)) if pattern else None
        if match:
            keys[modification].add((modification, str(gene).upper(), match.group(1)))
    return keys


def main() -> None:
    """分块扫描 qPTM 并保存其相对 CPTAC 共同位点的匹配覆盖。"""

    settings, names = CONFIG["qptm"], CONFIG["qptm"]["columns"]
    cptac = cptac_feature_keys()
    matched, qptm_sites, event_counts, matched_events = defaultdict(set), defaultdict(set), Counter(), Counter()
    usecols = [names[key] for key in ["organism", "modification", "gene", "position"]]
    input_path = configured_path("qptm_data_file")
    report_progress("qPTM-CPTAC feature overlap audit started")
    for chunk_number, chunk in enumerate(pd.read_csv(input_path, sep="\t", usecols=usecols, chunksize=settings["read_chunksize"], low_memory=False), start=1):
        filtered = chunk.loc[(chunk[names["organism"]] == settings["organism"]) & chunk[names["modification"]].isin(settings["cptac_modification_map"])]
        for raw_modification, group in filtered.groupby(names["modification"], sort=False):
            modification = settings["cptac_modification_map"][raw_modification]
            for gene, position in zip(group[names["gene"]], group[names["position"]]):
                if pd.isna(gene) or pd.isna(position):
                    continue
                key = (modification, str(gene).upper(), str(position))
                qptm_sites[modification].add(key)
                event_counts[modification] += 1
                if key in cptac[modification]:
                    matched[modification].add(key)
                    matched_events[modification] += 1
        if chunk_number % settings["progress_every_chunks"] == 0:
            report_progress(f"qPTM-CPTAC overlap processed chunks={chunk_number}")
    rows = []
    for raw_modification, modification in settings["cptac_modification_map"].items():
        rows.append({"qptm_modification": raw_modification, "cptac_modification": modification, "cptac_simple_features": len(cptac[modification]), "qptm_sites": len(qptm_sites[modification]), "matched_sites": len(matched[modification]), "cptac_feature_match_fraction": len(matched[modification]) / len(cptac[modification]) if cptac[modification] else 0.0, "qptm_site_match_fraction": len(matched[modification]) / len(qptm_sites[modification]) if qptm_sites[modification] else 0.0, "qptm_events": event_counts[modification], "matched_event_fraction": matched_events[modification] / event_counts[modification] if event_counts[modification] else 0.0})
    result = pd.DataFrame(rows)
    output_path = configured_path("qptm_cptac_overlap_audit")
    result.to_csv(output_path, index=False)
    report_progress(f"qPTM-CPTAC feature overlap audit completed; summary={output_path}")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
