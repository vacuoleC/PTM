import cptac
lscc = cptac.Lscc()

# clinical 表来自 mssm 源（不是 umich）
clin = lscc.get_clinical("mssm")
print(clin.shape)

# 先把所有列名打出来看全貌
# print(list(clin.columns))

"""
Downloading clinical_Pan-cancer.May2022.tsv.gz: 100%|█████████████████████████████████████████████████████████████| 243k/243k [00:04<00:00, 56.9kB/s]
(110, 124)
[
    'tumor_code', 
    'discovery_study', 
    'type_of_analyzed_samples', 
    'confirmatory_study', 
    'type_of_analyzed_samples', 
    'age', 
    'sex', 
    'race', 
    'ethnicity', 
    'ethnicity_race_ancestry_identified', 
    'Inferred ancestry', 
    'collection_in_us', 
    'participant_country', 
    'maternal_grandmother_country', 
    'maternal_grandfather_country', 
    'paternal_grandmother_country', 
    'paternal_grandfather_country', 
    'deaf_or_difficulty_hearing', 
    'blind_or_difficulty_seeing', 
    'difficulty_concentrating_remembering_or_making_decisions', 
    'difficulty_walking_or_climbing_stairs', 
    'difficulty_dressing_or_bathing', 
    'difficulty_doing_errands', 
    'consent_form_signed', 
    'case_stopped', 
    'tumor_site', 
    'tumor_site_other', 
    'tumor_laterality', 
    'tumor_focality', 
    'tumor_size_cm', 
    'histologic_type', 
    'histologic_grade', 
    'tumor_necrosis', 
    'margin_status', 
    'ajcc_tnm_cancer_staging_edition_used', 
    'pathologic_staging_primary_tumor_pt', 
    'pathologic_staging_regional_lymph_nodes_pn', 
    'number_of_lymph_nodes_examined', 
    'number_of_lymph_nodes_positive_for_tumor_by_he_staining', 
    'clinical_staging_distant_metastasis_cm', 
    'pathologic_staging_distant_metastasis_pm', 
    'specify_distant_metastasis_documented_sites', 
    'residual_tumor', 
    'tumor_stage_pathological', 
    'paraneoplastic_syndrome_present', 
    'ancillary_studies_immunohistochemistry_performed', 
    'ancillary_studies_immunohistochemistry_type_and_result', 
    'ancillary_studies_other_testing_performed', 
    'ancillary_studies_other_testing_type_and_result', 
    'performance_status_assessment_ecog_performance_status_score', 
    'performance_status_assessment_karnofsky_performance_status_score', 
    'number_of_lymph_nodes_positive_for_tumor_by_ihc_staining', 
    'perineural_invasion', 
    'height_at_time_of_surgery_cm', 
    'weight_at_time_of_surgery_kg', 
    'bmi', 
    'history_of_cancer', 
    'alcohol_consumption', 
    'tobacco_smoking_history', 
    'age_at_which_the_participant_started_smoking', 
    'age_at_which_the_participant_stopped_smoking', 
    'on_the_days_participant_smoked_how_many_cigarettes_did_he_she_usually_smoke', 
    'number_of_pack_years_smoked', 
    'was_the_participant_exposed_to_secondhand_smoke', 
    'exposure_to_secondhand_smoke_in_household_during_participants_childhood', 
    'exposure_to_secondhand_smoke_in_participants_current_household', 
    'number_of_years_participant_has_consumed_more_than_2_drinks_per_day_for_men_and_more_than_1_drink_per_day_for_women', 
    'cancer_type', 
    'history_source', 
    'history_of_any_treatment', 
    'medical_record_documentation_of_this_history_of_cancer_and_treatment', 
    'medical_condition', 
    'history_of_treatment', 
    'history_source', 
    'medication_name_vitamins_supplements', 
    'history_source', 
    'blood_collection_minimum_required_blood_collected', 
    'blood_collection_number_of_blood_tubes_collected', 
    'tumor_tissue_collection_tumor_type', 
    'tumor_tissue_collection_number_of_tumor_segments_collected', 
    'tumor_tissue_collection_clamps_used', 
    'tumor_tissue_collection_frozen_with_oct', 
    'normal_adjacent_tissue_collection_number_of_normal_segments_collected', 
    'follow_up_period', 
    'is_this_patient_lost_to_follow-up', 
    'vital_status_at_date_of_last_contact', 
    'number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_last_contact', 
    'number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_death', 
    'cause_of_death', 
    'number_of_days_from_date_of_collection_to_date_of_last_contact', 
    'number_of_days_from_date_of_collection_to_date_of_death', 
    'adjuvant_post-operative_radiation_therapy', 
    'adjuvant_post-operative_pharmaceutical_therapy', 
    'adjuvant_post-operative_immunological_therapy', 
    'tumor_status_at_date_of_last_contact_or_death', 
    'measure_of_success_of_outcome_at_the_completion_of_initial_first_course_treatment', 
    'measure_of_success_of_outcome_at_date_of_last_contact_or_death', 
    'ecog_performance_status_score_at_date_of_last_contact_or_death', 
    'karnofsky_performance_status_score_at_date_of_last_contact_or_death', 
    'performance_status_scale_timing_at_date_of_last_contact_or_death', 
    'measure_of_success_of_outcome_at_first_NTE', 
    'ecog_performance_status_score_at_first_NTE', 
    'karnofsky_performance_status_score_at_first_NTE', 
    'performance_status_scale_timing_at_first_NTE', 
    'new_tumor_after_initial_treatment', 
    'number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_new_tumor_event_after_initial_treatment', 
    'type_of_new_tumor', 
    'site_of_new_tumor', 
    'other_site_of_new_tumor', 
    'diagnostic_evidence_of_recurrence_or_relapse', 
    'additional_surgery_for_new_tumor_loco-regional', 
    'additional_surgery_for_new_tumor_metastasis', 
    'residual_tumor_after_surgery_for_new_tumor', 
    'additional_treatment_radiation_therapy_for_new_tumor', 
    'additional_treatment_pharmaceutical_therapy_for_new_tumor', 
    'additional_treatment_immuno_for_new_tumor', 
    'number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_additional_surgery_for_new_tumor_event_loco-regional', 
    'number_of_days_from_date_of_initial_pathologic_diagnosis_to_date_of_additional_surgery_for_new_tumor_event_metastasis', 
    'Recurrence-free survival, days', 
    'Recurrence-free survival from collection, days', 
    'Recurrence status (1, yes; 0, no)', 
    'Overall survival, days', 
    'Overall survival from collection, days', 
    'Survival status (1, dead; 0, alive)'
]
"""

# for col in [
#     "tumor_stage_pathological",
#     "pathologic_staging_primary_tumor_pt",
#     "pathologic_staging_regional_lymph_nodes_pn",
#     "histologic_grade",
#     "histologic_type",
# ]:
#     print("=" * 60)
#     print(col)
#     print(clin[col].value_counts(dropna=False))   # dropna=False 让 NaN 也现形

"""
============================================================
tumor_stage_pathological
tumor_stage_pathological
Stage II                                44
Stage I                                 41
Stage III                               21
Staging is not applicable or unknown     3
Stage IV                                 1
Name: count, dtype: int64
============================================================
pathologic_staging_primary_tumor_pt
pathologic_staging_primary_tumor_pt
pT2a    42
pT3     16
pT2b    15
pT1b     8
pT2      6
3        5
pT1a     4
pT1c     4
NaN      3
pT4      2
pT1      2
pT3A     1
2b       1
T1b      1
Name: count, dtype: int64
============================================================
pathologic_staging_regional_lymph_nodes_pn
pathologic_staging_regional_lymph_nodes_pn
pN0    65
pN1    16
pN2    15
0       4
pNX     3
NaN     3
1       2
N1      1
N0      1
Name: count, dtype: int64
============================================================
histologic_grade
histologic_grade
G2 Moderately differentiated                                         58
G3 Poorly differentiated                                             48
GX Grading is not applicable, cannot be assessed or not specified     3
G1 Well differentiated                                                1
Name: count, dtype: int64
============================================================
histologic_type
histologic_type
Squamous cell carcinoma                                             71
Keratinizing squamous cell carcinoma                                18
Non-keratinizing squamous cell carcinoma                            14
Basaloid squamous cell carcinoma                                     2
Adenosquamous carcinoma                                              1
Adenosquamous Carcinoma; at least 66% squamous component             1
Solid adenocarcinoma                                                 1
Spindle cell carcinoma with undifferentiated non small carcinoma     1
adenosquamous carcinoma                                              1
Name: count, dtype: int64
"""

# 主候选:grade G2 vs G3
g = clin["histologic_grade"]
g2g3 = g[g.str.startswith(("G2", "G3"), na=False)]
print("grade G2 vs G3:")
print(g2g3.str[:2].value_counts())   # 只取 G2/G3 前两字，看最终二分类计数

# 备胎:stage 早 vs 晚
s = clin["tumor_stage_pathological"]
early = s.isin(["Stage I", "Stage II"]).sum()
late  = s.isin(["Stage III", "Stage IV"]).sum()
print(f"\nstage 早(I+II)={early}  晚(III+IV)={late}")