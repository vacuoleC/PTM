import cptac
from cptac_setup import configure_cptac
from project_config import CONFIG, get_cohort_class


configure_cptac()

for name in CONFIG["datasets"]["focus_cohorts"]:
    print("=" * 60, name)
    cohort = get_cohort_class(cptac, name)()
    c = cohort.get_clinical(CONFIG["cptac"]["clinical_source"])
    for col in CONFIG["clinical"]["focus_columns"]:
        if col in c.columns:
            print(f"\n[{col}]")
            print(c[col].value_counts(dropna=False))
        else:
            print(f"\n[{col}] —— 该队列无此列")
