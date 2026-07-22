import cptac
for name, cls in [("LUAD", cptac.Luad), ("UCEC", cptac.Ucec)]:
    print("=" * 60, name)
    c = cls().get_clinical("mssm")
    for col in ["histologic_grade", "tumor_stage_pathological"]:
        if col in c.columns:
            print(f"\n[{col}]")
            print(c[col].value_counts(dropna=False))
        else:
            print(f"\n[{col}] —— 该队列无此列")