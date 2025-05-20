# verify_courses.py
# Quick data-health report for nu_courses_2024_25.csv
import json, pandas as pd

CSV_IN  = "nu_courses_2024_25.csv"
REPORT  = "verify_report.json"

df = pd.read_csv(CSV_IN)

summary = {
    "rows"             : len(df),
    "cols"             : list(df.columns),
    "empty_desc_rate"  : df["description"].eq("").mean(),
    "duplicate_keys"   : int(df.duplicated(["subject","catalog_num"]).sum()),
    "unique_subjects"  : int(df["subject"].nunique()),
    "subject_counts"   : df["subject"].value_counts().to_dict(),
}

print("=== NU Course Catalog – Verify Report ===")
print(f"Rows                       : {summary['rows']}")
print(f"Empty descriptions (%)     : {summary['empty_desc_rate']:.2%}")
print(f"Duplicate subject+number   : {summary['duplicate_keys']}")
print(f"Subjects covered           : {summary['unique_subjects']}/118\n")

(pd.Series(summary["subject_counts"])
     .sort_values(ascending=False)
     .head(15)
     .to_string(float_format="%.0f"))
print("\nTop-15 subjects by row-count printed above.")

with open(REPORT, "w") as f:
    json.dump(summary, f, indent=2)
print("Saved JSON summary →", REPORT)
