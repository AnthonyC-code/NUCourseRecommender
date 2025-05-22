# enrich_courses.py
# Clean + embed the catalog for downstream recommendation work
import re, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

RAW_CSV   = "nu_courses_2024_25.csv"
CLEAN_CSV = "courses_clean.csv"
EMB_NPY   = "embeddings.npy"
META_JSON = "meta.json"

df = pd.read_csv(RAW_CSV)

split_rx = re.compile(r"^([A-Z_]+)\s+([\dA-Z\-]+-?\d*)\s+(.+?)\s+\(([\d.]+|Variable) Unit")
cols = df["title_units"].str.extract(split_rx)
cols.columns = ["subject","catalog_num","title","units_raw"]

df[["subject","catalog_num","title"]] = cols[["subject","catalog_num","title"]]
df["units"] = pd.to_numeric(cols["units_raw"].where(cols["units_raw"]!="Variable"), errors="coerce")

df["prereq"] = df["description"].str.extract(r"Prerequisite[s]?:\s*(.*?)\.\s*$", expand=False)

df["description"] = df["description"].fillna("")
df["prereq"]      = df["prereq"].fillna("")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb   = model.encode(df["description"].tolist(), show_progress_bar=True, convert_to_numpy=True)

df.to_csv(CLEAN_CSV, index=False)
np.save(EMB_NPY, emb)

meta = {
    "rows"   : len(df),
    "emb_dim": int(emb.shape[1]),
    "model"  : "all-MiniLM-L6-v2",
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

print("   ", CLEAN_CSV)
print("   ", EMB_NPY)
print("   ", META_JSON)
