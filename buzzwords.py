# buzzwords.py  – extract data-driven “buzzwords” from course descriptions
import re, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_IN  = "courses_clean.csv"
PKL_OUT = "courses_with_buzz.pkl"

df = pd.read_csv(CSV_IN)
df["description"] = df["description"].fillna("")

tok_pat = r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
vec = TfidfVectorizer(
        lowercase=True,
        token_pattern=tok_pat,
        ngram_range=(1,3),
        stop_words="english",
        max_features=25_000,
        min_df=3 
)
tfidf  = vec.fit_transform(df["description"])
vocab  = vec.get_feature_names_out()

M = 5
top_kw = []
for row in range(tfidf.shape[0]):
    row_vec = tfidf.getrow(row)
    if row_vec.nnz == 0:
        top_kw.append([])
        continue
    idx = row_vec.indices[row_vec.data.argsort()[::-1][:M]]
    top_kw.append([vocab[i] for i in idx])

df["buzz"] = top_kw
df.to_pickle(PKL_OUT)
print(f"wrote {PKL_OUT} with buzzword lists for {len(df)} courses")
