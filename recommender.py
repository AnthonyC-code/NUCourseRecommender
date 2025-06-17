import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("courses_clean.csv")

emb_dept = np.load("proj_embeddings_dept.npy").astype("float32")
emb_buzz = np.load("proj_embeddings_buzz.npy").astype("float32")

emb_dept /= np.linalg.norm(emb_dept, axis=1, keepdims=True)
emb_buzz /= np.linalg.norm(emb_buzz, axis=1, keepdims=True)

def recommend(subject: str, cat_num: str, k=5, w_dept=0.7, w_buzz=0.3):
    row = df.index[(df.subject == subject) &
                   (df.catalog_num == cat_num)]
    if row.empty:
        return f"{subject} {cat_num} not found."
    i = row[0]

    blend = (
        w_dept * cosine_similarity(emb_dept[i:i+1], emb_dept).ravel()
      + w_buzz * cosine_similarity(emb_buzz[i:i+1], emb_buzz).ravel()
    )

    anchor_digit = int(cat_num.lstrip()[0])
    if anchor_digit == 1:
        allowed = {"1", "2"}
    elif anchor_digit == 2:
        allowed = {"1", "2", "3"}
    elif anchor_digit == 3:
        allowed = {"3"}
    else:
        allowed = {str(anchor_digit)}

    mask = df["catalog_num"].str.strip().str[0].isin(allowed)

    candidate_idx = np.where(mask)[0]
    ranked = candidate_idx[np.argsort(blend[candidate_idx])[::-1]]

    ranked = ranked[ranked != i][:k]

    return df.iloc[ranked][
        ["subject", "catalog_num", "title", "description"]
    ]


print(recommend("HISTORY","203-1",k=20))
