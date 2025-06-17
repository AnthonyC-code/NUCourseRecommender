import numpy as np, pandas as pd, json

df   = pd.read_csv("courses_clean.csv")           # has subject, catalog_num, title
embD = np.load("proj_embeddings_dept.npy")
embB = np.load("proj_embeddings_buzz.npy")

embD /= np.linalg.norm(embD, axis=1, keepdims=True)
embB /= np.linalg.norm(embB, axis=1, keepdims=True)
W_DEPT, W_BUZZ = 0.6, 0.4
vecs = W_DEPT*embD + W_BUZZ*embB
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # final unit vectors


sims = vecs @ vecs.T
np.fill_diagonal(sims, -1)             # don’t recommend itself
top_idx = np.argsort(sims, axis=1)[:, :-11:-1]   # last 10 cols reversed
course_ids = df.subject + "_" + df.catalog_num.astype(str)

recommend = {
    cid: [course_ids[j] for j in nbrs]
    for cid, nbrs in zip(course_ids, top_idx)
}
with open("recommend.json", "w") as f:
    json.dump(recommend, f)


meta = (
    df.assign(course_id=course_ids)
      .loc[:, ["course_id", "subject", "catalog_num", "title"]]
)
meta.to_json("meta.json", orient="records")
