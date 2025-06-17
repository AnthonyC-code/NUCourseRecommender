# buzz_to_pos.py
import pandas as pd, collections, random, torch
from torch.utils.data import Dataset
import numpy as np

df = pd.read_pickle("courses_with_buzz.pkl")
emb = np.load("embeddings.npy")

inv = collections.defaultdict(list)
for i, kws in enumerate(df["buzz"]):
    for kw in kws:
        inv[kw].append(i)

valid_kw = {k: v for k,v in inv.items() if 2 <= len(v) <= 100}

pos_lists = []
for i, kws in enumerate(df["buzz"]):
    pos = set()
    for kw in kws:
        pos.update(valid_kw.get(kw, []))
    pos.discard(i)
    pos_lists.append(list(pos))
