# utils.py  – triplet dataset helpers
import random, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, csv_path, emb_path):
        self.df   = pd.read_csv(csv_path)
        self.emb  = np.load(emb_path)
        self.all  = list(range(len(self.df)))

        # cache positive lists
        from positive_policy import positives
        self.pos_lists = [positives(self.df,i) for i in self.all]

        # keep only anchors that have >= 1 positives
        self.anchor_idx = [i for i,p in enumerate(self.pos_lists) if p]

    def __len__(self): return len(self.anchor_idx)

    def __getitem__(self, n):
        i   = self.anchor_idx[n]
        pos = random.choice(self.pos_lists[i])
        while True:
            neg = random.choice(self.all)
            if neg not in self.pos_lists[i]:
                break
        return (torch.from_numpy(self.emb[i]).float(),
                torch.from_numpy(self.emb[pos]).float(),
                torch.from_numpy(self.emb[neg]).float())


def collate(batch):
    a, p, n = zip(*batch)
    return torch.stack(a), torch.stack(p), torch.stack(n)
