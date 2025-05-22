import random, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PKL  = "courses_with_buzz.pkl" 
NPY  = "embeddings.npy"

OUT_PT  = "proj128_buzz.pt"
OUT_NPY = "proj_embeddings_buzz.npy"

BATCH  = 256
EPOCHS = 20
LR     = 1e-3
MARGIN = 0.3

device = torch.device("cpu")

class BuzzTriplet(Dataset):
    """positive shares ≥1 buzzword with anchor."""
    def __init__(self, df, emb):
        self.df  = df.reset_index(drop=True)
        self.emb = emb.astype("float32")
        self.all = list(range(len(df)))

        inv = {}
        for i,kws in enumerate(self.df["buzz"]):
            for kw in kws:
                inv.setdefault(kw, []).append(i)

        self.pos = []
        for kws in self.df["buzz"]:
            peers = set()
            for kw in kws:
                peers.update(inv[kw])
            self.pos.append(list(peers))

        self.anchor = [i for i,p in enumerate(self.pos) if len(p) > 1]

    def __len__(self): return len(self.anchor)

    def __getitem__(self,n):
        i = self.anchor[n]
        p = random.choice([j for j in self.pos[i] if j != i])

        anchor_kw = set(self.df.loc[i,"buzz"])
        while True:
            neg = random.choice(self.all)
            if anchor_kw.isdisjoint(self.df.loc[neg,"buzz"]):
                break

        return (torch.from_numpy(self.emb[i]),
                torch.from_numpy(self.emb[p]),
                torch.from_numpy(self.emb[neg]))

def collate(batch):
    a,p,n = zip(*batch)
    return torch.stack(a), torch.stack(p), torch.stack(n)

df   = pd.read_pickle(PKL)
emb  = np.load(NPY)

ds   = BuzzTriplet(df, emb)
dl   = DataLoader(ds, batch_size=BATCH, shuffle=True,
                  num_workers=0, collate_fn=collate)

print(f"Anchors:{len(ds):,}  total rows:{len(df):,}")

proj = nn.Sequential(
        nn.Linear(384,256), nn.ReLU(),
        nn.Linear(256,128)
).to(device)

loss_fn = nn.TripletMarginLoss(margin=MARGIN, p=2)
opt     = optim.Adam(proj.parameters(), lr=LR)

for epoch in range(1, EPOCHS+1):
    running = 0.0
    for a,p,n in tqdm(dl, desc=f"Buzz Epoch {epoch}", leave=False):
        a,p,n = a.to(device),p.to(device),n.to(device)
        la,lp,ln = proj(a),proj(p),proj(n)
        loss = loss_fn(la,lp,ln)
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()*a.size(0)
    print(f"Epoch {epoch:02d}  avg-loss {running/len(ds):.4f}")

with torch.no_grad():
    proj_emb = proj(torch.from_numpy(emb).to(device)).cpu().numpy()

torch.save(proj.state_dict(), OUT_PT)
np.save(OUT_NPY, proj_emb)
print(f"saved {OUT_PT} & {OUT_NPY}")
