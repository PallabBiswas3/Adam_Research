"""Smoke-only CE/global-loss/classwise-loss controls; NOT paper reproductions."""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from robust_learning.synthetic import make_long_tail,retention_by_stratum

def run(seed=0,method="small_loss",epochs=18,keep_fraction=.70):
    if method not in {"ce","small_loss","classwise_small_loss"}:
        raise ValueError("unknown control")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    d=make_long_tail(seed)
    x=torch.tensor(d["x"]);y=torch.tensor(d["observed_y"])
    model=torch.nn.Sequential(torch.nn.Linear(2,32),torch.nn.ReLU(),torch.nn.Linear(32,3))
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
    keep=torch.ones(len(x),dtype=torch.bool)
    for epoch in range(epochs):
        with torch.no_grad():
            losses=F.cross_entropy(model(x),y,reduction="none")
            if method=="small_loss" and epoch>=3:
                k=max(1,int(keep_fraction*len(y)))
                keep=torch.zeros_like(keep)
                keep[losses.argsort()[:k]]=True
            elif method=="classwise_small_loss" and epoch>=3:
                keep=torch.zeros_like(keep)
                for c in range(3):
                    indices=(y==c).nonzero().flatten()
                    k=max(1,int(keep_fraction*len(indices)))
                    keep[indices[losses[indices].argsort()[:k]]]=True
        opt.zero_grad()
        F.cross_entropy(model(x[keep]),y[keep]).backward()
        opt.step()
    return {"seed":seed,"baseline":method,"n":len(y),
        "observed_train_accuracy":float((model(x).argmax(-1)==y).float().mean()),
        "selection":retention_by_stratum(keep.numpy(),d["oracle"])}

if __name__=="__main__":
    print(json.dumps([run(0,m) for m in ["ce","small_loss","classwise_small_loss"]],indent=2))
