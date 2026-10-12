"""Controlled long-tail Gaussian data. Clean labels are evaluation-only."""
import numpy as np

def make_long_tail(seed=0, counts=(600,160,60), noise_rate=0.25, noise_kind="instance"):
    if len(counts)!=3 or min(counts)<1 or not 0<=noise_rate<=1:
        raise ValueError("invalid class counts/noise")
    if noise_kind not in {"instance","symmetric"}:
        raise ValueError("unsupported noise family")
    rng=np.random.default_rng(seed)
    centers=np.array([[-1.4,0.],[1.4,0.],[0.,1.6]])
    y=np.concatenate([np.full(n,c,dtype=np.int64) for c,n in enumerate(counts)])
    x=centers[y]+rng.normal(0,1.15,(len(y),2))
    prior=np.asarray(counts,dtype=float)/sum(counts)
    logp=-((x[:,None,:]-centers[None,:,:])**2).sum(-1)/(2*1.15**2)+np.log(prior)
    e=np.exp(logp-logp.max(-1,keepdims=True))
    posterior=e/e.sum(-1,keepdims=True)
    ambiguity=1-posterior[np.arange(len(y)),y]
    rate=np.full(len(y),noise_rate) if noise_kind=="symmetric" else noise_rate*(.25+.75*ambiguity)
    flipped=rng.random(len(y))<rate
    observed=y.copy()
    observed[flipped]=(observed[flipped]+rng.integers(1,3,flipped.sum()))%3
    idx=rng.permutation(len(y))
    return {"x":x[idx].astype(np.float32),"observed_y":observed[idx],
        "oracle":{"true_y":y[idx],"flipped":flipped[idx],"bayes_ambiguity":ambiguity[idx]}}

def retention_by_stratum(keep,oracle,hard_threshold=.5):
    keep=np.asarray(keep,dtype=bool)
    clean=~oracle["flipped"]
    hard=oracle["bayes_ambiguity"]>=hard_threshold
    tail=oracle["true_y"]==2
    masks={"clean_hard":clean&hard,"clean_easy":clean&~hard,
           "clean_tail":clean&tail,"noisy":~clean}
    return {name:{"n":int(mask.sum()),"retained":float(keep[mask].mean()) if mask.any() else None}
            for name,mask in masks.items()}
