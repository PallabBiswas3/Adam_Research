import torch


def hessian_vector_product(loss, parameters, vectors):
    """Autograd HVP; loss must retain its graph. No dense neural Hessian."""
    params, vecs = list(parameters), list(vectors)
    if len(params) != len(vecs) or any(p.shape != v.shape for p, v in zip(params, vecs)):
        raise ValueError("HVP vector shapes must match parameters")
    first = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    terms = [(g*v).sum() for g, v in zip(first, vecs) if g is not None and g.requires_grad]
    if not terms:
        return [torch.zeros_like(p) for p in params]
    second = torch.autograd.grad(sum(terms), params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if h is None else h for p, h in zip(params, second)]


def directional_curvature(loss, parameters, vectors, eps=1e-12):
    vecs = list(vectors)
    hv = hessian_vector_product(loss, parameters, vecs)
    denom = sum(v.square().sum() for v in vecs)
    if denom <= eps:
        return float("nan")
    return float(sum((v*h).sum() for v, h in zip(vecs, hv))/denom)
