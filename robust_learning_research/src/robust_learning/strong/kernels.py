"""Independent paper-equation update kernels; no training-label oracle inputs."""
import torch
import torch.nn.functional as F

from .cifar import batch_images
from .equations import adjusted_probability, ccs_loss, refine_targets, sharpen, semisupervised_loss


class DSSState:
    def __init__(self, examples, classes, prior, decay=.99):
        self.prior = prior
        self.decay = decay
        self.marginal = torch.ones(classes)/classes
        self.current = torch.ones(examples, classes)/classes
        self.history = []
        self.score = torch.zeros(examples, classes)

    def observe(self, indices, probability):
        # Histories stay on the host; only the current minibatch occupies VRAM.
        probability = probability.detach().cpu()
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        self.marginal = self.decay*self.marginal+(1-self.decay)*probability.mean(0)
        self.current[indices] = adjusted_probability(probability, self.marginal, self.prior).detach()

    def finish_epoch(self):
        if self.history:
            self.score += torch.sign(self.current-torch.stack(self.history)).sum(0)
        self.history.append(self.current.clone())

    def policy(self, observed, threshold=1.2815515655446004):
        keep = self.current.argmax(1) == observed
        t = len(self.history)
        variance = t*(t-1)*(2*t+5)/18
        z = (self.score-self.score.sign())/variance**.5 if t >= 2 else torch.zeros_like(self.score)
        exclude = z > threshold
        exclude[torch.arange(len(observed)), observed] = False
        return keep, exclude


def _mix(images, targets, mix, generator):
    permutation = torch.randperm(len(images), generator=generator).to(images.device)
    return mix*images+(1-mix)*images[permutation], mix*targets+(1-mix)*targets[permutation]


def mixup(images, targets, alpha, rng, generator):
    draw = float(rng.beta(alpha, alpha))
    return _mix(images, targets, max(draw, 1-draw), generator)


def dss_update(model, optimizer, state, indices, weak, observed, keep, exclude,
               strong=None, plus_config=None, rng=None, generator=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(weak)
    probability = logits.softmax(1).detach()
    state.observe(indices, probability)
    supervised = ccs_loss(logits, observed, exclude.to(logits.device), keep.to(logits.device))
    total = supervised
    consistency = logits.new_zeros(())
    if strong is not None:
        # DSS+ supplement Eq.6 uses p^(1/T); official code uses p^T and
        # adjusted targets. This kernel follows the published raw-probability equation.
        target = sharpen(probability, plus_config["temperature"]).detach()
        mixed, mixed_target = mixup(strong, target, plus_config["mixup_beta"], rng, generator)
        consistency = -(mixed_target*model(mixed).log_softmax(1)).sum(1).mean()
        total = supervised+plus_config["consistency_weight"]*consistency
    total.backward()
    optimizer.step()
    return {"loss": float(total.detach()), "supervised_loss": float(supervised.detach()),
            "consistency_loss": float(consistency.detach()), "accepted_labels": int(keep.sum())}


def dividemix_update(model, peer, optimizer, view, labeled_indices, unlabeled_indices,
                     peer_probability, weak_transform, config, rng, generator,
                     ramp_fraction=1.):
    model.train(); peer.eval()
    device = next(model.parameters()).device
    x1, x2 = [batch_images(view, labeled_indices, weak_transform).to(device) for _ in range(2)]
    u1, u2 = [batch_images(view, unlabeled_indices, weak_transform).to(device) for _ in range(2)]
    observed = torch.as_tensor(view.observed_y[labeled_indices], dtype=torch.long, device=device)
    weight = torch.as_tensor(peer_probability[labeled_indices], dtype=torch.float32, device=device)
    with torch.no_grad():
        px = (model(x1).softmax(1)+model(x2).softmax(1))/2
        tx = refine_targets(observed, px, weight, config["temperature"])
        pu = (model(u1).softmax(1)+model(u2).softmax(1)+peer(u1).softmax(1)+peer(u2).softmax(1))/4
        tu = sharpen(pu, config["temperature"]).detach()
    inputs = torch.cat([x1, x2, u1, u2]); target = torch.cat([tx, tx, tu, tu])
    inputs, target = mixup(inputs, target, config["mixup_alpha"], rng, generator)
    logits = model(inputs)
    n = len(labeled_indices)*2
    lx, lu, reg = semisupervised_loss(logits[:n], target[:n], logits[n:], target[n:])
    loss = lx+config["lambda_u"]*ramp_fraction*lu+config["regularizer_weight"]*reg
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return {"loss": float(loss.detach()), "labeled_ce": float(lx.detach()), "unlabeled_mse": float(lu.detach()),
            "prior_regularization": float(reg.detach()), "labeled_examples": len(labeled_indices), "unlabeled_examples": len(unlabeled_indices)}
