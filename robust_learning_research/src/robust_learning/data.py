"""Independent frozen corruption, clean held-out splits and sealed toy oracles."""
from dataclasses import dataclass

import numpy as np


def immutable(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TrainingView:
    x: np.ndarray
    observed_y: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class CleanSet:
    x: np.ndarray
    y: np.ndarray
    ids: np.ndarray


@dataclass(frozen=True)
class Oracle:
    true_y: np.ndarray
    flipped: np.ndarray
    ambiguity: np.ndarray
    flip_probability: np.ndarray


@dataclass(frozen=True)
class Problem:
    train: TrainingView
    validation: CleanSet
    test: CleanSet
    oracle: Oracle


def make_problem(data_seed, noise_seed, counts=(600, 160, 60), noise_rate=.3,
                 noise_kind="instance", task_seed=300, std=1.15):
    if len(counts) != 3 or min(counts) < 1 or not 0 <= noise_rate <= 1 or std <= 0:
        raise ValueError("invalid counts/rate/scale")
    if noise_kind not in {"symmetric", "class_conditional", "instance"}:
        raise ValueError("unsupported noise family")
    task = np.random.default_rng(task_seed)
    angle = task.uniform(-np.pi, np.pi)
    q = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centers = np.array([[-1.4, 0.], [1.4, 0.], [0., 1.6]])@q.T
    feature = np.random.default_rng(np.random.SeedSequence([data_seed, 401]))
    true_y = np.concatenate([np.full(n, c, dtype=np.int64) for c, n in enumerate(counts)])
    raw_x = centers[true_y]+feature.normal(0, std, (len(true_y), 2))
    prior = np.asarray(counts)/sum(counts)
    logp = -np.square(raw_x[:, None, :]-centers[None, :, :]).sum(-1)/(2*std**2)+np.log(prior)
    posterior = np.exp(logp-logp.max(1, keepdims=True))
    posterior /= posterior.sum(1, keepdims=True)
    ambiguity = 1-posterior[np.arange(len(true_y)), true_y]
    probability = noise_rate*(.25+.75*ambiguity) if noise_kind == "instance" else np.full(len(true_y), noise_rate)
    noise = np.random.default_rng(np.random.SeedSequence([noise_seed, 991]))
    flipped = noise.random(len(true_y)) < probability
    observed_y = true_y.copy()
    offsets = np.ones(flipped.sum(), dtype=np.int64) if noise_kind == "class_conditional" else noise.integers(1, 3, flipped.sum())
    observed_y[flipped] = (observed_y[flipped]+offsets)%3
    order = np.random.default_rng(np.random.SeedSequence([data_seed, 551])).permutation(len(true_y))
    # Normalization uses training FEATURES only, never labels/validation/test.
    mean, scale = raw_x.mean(0), raw_x.std(0).clip(1e-6)
    train = TrainingView(immutable(((raw_x[order]-mean)/scale).astype(np.float32)),
        immutable(observed_y[order]), immutable(np.arange(len(true_y))[order]))
    oracle = Oracle(*(immutable(a[order]) for a in (true_y, flipped, ambiguity, probability)))
    clean_sets = []
    for split in (1, 2):
        generator = np.random.default_rng(np.random.SeedSequence([data_seed, task_seed, 801+split]))
        y = np.repeat(np.arange(3), 120)
        x = centers[y]+generator.normal(0, std, (len(y), 2))
        clean_sets.append(CleanSet(immutable(((x-mean)/scale).astype(np.float32)), immutable(y),
            immutable(np.arange(len(y))+split*10_000_000)))
    return Problem(train, *clean_sets, oracle)
