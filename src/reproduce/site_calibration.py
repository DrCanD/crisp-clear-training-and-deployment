"""Calibration-only site rules and single-pass logit correction.

All arrays use example order shared by labels, participant IDs and teacher
logits. Sampled logits have shape (draws, examples, classes) for each site.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import ndtr

from .site_training import draw_logits


def _participants(labels, people, size):
    def to_numpy(values):
        return values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else np.asarray(values)

    labels, people = to_numpy(labels), to_numpy(people)
    if not size or labels.shape != (size,) or people.shape != (size,):
        raise ValueError('each calibration example must have one label and one participant ID')
    return labels, people


def projected_risks(model, x, y, *, strength=1., limit=512):
    """Return signed and induced-error-only predicted costs in percentage points.

    Every calibration example contributes to the denominator, including errors.
    The default frozen subset is the first 512 indices of the seed-597 permutation.
    Set limit=None to use all examples in input order.
    """
    if model.family != 'pspike' or not 0 < strength <= 1 or (limit is not None and limit < 1):
        raise ValueError('expected a P-SpikeSSM, positive amplitude and positive subset size')
    if limit is not None:
        indices = np.random.RandomState(597).permutation(len(y))[:limit]
        x, y = x[indices], y[indices]
    model.eval()
    with torch.enable_grad(), model.sampling(capture=True) as gates:
        logits = model(x)
        rivals = logits.detach().clone().scatter_(1, y[:, None], -float('inf'))
        margin = (logits.gather(1, y[:, None]) -
                  logits.gather(1, rivals.argmax(1, keepdim=True))).flatten()
        gradients = torch.autograd.grad(margin.sum(), [p for _, p in gates])
    correct = (logits.argmax(1) == y).detach().cpu().numpy()
    distance = margin.detach().double().cpu().numpy()
    risks = {'signed': [], 'loss_only': []}
    for site in (0, 1):
        variance = np.zeros(len(y))
        for (position, p), grad in zip(gates, gradients):
            if position == site:
                p = p.detach()
                variance += (grad.square() * (p * (1 - p))).flatten(1).sum(1).double().cpu().numpy()
        moving = variance > 0
        positive = correct.astype(float)
        positive[moving] = ndtr(distance[moving] / (strength * np.sqrt(variance[moving])))
        lost = np.where(correct, 1 - positive, 0.)
        rescued = np.where(correct, 0., positive)
        risks['signed'].append(100 * (lost - rescued).mean())
        risks['loss_only'].append(100 * lost.mean())
    return risks


def higher_risk_site(costs):
    """Frozen ranking convention for projected and absolute-margin risk scores."""
    return int(costs[1] > costs[0] + 1e-10)


def absolute_margin_costs(teacher, sampled):
    """Mean absolute change in the top-two difference for each site's half-strength draws."""
    z = torch.as_tensor(teacher)
    pair = z.argsort(dim=1, descending=True, stable=True)[:, :2].cpu().numpy()
    def margin(logits):
        selected = np.take_along_axis(logits, pair, axis=-1)
        return selected[:, 0] - selected[:, 1]
    clean = margin(teacher)
    return [float(np.mean([np.abs(margin(draw) - clean) for draw in draws], dtype=np.float64))
            for draws in sampled]


def rank_sampling_risks(model, x, y):
    """Compare the two frozen risk rules and signed risk on the same calibration subset.

    Returns predicted more costly sites (early=0, late=1) and both site scores.
    This is the original single-batch protocol, including top-two tie ordering.
    """
    indices = np.random.RandomState(597).permutation(len(y))[:512]
    x, y = x[indices], y[indices]
    scores = projected_risks(model, x, y, limit=None)
    with torch.inference_mode(), model.sampling():
        teacher = model(x)
        pair = teacher.argsort(dim=1, descending=True)[:, :2]
        def margin(logits):
            top = logits.gather(1, pair)
            return top[:, 0] - top[:, 1]
        clean = margin(teacher)
        scores['margin'] = []
        for site in (0, 1):
            values = []
            with model.sampling(site, .5):
                for seed in range(234500, 234508):
                    torch.manual_seed(seed)
                    values.append((margin(model(x)) - clean).abs().mean().item())
            scores['margin'].append(float(np.mean(values)))
    return {name: higher_risk_site(cost) for name, cost in scores.items()}, scores


def calibrate_sites(model, x, y, people):
    """Freeze the participant-balanced accuracy choice and the half-strength comparator."""
    y, people = _participants(y, people, len(x))
    teacher = draw_logits(model, x)[0]
    empirical = []
    margin_cost = []
    for site in (0, 1):
        draws = draw_logits(model, x, site, draws=8)
        hits = draws.argmax(-1) == y
        empirical.append(np.mean([hits[:, people == p].mean() for p in np.unique(people)]))
        mild = draw_logits(model, x, site, draws=8, strength=.5, first_seed=234500)
        margin_cost.extend(absolute_margin_costs(teacher, [mild]))
    return {'empirical': int(empirical[1] > empirical[0] + 1e-12),
            'margin': int(margin_cost[1] < margin_cost[0] - 1e-12)}


def _center(logits):
    values = np.asarray(logits, dtype=np.float64)
    return values - values.mean(axis=-1, keepdims=True)


def fit_readouts(samples, teacher, labels, people):
    """Fit raw, mean-shift, teacher-affine and supervised ridge maps at one site."""
    x, z = _center(samples), _center(teacher)
    if x.ndim != 3 or not x.shape[0] or x.shape[1:] != z.shape or z.shape[1] < 2:
        raise ValueError('expected aligned sampled and deterministic logits with at least two classes')
    classes = x.shape[-1]
    labels, people = _participants(labels, people, len(z))
    if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels >= classes)):
        raise ValueError('labels must be integer class indices')
    groups, inverse, counts = np.unique(people, return_inverse=True, return_counts=True)
    weights = 1 / (len(groups) * counts[inverse])
    per_example = x.mean(axis=0)
    mean = weights @ per_example
    centered = x - mean
    covariance = np.einsum('rni,n,rnj->ij', centered, weights / len(x), centered, optimize=True)
    identity = np.eye(classes)
    ridge = max(.1 * covariance.trace() / classes, 1e-8)
    maps = {'raw': (identity, np.zeros(classes)), 'shift': (identity, weights @ z - mean)}
    for name, target in [('affine', z), ('supervised', identity[labels])]:
        target_mean = weights @ target
        cross = (per_example - mean).T @ (weights[:, None] * (target - target_mean))
        prior = identity if name == 'affine' else np.zeros_like(identity)
        # Shrink the teacher map toward identity and the label map toward zero.
        matrix = prior + np.linalg.solve(covariance + ridge * identity,
                                         cross - (covariance if name == 'affine' else 0.))
        maps[name] = matrix, target_mean - mean @ matrix
    return maps


@dataclass
class CalibratedReadout:
    site: int
    matrix: np.ndarray
    intercept: np.ndarray

    def predict(self, sampled_logits):
        """Apply the frozen map to each draw from the selected site separately."""
        return (_center(sampled_logits) @ self.matrix + self.intercept).argmax(axis=-1)


def select_readouts(sampled, teacher, labels, people):
    """Select by held-out calibration participant, then refit on all calibration data."""
    labels, people = _participants(labels, people, len(teacher))
    groups = np.unique(people)
    if len(groups) < 2 or len(sampled) != 2:
        raise ValueError('site selection requires two sites and at least two calibration participants')
    scores = {}
    full = []
    for site, x in enumerate(sampled):
        fold_scores = {}
        for person in groups:
            held = people == person
            maps = fit_readouts(x[:, ~held], teacher[~held], labels[~held], people[~held])
            for name, (matrix, intercept) in maps.items():
                prediction = CalibratedReadout(site, matrix, intercept).predict(x[:, held])
                fold_scores.setdefault(name, []).append((prediction == labels[held]).mean())
        for name, values in fold_scores.items():
            scores.setdefault(name, []).append(np.mean(values))
        full.append(fit_readouts(x, teacher, labels, people))
    selected = {}
    for name, (early, late) in scores.items():
        site = int(late > early + 1e-12)
        selected[name] = CalibratedReadout(site, *full[site][name])
    return selected, scores


def crossed_interval(task_differences, *, seed=126911, draws=30000):
    """Equal-task mean and paired 95% percentile interval for seed-by-person contrasts.

    Supply one matrix of within-cell accuracy differences per task, in matched
    seed order. Resample seeds jointly across tasks and people within each task.
    """
    tasks = [np.asarray(matrix, dtype=float) for matrix in task_differences]
    if not tasks or draws < 1 or any(m.ndim != 2 or min(m.shape) == 0 for m in tasks):
        raise ValueError('expected nonempty seed-by-person matrices and positive draws')
    seeds = len(tasks[0])
    if any(len(matrix) != seeds for matrix in tasks):
        raise ValueError('all tasks must share the same seed order')
    rng = np.random.default_rng(seed)
    def weights(n):
        return rng.multinomial(n, np.full(n, 1 / n), size=draws) / n
    seed_weights = weights(seeds)
    totals = np.zeros(draws)
    for matrix in tasks:
        totals += ((seed_weights @ matrix) * weights(matrix.shape[1])).sum(axis=1)
    return np.mean([m.mean() for m in tasks]), np.percentile(totals / len(tasks), [2.5, 97.5])
