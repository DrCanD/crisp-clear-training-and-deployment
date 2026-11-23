"""Compact mean-field classifiers with a choice of sampled deployment site.

Site 0 precedes the P-SpikeSSM kernel and site 1 precedes its mixer, in
both blocks. For CRISP and LIF the site is the hidden layer index.
Inputs are tensors of shape (examples, time steps, features).
"""
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import crisp, lif, state_space


class SiteClassifier(nn.Module):
    def __init__(self, family, classes, *, width=32, features=128, steps=8,
                 source='external_code/pspikessm', device='cpu'):
        super().__init__()
        self.family = family
        self._site = None
        self._strength = 1.
        self._probes = None
        if family == 'pspike':
            self.net = state_space.build_model(source, n_in=features, n_classes=classes,
                                               timesteps=steps, hidden=width, device=device)
            for block in self.net.backbone.layers:
                def sample(y, uniform, block=block):
                    site = block._sampling_call
                    block._sampling_call = (site + 1) % 2
                    return self._gate(y.clamp(0, 1), site, uniform)
                block.stochSpikeGen = sample
        elif family == 'crisp':
            config = crisp.Config(n_in=features, n_hidden=width, k_modes=4,
                                  n_classes=classes, hidden_dropout=0.)
            self.net = crisp.DendStochAudioNet(config)
        elif family == 'lif':
            config = lif.Config(n_in=features, n_hidden=width, hidden_dropout=0.)
            self.net = lif.LIFNet(config, classes)
        else:
            raise ValueError('family must be pspike, crisp or lif')
        self.to(device)

    @contextmanager
    def sampling(self, site=None, strength=1., *, capture=False):
        """Scope a perturbation to one site, or capture mean-field probabilities."""
        if site not in (None, 0, 1) or not 0 <= strength <= 1:
            raise ValueError('expected site None, 0 or 1 and strength in [0, 1]')
        previous = self._site, self._strength, self._probes
        self._site, self._strength = site, strength
        self._probes = [] if capture else None
        try:
            yield self._probes
        finally:
            self._site, self._strength, self._probes = previous

    def _gate(self, p, site, uniform=None):
        if self._probes is not None:
            self._probes.append((site, p))
        if self._site != site:
            return p
        if uniform is None:
            uniform = torch.rand_like(p)
        return p + self._strength * ((uniform < p).to(p.dtype) - p)

    def forward(self, x):
        if self.family == 'pspike':
            return self.net(x)
        for site, layer in enumerate(self.net.layers):
            if self.family == 'crisp':
                current = layer.norm_apply(layer.dend(layer.proj(x)))
                x = self._gate(torch.sigmoid(current - layer.theta), site)
            else:
                current = layer.norm_apply(layer.proj(x))
                v = torch.zeros_like(current[:, 0])
                spikes = torch.zeros_like(v)
                alpha = layer.alpha()
                sequence = []
                for step in current.unbind(dim=1):
                    v = alpha * v * (1 - spikes) + step
                    spikes = self._gate(torch.sigmoid(v - layer.theta), site)
                    sequence.append(spikes)
                x = torch.stack(sequence, dim=1)
        if self.family == 'crisp':
            x = self.net.readout(x)
        return self.net.pool(self.net.ro(x))


def fit_site_model(model, x, y, seed, *, penalty='none', weight=1.,
                   continuation=False, epochs=None):
    """Fit in place. For matched continuations, load the same conventional fit first.

    Set the training seed before constructing each independently initialized
    model. Pass identical initial weights and seed when comparing penalties.
    """
    target = {'none': (), 'early': (0,), 'late': (1,), 'split': (0, 1)}
    if penalty not in target or weight < 0:
        raise ValueError('unknown variance penalty or negative weight')
    epochs = (6 if continuation else 24) if epochs is None else epochs
    if epochs < 1 or len(y) == 0 or len(x) != len(y):
        raise ValueError('expected positive epochs and aligned nonempty training inputs')
    order = torch.Generator().manual_seed(seed + (9000 if continuation else 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005 if continuation else .003,
                                  weight_decay=.0001)
    if continuation:
        model.eval()
        with model.sampling(), torch.no_grad():
            teacher = torch.cat([(model(batch) / 2).softmax(1) for batch in x.split(256)])
    model.train()
    for _ in range(epochs):
        for index in torch.randperm(len(y), generator=order).split(128):
            optimizer.zero_grad(set_to_none=True)
            with model.sampling(capture=bool(target[penalty])) as gates:
                logits = model(x[index])
            loss = F.cross_entropy(logits, y[index])
            if target[penalty]:
                terms = [(p * (1 - p)).mean() for site, p in gates if site in target[penalty]]
                loss = loss + weight * torch.stack(terms).mean()
            if continuation:
                loss = loss + 4 * F.kl_div((logits / 2).log_softmax(1), teacher[index],
                                           reduction='batchmean')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    model.eval()


@torch.inference_mode()
def draw_logits(model, x, site=None, *, draws=1, strength=1., first_seed=186000, batch_size=256):
    """One independent pass per draw, without voting. Use common seeds across sites."""
    if draws < 1 or batch_size < 1 or len(x) == 0:
        raise ValueError('expected positive draw count, batch size and number of inputs')
    model.eval()
    result = []
    with model.sampling(site, strength):
        for seed in range(first_seed, first_seed + draws):
            torch.manual_seed(seed)
            result.append(torch.cat([model(batch) for batch in x.split(batch_size)]).cpu().numpy())
    return np.stack(result)
