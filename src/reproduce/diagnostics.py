"""Sampling moments and operation accounting from actual forward passes."""
import math
import numpy as np
import torch
from torch import nn
from .models.crisp import DendriticSSM
from .training import set_all_seeds
OP_TYPES = ('AC_event', 'MAC_dense', 'dend_MAC', 'mem_MAC', 'RNG', 'CMP', 'SIG', 'ADD')

class OpCounter:
    """Counts operations of one forward pass by type, split into the deterministic PREFIX (everything up to layer 1's
    potential: input/stem, layer-1 projection and dendrite) and the SUFFIX (layer-1 soma onwards) for CRISP.
    Linear / Conv2d: if the input is non-negative integer-valued (spikes, event counts) -> AC_event = sum(input) x fan-out;
    else MAC_dense = (#output elements) x (inputs per output). DendriticSSM: dend_MAC = 2 K per channel per step.
    CRISP soma: 'sample' -> RNG + CMP per neuron-step (logistic-noise threshold), 'graded' -> SIG, 'heaviside' -> CMP.
    LIF layer: mem_MAC + CMP per neuron-step; LI readout: mem_MAC per class-step. Pooling: ADD. BN: folded (0)."""

    def __init__(self, model, family, ds):
        self.family, self.mode, self.n = (family, None, 0)
        self.c = {'prefix': dict.fromkeys(OP_TYPES, 0.0), 'suffix': dict.fromkeys(OP_TYPES, 0.0)}
        self.spikes = {}
        self.h = []
        prefix = set()
        if hasattr(model, 'stem'):
            prefix |= set(model.stem.modules())
        if len(model.layers):
            prefix |= {model.layers[0].proj}
        if family == 'crisp' and len(model.layers):
            prefix |= {model.layers[0].dend}
        self.prefix = prefix
        for m in model.modules():
            part = 'prefix' if m in prefix else 'suffix'
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                self.h.append(m.register_forward_hook(self._lin(part)))
            elif isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                self.h.append(m.register_forward_hook(self._pool(part)))
            elif isinstance(m, DendriticSSM):
                self.h.append(m.register_forward_hook(self._dend(part)))
        for i, layer in enumerate(model.layers):
            self.h.append(layer.register_forward_hook(self._soma(i)))
        if family == 'lif' and isinstance(getattr(model, 'ro', None), nn.Module):
            self.h.append(model.ro.register_forward_hook(self._li()))

    def _add(self, part, k, v):
        self.c[part][k] += float(v)

    def _lin(self, part):

        def fn(mod, inp, out):
            x = inp[0].detach()
            if isinstance(mod, nn.Conv2d):
                assert tuple(mod.stride) == (1, 1) and mod.groups == 1, 'counter assumes stride 1, groups 1'
                fan_out = mod.out_channels * mod.kernel_size[0] * mod.kernel_size[1]
                per_out = mod.in_channels * mod.kernel_size[0] * mod.kernel_size[1]
            else:
                fan_out = mod.out_features
                per_out = mod.in_features
            if float(x.min()) >= 0 and torch.equal(x, torch.round(x)):
                self._add(part, 'AC_event', x.double().sum().item() * fan_out)
            else:
                self._add(part, 'MAC_dense', out.numel() * per_out)
        return fn

    def _pool(self, part):

        def fn(mod, inp, out):
            self._add(part, 'ADD', inp[0].numel())
        return fn

    def _dend(self, part):

        def fn(mod, inp, out):
            B, T, C = inp[0].shape
            self._add(part, 'dend_MAC', 2 * mod.K * B * T * C)
        return fn

    def _soma(self, i):

        def fn(mod, inp, out):
            s = out[0]
            n = s.numel()
            part = 'suffix' if self.family == 'crisp' or i > 0 else 'prefix'
            if self.family == 'lif':
                self._add(part, 'mem_MAC', n)
                self._add(part, 'CMP', n)
            elif self.mode == 'sample':
                self._add(part, 'RNG', n)
                self._add(part, 'CMP', n)
            elif self.mode == 'graded':
                self._add(part, 'SIG', n)
            else:
                self._add(part, 'CMP', n)
            self.spikes[i] = self.spikes.get(i, 0.0) + float(s.detach().double().sum())
            self.spikes[f'{i}_n'] = self.spikes.get(f'{i}_n', 0.0) + n
        return fn

    def _li(self):

        def fn(mod, inp, out):
            self._add('suffix', 'mem_MAC', out.numel())
        return fn

    def batch_done(self, B, T, C):
        self._add('suffix', 'ADD', B * T * C)
        self.n += B

    def remove(self):
        for h in self.h:
            h.remove()
        self.h = []

    def result(self):
        per = {p: {k: v / max(self.n, 1) for k, v in d.items()} for p, d in self.c.items()}
        per['total'] = {k: per['prefix'][k] + per['suffix'][k] for k in OP_TYPES}
        L = len([k for k in self.spikes if isinstance(k, int)])
        per['rate_per_layer'] = [self.spikes[i] / max(self.spikes[f'{i}_n'], 1) for i in range(L)]
        per['spikes_per_inference'] = [self.spikes[i] / max(self.n, 1) for i in range(L)]
        return per

@torch.no_grad()
def count_ops(model, family, ds, loader, mode=None, beta=None, seed=0):
    """Operations per inference, averaged over the loader (one pass / one draw per input)."""
    model.eval()
    oc = OpCounter(model, family, ds)
    oc.mode = mode
    set_all_seeds(seed)
    try:
        for xb, yb in loader:
            out = model(xb, mode, beta) if family == 'crisp' else model(xb)
            T = xb.shape[1]
            oc.batch_done(xb.shape[0], T, out[0].shape[1])
    finally:
        oc.remove()
    return oc.result()

def counter_selftest(device="cpu"):
    """Hand-computed counts on tiny modules (raises on mismatch)."""
    torch.manual_seed(0)
    lin = nn.Linear(3, 2).to(device)
    conv = nn.Conv2d(1, 2, 3, padding=1, bias=False).to(device)
    dend = DendriticSSM(4, 3, zoh_b=True, ct_exact=True).to(device)

    class Tiny(nn.Module):

        def __init__(s):
            super().__init__()
            s.stem = nn.Sequential(conv)
            s.l = lin
            s.d = dend
            s.layers = []
    t = Tiny()
    oc = OpCounter(t, 'lif', 'shd')
    with torch.no_grad():
        lin(torch.tensor([[1.0, 0.0, 2.0]], device=device))
        lin(torch.tensor([[0.5, 0.0, 1.0]], device=device))
        conv(torch.rand(1, 1, 4, 4, device=device) + 0.1)
        conv(torch.tensor([[[[0.0, 2.0], [1.0, 0.0]]]], device=device))
        dend(torch.randn(1, 5, 4, device=device))
    oc.remove()
    c = {k: oc.c['prefix'][k] + oc.c['suffix'][k] for k in OP_TYPES}
    exp = {'AC_event': 6 + 54, 'MAC_dense': 6 + 288, 'dend_MAC': 120}
    got = {k: c[k] for k in exp}
    if got != {k: float(v) for k, v in exp.items()}:
        raise RuntimeError(f'[GATE] operation counter: {got} != {exp}')
    return exp

@torch.no_grad()
def sampling_statistics(model, loader, *, family='crisp', beta=5.):
    """Accumulate probabilities, Bernoulli variance and saturation on mean-field paths."""
    model.eval()
    device = next(model.parameters()).device
    accumulators = {}
    if family == 'crisp':
        for x, _ in loader:
            state = x.to(device).float()
            for index, layer in enumerate(model.layers):
                state = layer(state, 'graded', beta)[0]
                q = state.double()
                row = accumulators.setdefault(f'layer_{index + 1}', [0., 0., 0, 0])
                row[0] += float(q.sum())
                row[1] += float((q * (1 - q)).sum())
                row[2] += q.numel()
                row[3] += int(((q < .05) | (q > .95)).sum())
    elif family == 'pspikessm':
        old_mode, old_probe = model.sampling_mode, model.sampler_statistics
        model.sampling_mode = 'meanfield'
        model.sampler_statistics = {}
        try:
            for x, _ in loader:
                model(x.to(device).float())
            for (block, site), values in model.sampler_statistics.items():
                name = 'state_space_input' if site == 0 else 'mixer_input'
                accumulators[f'block_{block + 1}_{name}'] = values
        finally:
            model.sampler_statistics = old_probe
            model.sampling_mode = old_mode
    else:
        raise ValueError(f'No sampling moments for family {family}')
    return [dict(sampler=name, mean_probability=sp / n, mean_bernoulli_variance=sv / n,
                 saturated_fraction=nsat / n, unit_steps=n)
            for name, (sp, sv, n, nsat) in sorted(accumulators.items())]


def sampling_location_evaluation(model, loader, *, draws=(1, 5), eval_seed=2026):
    """Hold trained weights fixed while selecting P-SpikeSSM sampling sites."""
    from .models.state_space import evaluate
    original = (model.sampling_mode, model.sample_sites)
    result = {}
    try:
        model.sampling_mode = 'meanfield'
        reference = evaluate(model, loader)['acc']
        result['meanfield'] = {'acc': reference}
        for label, sites in (('both', (True, True)), ('state_space_input', (True, False)),
                             ('mixer_input', (False, True))):
            model.sample_sites = sites
            model.sampling_mode = 'meanfield'
            accuracy = evaluate(model, loader)['acc']
            if accuracy != reference:
                raise RuntimeError('Mean-field predictions changed with the sampling-site switch')
            model.sampling_mode = 'sample'
            for count in draws:
                set_all_seeds(eval_seed + count)
                result[f'{label}_{count}_draws'] = evaluate(model, loader, n_samples=count)
    finally:
        model.sampling_mode, model.sample_sites = original
    return result


def paired_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('Cannot summarize an empty sample')
    return {'mean': float(values.mean()),
            'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else None,
            'n': len(values)}


class SomaProbe:
    """Forward hooks: record each hidden layer's BN output V (pre-sigmoid) and its output s."""

    def __init__(self, model, layers=None):
        self.V, self.S, self.h = ({}, {}, [])
        for l, layer in enumerate(model.layers):
            if layers is not None and l not in layers:
                continue
            self.h.append(layer.norm.register_forward_hook(self._vh(l)))
            self.h.append(layer.register_forward_hook(self._sh(l)))

    def _vh(self, l):

        def fn(mod, inp, out):
            self.V[l] = out
        return fn

    def _sh(self, l):

        def fn(mod, inp, out):
            self.S[l] = out[0]
        return fn

    def remove(self):
        for h in self.h:
            h.remove()
        self.h = []

@torch.no_grad()
def linear_maps(model, T):
    """Per-layer linear maps of the MF network at this dt and length T: dendrite impulse response
    (reference module on a unit impulse) as a causal Toeplitz operator, BN gain, weights; readout Jacobian A."""
    device = next(model.parameters()).device
    consts = []
    tt = torch.arange(T, device=device)
    D = tt[:, None] - tt[None, :]
    for layer in model.layers:
        H = layer.proj.out_features
        imp = torch.zeros(1, T, H, device=device)
        imp[0, 0, :] = 1.0
        hk = layer.dend(imp)[0]
        Hm = (hk[D.clamp(min=0)] * (D >= 0).unsqueeze(-1).to(hk.dtype)).permute(2, 0, 1).contiguous()
        kap = (layer.norm_apply(torch.ones(1, 1, H, device=device)) - layer.norm_apply(torch.zeros(1, 1, H, device=device))).view(H)
        consts.append({'W': layer.proj.weight.detach(), 'Hm': Hm, 'kap': kap, 'theta': layer.theta.detach()})
    HL = model.layers[-1].proj.out_features
    C = model.readout.out_features
    with torch.enable_grad():
        sL = torch.zeros(1, T, HL, device=device, requires_grad=True)
        z = model.pool(model.ro(model.readout(sL)))
        A = torch.stack([torch.autograd.grad(z[0, c], sL, retain_graph=c < C - 1)[0][0] for c in range(C)])
    return (consts, A.detach(), z.detach()[0])

def upstream_gradients(gL, p, consts, beta):
    """gL: dz/ds_L for a block of classes, [1 or B, c, T, H_L]. Returns [dz/ds_l for l=0..L-1] at the MF point.
    V_l = kap * (h * (W s_{l-1} + b)) + const  ->  dz/ds_{l-1} = W^T kap H^T (dz/ds_l * beta p(1-p))."""
    L = len(p)
    gs = [None] * L
    gs[L - 1] = gL
    for l in range(L - 1, 0, -1):
        c = consts[l]
        w = gs[l] * (beta * p[l] * (1 - p[l])).unsqueeze(1)
        dzdU = torch.einsum('bcth,hts->bcsh', w, c['Hm']) * c['kap']
        gs[l - 1] = dzdU @ c['W']
    return gs

@torch.no_grad()
def moment_approximated_logits(model, V_L, v_prev, consts, beta):
    """Moment-propagated mean logit. Var(V_L) exact given independent spikes below (V_L affine in them);
    E[sigmoid] by the probit approximation."""
    c = consts[-1]
    q = v_prev @ (c['W'] ** 2).t()
    varV = torch.einsum('bsh,hts->bth', q, c['Hm'] ** 2) * c['kap'] ** 2
    m = torch.sigmoid(beta * (V_L - c['theta']) / torch.sqrt(1 + math.pi * beta ** 2 * varV / 8))
    return (model.pool(model.ro(model.readout(m))), varV)

def iter_sampled_logits(model, p1c, K, R, beta):
    """Deployment: every layer sampled. Layer 1 input is deterministic, so its p is computed once and
    sampled R times per forward (bitwise identical to model(x,'sample') for R=1; gated).
    Yields (z [R,B,C], v_real [R,B,T,H_L]) where v_real = p(1-p) of the REALIZED last-layer p."""
    if K < 1 or R < 1 or K % R:
        raise ValueError('Draw count must be positive and divisible by the draw chunk')
    B = p1c.shape[0]
    lastL = model.layers[-1]
    cap = {}
    h = lastL.norm.register_forward_hook(lambda m, i, o: cap.__setitem__('V', o))
    try:
        for _ in range(K // R):
            s = torch.bernoulli(p1c.repeat(R, 1, 1))
            for layer in model.layers[1:]:
                s, _, _ = layer(s, 'sample', beta)
            z = model.pool(model.ro(model.readout(s))).view(R, B, -1)
            pr = torch.sigmoid(beta * (cap['V'].view(s.shape) - lastL.theta)).clamp(1e-06, 1 - 1e-06)
            yield (z, (pr * (1 - pr)).view(R, B, *s.shape[1:]))
    finally:
        h.remove()

def grouped_mean_logit_decisions(Z, n):
    """Z [M, B, C] draws -> [B, M // n] int8: decision of each consecutive group of n draws (argmax of its mean logit)."""
    M, B, C = Z.shape
    m = M // n
    return Z[:m * n].view(m, n, B, C).mean(1).argmax(2).t().to(torch.int8)
