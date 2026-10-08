"""CLEAR: closed-form local eligibility and adjoint rule.

The arithmetic is retained from the archived local-learning implementation.
Within-layer derivatives are exact when supplied with the exact incoming
learning signal. Truncated and random cross-layer feedback are approximations.
The reverse-time adjoint is an offline operation, not an online learning rule.
Batch-normalization statistics must be calibrated on training data and frozen.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models.crisp import _shift, parallel_scan


@dataclass(frozen=True)
class LearningConfig:
    reg_coef: float = 0.02
    rate_target: float = 0.10


def adjoint_dend_K(g, a, mix):
    """Reverse-time scan, keeping K dimension. g:[B,T,C] -> dE/d(scan_input):[B,T,C,K]."""
    return torch.flip(parallel_scan(a, torch.flip(g.unsqueeze(-1) * mix, [1])), [1])

def dend_forward(d, xin):
    """Same ops, same order as the reference DendriticSSM.forward (ct_exact); returns the cache for the rule."""
    B, T, C = xin.shape
    xd = xin.unsqueeze(-1).expand(B, T, C, d.K)
    lam = -torch.exp(d.raw_lam); a = torch.exp(lam * d.dt); Bd = torch.expm1(lam * d.dt) / lam
    h = d._scan(a, xd * Bd)
    return (h * d.mix).sum(-1), {'xd': xd, 'h': h, 'a': a, 'Bd': Bd, 'lam': lam, 'dt': d.dt, 'mix': d.mix}

def dend_grads(g_y, c):
    """dE/d(raw_lam), dE/d(mix), dE/d(input) of a ct_exact dendrite from g_y = dE/d(output).
    The source eligibility formulas are applied independently to each layer."""
    a, Bd, lam, dt, h, xd, mix = c['a'], c['Bd'], c['lam'], c['dt'], c['h'], c['xd'], c['mix']
    gmix = torch.einsum('btc,btck->ck', g_y, h)
    adjK = adjoint_dend_K(g_y, a, mix)
    gBd = torch.einsum('bthk,bthk->hk', adjK, xd)
    u = parallel_scan(a, _shift(h, 1, 0.0))
    ga = torch.einsum('btc,btck->ck', g_y, u) * mix
    dBd_dlam = (dt * a * lam - torch.expm1(lam * dt)) / (lam * lam)
    graw = (ga * (dt * a) + gBd * dBd_dlam) * lam
    gx = (adjK * Bd).sum(-1)
    return graw, gmix, gx

def lag_kernel(c, k):
    """h(tau) = sum_k mix_k Bd_k a_k^tau for tau < k: the dendrite's input->output impulse response."""
    tau = torch.arange(k, device=c['a'].device, dtype=c['a'].dtype)
    return (c['mix'].unsqueeze(-1) * c['Bd'].unsqueeze(-1) * c['a'].unsqueeze(-1) ** tau).sum(1)

def truncated_adjoint(g_y, c, k):
    """sum_{tau<k} h(tau) g_y(t+tau): the exact dE/d(input) when k >= T, CLEAR's instantaneous signal when k = 1."""
    B, T, H = g_y.shape
    if k < 1:
        raise ValueError("The adjoint lag count must be positive.")
    k = min(k, T); ker = lag_kernel(c, k); out = torch.zeros_like(g_y)
    for tau in range(k):
        out[:, :T - tau] += ker[:, tau] * g_y[:, tau:]
    return out

class ClearNet(nn.Module):
    """Original CRISP parameters with frozen batch-normalization statistics.

    The explicit forward is differentiated by the local rule. Fixed random
    feedback supports the approximate comparison. Evaluation and checkpoints
    use the original network (self.net) without changing its parameter names.
    """
    def __init__(self, net, seed=0, learning_config=None):
        super().__init__()
        if net.readout_pool != 'mean':
            raise ValueError("The local rule assumes the mean-pool readout.")
        if not net.layers or any(not layer.dend.ct_exact for layer in net.layers) or not net.ro.ct_exact:
            raise ValueError("CLEAR requires continuous-time exact dendrites.")
        self.net = net; self.q = float(net.layers[0].drop.p)
        self.ccfg = learning_config or LearningConfig()
        if any(not isinstance(layer.norm, nn.BatchNorm1d) for layer in net.layers):
            raise ValueError("CLEAR requires batch normalization with frozen statistics.")
        g = torch.Generator().manual_seed(seed + 7919)
        for i, layer in enumerate(net.layers):
            Ho, Hi = layer.proj.weight.shape
            self.register_buffer(f'Bfb{i}', (torch.randn(Ho, Hi, generator=g) / math.sqrt(Ho)).to(layer.proj.weight))

    def forward_explicit(self, x, beta, masks=None, probes=None):
        """masks: per-layer dropout masks (already / (1-q)) or None (eval). probes: zero tensors added to each
        layer output (for exact learning signals from autograd)."""
        s = x; caches = []
        for l, layer in enumerate(self.net.layers):
            xin = F.linear(s, layer.proj.weight, layer.proj.bias)
            y, c = dend_forward(layer.dend, xin)
            B, T, H = y.shape; n = layer.norm
            V = F.batch_norm(y.reshape(B * T, H), n.running_mean, n.running_var, n.weight, n.bias, False, 0.0, n.eps).reshape(B, T, H)
            p = torch.sigmoid(beta * (V - layer.theta))
            so = p * masks[l] if masks is not None else p
            if probes is not None: so = so + probes[l]
            c.update({'s_in': s, 'y': y, 'p': p, 'm': None if masks is None else masks[l]})
            caches.append(c); s = so
        u = F.linear(s, self.net.readout.weight, self.net.readout.bias)
        yro, cro = dend_forward(self.net.ro, u); cro['s_in'] = s
        return self.net.pool(yro), caches, cro

    def make_masks(self, x, beta):
        if self.q <= 0: return None
        B, T = x.shape[:2]
        return [(torch.rand(B, T, layer.proj.out_features, device=x.device, dtype=x.dtype) >= self.q).to(x.dtype) / (1 - self.q)
                for layer in self.net.layers]

    def loss_of(self, z, y, caches):
        ccfg = self.ccfg
        mr = torch.stack([c['p'].mean() for c in caches]).mean()
        return F.cross_entropy(z, y) + ccfg.reg_coef * (mr - ccfg.rate_target) ** 2

    @torch.no_grad()
    def local_grads(self, z, y, caches, cro, beta, rule, Lsig=None):
        """rule: 'exact' (reverse adjoint across layers == BPTT), 'clear_sym', 'clear_random', 'clear_sym_k<k>'.
        Lsig: optional dict l -> exact dE/ds_l supplied from autograd for a
        task loss with no explicit intermediate-rate penalty. The rate penalty
        is added internally only when Lsig is absent."""
        if rule not in {"exact", "clear_sym", "clear_random"}:
            if not rule.startswith("clear_sym_k") or not rule.removeprefix("clear_sym_k").isdigit() or int(rule.removeprefix("clear_sym_k")) < 1:
                raise ValueError("Use exact, clear_sym, clear_random, or clear_sym_k followed by a positive lag count.")
        ccfg = self.ccfg; B = z.shape[0]; L = len(caches); names = {}
        dl = F.softmax(z, 1); dl[torch.arange(B, device=z.device), y] -= 1.0; dl = dl / B
        T = cro['h'].shape[1]
        graw, gmix, gx = dend_grads((dl / T).unsqueeze(1).expand(B, T, dl.shape[1]), cro)
        names['ro.raw_lam'] = graw; names['ro.mix'] = gmix
        names['readout.weight'] = torch.einsum('btc,bth->ch', gx, cro['s_in']); names['readout.bias'] = gx.sum((0, 1))
        Ls = gx @ self.net.readout.weight
        mr = float(torch.stack([c['p'].mean() for c in caches]).mean())
        dreg = 2 * ccfg.reg_coef * (mr - ccfg.rate_target) / L
        k_lag = int(rule.split('_k')[1]) if '_k' in rule else 1
        for l in range(L - 1, -1, -1):
            c = caches[l]; layer = self.net.layers[l]; n = layer.norm
            if Lsig is not None: Ls = Lsig[l]
            dEdp = Ls * c['m'] if c['m'] is not None else Ls
            if Lsig is None: dEdp = dEdp + dreg / c['p'].numel()
            p = c['p']; gV = dEdp * beta * p * (1 - p)
            inv = torch.rsqrt(n.running_var + n.eps)
            names[f'layers.{l}.theta'] = -gV.sum((0, 1))
            names[f'layers.{l}.norm.weight'] = (gV * (c['y'] - n.running_mean) * inv).sum((0, 1))
            names[f'layers.{l}.norm.bias'] = gV.sum((0, 1))
            gy = gV * (n.weight * inv)
            graw, gmix, gx = dend_grads(gy, c)
            names[f'layers.{l}.dend.raw_lam'] = graw; names[f'layers.{l}.dend.mix'] = gmix
            names[f'layers.{l}.proj.weight'] = torch.einsum('bth,bti->hi', gx, c['s_in']); names[f'layers.{l}.proj.bias'] = gx.sum((0, 1))
            if l > 0 and Lsig is None:
                if rule == 'exact':
                    Ls = gx @ layer.proj.weight
                else:
                    Wfb = layer.proj.weight if 'sym' in rule else getattr(self, f'Bfb{l}')
                    Ls = truncated_adjoint(gy, c, k_lag) @ Wfb
        return names

def gate_explicit(cn, xb, beta):
    cn.net.eval()
    with torch.no_grad():
        z_ref, _, _ = cn.net(xb, "graded", beta)
        z_ex, _, _ = cn.forward_explicit(xb, beta)
    if not torch.equal(z_ref, z_ex):
        raise RuntimeError(f"[GATE] explicit forward != reference forward (max diff {float((z_ref-z_ex).abs().max()):.2e})")

def bptt_grads(cn, xb, yb, beta, masks):
    z, caches, cro = cn.forward_explicit(xb, beta, masks)
    loss = cn.loss_of(z, yb, caches)
    ps = dict(cn.net.named_parameters())
    gs = torch.autograd.grad(loss, list(ps.values()))
    return dict(zip(ps.keys(), gs)), z.detach(), caches, cro


def check_gradient_exactness(modes=(1, 4, 8, 16), lengths=(25, 100, 400),
                             time_steps=(0.25, 0.5, 1.0), device="cpu"):
    """Reproduce the 36-setting float64 gradient comparison.

    Each setting compares every parameter with autograd and verifies that a
    cross-layer adjoint with k=T agrees with the complete reverse adjoint.
    This is a numerical implementation check, separate from the Lean proofs.
    """
    from .models.crisp import Config, DendStochAudioNet

    rows = []
    total = len(modes) * len(lengths) * len(time_steps)
    worst = 0.0
    for modes_count in modes:
        for length in lengths:
            for dt in time_steps:
                torch.manual_seed(1000 + modes_count * 7 + length)
                config = Config(n_in=20, n_hidden=12, k_modes=modes_count,
                                n_classes=5, hidden_dropout=0.2)
                net = DendStochAudioNet(config).double().to(device)
                with torch.no_grad():
                    for layer in net.layers:
                        layer.dend.raw_lam.add_(0.3 * torch.randn_like(layer.dend.raw_lam))
                        layer.theta.normal_(0, 0.3)
                        layer.norm.running_mean.normal_(0, 0.5)
                        layer.norm.running_var.uniform_(0.5, 2.0)
                        layer.norm.weight.uniform_(0.5, 1.5)
                        layer.norm.bias.normal_(0, 0.2)
                    net.ro.raw_lam.add_(0.3 * torch.randn_like(net.ro.raw_lam))
                net.set_dt(dt)
                clear = ClearNet(net, seed=0).to(device)
                x = torch.rand(3, length, 20, dtype=torch.float64, device=device) * 2
                target = torch.randint(0, 5, (3,), device=device)
                beta = 3.0
                masks = clear.make_masks(x, beta)
                reference, logits, caches, readout_cache = bptt_grads(clear, x, target, beta, masks)
                exact = clear.local_grads(logits, target, caches, readout_cache, beta, "exact")
                complete = clear.local_grads(logits, target, caches, readout_cache,
                                             beta, f"clear_sym_k{length}")
                relative = {key: float((exact[key] - value).norm() / (value.norm() + 1e-300))
                            for key, value in reference.items()}
                complete_error = max(float((complete[key] - value).norm() / (value.norm() + 1e-300))
                                     for key, value in exact.items())
                maximum = max(relative.values())
                worst = max(worst, maximum, complete_error)
                rows.append({"modes": modes_count, "length": length, "time_step": dt,
                             "max_relative_error": maximum,
                             "complete_adjoint_relative_error": complete_error,
                             "parameter_relative_errors": relative})
                print(f"[{len(rows)}/{total}] modes={modes_count} length={length} dt={dt:g} "
                      f"relative error={maximum:.3e}", flush=True)
    if worst >= 1e-9:
        raise RuntimeError(f"Local-gradient verification failed: worst relative error {worst:.3e}")
    return {"dtype": "float64", "rows": rows, "worst_relative_error": worst,
            "scope": "Exact incoming learning signals and complete cross-layer adjoints."}
