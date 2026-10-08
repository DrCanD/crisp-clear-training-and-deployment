"""Moment propagation and Monte Carlo diagnostics for two-layer CRISP models.

The Gaussian and Cantelli curves use estimated or approximated moments. They
are descriptive diagnostics, separate from the exact vote-test certificate.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import ndtr
from scipy.stats import binom

from .datasets import load_data, ArrLoader, data_fingerprint
from .models.crisp import load_ckpt
from .training import set_all_seeds
from .paths import ROOT, resolve_path
from .diagnostics import (SomaProbe, linear_maps, upstream_gradients,
                          moment_approximated_logits, iter_sampled_logits, paired_summary)

CLASS_CHUNK = 8

@dataclass
class SamplingSettings:
    K: int = 128
    KL: int = 128
    draw_chunk: int = 8
    batch_size: int = 128
    group_Ns: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    mc_check_M: int = 256

@dataclass
class AnalysisSettings:
    cert_Ns: list = field(default_factory=lambda: [1,2,4,8,16,32,64,128,256,512,1024])
    eps: list = field(default_factory=lambda: [.01,.05])
    calib_edges: list = field(default_factory=lambda: [0,1e-4,1e-3,1e-2,.05,.1,.2,.35,.5,.75,1.0001])
    coverage_alpha: float = .01
    fixed_Ns: list = field(default_factory=lambda: [1,2,3,4,6,8,12,16,24,32,48,64,96,128])
    soft_taus: list = field(default_factory=lambda: [.3,.5,.7,.8,.9,.95,.99,.999])
    z_grid: list = field(default_factory=lambda: [.5,1.,1.5,2.,2.5,3.,4.,5.])
    plan_eps: list = field(default_factory=lambda: [.2,.1,.05,.02,.01,.001])
    budgets: list = field(default_factory=lambda: [1.5,2,3,4,6,8,12,16])

def draw_last_layer(model, pLc, KL, R):
    """Sample only the last hidden layer, with mean-field activity below. Returns [KL,B,C]."""
    B = pLc.shape[0]
    out = []
    for _ in range(KL // R):
        s = torch.bernoulli(pLc.repeat(R, 1, 1))
        out.append(model.pool(model.ro(model.readout(s))).view(R, B, -1))
    return torch.cat(out, 0)

def group_counts(Z, cs, y, Ns):
    K, B, C = Z.shape
    fl, co = ([], [])
    for N in Ns:
        G = K // N
        dec = Z[:G * N].view(G, N, B, C).mean(1).argmax(2)
        fl.append((dec != cs).sum(0))
        co.append((dec == y).sum(0))
    return (torch.stack(fl, 1), torch.stack(co, 1))

def moments(Z, cs):
    K, B, C = Z.shape
    d = Z - Z.gather(2, cs.view(1, B, 1).expand(K, B, 1))
    return (Z.mean(0), Z.var(0, unbiased=True), d.mean(0), d.var(0, unbiased=True))

def rel_err(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))

def check_model_identities(model, xb, U, settings):
    """Run on the first batch of every unit (also on resume). Raises on failure."""
    device = next(model.parameters()).device
    beta = U['beta']
    L = len(model.layers)
    B = xb.shape[0]
    out = {}
    tol_jac = 0.02 if device.type == 'cuda' and torch.backends.cuda.matmul.allow_tf32 else 0.0001
    pr = SomaProbe(model)
    with torch.no_grad():
        zbar, _, _ = model(xb, 'graded', beta)
    p = [pr.S[l] for l in range(L)]
    V = [pr.V[l].view_as(p[l]) for l in range(L)]
    pr.remove()
    for l in range(L):
        if not torch.equal(torch.sigmoid(beta * (V[l] - model.layers[l].theta)), p[l]):
            raise RuntimeError(f'[probability check] layer {l}: sigmoid(beta(V-theta)) from the BN hook != layer output')
    out['probability_capture'] = 'bitwise'
    p1c = p[0].clamp(1e-06, 1 - 1e-06)
    with torch.no_grad():
        set_all_seeds(12345)
        z_ref, _, _ = model(xb, 'sample', beta)
        set_all_seeds(12345)
        z_pw, _ = next(iter_sampled_logits(model, p1c, 1, 1, beta))
    if not torch.equal(z_ref, z_pw[0]):
        raise RuntimeError(f"[cached sampling check] piecewise sampling != model(x,'sample') (max diff {float((z_ref - z_pw[0]).abs().max()):.3e})")
    out['cached_sampling_identity'] = 'bitwise'
    with torch.no_grad():
        set_all_seeds(777)
        a = torch.cat([z for z, _ in iter_sampled_logits(model, p1c, 2 * settings.draw_chunk, settings.draw_chunk, beta)])
        set_all_seeds(777)
        b = torch.cat([z for z, _ in iter_sampled_logits(model, p1c, 2 * settings.draw_chunk, settings.draw_chunk, beta)])
    if not torch.equal(a, b):
        raise RuntimeError('[repeatability check] same seed gave different draws')
    out['sampling_repeatability'] = 'bitwise'
    C = U['A'].shape[0]
    cls = sorted(set([0, C // 2, C - 1]))
    probes = []

    def addp(mod, inp, o):
        e = torch.zeros_like(o[0], requires_grad=True)
        probes.append(e)
        return (o[0] + e, o[1], o[2])
    hs = [layer.register_forward_hook(addp) for layer in model.layers]
    with torch.enable_grad():
        zz, _, _ = model(xb, 'graded', beta)
        errs = []
        for c in cls:
            g_ag = torch.autograd.grad(zz[:, c].sum(), probes, retain_graph=True)
            with torch.no_grad():
                g_an = upstream_gradients(U['A'][c:c + 1].unsqueeze(0), p, U['consts'], beta)
            errs += [rel_err(g_an[l].expand(B, 1, *g_an[l].shape[2:])[:, 0], g_ag[l]) for l in range(L)]
    for h in hs:
        h.remove()
    out['jacobian_max_relative_error'] = max(errs)
    if max(errs) > tol_jac:
        raise RuntimeError(f'[Jacobian check] closed-form Jacobian vs autograd rel err {max(errs):.2e} > {tol_jac}')
    with torch.no_grad():
        zl = model.pool(model.ro(model.readout(p[-1])))
        zl_lin = torch.einsum('bth,cth->bc', p[-1], U['A']) + U['z0']
    out['readout_linearity_relative_error'] = rel_err(zl_lin, zl)
    if out['readout_linearity_relative_error'] > tol_jac:
        raise RuntimeError(f"[readout linearity check] readout not linear: {out['readout_linearity_relative_error']:.2e}")
    with torch.no_grad():
        v_prev = p[-2].clamp(1e-06, 1 - 1e-06) * (1 - p[-2].clamp(1e-06, 1 - 1e-06))
        _, varV = moment_approximated_logits(model, V[-1], v_prev, U['consts'], beta)
        cap = {}
        lastL = model.layers[-1]
        h = lastL.norm.register_forward_hook(lambda m, i, o: cap.__setitem__('V', o))
        set_all_seeds(4242)
        acc = [torch.zeros_like(varV, dtype=torch.float64) for _ in range(4)]
        M = 0
        R = settings.draw_chunk
        pin = p[-2].clamp(1e-06, 1 - 1e-06)
        while M < settings.mc_check_M:
            s = torch.bernoulli(pin.repeat(R, 1, 1))
            lastL(s, 'graded', beta)
            Vd = (cap['V'].view(R, *V[-1].shape) - V[-1].unsqueeze(0)).double()
            for k_ in range(4):
                acc[k_] += (Vd ** (k_ + 1)).sum(0)
            M += R
        h.remove()
        m1 = acc[0] / M
        mc_var = ((acc[1] - acc[0] ** 2 / M) / (M - 1)).float()
        mc_m4 = (acc[3] / M - 4 * m1 * acc[2] / M + 6 * m1 ** 2 * acc[1] / M - 3 * m1 ** 4).float()
        big = varV > varV.flatten().quantile(0.5) if varV.numel() < 16000000 else varV > varV.mean()
        ratio = float(mc_var[big].mean() / varV[big].mean())
        med = float(((mc_var[big] - varV[big]).abs() / varV[big]).median())
    se_rel = math.sqrt(2.0 / max(M - 1, 1))
    out['preactivation_variance_mc_ratio'] = ratio
    out['preactivation_variance_median_relative_error'] = med
    out['preactivation_variance_draws'] = M
    if L >= 3:
        out['preactivation_variance_scope'] = 'L>=3: Var(V_L) uses only the Bernoulli variance of layer L-1 (upstream variance ignored)'
    elif abs(ratio - 1) > 0.05 + 3 * se_rel / math.sqrt(max(1, int(big.sum()))) or med > 2.0 * se_rel:
        raise RuntimeError(f'[preactivation variance check] exact Var(V_L) vs Monte Carlo: mean ratio {ratio:.3f}, median rel err {med:.3f} (MC rel s.e. per element {se_rel:.3f})')
    return out

def measure_batch(model, xb, yb, U, seed, settings):
    """All per-input quantities for one batch. Returns dict of CPU numpy arrays."""
    device = next(model.parameters()).device
    beta = U['beta']
    L = len(model.layers)
    B = xb.shape[0]
    K, KL, R = (settings.K, settings.KL, settings.draw_chunk)
    A, consts = (U['A'], U['consts'])
    C = A.shape[0]
    y = yb.to(device)
    with torch.no_grad():
        pr = SomaProbe(model)
        zbar, _, _ = model(xb, 'graded', beta)
        p = [pr.S[l] for l in range(L)]
        V = [pr.V[l].view_as(p[l]) for l in range(L)]
        pr.remove()
        pc = [q.clamp(1e-06, 1 - 1e-06) for q in p]
        v = [q * (1 - q) for q in pc]
        cs = zbar.argmax(1)
        gcs = upstream_gradients(A[cs].unsqueeze(1), p, consts, beta)
        var = torch.zeros(L, B, C, device=device)
        dvar = torch.zeros(L, B, C, device=device)
        for c0 in range(0, C, CLASS_CHUNK):
            c1 = min(C, c0 + CLASS_CHUNK)
            gs = upstream_gradients(A[c0:c1].unsqueeze(0), p, consts, beta)
            for l in range(L):
                vl = v[l].unsqueeze(1)
                var[l, :, c0:c1] = (gs[l] ** 2 * vl).sum((2, 3))
                dvar[l, :, c0:c1] = ((gs[l] - gcs[l]) ** 2 * vl).sum((2, 3))
            del gs
        zmp, varV = moment_approximated_logits(model, V[-1], v[-2], consts, beta)
        cs_mp = zmp.argmax(1)
        set_all_seeds(seed)
        Z0 = draw_last_layer(model, pc[-1], KL, R)
        Zf = torch.empty(K, B, C, device=device)
        seq = {k: torch.empty(K, B, device=device) for k in ('margin', 'maxsoft', 'empse', 'anase')}
        amax = torch.empty(K, B, dtype=torch.long, device=device)
        Zsum = torch.zeros(B, C, device=device)
        S = torch.zeros_like(v[-1])
        n = 0
        ar = torch.arange(B, device=device)
        for z, vr in iter_sampled_logits(model, pc[0], K, R, beta):
            for r in range(z.shape[0]):
                Zf[n] = z[r]
                Zsum += z[r]
                S += vr[r]
                n += 1
                M = Zsum / n
                tv, ti = M.topk(2, dim=1)
                c1, c2 = (ti[:, 0], ti[:, 1])
                seq['margin'][n - 1] = tv[:, 0] - tv[:, 1]
                seq['maxsoft'][n - 1] = F.softmax(M, 1).max(1).values
                amax[n - 1] = c1
                Ad = A[c1] - A[c2]
                seq['anase'][n - 1] = torch.sqrt((Ad * Ad * S).sum((1, 2))) / n
                if n >= 2:
                    D = Zf[:n, ar, c1] - Zf[:n, ar, c2]
                    seq['empse'][n - 1] = D.std(0, unbiased=True) / math.sqrt(n)
                else:
                    seq['empse'][n - 1] = float('inf')
        fF, cF = group_counts(Zf, cs, y, settings.group_Ns)
        fP, _ = group_counts(Zf, cs_mp, y, settings.group_Ns)
        f0, c0_ = group_counts(Z0, cs, y, settings.group_Ns)
        mF, vF, dmF, dvF = moments(Zf, cs)
        m0, v0, dm0, dv0 = moments(Z0, cs)
    npy = lambda t: t.detach().cpu().numpy()
    extra = {'fP': npy(fP).astype(np.int16)}
    return extra | {'y': npy(yb).astype(np.int16), 'cs': npy(cs).astype(np.int16), 'zbar': npy(zbar), 'zmp': npy(zmp), 'var': npy(var.permute(1, 0, 2)), 'dvar': npy(dvar.permute(1, 0, 2)), 'mF': npy(mF), 'vF': npy(vF), 'dmF': npy(dmF), 'dvF': npy(dvF), 'm0': npy(m0), 'v0': npy(v0), 'dm0': npy(dm0), 'dv0': npy(dv0), 'fF': npy(fF).astype(np.int16), 'cF': npy(cF).astype(np.int16), 'f0': npy(f0).astype(np.int16), 'c0': npy(c0_).astype(np.int16), 'seq_margin': npy(seq['margin'].t()), 'seq_maxsoft': npy(seq['maxsoft'].t()), 'seq_empse': npy(seq['empse'].t()), 'seq_anase': npy(seq['anase'].t()), 'seq_amax': npy(amax.t()).astype(np.int8), 'rate': np.stack([npy(q.mean((1, 2))) for q in p], 1), 'varV_mean': npy(varV.mean((1, 2)))}

def r2(meas, pred):
    meas = np.asarray(meas, np.float64).ravel()
    pred = np.asarray(pred, np.float64).ravel()
    ss_tot = ((meas - meas.mean()) ** 2).sum()
    return float(1 - ((meas - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float('nan')

def r2_log(meas, pred, floor=1e-08):
    m = (meas > floor) & (pred > floor)
    return (r2(np.log10(meas[m]), np.log10(pred[m])), float(m.mean()))

def r2_ceiling(pred, K, seed=0):
    rng = np.random.default_rng(seed)
    return r2(pred * rng.chisquare(K - 1, size=pred.shape) / (K - 1), pred)

def flip_probs(mu, s2, N, mask):
    """mu: mean of D_j = z_j - z_c* (<0 when c* wins), s2: Var(D_j) per draw, N draws averaged.
    Returns P_max (largest single-pair Gaussian term), P_union (Gaussian union), Cantelli union bound."""
    s2N = np.maximum(s2, 0) / N
    sd = np.sqrt(s2N)
    with np.errstate(divide='ignore', invalid='ignore'):
        zz = np.where(sd > 0, mu / np.where(sd > 0, sd, 1), np.where(mu >= 0, np.inf, -np.inf))
        P = ndtr(zz)
        cant = np.where(mu < 0, s2N / (s2N + mu ** 2), 1.0)
    P[mask] = 0
    cant[mask] = 0
    return (P.max(1), np.minimum(1, P.sum(1)), np.minimum(1, cant.sum(1)))

def first_true(mask):
    K = mask.shape[1]
    anyv = mask.any(1)
    return np.where(anyv, mask.argmax(1), K - 1)

def analyze_moments(D, K, KL, Ns, analysis):
    n, C = D['zbar'].shape
    y = D['y'].astype(np.int64)
    cs = D['cs'].astype(np.int64)
    rows = np.arange(n)
    mask = np.zeros((n, C), bool)
    mask[rows, cs] = True
    varL = D['var'][:, -1, :]
    varFO = D['var'].sum(1)
    dvarL = D['dvar'][:, -1, :]
    dvarFO = D['dvar'].sum(1)
    correct = cs == y
    res = {'n_test': int(n), 'K': int(K), 'mf_acc': float(correct.mean()), 'single_draw_acc': float(D['cF'][:, 0].sum() / (n * K)), 'rate_per_layer': [float(v) for v in D['rate'].mean(0)]}
    res['single_draw_gap'] = res['single_draw_acc'] - res['mf_acc']
    se = np.sqrt(D['vF'] / K)
    ok = se > 0
    zF = (D['mF'] - D['zbar'])[ok] / se[ok]
    zMP = (D['mF'] - D['zmp'])[ok] / se[ok]
    se0 = np.sqrt(D['v0'] / KL)
    ok0 = se0 > 0
    z0 = (D['m0'] - D['zbar'])[ok0] / se0[ok0]
    res['bias'] = {'full_signed_bias_mean': float((D['mF'] - D['zbar']).mean()), 'full_bias_to_noise': float(np.abs(D['mF'] - D['zbar']).mean() / np.sqrt(D['vF']).mean()), 'full_frac_absz_gt3_vs_MF': float((np.abs(zF) > 3).mean()), 'full_frac_absz_gt3_vs_MP': float((np.abs(zMP) > 3).mean()), 'L0regime_frac_absz_gt3_vs_MF': float((np.abs(z0) > 3).mean()), 'expected_frac_if_unbiased': 0.0027, 'MP_bias_R2': r2(D['mF'] - D['zbar'], D['zmp'] - D['zbar']), 'mp_changes_decision_frac': float((D['zmp'].argmax(1) != cs).mean())}
    nd = ~mask
    res['variance'] = {'L0regime_R2_pred_vs_meas': r2(D['v0'], varL), 'L0regime_R2_ceiling_K': r2_ceiling(varL, KL), 'full_R2_FO': r2(D['vF'], varFO), 'full_R2_lastlayer_only': r2(D['vF'], varL), 'full_R2_ceiling_K': r2_ceiling(varFO, K), 'full_logR2_FO': r2_log(D['vF'], varFO)[0], 'full_logR2_lastlayer_only': r2_log(D['vF'], varL)[0], 'full_median_ratio_meas_over_FO': float(np.median(D['vF'][varFO > 0] / varFO[varFO > 0])), 'full_median_ratio_meas_over_L0': float(np.median(D['vF'][varL > 0] / varL[varL > 0])), 'diff_R2_FO': r2(D['dvF'][nd], dvarFO[nd]), 'diff_R2_lastlayer_only': r2(D['dvF'][nd], dvarL[nd]), 'predicted_share_of_upstream_layers_median': float(np.median(1 - varL[varFO > 0] / varFO[varFO > 0])), 'mean_pred_var_FO': float(varFO.mean()), 'mean_meas_var_full': float(D['vF'].mean())}
    mu_mf = D['zbar'] - D['zbar'][rows, cs][:, None]
    mu_mp = D['zmp'] - D['zmp'][rows, cs][:, None]
    flip_calibration = {}
    for iN, N in enumerate(Ns):
        G = K // N
        if G < 2:
            continue
        blk = {}
        for name, mu, s2, fl, G_ in (('L0regime', mu_mf, dvarL, D['f0'][:, iN], KL // N), ('full_MF', mu_mf, dvarFO, D['fF'][:, iN], G), ('full_MP', mu_mp, dvarFO, D['fF'][:, iN], G), ('full_lastlayer_only', mu_mf, dvarL, D['fF'][:, iN], G)):
            pmax, pun, cant = flip_probs(mu, s2, N, mask)
            fl = fl.astype(np.int64)
            obs = fl.sum() / (n * G_)
            pv = binom.sf(fl - 1, G_, np.clip(cant, 0, 1))
            viol = (fl > 0) & (pv < analysis.coverage_alpha)
            pvg = binom.sf(fl - 1, G_, np.clip(pun, 0, 1))
            violg = (fl > 0) & (pvg < analysis.coverage_alpha)
            edges = np.array(analysis.calib_edges)
            bi = np.clip(np.digitize(pun, edges) - 1, 0, len(edges) - 2)
            calib = []
            for b in range(len(edges) - 1):
                m_ = bi == b
                if m_.sum() == 0:
                    continue
                calib.append({'bin': [float(edges[b]), float(edges[b + 1])], 'n': int(m_.sum()), 'pred_max': float(pmax[m_].mean()), 'pred_union': float(pun[m_].mean()), 'observed': float(fl[m_].sum() / (m_.sum() * G_))})
            blk[name] = {'groups': int(G_), 'observed_flip_rate': float(obs), 'pred_max_mean': float(pmax.mean()), 'pred_union_mean': float(pun.mean()), 'cantelli_mean': float(cant.mean()), 'cantelli_violation_frac': float(viol.mean()), 'gauss_union_violation_frac': float(violg.mean()), 'alpha': analysis.coverage_alpha, 'calibration': calib}
        flip_calibration[str(N)] = blk
    res['flip_calibration'] = flip_calibration
    approximate_acceptance = {}
    for name, mu, s2, kind in (('gauss_MF', mu_mf, dvarFO, 'g'), ('gauss_MP', mu_mp, dvarFO, 'g'), ('cantelli_MF', mu_mf, dvarFO, 'c'), ('cantelli_MP', mu_mp, dvarFO, 'c'), ('gauss_lastlayer_only', mu_mf, dvarL, 'g')):
        tab = {}
        for N in analysis.cert_Ns:
            pmax, pun, cant = flip_probs(mu, s2, N, mask)
            P = pun if kind == 'g' else cant
            tab[str(N)] = {str(e): float((correct & (P <= e)).mean()) for e in analysis.eps}
            if N in Ns and K // N >= 2:
                iN = Ns.index(N)
                G = K // N
                fl = D['fF'][:, iN].astype(np.int64)
                for e in analysis.eps:
                    cert = P <= e
                    tab[str(N)][f'pooled_flip_among_certified_{e}'] = float(fl[cert].sum() / max(1, cert.sum() * G))
                    pv = binom.sf(fl - 1, G, e)
                    tab[str(N)][f'violation_frac_{e}'] = float((cert & (fl > 0) & (pv < analysis.coverage_alpha)).sum() / max(1, cert.sum()))
        approximate_acceptance[name] = tab
    approximate_acceptance['empirical_acc_at_N'] = {str(N): float(D['cF'][:, i].sum() / (n * (K // N))) for i, N in enumerate(Ns)}
    approximate_acceptance['empirical_flip_at_N'] = {str(N): float(D['fF'][:, i].sum() / (n * (K // N))) for i, N in enumerate(Ns)}
    res['approximate_acceptance'] = approximate_acceptance
    am = D['seq_amax'].astype(np.int64)
    mg = D['seq_margin']
    ms = D['seq_maxsoft']
    es = D['seq_empse']
    an = D['seq_anase']

    def ev(stop):
        dec = am[rows, stop]
        return {'mean_n': float((stop + 1).mean()), 'acc': float((dec == y).mean()), 'agree_MF': float((dec == cs).mean())}
    rules = {'fixed_draws': [], 'softmax_confidence': [], 'empirical_standard_error': [], 'conditional_standard_error': [], 'crossfit_standard_error': [], 'moment_propagation_planner': []}
    for N in analysis.fixed_Ns:
        if N <= K:
            rules['fixed_draws'].append({'param': N, **ev(np.full(n, N - 1))})
    for tau in analysis.soft_taus:
        rules['softmax_confidence'].append({'param': tau, **ev(first_true(ms >= tau))})
    with np.errstate(divide='ignore', invalid='ignore'):
        zr_emp = np.where(np.isfinite(es) & (es > 0), mg / es, np.where((es == 0) & (mg > 0), np.inf, 0.0))
        zr_an = np.where(an > 0, mg / an, np.where((an == 0) & (mg > 0), np.inf, 0.0))
    j2 = np.argsort(-D['zbar'], 1)[:, 1]
    ratio = dvarFO[rows, j2] / np.maximum(dvarL[rows, j2], 1e-30)
    fold = rows % 2
    kap = np.array([math.sqrt(float(np.median(ratio[fold != f]))) for f in (0, 1)])
    kap_i = kap[fold]
    res['crossfit_variance_correction'] = [float(k) for k in kap]
    for zt in analysis.z_grid:
        rules['empirical_standard_error'].append({'param': zt, **ev(first_true(zr_emp >= zt))})
        rules['conditional_standard_error'].append({'param': zt, **ev(first_true(zr_an >= zt))})
        rules['crossfit_standard_error'].append({'param': zt, **ev(first_true(zr_an >= zt * kap_i[:, None]))})
    Pn = np.stack([flip_probs(mu_mp, dvarFO, N, mask)[1] for N in range(1, K + 1)], 1)
    for e in analysis.plan_eps:
        rules['moment_propagation_planner'].append({'param': e, **ev(first_true(Pn <= e))})
    matched = {}
    for rn, pts in rules.items():
        xs = np.array([q['mean_n'] for q in pts])
        ys = np.array([q['acc'] for q in pts])
        ux = np.unique(xs)
        uy = np.array([ys[xs == u].mean() for u in ux])
        matched[rn] = {str(b): float(np.interp(b, ux, uy)) if ux.min() <= b <= ux.max() else None for b in analysis.budgets}
    res['stopping_rules'] = {'rules': rules, 'acc_at_matched_mean_n': matched}
    dmp = D['zmp'].argmax(1)
    dK = D['mF'].argmax(1)
    dis = cs != dmp
    decision_comparison = {'acc_MF': float(correct.mean()), 'acc_MP': float((dmp == y).mean()), f'acc_mean_of_{K}_draws': float((dK == y).mean()), f'agree_{K}mean_with_MF': float((dK == cs).mean()), f'agree_{K}mean_with_MP': float((dK == dmp).mean()), 'frac_MF_ne_MP': float(dis.mean()), 'n_MF_ne_MP': int(dis.sum())}
    if dis.any():
        decision_comparison['on_MF_ne_MP'] = {f'{K}mean_eq_MP': float((dK[dis] == dmp[dis]).mean()), f'{K}mean_eq_MF': float((dK[dis] == cs[dis]).mean()), 'MP_correct': float((dmp[dis] == y[dis]).mean()), 'MF_correct': float((cs[dis] == y[dis]).mean())}
    if 'fP' in D:
        decision_comparison['flip_vs_MF_at_N'] = {str(N): float(D['fF'][:, i].sum() / (n * (K // N))) for i, N in enumerate(Ns)}
        decision_comparison['flip_vs_MP_at_N'] = {str(N): float(D['fP'][:, i].sum() / (n * (K // N))) for i, N in enumerate(Ns)}
    res['decision_comparison'] = decision_comparison
    return res


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(config):
    """Measure fresh moments of the declared checkpoints, with per-batch resume.

    Batch arrays retain mathematical notation: var/dvar are per-layer logit and
    margin variances; mF/vF and m0/v0 are empirical full-network and last-layer
    mean/variance; zbar/zmp are mean-field and moment-propagated logits.
    """
    dataset = config['dataset']['name']
    settings = SamplingSettings(**config.get('sampling', {}))
    analysis = AnalysisSettings(**config.get('analysis', {}))
    if settings.K < 4 or settings.KL < 4 or settings.draw_chunk < 1:
        raise ValueError('At least four draws and a positive draw chunk are required')
    if settings.K % settings.draw_chunk or settings.KL % settings.draw_chunk:
        raise ValueError('Both draw counts must be divisible by the draw chunk')
    if not settings.group_Ns or settings.group_Ns[0] != 1 or any(
            n < 1 or n > min(settings.K, settings.KL) for n in settings.group_Ns):
        raise ValueError('Group sizes must start with one and fit both draw counts')
    data = load_data(config['dataset']['path'], dataset, config['dataset'].get('fingerprint'))
    x, y = data[2]
    limit = int(config.get('test_limit', 0))
    if limit:
        x, y = x[:limit], y[:limit]
    directory = resolve_path(config.get('output', f'outputs/moment_diagnostics_{dataset}'))
    directory.mkdir(parents=True, exist_ok=True)
    config_file = directory / 'configuration.json'
    if config_file.exists() and json.loads(config_file.read_text()) != _json_safe(config):
        raise ValueError('Existing moment outputs use a different configuration')
    _write_json(config_file, config)
    device = torch.device(config.get('device', 'cpu'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for seed in config.get('seeds', [42, 123, 999]):
        path = resolve_path(config['checkpoint'].format(seed=seed))
        if not path.is_file():
            raise FileNotFoundError('Run deployment_gap before the moment diagnostic: ' + path.relative_to(ROOT).as_posix())
        model, architecture = load_ckpt(path, device=device)
        model.eval()
        if len(model.layers) != 2:
            raise ValueError('This archived moment-propagation protocol requires exactly two hidden layers')
        for key, expected in config.get('model_contract', {}).items():
            if getattr(architecture, key) != expected:
                raise ValueError(f'Checkpoint architecture mismatch: {key}')
        if int(x.shape[1]) != int(config.get('timesteps', 100)):
            raise ValueError('Prepared data do not match the declared number of time bins')
        if architecture.n_classes != {'shd': 20, 'ssc': 35}[dataset]:
            raise ValueError('Checkpoint class count does not match the dataset')
        model.set_dt(float(config.get('dt', 1.)))
        unit = directory / f'seed_{seed}'
        unit.mkdir(exist_ok=True)
        draw_seed = int(config['draw_seeds'][str(seed)])
        identity = {'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'checkpoint': path.relative_to(ROOT).as_posix(), 'draw_seed': draw_seed,
                    'data_fingerprint': data_fingerprint(data), 'configuration': config}
        identity_file = unit / 'identity.json'
        if identity_file.exists() and json.loads(identity_file.read_text()) != _json_safe(identity):
            raise ValueError('Checkpoint or data changed; choose a new output directory')
        _write_json(identity_file, identity)
        consts, readout, zero = linear_maps(model, x.shape[1])
        constants = {'beta': float(architecture.beta_end), 'consts': consts, 'A': readout, 'z0': zero}
        gates = check_model_identities(model, x[:settings.batch_size].to(device).float(), constants, settings)
        _write_json(unit / 'identity_checks.json', gates)
        arrays = {}
        loader = ArrLoader(x, y, settings.batch_size, device=device)
        for index, (xb, yb) in enumerate(loader):
            block_file = unit / f'batch_{index:05d}.npz'
            if block_file.exists():
                with np.load(block_file, allow_pickle=False) as stored:
                    block = {key: stored[key] for key in stored.files}
            else:
                block = measure_batch(model, xb, yb, constants, draw_seed + index, settings)
                temporary = block_file.with_suffix('.tmp')
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle, **block)
                temporary.replace(block_file)
            for key, value in block.items():
                arrays.setdefault(key, []).append(value)
            print(f'Moment diagnostics: {dataset}, seed {seed}, batch {index + 1}/{len(loader)}', flush=True)
        arrays = {key: np.concatenate(values) for key, values in arrays.items()}
        report = analyze_moments(arrays, settings.K, settings.KL, settings.group_Ns, analysis)
        report.update(seed=seed, checkpoint_sha256=identity['checkpoint_sha256'],
                      status='fresh_checkpoint_evaluation',
                      scope='Gaussian and Cantelli curves use approximate moments; these are not vote-test certificates',
                      undefined_statistics='Nonfinite or undefined summary statistics are represented by null')
        _write_json(unit / 'summary.json', report)
        rows.append(report)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    fields = {
        'all_layer_r2': ('variance', 'full_R2_FO'),
        'last_layer_r2': ('variance', 'full_R2_lastlayer_only'),
        'lower_layer_variance_fraction': ('variance', 'predicted_share_of_upstream_layers_median'),
        'mf_tail_fraction': ('bias', 'full_frac_absz_gt3_vs_MF'),
        'mp_tail_fraction': ('bias', 'full_frac_absz_gt3_vs_MP'),
    }
    aggregate = {key: paired_summary([row[group][field] for row in rows])
                 for key, (group, field) in fields.items()}
    result = {'experiment': 'moment_diagnostics', 'dataset': dataset,
              'status': 'fresh_checkpoint_evaluation', 'rows': rows, 'summary': aggregate}
    _write_json(directory / 'results.json', result)
    return _json_safe(result)
