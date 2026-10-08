"""Conditional last-layer variance: independent CRISP versus reset-coupled LIF.

The LIF diagnostic adds a stochastic threshold to the trained deterministic
baseline. It is a conditional variance experiment, not the deployed LIF model.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from .datasets import load_data, ArrLoader, data_fingerprint
from .models import crisp, lif
from .training import set_all_seeds
from .paths import ROOT, resolve_path
from .moment_diagnostics import r2, _write_json, _json_safe
from .diagnostics import paired_summary

def lif_sampled_layer(layer, s_in, beta_n):
    """LIF v2 layer recursion (hard reset, as the reference); beta_n None -> the reference deterministic threshold."""
    I = layer.norm_apply(layer.proj(s_in))
    B, T, d = I.shape
    a = layer.alpha()
    v = torch.zeros(B, d, device=I.device, dtype=I.dtype)
    s_prev = torch.zeros_like(v)
    out = []
    for t in range(T):
        v = a * v * (1 - s_prev) + I[:, t]
        s = (v - layer.theta > 0).to(I.dtype) if beta_n is None else torch.bernoulli(torch.sigmoid(beta_n * (v - layer.theta)))
        out.append(s)
        s_prev = s
    return torch.stack(out, 1)

def lif_deterministic_probabilities(layer, s_in, beta_n):
    """closed-form analogue for LIF: firing probability along the DETERMINISTIC trajectory (what one can compute without sampling)"""
    I = layer.norm_apply(layer.proj(s_in))
    B, T, d = I.shape
    a = layer.alpha()
    v = torch.zeros(B, d, device=I.device, dtype=I.dtype)
    s_prev = torch.zeros_like(v)
    ps = []
    for t in range(T):
        v = a * v * (1 - s_prev) + I[:, t]
        ps.append(torch.sigmoid(beta_n * (v - layer.theta)))
        s_prev = (v - layer.theta > 0).to(I.dtype)
    return torch.stack(ps, 1)

def measure_conditional_variance(fam, model, beta, X, y, beta_n, seed, draws, batch_size):
    """returns per-(input, class) measured / predicted variances and the one-draw accuracy"""
    device = next(model.parameters()).device
    C = int(model.ro.proj.out_features) if fam == 'lif' else int(model.readout.out_features)
    M, ch = (draws, batch_size)
    meas, pmc, pex, pdt, half_a, half_b, acc1 = ([], [], [], [], [], [], [])
    set_all_seeds(seed)
    for i in range(0, len(y), ch):
        xb = X[i:i + ch].to(device).float()
        yb = y[i:i + ch].to(device)
        B = xb.shape[0]
        with torch.no_grad():
            if fam == 'crisp':
                L0, L1 = model.layers
                s1, _, _ = L0(xb, 'graded', beta)
                p2 = L1(s1, 'graded', beta)[0]
                s1e = s1.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, *s1.shape[1:])
                s2 = L1(s1e, 'sample', beta)[0]
                readout = lambda s: model.pool(model.ro(model.readout(s)))
            else:
                L0, L1 = model.layers
                s1 = L0(xb)[0]
                s1e = s1.unsqueeze(1).expand(-1, M, -1, -1).reshape(B * M, *s1.shape[1:])
                s2 = lif_sampled_layer(L1, s1e, beta_n)
                readout = lambda s: model.pool(model.ro(s))
                p2 = None
                p_det = lif_deterministic_probabilities(L1, s1, beta_n)
            zd = readout(s2).view(B, M, C)
            s2v = s2.view(B, M, *s2.shape[1:])
            phat = s2v.mean(1)
        sv = (phat if p2 is None else p2).detach().clone().requires_grad_(True)
        z = readout(sv)
        R2 = torch.stack([torch.autograd.grad(z[:, c].sum(), sv, retain_graph=c < C - 1)[0] for c in range(C)], 1) ** 2
        with torch.no_grad():
            meas.append(zd.var(1, unbiased=True).cpu())
            pmc.append((R2 * (phat * (1 - phat)).unsqueeze(1)).sum((2, 3)).cpu())
            if p2 is not None:
                pex.append((R2 * (p2 * (1 - p2)).unsqueeze(1)).sum((2, 3)).cpu())
            else:
                pdt.append((R2 * (p_det * (1 - p_det)).unsqueeze(1)).sum((2, 3)).cpu())
            half_a.append(zd[:, :M // 2].var(1, unbiased=True).cpu())
            half_b.append(zd[:, M // 2:].var(1, unbiased=True).cpu())
            acc1.append((zd.argmax(2) == yb[:, None]).float().mean(1).cpu())
        del s2, s2v, zd, R2, sv, z
    cat = lambda L: torch.cat(L).numpy() if L else None
    return {'meas': cat(meas), 'pred_mc': cat(pmc), 'pred_exact': cat(pex), 'pred_det': cat(pdt), 'half_a': cat(half_a), 'half_b': cat(half_b), 'acc_one_draw': float(torch.cat(acc1).mean())}

def summarize_conditional_variance(u):
    m, p = (u['meas'].ravel(), u['pred_mc'].ravel())
    keep = p > 1e-12 * max(p.max(), 1e-30)
    m, p = (m[keep], p[keep])
    ratio = m / p
    ha, hb = (u['half_a'].ravel()[keep], u['half_b'].ravel()[keep])
    ok = (m > 0) & (ha > 0) & (hb > 0)
    lr = np.log(m[ok] / p[ok])
    lh = np.log(ha[ok] / hb[ok])
    sd_mc = float(np.std(lh, ddof=1) / 2)
    sd_lr = float(np.std(lr, ddof=1))
    out = {'r2_mc_marginals': r2(m, p), 'ratio_median': float(np.median(ratio)), 'ratio_p10': float(np.percentile(ratio, 10)), 'ratio_p90': float(np.percentile(ratio, 90)), 'split_half_r2': r2(ha, hb), 'log_ratio_mean': float(lr.mean()), 'log_ratio_sd': sd_lr, 'log_ratio_sd_mc': sd_mc, 'log_ratio_sd_excess': float(math.sqrt(max(sd_lr ** 2 - sd_mc ** 2, 0.0))), 'n_pairs': int(keep.sum()), 'acc_one_draw': u['acc_one_draw']}
    if u['pred_exact'] is not None:
        pe = u['pred_exact'].ravel()[keep]
        out['r2_closed_form'] = r2(m, pe)
        out['ratio_median_closed_form'] = float(np.median(m / pe))
    if u['pred_det'] is not None:
        pdd = u['pred_det'].ravel()[keep]
        ok2 = pdd > 0
        out['r2_closed_form'] = r2(m, pdd)
        out['ratio_median_closed_form'] = float(np.median(m[ok2] / pdd[ok2]))
    return out


def run(config):
    dataset = config['dataset']['name']
    if dataset != 'shd':
        raise ValueError('The archived conditional-variance experiment uses SHD')
    draws = int(config.get('draws', 256))
    batch_size = int(config.get('batch_size', 8))
    if draws < 4 or draws % 2 or batch_size < 1:
        raise ValueError('Draw count must be even and at least four; batch size must be positive')
    data = load_data(config['dataset']['path'], dataset, config['dataset'].get('fingerprint'))
    x, y = data[2]
    indices = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))[:int(config.get('n_inputs', 256))]
    subset, labels = x[indices], y[indices]
    directory = resolve_path(config.get('output', 'outputs/conditional_variance_shd'))
    directory.mkdir(parents=True, exist_ok=True)
    identity_file = directory / 'configuration.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != _json_safe(config):
        raise ValueError('Existing conditional-variance outputs use another protocol')
    _write_json(identity_file, config)
    device = torch.device(config.get('device', 'cpu'))
    test = ArrLoader(x, y, int(config.get('evaluation_batch_size', 128)), device=device)
    rows, pairs, sequence = [], [], 0
    for job in config['jobs']:
        family = job['family']
        if family not in ('crisp', 'lif'):
            raise ValueError('Conditional variance supports CRISP and LIF')
        for seed in config.get('seeds', [42, 123, 999]):
            checkpoint = resolve_path(job['checkpoint'].format(seed=seed))
            record = resolve_path(job['result'].format(seed=seed))
            if not checkpoint.is_file() or not record.is_file():
                raise FileNotFoundError('Complete the SHD sparsity experiment before conditional variance')
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            if family == 'crisp':
                model, architecture = crisp.load_ckpt(checkpoint, device=device)
                beta = float(architecture.beta_end)
            else:
                options = {key: value for key, value in saved['cfg'].items() if key in lif.Config.__dataclass_fields__}
                model = lif.build_model(options, int(saved.get('n_classes', 20)), device=device)
                model.load_state_dict(saved['state'], strict=True)
                beta = None
            model.eval()
            if len(model.layers) != 2:
                raise ValueError('Conditional diagnostic requires exactly two hidden layers')
            torch.backends.cuda.matmul.allow_tf32 = bool(config.get('tf32_evaluation', True))
            torch.backends.cudnn.allow_tf32 = bool(config.get('tf32_evaluation', True))
            expected_row = json.loads(record.read_text())['test']
            if family == 'crisp':
                accuracy = crisp.evaluate(model, test, 'graded', beta)['acc']
                expected = expected_row['meanfield']['acc']
            else:
                accuracy = lif.evaluate(model, test)['acc']
                expected = expected_row['deterministic']['acc']
            if abs(accuracy - expected) * len(y) > int(config.get('accuracy_tolerance_inputs', 0)) + 1e-9:
                raise RuntimeError('Checkpoint deterministic accuracy does not match its source result')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if family == 'lif':
                with torch.no_grad():
                    upstream = model.layers[0](subset[:4].to(device).float())[0]
                    expected_spikes = model.layers[1](upstream)[0]
                    if not torch.equal(lif_sampled_layer(model.layers[1], upstream, None), expected_spikes):
                        raise RuntimeError('Diagnostic LIF reset recurrence differs from the baseline')
            for noise_beta in ([None] if family == 'crisp' else config.get('lif_slopes', [5., 10.])):
                sequence += 1
                print(f'Conditional variance: {job["name"]}, seed {seed}, slope {noise_beta or beta}', flush=True)
                values = measure_conditional_variance(
                    family, model, beta, subset, labels, noise_beta,
                    int(config.get('draw_seed', 909)) + sequence, draws, batch_size,
                )
                statistics = summarize_conditional_variance(values)
                statistics.update(family=family, condition=job['name'], seed=seed,
                                  slope=noise_beta if noise_beta is not None else beta,
                                  deterministic_accuracy=accuracy,
                                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                label = f'{job["name"]}_seed_{seed}_slope_{statistics["slope"]:g}'
                np.savez_compressed(directory / f'{label}.npz', **{key: value for key, value in values.items() if value is not None})
                _write_json(directory / f'{label}.json', statistics)
                rows.append(statistics)
                computable = values['pred_exact'] if values['pred_exact'] is not None else values['pred_det']
                for row_index in range(len(indices)):
                    for category in range(values['meas'].shape[1]):
                        pairs.append({'family': family, 'condition': job['name'], 'seed': seed,
                                      'slope': statistics['slope'], 'input_index': int(indices[row_index]),
                                      'class_index': category, 'measured_variance': float(values['meas'][row_index, category]),
                                      'independent_marginal_variance': float(values['pred_mc'][row_index, category]),
                                      'computable_variance': float(computable[row_index, category])})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    with (directory / 'variance_pairs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    groups = {}
    measures = ('r2_mc_marginals','r2_closed_form','ratio_median_closed_form','ratio_median',
                'split_half_r2','log_ratio_mean','log_ratio_sd','log_ratio_sd_mc',
                'log_ratio_sd_excess','acc_one_draw','deterministic_accuracy')
    for condition, slope in sorted({(row['condition'], row['slope']) for row in rows}):
        selected = [row for row in rows if row['condition'] == condition and row['slope'] == slope]
        groups[f'{condition}_slope_{slope:g}'] = {key: paired_summary([row[key] for row in selected]) for key in measures}
    result = {'experiment': 'conditional_variance', 'dataset': dataset,
              'data_fingerprint': data_fingerprint(data), 'status': 'fresh_checkpoint_evaluation',
              'rows': rows, 'summary': groups}
    _write_json(directory / 'results.json', result)
    return _json_safe(result)
