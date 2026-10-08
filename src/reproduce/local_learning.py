"""Frozen-normalization CRISP training with BPTT and CLEAR feedback rules.

Training arithmetic, calibration order, gradient diagnostics, and random-number
handling are retained from the original fixed-epoch comparison.
"""
import copy
import gc
import json
import time
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from .datasets import ArrLoader, load_data
from .models.crisp import Config, DendStochAudioNet, anneal, evaluate, recalibrate_bn
from .learning import ClearNet, gate_explicit, bptt_grads
from .training import augment, set_all_seeds, save_torch_atomic
from .paths import resolve_path

def cosine(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-300))

def layer_of(name):
    return name.split('.')[1] if name.startswith('layers.') else 'readout'

def grouped_cos(g_rule, g_ref):
    """cosine per layer (all params of that layer concatenated) and for proj.weight alone."""
    out = {}
    groups = sorted(set(layer_of(k) for k in g_ref))
    for gname in groups:
        ks = [k for k in g_ref if layer_of(k) == gname]
        a = torch.cat([g_rule[k].flatten() for k in ks]); b = torch.cat([g_ref[k].flatten() for k in ks])
        out[f'layer{gname}' if gname != 'readout' else 'readout'] = cosine(a, b)
    for k in g_ref:
        if k.endswith('proj.weight'): out[k] = cosine(g_rule[k], g_ref[k])
    return out

def ref_config(ds, cfg):
    return Config(**dict(cfg.arch, dataset=ds, n_in=cfg.n_in, n_classes=cfg.n_classes_of[ds],
                         lr=cfg.lr, weight_decay=cfg.weight_decay, grad_clip=cfg.grad_clip, batch_size=cfg.batch_size,
                         n_epochs=cfg.n_epochs, patience=cfg.patience, dt_factor=1, seeds=list(cfg.seeds)))

def build_clearnet(ds, seed, Xtr, ytr, cfg, device):
    """Reference net (seeded init) -> BN statistics calibrated on shuffled train batches (MF forward, beta_start),
    then frozen. Identical for every rule given the seed."""
    ccfg = ref_config(ds, cfg); set_all_seeds(seed)
    net = DendStochAudioNet(ccfg).to(device)
    idx = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed))[:cfg.bn_calib_batches * cfg.batch_size]
    net = recalibrate_bn(net, ArrLoader(Xtr[idx], ytr[idx], cfg.batch_size, device=device), ccfg.beta_start, cfg.bn_calib_batches)
    cn = ClearNet(net, seed).to(device); cn.ccfg = ccfg
    return cn

def eval_clear(cn, X, y, beta, cfg):
    net = cn.net; net.eval(); ld = ArrLoader(X, y, cfg.batch_size, device=next(cn.parameters()).device); out = {'graded': evaluate(net, ld, 'graded', beta)}
    for n in cfg.eval_sample_ns:
        set_all_seeds(cfg.eval_seed + n); out[f'sampled_n{n}'] = evaluate(net, ld, 'sample', beta, n_samples=n)
    return out

def train_rule(ds, rule, seed, data, ck_dir, desc, cfg, device):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    cn = build_clearnet(ds, seed, Xtr, ytr, cfg, device); ccfg = cn.ccfg; net = cn.net
    gate_explicit(cn, Xva[:16].to(device).float(), ccfg.beta_end)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    tr = ArrLoader(Xtr, ytr, cfg.batch_size, shuffle=True, seed=seed, device=device); va = ArrLoader(Xva, yva, cfg.batch_size, device=device)
    ck_dir.mkdir(parents=True, exist_ok=True); ck = ck_dir / 'latest.pt'
    start, best_val, best_state, wait, ep_times, cos_log = 0, -1.0, copy.deepcopy(net.state_dict()), 0, [], []
    best_ep = 0
    if ck.exists():
        try:
            c = torch.load(ck, map_location=device, weights_only=False)
            if c.get('done', False):
                net.load_state_dict(c['best_state']); print(f"      [done] {desc}: from checkpoint", flush=True)
                return c['info'], cn
            net.load_state_dict(c['model']); opt.load_state_dict(c['opt']); sched.load_state_dict(c['sched'])
            start = c['epoch'] + 1; best_val = c['best_val']; best_state = c['best_state']; wait = c['wait']
            ep_times = c['ep_times']; cos_log = c['cos_log']; best_ep = c.get('best_ep', 0); torch.set_rng_state(c['rng_cpu'].cpu())
            if torch.cuda.is_available() and c.get('rng_cuda') is not None: torch.cuda.set_rng_state_all([state.cpu() for state in c['rng_cuda']])
            print(f"      [RESUME] {desc} ep {start} (best_val {best_val:.4f})", flush=True)
        except Exception as e:
            raise RuntimeError("Unable to resume the local-learning checkpoint") from e
    xc = torch.cat([Xva[i * cfg.batch_size:(i + 1) * cfg.batch_size] for i in range(cfg.cos_batches)]).to(device).float()
    yc = torch.cat([yva[i * cfg.batch_size:(i + 1) * cfg.batch_size] for i in range(cfg.cos_batches)]).to(device)
    nb = len(tr); pbar = tqdm(total=(cfg.n_epochs - start) * nb, desc=desc, leave=False, dynamic_ncols=True)
    epoch = max(start - 1, 0)
    for epoch in range(start, cfg.n_epochs):
        beta = anneal(epoch, cfg.n_epochs, ccfg.beta_start, ccfg.beta_end, ccfg.anneal_frac)
        net.train(); tr.set_epoch(epoch); t0 = time.time(); seen = corr = 0; loss_sum = 0.0
        if epoch % cfg.cos_every == 0 or cfg.smoke:
            set_all_seeds(cfg.eval_seed + epoch)
            masks = cn.make_masks(xc, beta)
            g_ref, z, caches, cro = bptt_grads(cn, xc, yc, beta, masks)
            ent = {'epoch': epoch}
            for rr in sorted(set([rule, 'exact']) - {'bptt'}):
                ent[rr] = grouped_cos(cn.local_grads(z, yc, caches, cro, beta, rr), g_ref)
            cos_log.append(ent)
        for xb, yb in tr:
            xb = augment(xb, cfg.aug); yb = yb.to(device)
            masks = cn.make_masks(xb, beta)
            opt.zero_grad(set_to_none=True)
            if rule == 'bptt':
                z, caches, _ = cn.forward_explicit(xb, beta, masks); loss = cn.loss_of(z, yb, caches)
                if not torch.isfinite(loss): raise FloatingPointError("Non-finite local-learning loss")
                loss.backward()
            else:
                with torch.no_grad():
                    z, caches, cro = cn.forward_explicit(xb, beta, masks); loss = cn.loss_of(z, yb, caches)
                    if not torch.isfinite(loss): raise FloatingPointError("Non-finite local-learning loss")
                    grads = cn.local_grads(z, yb, caches, cro, beta, rule)
                for nm, p_ in net.named_parameters(): p_.grad = grads[nm]
            if cfg.grad_clip: nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            seen += yb.size(0); corr += (z.argmax(1) == yb).sum().item(); loss_sum += float(loss.detach()) * yb.size(0); pbar.update(1)
        sched.step()
        va_acc = evaluate(net, va, 'graded', beta)['acc']
        if va_acc > best_val: best_val = va_acc; best_state = copy.deepcopy(net.state_dict()); wait = 0; best_ep = epoch + 1
        else: wait += 1
        dt_ep = time.time() - t0; ep_times.append(dt_ep)
        pbar.set_postfix_str(f"ep {epoch+1}/{cfg.n_epochs} train {corr/max(seen,1):.3f} val {va_acc:.4f} wait {wait}")
        if (epoch + 1) % 10 == 0 or epoch == cfg.n_epochs - 1 or cfg.smoke:
            last = cos_log[-1] if cos_log else {}
            cs_ = last.get(rule if rule != 'bptt' else 'exact', {})
            pbar.write(f"      ep {epoch+1:>3}/{cfg.n_epochs} loss {loss_sum/max(seen,1):.3f} train {corr/max(seen,1):.3f} "
                       f"val {va_acc:.4f} wait {wait} ({dt_ep:.0f}s) | cos to BPTT @ep{last.get('epoch', '-')}: "
                       + " ".join(f"{k} {v:.3f}" for k, v in cs_.items() if k.startswith('layer') or k == 'readout'))
        save_torch_atomic({'epoch': epoch, 'model': net.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                           'best_val': best_val, 'best_state': best_state, 'wait': wait, 'ep_times': ep_times,
                           'cos_log': cos_log, 'best_ep': best_ep, 'done': False, 'rng_cpu': torch.get_rng_state(),
                           'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, ck)
        if wait >= cfg.patience and not cfg.fixed_epochs:
            pbar.write(f"      [stop] patience {cfg.patience} @ ep {epoch+1}"); break
    pbar.close()
    if cfg.fixed_epochs:                              # the model after the last epoch, evaluated next to the best-validation one
        save_torch_atomic({'state': copy.deepcopy(net.state_dict()), 'epoch': epoch + 1}, ck_dir / 'last_model.pt')
    net.load_state_dict(best_state); net.eval()
    info = {'rule': rule, 'seed': seed, 'best_val': best_val, 'best_epoch': best_ep, 'stopped_ep': epoch + 1, 'mean_epoch_s': float(np.mean(ep_times)) if ep_times else None,
            'cos_log': cos_log, 'n_params': sum(p_.numel() for p_ in net.parameters())}
    save_torch_atomic({'state': best_state, 'cfg': asdict(ccfg), 'train': {'rule': rule, 'seed': seed, 'bn': 'frozen after calibration'},
                       'info': info}, ck_dir / 'best_model.pt')
    save_torch_atomic({'epoch': epoch, 'best_state': best_state, 'done': True, 'info': info}, ck)
    return info, cn


RULES = {
    'bptt': 'bptt',
    'clear_symmetric_lag_1': 'clear_sym',
    'clear_random_feedback': 'clear_random',
    'clear_symmetric_lag_8': 'clear_sym_k8',
    'clear_symmetric_lag_32': 'clear_sym_k32',
}


def run(config):
    """Train explicit rule/seed jobs and evaluate best-validation and final models."""
    dataset = config['dataset']['name']
    if dataset not in ('shd', 'ssc'):
        raise ValueError('Local-learning training supports SHD and SSC')
    defaults = dict(
        lr=.001, weight_decay=.0001, grad_clip=1., batch_size=128,
        n_epochs=100, patience=20, fixed_epochs=True, bn_calib_batches=30,
        cos_every=10, cos_batches=2, eval_sample_ns=[1, 5], eval_seed=2026,
        aug={'aug_jitter': 4, 'aug_chdrop': .05, 'aug_tmask': 20}, tf32=True,
    )
    unknown = set(config.get('training', {})) - defaults.keys()
    if unknown:
        raise ValueError(f'Unknown local-learning settings: {sorted(unknown)}')
    defaults.update(config.get('training', {}))
    defaults.update(
        n_in=700, n_classes_of={'shd': 20, 'ssc': 35},
        arch=dict(complex_poles=False, zoh_b=True, norm_type='batchnorm',
                  readout_pool='mean', ct_exact=True, n_layers=2, k_modes=8,
                  n_hidden=256, hidden_dropout=.2, beta_start=1., beta_end=5.,
                  anneal_frac=.7, reg_coef=.02, rate_target=.1),
        seeds=config.get('seeds', [42, 123, 999]), smoke=False,
    )
    defaults['arch'].update(config.get('model', {}))
    cfg = SimpleNamespace(**defaults)
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    rules = config.get('rules', list(RULES) if dataset == 'shd' else list(RULES)[:3])
    if not rules or any(rule not in RULES for rule in rules):
        raise ValueError(f'rules must select from {list(RULES)}')
    if cfg.n_epochs < 1 or cfg.cos_every < 1 or cfg.cos_batches < 1:
        raise ValueError('Epoch and diagnostic counts must be positive')
    data = load_data(config['dataset']['path'], dataset,
                     config['dataset'].get('fingerprint'))
    directory = resolve_path(config.get('output', f'outputs/local_learning_{dataset}'))
    directory.mkdir(parents=True, exist_ok=True)
    protocol = directory / 'configuration.json'
    if protocol.exists() and json.loads(protocol.read_text()) != config:
        raise ValueError('Output directory belongs to a different learning protocol')
    protocol.write_text(json.dumps(config, indent=2) + '\n')
    rows = []
    for label in rules:
        for seed in cfg.seeds:
            print(f'Local learning: {dataset}, {label}, seed {seed}', flush=True)
            checkpoint = directory / 'checkpoints' / label / f'seed_{seed}'
            info, model = train_rule(dataset, RULES[label], seed, data,
                                     checkpoint, f'{dataset} {label} seed {seed}', cfg, device)
            row = {'rule': label, 'seed': seed, 'training': info,
                   'best_validation_model': eval_clear(model, *data[2], model.ccfg.beta_end, cfg)}
            if cfg.fixed_epochs:
                final = torch.load(checkpoint / 'last_model.pt', map_location=device, weights_only=False)
                model.net.load_state_dict(final['state'])
                row['last_epoch_model'] = eval_clear(model, *data[2], model.ccfg.beta_end, cfg)
                row['last_epoch'] = final['epoch']
            rows.append(row)
            (directory / f'{label}_seed_{seed}.json').write_text(json.dumps(row, indent=2) + '\n')
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    result = {'experiment': 'local_learning', 'dataset': dataset, 'rows': rows}
    (directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
