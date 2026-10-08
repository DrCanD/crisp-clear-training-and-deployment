"""CRISP training with the archived matched recipe and explicit paths.

Mean-field, sampled straight-through and threshold surrogate-gradient rules
share one model. Model selection uses validation data. Corrupt resume files
and nonfinite losses fail explicitly; finite-update arithmetic is preserved.
"""
import copy
import os
import random
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from .paths import resolve_path
from .datasets import ArrLoader, pool_batch, temporal_pool_exact, load_data
from .models.crisp import Config, build_model, anneal, evaluate

def set_all_seeds(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def save_torch_atomic(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def augment(X, trial):
    """Apply augmentation in-place-ish on a [B,T,C] GPU batch. Training only."""
    B, T, C = X.shape
    if trial['aug_jitter'] > 0:
        ms = trial['aug_jitter']
        shifts = torch.randint(-ms, ms + 1, (B,), device=X.device)
        idx = (torch.arange(T, device=X.device).unsqueeze(0) - shifts.unsqueeze(1)) % T
        X = torch.gather(X, 1, idx.unsqueeze(-1).expand(B, T, C))
    if trial['aug_chdrop'] > 0:
        mask = (torch.rand(B, 1, C, device=X.device) > trial['aug_chdrop']).float()
        X = X * mask
    if trial['aug_tmask'] > 0:
        w = int(torch.randint(0, trial['aug_tmask'] + 1, (1,)))
        if w > 0:
            starts = torch.randint(0, max(1, T - w), (B,), device=X.device)
            tidx = torch.arange(T, device=X.device).unsqueeze(0)
            m = (tidx >= starts.unsqueeze(1)) & (tidx < (starts + w).unsqueeze(1))
            X = X * (~m).float().unsqueeze(-1)
    return X

TRAIN_MODE_OF = {"meanfield": "meanfield", "sample_ste": "sample_ste", "heaviside_sg": "heaviside_sg"}
DEPLOY_OF = {"meanfield": "graded", "sample_ste": "sample", "heaviside_sg": "heaviside"}

@dataclass
class TrainConfig:
    lr: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    batch_size: int = 128
    n_epochs: int = 100
    patience: int = 20
    schedule: str = "cosine"
    eval_seed: int = 2026
    tf32: bool = True
    aug: dict = field(default_factory=lambda: {"aug_jitter": 4, "aug_chdrop": 0.05, "aug_tmask": 20})
    coarse_aug_domain: str = "fine_then_pool"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Matched slope-schedule control validates at the same fixed slope.
    validation_beta: float | None = None
    preserve_validation_rng: bool = False


@torch.no_grad()
def eval_mode(model, loader, mode, beta, n_samples=1):
    """Evaluate probabilities, sampled predictions or threshold spikes."""
    if mode in ('graded', 'sample'):
        return evaluate(model, loader, mode, beta, n_samples=n_samples)
    model.eval()
    yp, yt, rates = ([], [], [])
    for xb, yb in loader:
        lg, mr, _ = model(xb.to(next(model.parameters()).device), 'heaviside', beta)
        yp.append(lg.argmax(1).cpu().numpy())
        yt.append(yb.numpy())
        rates.append(float(mr))
    yp = np.concatenate(yp)
    yt = np.concatenate(yt)
    f1s = []
    for cc in np.unique(yt):
        tp = np.sum((yp == cc) & (yt == cc))
        fp = np.sum((yp == cc) & (yt != cc))
        fn = np.sum((yp != cc) & (yt == cc))
        pr = tp / (tp + fp + 1e-09)
        rc = tp / (tp + fn + 1e-09)
        f1s.append(2 * pr * rc / (pr + rc + 1e-09))
    return {'acc': float((yp == yt).mean()), 'macro_f1': float(np.mean(f1s)), 'firing_rate': float(np.mean(rates))}

def train_one(seed, data, rule, ccfg, cfg, checkpoint_dir, train_pool=1):
    """CRISP forward with a slope schedule and epoch checkpoints.
    train_pool > 1: data[0] is the T100 train set; each batch is augmented at T100, then pooled x train_pool."""
    device = torch.device(cfg.device)
    ck_dir = resolve_path(checkpoint_dir)
    desc = f"crisp_{ccfg.dataset}_{rule}_seed{seed}"
    if rule not in TRAIN_MODE_OF:
        raise ValueError(f"Unknown training rule: {rule}")
    if cfg.schedule != "cosine":
        raise ValueError("The archived recipe uses the cosine learning-rate schedule")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    T_model = int(Xtr.shape[1]) // train_pool
    set_all_seeds(seed)
    tr = ArrLoader(Xtr, ytr, cfg.batch_size, shuffle=True, seed=seed, device=device)
    va = ArrLoader(Xva, yva, cfg.batch_size, device=device)
    model = build_model(ccfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.n_epochs)
    n_params = sum((p.numel() for p in model.parameters()))
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / 'latest.pt'
    start, best_val, best_state, wait, ep_times = (0, -1.0, copy.deepcopy(model.state_dict()), 0, [])
    if ck.exists():
        try:
            c = torch.load(ck, map_location=device, weights_only=False)
            if c.get('done', False):
                model.load_state_dict(c['best_state'])
                model.eval()
                print(f'      [done] {desc}: from checkpoint', flush=True)
                return (c['train_info'], model)
            model.load_state_dict(c['model'])
            opt.load_state_dict(c['opt'])
            sched.load_state_dict(c['sched'])
            start = c['epoch'] + 1
            best_val = c['best_val']
            best_state = c['best_state']
            wait = c.get('wait', 0)
            ep_times = c.get('ep_times', [])
            torch.set_rng_state(c['rng_cpu'].cpu())
            if torch.cuda.is_available() and c.get('rng_cuda') is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in c['rng_cuda']])
            print(f'      [RESUME] {desc} ep {start} (best_val {best_val:.4f})', flush=True)
        except Exception as e:
            raise RuntimeError("Existing checkpoint could not be restored; preserving it") from e
    train_mode, val_mode = (TRAIN_MODE_OF[rule], DEPLOY_OF[rule])
    print(f"      [model] {n_params / 1000.0:.0f}K params | {desc} | T={T_model} | train '{train_mode}', validate '{val_mode}'" + (f' | augment at T{int(Xtr.shape[1])}, then pool x{train_pool}' if train_pool > 1 else ''), flush=True)
    nb = len(tr)
    pbar = tqdm(total=(cfg.n_epochs - start) * nb, desc=desc, leave=False, dynamic_ncols=True)
    epoch = max(start - 1, 0)
    for epoch in range(start, cfg.n_epochs):
        beta = anneal(epoch, cfg.n_epochs, ccfg.beta_start, ccfg.beta_end, ccfg.anneal_frac)
        t0 = time.time()
        model.train()
        tr.set_epoch(epoch)
        seen = corr = 0
        loss_sum = 0.0
        for xb, yb in tr:
            xb = augment(xb.to(device), cfg.aug)
            yb = yb.to(device)
            if train_pool > 1:
                xb = pool_batch(xb, train_pool)
            opt.zero_grad(set_to_none=True)
            logits, mr, _ = model(xb, train_mode, beta)
            loss = F.cross_entropy(logits, yb) + ccfg.reg_coef * (mr - ccfg.rate_target) ** 2
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; preserving the last checkpoint")
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            seen += yb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
            loss_sum += loss.item() * yb.size(0)
            pbar.update(1)
        sched.step()
        rng_before_validation = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        set_all_seeds(cfg.eval_seed + epoch)
        selected_beta = beta if cfg.validation_beta is None else cfg.validation_beta
        va_acc = eval_mode(model, va, val_mode, selected_beta)['acc']
        if cfg.preserve_validation_rng:
            random.setstate(rng_before_validation[0])
            np.random.set_state(rng_before_validation[1])
            torch.set_rng_state(rng_before_validation[2])
            if rng_before_validation[3] is not None:
                torch.cuda.set_rng_state_all(rng_before_validation[3])
        if va_acc > best_val:
            best_val = va_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        dt_ep = time.time() - t0
        ep_times.append(dt_ep)

        pbar.set_postfix_str(f'ep {epoch + 1}/{cfg.n_epochs} beta {beta:.2f} train {corr / max(seen, 1):.3f} val {va_acc:.4f} wait {wait}')
        if epoch % 5 == 0 or epoch == cfg.n_epochs - 1 :
            pbar.write(f'      ep {epoch + 1:>3}/{cfg.n_epochs} loss {loss_sum / max(seen, 1):.3f} train {corr / max(seen, 1):.3f} val({val_mode}) {va_acc:.4f} beta {beta:.2f} wait {wait} ({dt_ep:.0f}s)')
        save_torch_atomic({'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'best_val': best_val, 'best_state': best_state, 'wait': wait, 'ep_times': ep_times, 'done': False, 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, ck)
        if wait >= cfg.patience:
            pbar.write(f'      [stop] patience {cfg.patience} @ ep {epoch + 1}')
            break
    pbar.close()
    model.load_state_dict(best_state)
    model.eval()
    info = {'seed': seed, 'rule': rule, 'best_val': best_val, 'val_mode': val_mode, 'stopped_ep': epoch + 1, 'n_params': n_params, 'T': T_model, 'mean_epoch_s': float(np.mean(ep_times)) if ep_times else None, 'n_epochs_run': len(ep_times)}
    save_torch_atomic({'state': best_state, 'cfg': asdict(ccfg), 'train': {'rule': rule, 'seed': seed, 'recipe': {'lr': cfg.lr, 'weight_decay': cfg.weight_decay, 'schedule': cfg.schedule, 'aug': cfg.aug, 'batch_size': cfg.batch_size, 'n_epochs': cfg.n_epochs, 'patience': cfg.patience, 'aug_domain': cfg.coarse_aug_domain if train_pool > 1 else 'native'}}, 'info': info}, ck_dir / 'best_model.pt')
    save_torch_atomic({'epoch': epoch, 'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'best_val': best_val, 'best_state': best_state, 'wait': wait, 'ep_times': ep_times, 'done': True, 'train_info': info}, ck)
    return (info, model)
