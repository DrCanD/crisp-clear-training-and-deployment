"""Matched leaky integrate-and-fire baseline, preserving the archived recurrence."""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class Config:
    n_in: int = 700
    n_hidden: int = 256
    n_layers: int = 2
    hidden_dropout: float = 0.20
    norm_type: str = "batchnorm"
    surrogate_beta: float = 5.0
    learnable_tau: bool = True
    tau_init_alpha: float = 0.70
    theta_init: float = 0.0
    readout_type: str = "li"
    readout_pool: str = "mean"
    readout_learnable_tau: bool = True
    reg_scope: str = "hidden"

class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward, fast-sigmoid surrogate backward."""

    @staticmethod
    def forward(ctx, v, beta):
        ctx.save_for_backward(v)
        ctx.beta = beta
        return (v > 0).float()

    @staticmethod
    def backward(ctx, grad_out):
        v, = ctx.saved_tensors
        beta = ctx.beta
        sg = beta / (2 * (1 + beta * v.abs()) ** 2)
        return (grad_out * sg, None)

def _raw_tau_for_alpha(alpha):
    tau = -1.0 / math.log(alpha)
    return math.log(math.expm1(tau))

class LIFLayer(nn.Module):
    """linear proj -> BN -> per-step LIF with hard reset. alpha = exp(-dt/tau), dt held for dt-transfer."""

    def __init__(self, d_in, d_out, dropout, norm_type, surrogate_beta, learnable_tau, alpha_init, theta_init):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.norm_type = norm_type
        self.norm = nn.LayerNorm(d_out) if norm_type == 'layernorm' else nn.BatchNorm1d(d_out) if norm_type == 'batchnorm' else nn.Identity()
        self.theta = nn.Parameter(torch.full((d_out,), float(theta_init)))
        self.drop = nn.Dropout(dropout)
        self.beta = surrogate_beta
        self.dt = 1.0
        raw = torch.full((d_out,), _raw_tau_for_alpha(alpha_init))
        if learnable_tau:
            self.raw_tau = nn.Parameter(raw)
        else:
            self.register_buffer('raw_tau', raw)

    def alpha(self):
        tau = F.softplus(self.raw_tau) + 0.001
        return torch.exp(-self.dt / tau)

    def norm_apply(self, V):
        if self.norm_type == 'batchnorm':
            B, T, d = V.shape
            return self.norm(V.reshape(B * T, d)).reshape(B, T, d)
        return self.norm(V)

    def forward(self, s_in):
        I = self.norm_apply(self.proj(s_in))
        B, T, d = I.shape
        a = self.alpha()
        v = torch.zeros(B, d, device=I.device, dtype=I.dtype)
        s_prev = torch.zeros(B, d, device=I.device, dtype=I.dtype)
        spikes = []
        for t in range(T):
            v = a * v * (1 - s_prev) + I[:, t]
            s = SurrogateSpike.apply(v - self.theta, self.beta)
            spikes.append(s)
            s_prev = s
        S = torch.stack(spikes, 1)
        S = self.drop(S)
        return (S, S.mean())

class LIReadout(nn.Module):
    """Linear proj -> leaky integrator, no threshold, no reset. Returns membrane [B,T,C]."""

    def __init__(self, d_in, n_classes, learnable_tau, alpha_init):
        super().__init__()
        self.proj = nn.Linear(d_in, n_classes)
        self.dt = 1.0
        raw = torch.full((n_classes,), _raw_tau_for_alpha(alpha_init))
        if learnable_tau:
            self.raw_tau = nn.Parameter(raw)
        else:
            self.register_buffer('raw_tau', raw)

    def alpha(self):
        tau = F.softplus(self.raw_tau) + 0.001
        return torch.exp(-self.dt / tau)

    def forward(self, s_in):
        I = self.proj(s_in)
        B, T, C = I.shape
        a = self.alpha()
        v = torch.zeros(B, C, device=I.device, dtype=I.dtype)
        vs = []
        for t in range(T):
            v = a * v + I[:, t]
            vs.append(v)
        return torch.stack(vs, 1)

class LIFNet(nn.Module):

    def __init__(self, cfg, n_classes):
        super().__init__()
        self.readout_pool = cfg.readout_pool
        self.reg_scope = cfg.reg_scope
        self.readout_type = cfg.readout_type
        dims = [cfg.n_in] + [cfg.n_hidden] * cfg.n_layers
        self.layers = nn.ModuleList([LIFLayer(dims[i], dims[i + 1], cfg.hidden_dropout, cfg.norm_type, cfg.surrogate_beta, cfg.learnable_tau, cfg.tau_init_alpha, cfg.theta_init) for i in range(cfg.n_layers)])
        if cfg.readout_type == 'li':
            self.ro = LIReadout(cfg.n_hidden, n_classes, cfg.readout_learnable_tau, cfg.tau_init_alpha)
        elif cfg.readout_type == 'lif':
            self.readout = nn.Linear(cfg.n_hidden, n_classes)
            self.ro = LIFLayer(n_classes, n_classes, 0.0, 'none', cfg.surrogate_beta, cfg.learnable_tau, cfg.tau_init_alpha, cfg.theta_init)
        else:
            raise ValueError(cfg.readout_type)

    def pool(self, x):
        return x.mean(dim=1) if self.readout_pool == 'mean' else x.max(dim=1).values

    def forward(self, x):
        s = x
        rates = []
        for layer in self.layers:
            s, r = layer(s)
            rates.append(r)
        if self.readout_type == 'li':
            out = self.ro(s)
        else:
            out, rr = self.ro(self.readout(s))
            if self.reg_scope == 'all':
                rates.append(rr)
        logits = self.pool(out)
        return (logits, torch.stack(rates).mean())

    def set_dt(self, dt):
        for m in self.modules():
            if isinstance(m, (LIFLayer, LIReadout)):
                m.dt = float(dt)


def build_model(options, n_classes, device='cpu'):
    return LIFNet(Config(**options), n_classes).to(device)


@torch.no_grad()
def evaluate(model, loader):
    import numpy as np
    model.eval()
    device = next(model.parameters()).device
    predicted, target, rates = [], [], []
    for x, y in loader:
        logits, rate = model(x.to(device).float())
        predicted.append(logits.argmax(1).cpu().numpy())
        target.append(y.cpu().numpy())
        rates.append(float(rate))
    predicted, target = np.concatenate(predicted), np.concatenate(target)
    f1 = []
    for category in np.unique(target):
        tp = np.sum((predicted == category) & (target == category))
        fp = np.sum((predicted == category) & (target != category))
        fn = np.sum((predicted != category) & (target == category))
        precision, recall = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
        f1.append(2 * precision * recall / (precision + recall + 1e-9))
    return {'acc': float((predicted == target).mean()), 'macro_f1': float(np.mean(f1)),
            'firing_rate': float(np.mean(rates))}


def install_zoh_hooks(model, factor, scale_spike_rate=False):
    """Preserve the original LIF zero-order-hold input-gain policies."""
    def gain(module):
        tau = F.softplus(module.raw_tau.detach()) + 1e-3
        coarse = torch.exp(-1.0 / tau)
        fine = torch.exp(-(1.0 / factor) / tau)
        return (1 - fine) / (1 - coarse)
    handles = []
    for index, layer in enumerate(model.layers):
        g = gain(layer)
        handles.append(layer.norm.register_forward_hook(lambda module, inputs, output, g=g: output * g))
        if scale_spike_rate and index >= 1:
            handles.append(layer.proj.register_forward_pre_hook(lambda module, inputs: (inputs[0] * factor,)))
    g = gain(model.ro)
    handles.append(model.ro.proj.register_forward_hook(lambda module, inputs, output, g=g: output * g))
    if scale_spike_rate:
        handles.append(model.ro.proj.register_forward_pre_hook(lambda module, inputs: (inputs[0] * factor,)))
    return handles


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches=30):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.train()
            module.reset_running_stats()
            module.momentum = None
    device = next(model.parameters()).device
    for index, (x, _) in enumerate(loader):
        if index >= n_batches:
            break
        model(x.to(device).float())
    model.eval()
    return model


def transfer_evaluation(base, factor, coarse_loader, fine_loader, calibration_loader, n_batches=30):
    import copy
    results = {}
    base.set_dt(1.)
    results['coarse_native'] = evaluate(base, coarse_loader)['acc']
    results['naive_same_dt'] = evaluate(base, fine_loader)['acc']
    for name, use_gain, use_rate in (('retimed', False, False), ('zoh_gain', True, False),
                                     ('zoh_gain_rate', True, True)):
        for calibrate in (False, True):
            model = copy.deepcopy(base)
            model.set_dt(1. / factor)
            if use_gain:
                install_zoh_hooks(model, factor, use_rate)
            if calibrate:
                recalibrate_bn(model, calibration_loader, n_batches)
            key = name + ('_bn_reestimated' if calibrate else '')
            results[key] = evaluate(model, fine_loader)['acc']
    free = ('naive_same_dt', 'retimed', 'zoh_gain', 'zoh_gain_rate')
    best = max(free, key=lambda key: results[key])
    results['test_oracle_best_data_free'] = results[best]
    results['test_oracle_policy'] = best
    return results


def train_one(seed, data, options, cfg, checkpoint_dir, *, n_classes,
              train_pool=1, reg_coef=.02, rate_target=.1):
    """Matched AdamW/cosine training with validation-only model selection."""
    import copy
    import time
    import numpy as np
    from ..datasets import ArrLoader, pool_batch
    from ..paths import resolve_path
    from ..training import augment, save_torch_atomic, set_all_seeds
    (train_x, train_y), (val_x, val_y), _ = data
    set_all_seeds(seed)
    if str(cfg.device).startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    model = build_model(options, n_classes, cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    train = ArrLoader(train_x, train_y, cfg.batch_size, shuffle=True, seed=seed, device=cfg.device)
    validation = ArrLoader(val_x, val_y, cfg.batch_size, device=cfg.device)
    directory = resolve_path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'latest.pt'
    start, best_validation, best_epoch, wait = 0, -1., 0, 0
    best_state, times = copy.deepcopy(model.state_dict()), []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
        if saved.get('done'):
            model.load_state_dict(saved['best_state'])
            model.eval()
            return saved['info'], model
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best_validation = saved['epoch'] + 1, saved['best_validation']
        best_epoch, wait = saved['best_epoch'], saved['wait']
        best_state, times = saved['best_state'], saved['times']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if torch.cuda.is_available() and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng_cuda']])
    for epoch in range(start, cfg.n_epochs):
        t0 = time.perf_counter()
        model.train()
        train.set_epoch(epoch)
        loss_sum, seen = 0., 0
        for x, y in train:
            x = augment(x, cfg.aug)
            if train_pool > 1:
                x = pool_batch(x, train_pool)
            y = y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits, rate = model(x)
            loss = F.cross_entropy(logits, y) + reg_coef * (rate - rate_target) ** 2
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; last checkpoint is preserved')
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()
        set_all_seeds(cfg.eval_seed + epoch)
        accuracy = evaluate(model, validation)['acc']
        if accuracy > best_validation:
            best_validation, best_epoch, wait = accuracy, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        times.append(time.perf_counter() - t0)
        print(f'LIF seed {seed}: epoch {epoch + 1}/{cfg.n_epochs}, '
              f'loss {loss_sum / max(seen, 1):.5f}, validation {accuracy:.5f}, best {best_validation:.5f}', flush=True)
        save_torch_atomic({
            'epoch': epoch, 'state': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'best_validation': best_validation,
            'best_epoch': best_epoch, 'wait': wait, 'best_state': best_state,
            'times': times, 'done': False, 'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, checkpoint)
        if wait >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    info = {'seed': seed, 'best_val': best_validation, 'best_epoch': best_epoch,
            'n_epochs_run': len(times), 'mean_epoch_s': float(np.mean(times))}
    save_torch_atomic({'state': best_state, 'cfg': options, 'n_classes': n_classes, 'info': info}, directory / 'best_model.pt')
    save_torch_atomic({'done': True, 'best_state': best_state, 'info': info}, checkpoint)
    return info, model
