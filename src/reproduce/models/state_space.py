"""Pinned P-SpikeSSM model and explicit sampling interventions.

The upstream implementation is loaded from an unmodified local checkout.
This module never installs dependencies or downloads code at import time.
"""
import copy
import hashlib
import importlib
import logging
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

UPSTREAM_URL = 'https://github.com/NeuroCompLab-psu/PSpikeSSMs.git'
UPSTREAM_COMMIT = '8f7a954852ae29f4ef961bac341321f7f9330c51'
UPSTREAM_FILE_MD5 = {'src/models/functional/cauchy.py': '3745a416d902bc27a354e4189d11de3d', 'src/models/functional/krylov.py': '58877cc62f58530fba1672e91339c2bd', 'src/models/functional/toeplitz.py': '435e56203d961af249647db861ee71f3', 'src/models/functional/vandermonde.py': 'c7a743c58cf2a2949fa53d2df804ec0f', 'src/models/hippo/hippo.py': '0805e38a540c51e67c022f8354e9510d', 'src/models/nn/__init__.py': '77f6178fb7494bec15ac7203e17708f5', 'src/models/nn/activation.py': 'a327e42ba7a07085fdd88f485fec420f', 'src/models/nn/dropout.py': '3e03be085c634fe23e80dd3cbfb2551c', 'src/models/nn/linear.py': '44eeafca6ca62274e9abd712c09ac09d', 'src/models/nn/normalization.py': 'a3adec5e814bb25c60b970b4d577f834', 'src/models/nn/residual.py': 'b54f2c7ef384c4f0242bd6c4733d78f1', 'src/models/sequence/__init__.py': '09983990d0fb821f6d65f399e0f3d0e9', 'src/models/sequence/backbones/block.py': '76f33784363ac6194a04dd479e9cf37c', 'src/models/sequence/backbones/model.py': 'cafef1b1b8de2ac4cfba01943a5b6701', 'src/models/sequence/base.py': '1e65b9cd0f98b23935d134d19f91a0d2', 'src/models/sequence/kernels/__init__.py': 'a705a44c4bd02013f5f932c6bdf9d831', 'src/models/sequence/kernels/dplr.py': 'fe3c66103c2e632905ab2f3a77592c98', 'src/models/sequence/kernels/fftconv.py': 'bab748f3fafae3fb5ea7bbf1db2979fc', 'src/models/sequence/kernels/kernel.py': '5edf6c42dc4c48e70308d08dab353f19', 'src/models/sequence/kernels/ssm.py': '2bca7140acfd69e0f73d974a5096c7f9', 'src/models/sequence/modules/pool.py': 'c73df6360d6d70b9c529b9c3ad4e17c2', 'src/models/sequence/modules/s4block.py': 'd6902afc4cd4777d8cc9a5888049e608', 'src/utils/__init__.py': '4a83f82f60ef8e4d378bf45d4d945d1a', 'src/utils/config.py': '919f494594f8512b7054e435576261ac', 'src/utils/registry.py': 'e592fc671eec1bd15e8163a01e6137a7'}

def import_upstream(source):
    """Verify the pinned commit and model files before importing upstream code."""
    from ..paths import resolve_path
    source = resolve_path(source)
    if not (source / '.git').exists():
        raise FileNotFoundError(
            f"Pinned P-SpikeSSM source is missing. Clone {UPSTREAM_URL} into "
            "external_code/pspikessm and check out the commit in the README."
        )
    head = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f'P-SpikeSSM commit {head} does not match {UPSTREAM_COMMIT}')
    dirty = subprocess.run(['git', '-C', str(source), 'diff', '--name-only', 'HEAD'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError('P-SpikeSSM checkout contains modified tracked files')
    for filename, expected in UPSTREAM_FILE_MD5.items():
        path = source / filename
        if not path.is_file() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'P-SpikeSSM source identity failed: {filename}')
    existing = sys.modules.get('src')
    identity = (str(source), UPSTREAM_COMMIT)
    if existing is not None and getattr(existing, '_reproduction_source', None) != identity:
        raise RuntimeError('Unverified src.* modules are already loaded; start a fresh Python process')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    # Upstream imports Lightning only for the logger used by its model modules.
    # The archived runner used this same logging-only compatibility module.
    try:
        importlib.import_module('pytorch_lightning')
    except ImportError:
        logger_module = types.ModuleType('src.utils.train')
        logger_module.get_logger = lambda name=__name__: logging.getLogger(name)
        sys.modules['src.utils.train'] = logger_module
        utilities = importlib.import_module('src.utils')
        utilities.train = logger_module
    logging.getLogger('src.models.sequence.kernels.ssm').setLevel(logging.ERROR)
    from src.models.sequence.backbones.model import SequenceModel
    import yaml
    base = yaml.safe_load((source / 'configs/model/base.yaml').read_text())
    layer = yaml.safe_load((source / 'configs/model/layer/pSpikeSSM.yaml').read_text())
    layer = copy.deepcopy(layer)
    layer.update(base['layer'])
    layer['bidirectional'] = False
    layer['tie_dropout'] = base['tie_dropout']
    sys.modules['src']._reproduction_source = identity
    return SequenceModel, base, layer


class PSpikeNet(nn.Module):
    """Official causal two-block backbone with the matched input/output maps."""
    def __init__(self, sequence_model, base, layer, hidden, n_in, n_classes, timesteps):
        super().__init__()
        layer = copy.deepcopy(layer)
        layer['l_max'] = timesteps
        self.encoder = nn.Linear(n_in, hidden)
        self.backbone = sequence_model(
            d_model=hidden, n_layers=base['n_layers'], transposed=False,
            dropout=base['dropout'], tie_dropout=base['tie_dropout'],
            prenorm=base['prenorm'], bidirectional=False, layer=layer,
            residual=base['residual'], norm=base['norm'],
            pool=copy.deepcopy(base['pool']), track_norms=False,
        )
        self.decoder = nn.Linear(hidden, n_classes)
        attach_sampling_controls(self)

    def forward(self, x, rate=1.0):
        y, _ = (self.backbone(self.encoder(x)) if rate == 1.0
                else self.backbone(self.encoder(x), rate=rate))
        return self.decoder(y.mean(1))


def build_model(source, *, n_in=700, n_classes=20, timesteps=100, hidden=200, device='cpu'):
    sequence_model, base, layer = import_upstream(source)
    return PSpikeNet(sequence_model, base, layer, hidden, n_in, n_classes, timesteps).to(device)


def attach_sampling_controls(model):
    """Preserve upstream straight-through sampling and expose slope/site controls.

    With slope=1 and both sites enabled, probabilities are upstream clamp(y,0,1).
    Sites are ordered as the S4 input and the mixer input within each block.
    """
    model.sampling_mode = 'sample'
    model.sampler_slope = 1.0
    model.sample_sites = (True, True)
    model.sampler_statistics = None
    for index, block in enumerate(model.backbone.layers):
        block._sampling_call = 0
        def generate(y, random_uniform, _block=block, _index=index, _model=model):
            site = _block._sampling_call
            _block._sampling_call = (site + 1) % 2
            slope = float(_model.sampler_slope)
            p = (torch.clamp(y, 0, 1) if slope == 1.0 else
                 torch.clamp(0.5 + slope * (y - 0.5), 0, 1))
            if _model.sampler_statistics is not None:
                with torch.no_grad():
                    q = p.detach().double()
                    row = _model.sampler_statistics.setdefault((_index, site), [0., 0., 0, 0])
                    row[0] += float(q.sum())
                    row[1] += float((q * (1 - q)).sum())
                    row[2] += q.numel()
                    row[3] += int(((q < 0.05) | (q > 0.95)).sum())
            if _model.sampling_mode == 'meanfield' or not _model.sample_sites[site]:
                return p
            spikes = torch.where(p > random_uniform, torch.ones_like(p), torch.zeros_like(p))
            return _block.Replace.apply(p, spikes)
        block.stochSpikeGen = generate
    def reset_calls(module, inputs):
        for block in model.backbone.layers:
            block._sampling_call = 0
    model.backbone.register_forward_pre_hook(reset_calls)
    return model


def make_optimizer(model, lr=0.001, weight_decay=0.0001):
    """Keep upstream per-parameter S4 learning-rate and weight-decay groups."""
    all_parameters = list(model.parameters())
    normal = [p for p in all_parameters if not hasattr(p, '_optim')]
    default = {'lr': lr, 'weight_decay': weight_decay}
    optimizer = torch.optim.AdamW(normal, **default)
    groups = [getattr(p, '_optim') for p in all_parameters if hasattr(p, '_optim')]
    groups = [dict(group) for group in sorted(dict.fromkeys(frozenset(g.items()) for g in groups))]
    for group in groups:
        optimizer.add_param_group({
            'params': [p for p in all_parameters if getattr(p, '_optim', None) == group],
            **default, **group,
        })
    return optimizer, groups


@torch.no_grad()
def evaluate(model, loader, n_samples=1, rate=1.0):
    if n_samples < 1:
        raise ValueError('n_samples must be positive')
    model.eval()
    predictions, labels = [], []
    device = next(model.parameters()).device
    for x, y in loader:
        x = x.to(device).float()
        probabilities = sum(F.softmax(model(x, rate=rate), dim=1) for _ in range(n_samples))
        predictions.append(probabilities.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    if not labels:
        raise ValueError('Evaluation data are empty')
    predicted, target = np.concatenate(predictions), np.concatenate(labels)
    f1 = []
    for category in np.unique(target):
        tp = np.sum((predicted == category) & (target == category))
        fp = np.sum((predicted == category) & (target != category))
        fn = np.sum((predicted != category) & (target == category))
        precision, recall = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
        f1.append(2 * precision * recall / (precision + recall + 1e-9))
    return {'acc': float((predicted == target).mean()), 'macro_f1': float(np.mean(f1))}


def train_one(seed, data, model_options, cfg, checkpoint_dir, *, rule='meanfield',
              beta_start=1., beta_end=1., anneal_frac=.7, train_pool=1):
    """Train the pinned backbone using the archived matched recipe."""
    import time
    import random
    from dataclasses import asdict
    from ..datasets import ArrLoader, pool_batch
    from ..paths import resolve_path
    from ..training import augment, save_torch_atomic, set_all_seeds
    from .crisp import anneal
    if rule not in ('meanfield', 'sample_ste'):
        raise ValueError('P-SpikeSSM rule must be meanfield or sample_ste')
    (train_x, train_y), (val_x, val_y), _ = data
    set_all_seeds(seed)
    if str(cfg.device).startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
        torch.backends.cudnn.allow_tf32 = cfg.tf32
    model = build_model(**model_options, device=cfg.device)
    optimizer, groups = make_optimizer(model, cfg.lr, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)
    train = ArrLoader(train_x, train_y, cfg.batch_size, shuffle=True, seed=seed, device=cfg.device)
    validation = ArrLoader(val_x, val_y, cfg.batch_size, device=cfg.device)
    directory = resolve_path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / 'latest.pt'
    protocol = {'seed': seed, 'model': model_options, 'training': asdict(cfg), 'rule': rule,
                'beta_start': beta_start, 'beta_end': beta_end, 'anneal_frac': anneal_frac,
                'train_pool': train_pool}
    start, best_validation, best_epoch, wait = 0, -1., 0, 0
    best_state, epoch_times = copy.deepcopy(model.state_dict()), []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=cfg.device, weights_only=False)
        if saved.get('protocol') != protocol:
            raise RuntimeError('Existing checkpoint belongs to a different protocol')
        if saved.get('done'):
            model.load_state_dict(saved['best_state'])
            model.eval()
            return saved['info'], model
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best_validation = saved['epoch'] + 1, saved['best_validation']
        best_epoch, wait = saved['best_epoch'], saved['wait']
        best_state, epoch_times = saved['best_state'], saved['epoch_times']
        torch.set_rng_state(saved['rng_cpu'].cpu())
        if torch.cuda.is_available() and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['rng_cuda']])
    for epoch in range(start, cfg.n_epochs):
        t0 = time.perf_counter()
        model.train()
        model.sampling_mode = 'meanfield' if rule == 'meanfield' else 'sample'
        model.sampler_slope = anneal(epoch, cfg.n_epochs, beta_start, beta_end, anneal_frac)
        train.set_epoch(epoch)
        loss_sum, seen = 0., 0
        for x, y in train:
            x = augment(x, cfg.aug)
            if train_pool > 1:
                x = pool_batch(x, train_pool)
            y = y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; last checkpoint is preserved')
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)
        scheduler.step()
        rng_python, rng_numpy = random.getstate(), np.random.get_state()
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        old_slope = model.sampler_slope
        if cfg.validation_beta is not None:
            model.sampler_slope = cfg.validation_beta
        set_all_seeds(cfg.eval_seed + epoch)
        accuracy = evaluate(model, validation)['acc']
        model.sampler_slope = old_slope
        if cfg.preserve_validation_rng:
            random.setstate(rng_python)
            np.random.set_state(rng_numpy)
            torch.set_rng_state(rng_cpu)
            if rng_cuda is not None:
                torch.cuda.set_rng_state_all(rng_cuda)
        if accuracy > best_validation:
            best_validation, best_epoch, wait = accuracy, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        epoch_times.append(time.perf_counter() - t0)
        print(f'P-SpikeSSM seed {seed}: epoch {epoch + 1}/{cfg.n_epochs}, '
              f'loss {loss_sum / max(seen, 1):.5f}, validation {accuracy:.5f}, '
              f'best {best_validation:.5f}', flush=True)
        save_torch_atomic({
            'protocol': protocol, 'epoch': epoch, 'state': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'best_validation': best_validation, 'best_epoch': best_epoch, 'wait': wait,
            'best_state': best_state, 'epoch_times': epoch_times, 'done': False,
            'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, checkpoint)
        if wait >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    model.sampler_slope = beta_end
    info = {'seed': seed, 'best_val': best_validation, 'best_epoch': best_epoch,
            'n_epochs_run': len(epoch_times), 'mean_epoch_s': float(np.mean(epoch_times)),
            'param_groups': groups}
    save_torch_atomic({'state': best_state, 'protocol': protocol, 'info': info}, directory / 'best_model.pt')
    save_torch_atomic({'done': True, 'protocol': protocol, 'best_state': best_state, 'info': info}, checkpoint)
    return info, model
