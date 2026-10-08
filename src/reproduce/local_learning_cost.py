"""Measure inference, BPTT and CLEAR costs with the original synthetic workload.

This is a forward/backward cost comparison, without optimizer updates. Batch
normalization uses its initial frozen statistics; dropout masks are absent.
CLEAR uses the offline reverse-time adjoint with one-lag cross-layer feedback.
CUDA memory is the allocated peak above the pre-call baseline, not total
hardware memory. CPU memory is unavailable and is recorded as null.
"""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .learning import ClearNet, LearningConfig, gate_explicit
from .models.crisp import Config, DendStochAudioNet
from .paths import resolve_path


def measure(function, device, repeats):
    """One discarded warmup, followed by the original median of repeats."""
    cuda = device.type == 'cuda'
    times, memory = [], []
    for _ in range(repeats + 1):
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        start = time.perf_counter()
        function()
        if cuda:
            torch.cuda.synchronize(device)
            memory.append((torch.cuda.max_memory_allocated(device) - baseline) / 2 ** 20)
        times.append((time.perf_counter() - start) * 1000)
    return {'median_ms': float(np.median(times[1:])),
            'peak_increment_MB': float(np.median(memory[1:])) if cuda else None,
            'times_ms': times[1:], 'peak_increments_MB': memory[1:] if cuda else None,
            'warmup_ms': times[0], 'warmup_count': 1, 'repeats': repeats}


def _write(result, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(config):
    """Run the sequence-length sweep specified by the local_learning_cost config."""
    device = torch.device(config.get('device', 'cuda'))
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('The archived local-learning cost protocol supports CPU and CUDA')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    options = config.get('cost', {})
    lengths = tuple(int(value) for value in options.get('timesteps', [100, 250, 500, 1000, 2000]))
    batch_size = int(options.get('batch_size', 32))
    repeats = int(options.get('repeats', 3))
    seed = int(config.get('seed', 0))
    if not lengths or min(*lengths, batch_size, repeats) < 1:
        raise ValueError('Timesteps, batch_size and repeats must be positive')
    model_config = Config(**config.get('model', {}))
    if (not model_config.ct_exact or not model_config.zoh_b
            or model_config.norm_type != 'batchnorm' or model_config.readout_pool != 'mean'):
        raise ValueError('The cost protocol requires exact continuous-time dendrites, batch normalization and mean pooling')
    tf32 = bool(options.get('tf32', True))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    output = resolve_path(config.get('output', 'outputs/local_learning_cost'))
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'cost.json'
    identity = {'configuration': config, 'torch': torch.__version__, 'numpy': np.__version__,
                'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if target.exists():
        stored = json.loads(target.read_text(encoding='utf-8'))
        if stored.get('fingerprint') != fingerprint:
            raise RuntimeError('Output contains a different cost experiment; choose another relative output path')
        if stored.get('status') == 'complete':
            print('[complete] Reusing the recorded local-learning cost sweep', flush=True)
            return stored
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    net = DendStochAudioNet(model_config).to(device).eval()
    clear = ClearNet(net, seed=0, learning_config=LearningConfig(
        reg_coef=model_config.reg_coef, rate_target=model_config.rate_target)).to(device)
    beta = model_config.beta_end
    result = {'status': 'running', 'fingerprint': fingerprint, 'metadata': identity,
              'model': asdict(model_config), 'rows': [], 'checks': [],
              'protocol': {'input': 'Uniform [0,1) synthetic sequences; random class labels',
                           'normalization': 'Initial frozen running statistics; no data calibration',
                           'dropout': 'Disabled; no masks passed to the explicit forward',
                           'bptt': 'Same cross-entropy and rate penalty; backward then zero_grad; no optimizer step',
                           'clear': 'No autograd graph; offline reverse adjoints; clear_sym cross-layer feedback',
                           'inference': 'Same explicit forward and caches, without an autograd graph',
                           'timing': 'One warmup discarded; median of synchronized calls',
                           'memory': 'Median CUDA peak allocated increment above each call baseline; null on CPU'}}
    _write(result, target)
    for index, length in enumerate(lengths, 1):
        print(f'[{index}/{len(lengths)}] local learning cost: T={length}, B={batch_size}', flush=True)
        try:
            x = torch.rand(batch_size, length, model_config.n_in, device=device)
            labels = torch.randint(0, model_config.n_classes, (batch_size,), device=device)
            gate_explicit(clear, x[:min(2, batch_size)], beta)
            result['checks'].append({'timesteps': length, 'explicit_forward_bitwise': True})

            def inference():
                with torch.no_grad():
                    clear.forward_explicit(x, beta)

            def bptt():
                logits, caches, _ = clear.forward_explicit(x, beta)
                clear.loss_of(logits, labels, caches).backward()
                net.zero_grad(set_to_none=True)

            def local():
                with torch.no_grad():
                    logits, caches, readout_cache = clear.forward_explicit(x, beta)
                    clear.local_grads(logits, labels, caches, readout_cache, beta, 'clear_sym')

            measured = {name: measure(function, device, repeats)
                        for name, function in [('inference', inference), ('bptt', bptt), ('clear', local)]}
            row = {'T': length, 'B': batch_size, 'measurements': measured}
            for name, values in measured.items():
                row[name + '_ms'] = values['median_ms']
                row[name + '_MB'] = values['peak_increment_MB']
            result['rows'].append(row)
            print('  ' + ' | '.join(f"{name}: {values['median_ms']:.3f} ms" +
                                   (f", {values['peak_increment_MB']:.3f} MB" if values['peak_increment_MB'] is not None else ', memory unavailable')
                                   for name, values in measured.items()), flush=True)
            _write(result, target)
        except torch.cuda.OutOfMemoryError:
            result['status'] = 'partial'
            result['stopped'] = {'timesteps': length, 'reason': 'out of memory'}
            _write(result, target)
            print(f'[out of memory] Sweep stopped at T={length}; completed measurements preserved', flush=True)
            return result
        finally:
            net.zero_grad(set_to_none=True)
            if 'x' in locals():
                del x
            if 'labels' in locals():
                del labels
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    result['status'] = 'complete'
    _write(result, target)
    return result
