"""Named reproduction workflows with explicit datasets and output locations.

All experiment variations are declared in YAML jobs. New runs are saved separately
from reference evidence. A checkpoint path selects evaluation of an existing model;
otherwise the declared training protocol is run before evaluation.
"""
import copy
import hashlib
import json
import re
from dataclasses import asdict

import numpy as np
import torch

from .paths import resolve_path, ROOT
from .datasets import ArrLoader, load_data, data_fingerprint, temporal_pool_exact
from .training import TrainConfig, train_one, set_all_seeds, eval_mode
from .models import crisp, state_space, lif
from .diagnostics import count_ops, sampling_statistics, sampling_location_evaluation, paired_summary

EXPERIMENTS = {
    'deployment_gap', 'sampling_locations', 'sampler_sharpening', 'slope_schedule',
    'sampling_variance', 'time_step_transfer', 'sparsity',
}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _loader(part, cfg, *, shuffle=False, seed=0):
    return ArrLoader(*part, cfg.batch_size, shuffle=shuffle, seed=seed, device=cfg.device)


def _validate_job(job):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_]*', job.get('name', '')):
        raise ValueError('Each job needs a descriptive lowercase name containing letters, digits or underscores')
    if job.get('family', 'crisp') not in ('crisp', 'pspikessm', 'lif'):
        raise ValueError('Supported model families are crisp, pspikessm and lif')
    factor = job.get('factor', 1)
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError('The time-step factor must be a positive integer')


def _options(config, job, data):
    family = job.get('family', 'crisp')
    dataset = config['dataset']['name']
    train_options = dict(config.get('training', {}), **job.get('training', {}))
    train_options['device'] = config.get('device', 'cpu')
    train_config = TrainConfig(**train_options)
    settings = dict(config.get('model', {}), **job.get('model', {}))
    n_classes = {'shd': 20, 'ssc': 35, 'dvs_gesture': 11}[dataset]
    timesteps = int(data[0][0].shape[1]) // job.get('factor', 1)
    beta_start = float(job.get('beta_start', 1.))
    beta_end = float(job.get('beta_end', 5. if family == 'crisp' else 1.))
    anneal_frac = float(job.get('anneal_frac', .7))
    if family == 'crisp':
        settings.update(dataset=dataset, n_in=int(data[0][0].shape[-1]), n_classes=n_classes,
                        beta_start=beta_start, beta_end=beta_end, anneal_frac=anneal_frac,
                        device=train_config.device)
        for name in ('reg_coef', 'rate_target'):
            if name in job:
                settings[name] = job[name]
        settings.update({key: getattr(train_config, key) for key in
                         ('lr', 'weight_decay', 'grad_clip', 'batch_size', 'n_epochs', 'patience')})
        settings.update(seeds=config.get('seeds', [42, 123, 999]), dt_factor=job.get('factor', 1))
        architecture = crisp.Config(**settings)
    elif family == 'pspikessm':
        architecture = {
            'source': config.get('official_source', 'external_code/pspikessm'),
            'n_in': int(data[0][0].shape[-1]), 'n_classes': n_classes,
            'timesteps': timesteps, 'hidden': int(job.get('hidden', 200)),
        }
    else:
        fields = lif.Config.__dataclass_fields__
        architecture = {key: value for key, value in settings.items() if key in fields}
        architecture['n_in'] = int(data[0][0].shape[-1])
    return family, architecture, train_config, beta_start, beta_end, anneal_frac


def _load_or_train(config, job, seed, data, directory):
    family, architecture, training, beta_start, beta_end, fraction = _options(config, job, data)
    factor = job.get('factor', 1)
    checkpoint = job.get('checkpoint')
    if checkpoint:
        checkpoint = checkpoint.format(seed=seed)
        path = resolve_path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'Model checkpoint is missing: {checkpoint}')
        if job.get('checkpoint_sha256'):
            expected = job['checkpoint_sha256']
            expected = expected.get(str(seed)) if isinstance(expected, dict) else expected
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError('Checkpoint SHA-256 mismatch')
        if family == 'crisp':
            model, architecture = crisp.load_ckpt(path, device=training.device)
        else:
            blob = torch.load(path, map_location=training.device, weights_only=False)
            if family == 'pspikessm':
                model = state_space.build_model(**architecture, device=training.device)
            else:
                model = lif.build_model(architecture, {'shd': 20, 'ssc': 35}[config['dataset']['name']], training.device)
            model.load_state_dict(blob['state'], strict=True)
            model.eval()
        if family == 'pspikessm':
            model.sampler_slope = beta_end
        return model, architecture, training, {'source': 'checkpoint', 'checkpoint': checkpoint}
    if config['experiment'] in ('sampling_variance', 'sampling_locations') and not job.get('train', False):
        raise ValueError('This diagnostic needs a checkpoint or an explicit job train: true setting')
    coarse_data = data
    if factor > 1:
        if data[0][0].shape[1] % factor:
            raise ValueError('The time-step factor must divide the original sequence length')
        coarse_data = (data[0], (temporal_pool_exact(data[1][0], factor), data[1][1]),
                       (temporal_pool_exact(data[2][0], factor), data[2][1]))
    rule = job.get('rule', 'meanfield')
    checkpoint_dir = directory / 'checkpoints' / job['name'] / f'seed_{seed}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    identity = {'dataset': config['dataset'], 'fingerprint': data_fingerprint(data),
                'job': job, 'seed': seed, 'training': asdict(training),
                'architecture': asdict(architecture) if family == 'crisp' else architecture}
    protocol_file = checkpoint_dir / 'protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != identity:
        raise RuntimeError('Existing output directory belongs to a different training protocol')
    _write_json(protocol_file, identity)
    if family == 'crisp':
        info, model = train_one(seed, coarse_data, rule, architecture, training, checkpoint_dir, train_pool=factor)
    elif family == 'pspikessm':
        info, model = state_space.train_one(
            seed, coarse_data, architecture, training, checkpoint_dir, rule=rule,
            beta_start=beta_start, beta_end=beta_end, anneal_frac=fraction, train_pool=factor,
        )
    else:
        info, model = lif.train_one(
            seed, coarse_data, architecture, training, checkpoint_dir,
            n_classes={'shd': 20, 'ssc': 35}[config['dataset']['name']], train_pool=factor,
            reg_coef=float(job.get('reg_coef', .02)), rate_target=float(job.get('rate_target', .1)),
        )
    return model, architecture, training, info


def _deployment(model, family, loader, beta, evaluation_seed=2026, draws=(1, 5)):
    result = {}
    if family == 'lif':
        return {'deterministic': lif.evaluate(model, loader)}
    if family == 'crisp':
        result['meanfield'] = eval_mode(model, loader, 'graded', beta)
        for count in draws:
            set_all_seeds(evaluation_seed + count)
            result[f'sampled_{count}_draws'] = eval_mode(model, loader, 'sample', beta, n_samples=count)
        result['threshold'] = eval_mode(model, loader, 'heaviside', beta)
    else:
        model.sampling_mode = 'meanfield'
        model.sampler_slope = beta
        result['meanfield'] = state_space.evaluate(model, loader)
        model.sampling_mode = 'sample'
        for count in draws:
            set_all_seeds(evaluation_seed + count)
            result[f'sampled_{count}_draws'] = state_space.evaluate(model, loader, n_samples=count)
    result['deployment_gap_pp'] = 100 * (result['sampled_1_draws']['acc'] - result['meanfield']['acc'])
    return result


def _transfer(model, family, data, cfg, factor, beta, config):
    coarse = _loader((temporal_pool_exact(data[2][0], factor), data[2][1]), cfg)
    fine = _loader(data[2], cfg)
    calibration = _loader(data[0], cfg, shuffle=True, seed=int(config.get('bn_shuffle_seed', 0)))
    count = int(config.get('bn_recalibration_batches', 30))
    if family == 'lif':
        output = lif.transfer_evaluation(model, factor, coarse, fine, calibration, count)
        first_batches = lif.transfer_evaluation(model, factor, coarse, fine, _loader(data[0], cfg), count)
        for key, value in first_batches.items():
            if key.endswith('_bn_reestimated'):
                output[key.replace('_bn_reestimated', '_bn_first_batches')] = value
        return output
    if family == 'crisp':
        model.set_dt(1.)
        output = {'coarse_native': eval_mode(model, coarse, 'graded', beta)['acc'],
                  'naive_same_dt': eval_mode(model, fine, 'graded', beta)['acc']}
        model.set_dt(1. / factor)
        output['retimed'] = eval_mode(model, fine, 'graded', beta)['acc']
        set_all_seeds(cfg.eval_seed)
        output['retimed_sampled_1_draw'] = eval_mode(model, fine, 'sample', beta)['acc']
        recalibrated = crisp.recalibrate_bn(model, calibration, beta, n_batches=count)
        output['retimed_bn_reestimated'] = eval_mode(recalibrated, fine, 'graded', beta)['acc']
        first_calibrated = crisp.recalibrate_bn(model, _loader(data[0], cfg), beta, n_batches=count)
        output['retimed_bn_first_batches'] = eval_mode(first_calibrated, fine, 'graded', beta)['acc']
        model.set_dt(1.)
        return output
    output = {}
    convolution = next((module for module in model.modules() if type(module).__name__ == 'FFTConv'), None)
    if convolution is None:
        raise RuntimeError('Pinned state-space convolution is missing')
    with torch.no_grad():
        coarse_kernel = convolution.kernel(L=convolution.L, rate=1.)[0]
        fine_kernel = convolution.kernel(L=convolution.L * factor, rate=1. / factor)[0]
        coarse_dt = convolution.kernel._get_params(1.)[0]
        fine_dt = convolution.kernel._get_params(1. / factor)[0]
        deviation = float((fine_dt / coarse_dt - 1. / factor).abs().max())
    if fine_kernel.shape[-1] != factor * coarse_kernel.shape[-1] or deviation >= 1e-6:
        raise RuntimeError('Upstream sampling-rate argument failed its kernel-length/time-step check')
    output['kernel_transfer_check'] = {'coarse_length': coarse_kernel.shape[-1],
                                       'fine_length': fine_kernel.shape[-1], 'dt_ratio_max_error': deviation}
    model.sampling_mode = 'sample'
    for draws in (1, 5):
        set_all_seeds(cfg.eval_seed + draws)
        output[f'coarse_native_{draws}_draws'] = state_space.evaluate(model, coarse, n_samples=draws)['acc']
        set_all_seeds(cfg.eval_seed + draws)
        output[f'retimed_{draws}_draws'] = state_space.evaluate(model, fine, n_samples=draws, rate=1. / factor)['acc']
    set_all_seeds(cfg.eval_seed + 1)
    output['naive_same_dt'] = state_space.evaluate(model, fine)['acc']
    batchnorm = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm1d)]
    saved = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone(), m.momentum) for m in batchnorm]
    try:
        model.eval()
        for module in batchnorm:
            module.train()
            module.reset_running_stats()
            module.momentum = None
        with torch.no_grad():
            for index, (x, _) in enumerate(calibration):
                if index >= count:
                    break
                model(x, rate=1. / factor)
        model.eval()
        set_all_seeds(cfg.eval_seed + 1)
        output['retimed_bn_reestimated'] = state_space.evaluate(model, fine, rate=1. / factor)['acc']
    finally:
        for module, (mean, variance, batches, momentum) in zip(batchnorm, saved):
            module.running_mean.copy_(mean)
            module.running_var.copy_(variance)
            module.num_batches_tracked.copy_(batches)
            module.momentum = momentum
        model.eval()
    set_all_seeds(cfg.eval_seed + 1)
    restored = state_space.evaluate(model, coarse)['acc']
    if restored != output['coarse_native_1_draws']:
        raise RuntimeError('Batch-normalization statistics failed their restoration check')
    return output


def run(config):
    """Run every declared job/seed and return newly measured result rows."""
    experiment = config.get('experiment')
    if experiment == 'dvs_gesture':
        from .vision import run as run_vision
        return run_vision(config)
    if experiment == 'runtime_benchmark':
        from .benchmarks import run as run_benchmark
        return run_benchmark(config)
    if experiment not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {experiment!r}; supported: {sorted(EXPERIMENTS)}')
    dataset = config.get('dataset', {})
    if not {'name', 'path'} <= set(dataset):
        raise ValueError('dataset must supply name and a repository-relative prepared-cache path')
    jobs = config.get('jobs', [])
    if not jobs:
        raise ValueError('At least one explicit experiment job is required')
    if len({j.get('name') for j in jobs}) != len(jobs):
        raise ValueError('Job names must be unique')
    for job in jobs:
        _validate_job(job)
    seeds = config.get('seeds', [42, 123, 999])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be a nonempty list of distinct integers')
    data = load_data(dataset['path'], dataset=dataset['name'], expected_fingerprint=dataset.get('fingerprint'))
    directory = resolve_path(config.get('output', f'outputs/{experiment}'))
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / 'configuration.json', config)
    tasks = [(job, seed) for job in jobs for seed in seeds]
    if experiment == 'slope_schedule':
        # Every arm is trained and its validation-selected checkpoint fixed before
        # any held-out evaluation. No model selection uses test accuracy.
        sealed = []
        for index, (job, seed) in enumerate(tasks):
            print(f'Slope schedule training {index + 1}/{len(tasks)}: {job["name"]}, seed {seed}', flush=True)
            model, _, _, _ = _load_or_train(config, job, seed, data, directory)
            del model
            selected = (resolve_path(job['checkpoint'].format(seed=seed)) if job.get('checkpoint') else
                        directory / 'checkpoints' / job['name'] / f'seed_{seed}' / 'best_model.pt')
            sealed.append({'job': job['name'], 'seed': seed,
                           'checkpoint': selected.relative_to(ROOT).as_posix(),
                           'sha256': hashlib.sha256(selected.read_bytes()).hexdigest()})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _write_json(directory / 'validation_selection.json', {
            'rule': 'Every declared training run completed before any test evaluation',
            'checkpoints': sealed,
        })
    rows = []
    for index, (job, seed) in enumerate(tasks):
        print(f'{experiment}: job {index + 1}/{len(tasks)}, {job["name"]}, seed {seed}', flush=True)
        if experiment == 'slope_schedule':
            selected = next(item for item in sealed if item['job'] == job['name'] and item['seed'] == seed)
            if hashlib.sha256(resolve_path(selected['checkpoint']).read_bytes()).hexdigest() != selected['sha256']:
                raise RuntimeError('Validation-selected checkpoint changed before test evaluation')
        model, architecture, cfg, info = _load_or_train(config, job, seed, data, directory)
        family = job.get('family', 'crisp')
        beta = float(job.get('beta_end', getattr(architecture, 'beta_end', 1.)))
        test = _loader(data[2], cfg)
        row = {'job': job['name'], 'family': family, 'seed': seed, 'training': info}
        if experiment == 'sampling_locations':
            if family != 'pspikessm':
                raise ValueError('Sampling-location interventions require a P-SpikeSSM model')
            row['test'] = sampling_location_evaluation(model, test, eval_seed=cfg.eval_seed)
        elif experiment == 'time_step_transfer':
            factor = job.get('factor', 1)
            if factor < 2:
                row['test'] = _deployment(model, family, test, beta, cfg.eval_seed)
            else:
                row['transfer'] = _transfer(model, family, data, cfg, factor, beta, config)
            row['factor'] = factor
        else:
            row['test'] = _deployment(model, family, test, beta, cfg.eval_seed)
            if experiment == 'sampling_variance':
                row['sampler_moments'] = sampling_statistics(model, test, family=family, beta=beta)
                if family == 'crisp':
                    set_all_seeds(cfg.eval_seed)
                    row['last_layer_variance'] = crisp.analytic_variance(
                        model, test, beta, subset=int(config.get('variance_subset', 512)),
                        M=int(config.get('variance_draws', 64)))
            if experiment == 'sparsity':
                if family == 'pspikessm':
                    raise ValueError('Archived sparsity operation accounting supports CRISP and LIF')
                mode = 'sample' if family == 'crisp' else None
                row['operations'] = count_ops(model, family, dataset['name'], test,
                                              mode=mode, beta=beta, seed=cfg.eval_seed)
        rows.append(row)
        _write_json(directory / 'results' / f'{job["name"]}_seed_{seed}.json', row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {}
    for job in jobs:
        selected = [row for row in rows if row['job'] == job['name']]
        gaps = [row['test']['deployment_gap_pp'] for row in selected
                if 'deployment_gap_pp' in row.get('test', {})]
        if gaps:
            summary[job['name']] = {'deployment_gap_pp': paired_summary(gaps)}
    if experiment == 'slope_schedule':
        for family in sorted({job.get('family', 'crisp') for job in jobs}):
            fixed = [job for job in jobs if job.get('family', 'crisp') == family and job.get('beta_start') == 5.]
            annealed = [job for job in jobs if job.get('family', 'crisp') == family and job.get('beta_start') == 1.]
            if len(fixed) == len(annealed) == 1:
                pairs = []
                for seed in seeds:
                    a = next(row for row in rows if row['job'] == annealed[0]['name'] and row['seed'] == seed)['test']
                    b = next(row for row in rows if row['job'] == fixed[0]['name'] and row['seed'] == seed)['test']
                    pairs.append({
                        'seed': seed, 'meanfield_difference_pp': 100 * (a['meanfield']['acc'] - b['meanfield']['acc']),
                        'sampled_difference_pp': 100 * (a['sampled_1_draws']['acc'] - b['sampled_1_draws']['acc']),
                        'deployment_gap_difference_pp': a['deployment_gap_pp'] - b['deployment_gap_pp'],
                    })
                summary[f'{family}_annealed_minus_fixed'] = {
                    'paired_values': pairs,
                    **{key: paired_summary([pair[key] for pair in pairs]) for key in
                       ('meanfield_difference_pp', 'sampled_difference_pp', 'deployment_gap_difference_pp')},
                }
    result = {'experiment': experiment, 'dataset': dataset['name'],
              'data_fingerprint': data_fingerprint(data), 'rows': rows, 'summary': summary}
    _write_json(directory / 'results.json', result)
    return result
