"""Command-line entry points for preparation, experiments and verification."""
import argparse
import importlib
import json
import subprocess
import sys

from .paths import ROOT, resolve_path


def read_config(filename):
    import yaml
    config = yaml.safe_load(resolve_path(filename).read_text())
    if not isinstance(config, dict) or not isinstance(config.get('experiment'), str):
        raise ValueError('The YAML file must contain an experiment mapping')
    for key in ('output', 'official_source'):
        if key in config:
            resolve_path(config[key])
    if isinstance(config.get('dataset'), dict) and 'path' in config['dataset']:
        resolve_path(config['dataset']['path'])
    for job in config.get('jobs', []):
        if 'checkpoint' in job:
            resolve_path(job['checkpoint'])
    return config


def fetch_state_space():
    from .models.state_space import UPSTREAM_URL, UPSTREAM_COMMIT, import_upstream
    source = resolve_path('external_code/pspikessm')
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', UPSTREAM_URL, str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', UPSTREAM_COMMIT], check=True)
    import_upstream(source)
    print(f'P-SpikeSSM verified at {UPSTREAM_COMMIT}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='CRISP/CLEAR reproduction commands')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-data', help='Download and prepare the registered dataset')
    prepare.add_argument('dataset', choices=('shd', 'ssc', 'dvs_gesture'))
    prepare.add_argument('--no-download', action='store_true')
    prepare.add_argument('--raw-directory', default='data/raw/dvs_gesture', help='Extracted DVS128 Gesture recordings')
    prepare.add_argument('--split-by', choices=('time', 'number'), default='time')
    commands.add_parser('fetch-state-space', help='Fetch and verify the pinned P-SpikeSSM source')
    commands.add_parser('fetch-spsn', help='Fetch and verify the pinned SPSN benchmark source')
    run = commands.add_parser('run', help='Run a YAML experiment configuration')
    run.add_argument('config')
    run.add_argument('--device', choices=('cpu', 'cuda', 'xla'))
    analyse = commands.add_parser('analyse-reference', help='Recalculate summaries from stored numeric evidence')
    analyse.add_argument('--output', default='outputs/reference_analysis')
    gradients = commands.add_parser('check-gradients', help='Check 36 exact local-gradient settings')
    gradients.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    gradients.add_argument('--output', default='outputs/local_gradients/verification.json')
    commands.add_parser('test', help='Run implementation and integrity tests')
    proofs = commands.add_parser('check-proofs', help='Kernel-check the Lean proofs')
    proofs.add_argument('--setup', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'prepare-data':
        if args.dataset == 'dvs_gesture':
            from .vision_preparation import prepare_gesture
            prepare_gesture(raw_directory=args.raw_directory, split_by=args.split_by)
        else:
            from .data_preparation import prepare_audio
            prepare_audio(args.dataset, download=not args.no_download)
    elif args.command == 'fetch-state-space':
        fetch_state_space()
    elif args.command == 'fetch-spsn':
        from .benchmarks import load_spsn
        load_spsn('external_code/spsn', fetch=True)
    elif args.command == 'run':
        config = read_config(args.config)
        if args.device:
            config['device'] = args.device
        module = {'local_learning': 'local_learning', 'local_learning_cost': 'local_learning_cost',
                  'moment_diagnostics': 'moment_diagnostics', 'conditional_variance': 'conditional_variance',
                  'dvs_gesture': 'vision',
                  'certification': 'certification_experiment',
                  'runtime_benchmark': 'benchmarks'}.get(config['experiment'], 'experiments')
        result = importlib.import_module(f'.{module}', __package__).run(config)
        count = result.get('runs', len(result.get('rows', result.get('units', []))))
        print(json.dumps({'experiment': config['experiment'],
                          'output': config.get('output'),
                          'completed_units': count}, indent=2))
    elif args.command == 'analyse-reference':
        from .analysis import analyse
        result = analyse(resolve_path(args.output))
        print(json.dumps(result, indent=2))
    elif args.command == 'check-gradients':
        from .learning import check_gradient_exactness
        result = check_gradient_exactness(device=args.device)
        output = resolve_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n')
        print(f'Saved {args.output}')
    elif args.command == 'test':
        subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, check=True)
    elif args.command == 'check-proofs':
        command = [sys.executable, str(ROOT / 'proofs' / 'verify.py')]
        if args.setup:
            command.append('--setup')
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
