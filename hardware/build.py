#!/usr/bin/env python3
"""Build or simulate the KV260 accelerator with relative project paths.

AMD Vitis HLS and Vivado 2025.2, the KV260 board files, and bootgen must be on
PATH for synthesis. Portable C++ verification needs only a C++14 compiler.
All generated source copies, projects, firmware and logs remain under outputs/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from verify_integer import repository_path, check_hashes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path, base):
    return os.path.relpath(path, base).replace(os.sep, '/')


def tool(name):
    for candidate in (name, name + '.bat', name + '.exe'):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(f'{name} is not on PATH. Open the configured compiler or AMD tool environment first.')


def run(command, directory, logfile):
    print('[run] ' + ' '.join(map(str, command)), flush=True)
    started = time.monotonic()
    with logfile.open('w') as log:
        process = subprocess.Popen(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    lines = logfile.read_text(errors='replace').splitlines()
                    last = lines[-1] if lines else 'Tool started; waiting for its first log entry'
                    print(f'[progress] {time.monotonic() - started:.0f} s: {last[-180:]}', flush=True)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    if process.returncode:
        raise RuntimeError(f'Tool failed with exit code {process.returncode}: {relative(logfile, ROOT)}\n' + logfile.read_text(errors='replace')[-4000:])
    print(f'[complete] {relative(logfile, ROOT)}', flush=True)


def save(path, record):
    path.write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('sampled', 'mf'), required=True)
    parser.add_argument('--stage', choices=('verify', 'csim', 'synth', 'package', 'implement', 'all'), default='all')
    parser.add_argument('--fixture', default='data/reference/hardware/fixture')
    parser.add_argument('--output', default='outputs/hardware/build')
    parser.add_argument('--inputs', type=int, default=8, help='Number of fixture inputs used by the C++ testbench')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--pl-mhz', type=int, default=100)
    parser.add_argument('--cosim', action='store_true')
    args = parser.parse_args()
    if args.inputs < 1 or args.inputs > 64 or args.jobs < 1:
        parser.error('--inputs must be 1..64; --jobs must be positive')
    fixture = repository_path(args.fixture)
    check_hashes(fixture)
    params = json.loads((fixture / 'parameters.json').read_text())
    if params['key'] != '344b5f19':
        raise ValueError('The included accelerator is the archived SHD integer model; a different package needs regenerated parameters and validation')
    variant = args.variant
    number = 1 if variant == 'sampled' else 2
    work = repository_path(args.output) / variant
    hls = work / 'hls'
    vivado = work / 'vivado'
    logs = work / 'logs'
    for path in (hls / 'src', hls / 'tb', vivado, logs):
        path.mkdir(parents=True, exist_ok=True)
    sources = sorted((HERE / 'hls').glob('*')) + [HERE / 'tcl/build_design.tcl']
    source_digest = hashlib.sha256(b''.join(p.name.encode() + p.read_bytes() for p in sources if p.is_file())).hexdigest()
    identity = {'source_sha256': source_digest, 'fixture_sha256': sha(fixture / 'sha256.json'),
                'variant': variant, 'pl_mhz': args.pl_mhz, 'csim_inputs': args.inputs}
    state_path = work / 'build_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity:
            raise RuntimeError('Build inputs changed. Choose a new repository-relative --output directory; existing build outputs are preserved.')
    else:
        state = {'identity': identity, 'completed_steps': []}
    for source in sources:
        if source.suffix == '.tcl':
            destination = vivado / 'build_design.tcl'
        elif source.name == 'verify_accelerator.cpp':
            destination = hls / 'tb' / source.name
        else:
            destination = hls / 'src' / source.name
        shutil.copyfile(source, destination)
    fixture_relative = relative(fixture, hls)
    cfg = '\n'.join([
        'part=xck26-sfvc784-2LV-c', '[hls]', 'flow_target=vivado', 'package.output.format=ip_catalog',
        'package.output.syn=false', 'syn.top=crisp_top', 'syn.file=src/accelerator.cpp',
        f'syn.file_cflags=src/accelerator.cpp,-Isrc -std=c++14 -DUSE_AP_INT -DCRISP_VARIANT={number}',
        'tb.file=tb/verify_accelerator.cpp',
        f'tb.file_cflags=tb/verify_accelerator.cpp,-Isrc -std=c++14 -DUSE_AP_INT -DCRISP_VARIANT={number}',
        f'tb.file={fixture_relative}', f'csim.argv={fixture.name} {args.inputs}',
        'csim.clean=true', 'csim.O=true', f'cosim.argv={fixture.name} 1', 'cosim.trace_level=port',
        'clock=10.0', 'clock_uncertainty=1.0', 'syn.compile.pipeline_loops=0', 'syn.rtl.reset=control', ''])
    (hls / 'design.cfg').write_text(cfg)
    def execute_step(step, command, cwd):
        run(command, cwd, logs / f'{step}.log')
        if step in ('verify', 'csim', 'cosim'):
            text = (logs / f'{step}.log').read_text(errors='replace')
            if 'PASS' not in text or '[FAIL]' in text:
                raise RuntimeError(f'{step} did not pass the integer fixture')
        if step not in state['completed_steps']:
            state['completed_steps'].append(step)
        save(state_path, state)
    if args.stage == 'verify':
        executable = work / ('verify_accelerator.exe' if os.name == 'nt' else 'verify_accelerator')
        run([tool('g++'), '-std=c++14', '-O2', '-Wno-unknown-pragmas', f'-DCRISP_VARIANT={number}',
             '-Ihls/src', 'hls/src/accelerator.cpp', 'hls/tb/verify_accelerator.cpp', '-o', relative(executable, work)], work, logs / 'compile.log')
        execute_step('verify', [str(executable.resolve()), relative(fixture, work), str(args.inputs)], work)
        return
    order = ['csim', 'synth'] + (['cosim'] if args.cosim else []) + ['package', 'implement']
    selected = order if args.stage == 'all' else [args.stage]
    for stage in selected:
        prerequisites = {'synth': ['csim'], 'cosim': ['csim', 'synth'], 'package': ['csim', 'synth'], 'implement': ['package']}
        missing = [name for name in prerequisites.get(stage, []) if name not in state['completed_steps']]
        if missing:
            raise RuntimeError(f'{stage} requires completed stages: {missing}')
        if stage != 'implement':
            command = [tool('v++'), '-c', '--mode', 'hls'] if stage == 'synth' else [tool('vitis-run'), '--mode', 'hls', '--' + stage]
            execute_step(stage, command + ['--config', 'design.cfg', '--work_dir', 'work'], hls)
            continue
        ip = hls / 'work/hls/impl/ip'
        if not (ip / 'component.xml').is_file():
            raise FileNotFoundError('Packaged HLS IP is missing')
        previous = {k: os.environ.get(k) for k in ('CRISP_VARIANT', 'CRISP_IP_DIR', 'CRISP_JOBS', 'CRISP_PL_MHZ')}
        os.environ.update(CRISP_VARIANT=variant, CRISP_IP_DIR=relative(ip, vivado), CRISP_JOBS=str(args.jobs), CRISP_PL_MHZ=str(args.pl_mhz))
        try:
            execute_step('implement', [tool('vivado'), '-mode', 'batch', '-source', 'build_design.tcl', '-log', 'vivado.log', '-journal', 'vivado.jou'], vivado)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        status = (vivado / f'reports_{variant}/timing_status.txt').read_text()
        if 'TIMING MET' not in status or 'NOT MET' in status:
            raise RuntimeError('Timing did not close; firmware packaging stopped')
        headers = list(ip.rglob('xcrisp_top_hw.h'))
        if len(headers) != 1:
            raise RuntimeError('Expected exactly one generated HLS register-map header')
        registers = work / f'registers_{variant}.json'
        run([sys.executable, relative(HERE / 'parse_registers.py', work), relative(headers[0], work), relative(registers, work)], work, logs / 'registers.log')
        firmware = work / f'firmware_{variant}'
        binary = firmware / f'crisp_{variant}.bit.bin'
        run([tool('bootgen'), '-image', f'crisp_{variant}.bif', '-arch', 'zynqmp', '-o', relative(binary, vivado), '-w'], vivado, logs / 'bootgen.log')
        if not binary.is_file() or binary.stat().st_size < 100000:
            raise RuntimeError('Generated firmware is absent or incomplete')
        (firmware / 'shell.json').write_text('{"shell_type":"XRT_FLAT","num_slots":"1"}\n')
        receipt = {'variant': variant, 'package': params['key'], 'source_sha256': source_digest,
                   'completed': True, 'bitstream_sha256': sha(binary), 'regmap_sha256': sha(registers),
                   'overlay_source_sha256': sha(firmware / 'pl.dtsi'), 'timing_status': status,
                   'clock_request_hz': int((firmware / 'pl_clk_hz.txt').read_text()), 'pl_mhz_requested': args.pl_mhz,
                   'fixture_sha256': identity['fixture_sha256'], 'steps': state['completed_steps']}
        save(work / f'build_receipt_{variant}.json', receipt)
        print('[firmware] ' + relative(firmware, ROOT), flush=True)


if __name__ == '__main__':
    main()
