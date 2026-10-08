#!/usr/bin/env python3
"""Verify or measure the installed KV260 integer accelerator.

The matching firmware, overlay and timing-qualified build receipt must already
be present under the selected build directory. No synthesis or installation is performed.
File arguments are repository-relative. Physical board execution requires root.
"""
import argparse, glob, hashlib, json, math, os, platform, random, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from board_interface import Crisp, Package, R, CMD_RUN, CMD_NOP, MODE_NAMES, MODE_D0, SLOT_MAX

ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / 'hardware'
from paths import repository_path, check_hashes
def utc(): return datetime.now(timezone.utc).isoformat()
try: sys.stdout.reconfigure(line_buffering=True)
except Exception: pass
def fmt_dur(sec):
    sec = max(0, int(sec)); h, m_ = divmod(sec, 3600); m_, s_ = divmod(m_, 60)
    return f'{h} h {m_:02d} min' if h else f'{m_} min {s_:02d} s'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjson(p): return json.loads(Path(p).read_text())
def save(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); tmp.replace(p)
def require(ok, text):
    if not ok: raise RuntimeError(text)
def s32(v): return v - (1 << 32) if v >= (1 << 31) else v

def clock_hz(receipt):
    p = Path(os.sep).joinpath('sys/kernel/debug/clk/pl0_ref/clk_rate'); require(p.exists(), 'Actual PL clock unavailable; mount the board debug filesystem first.')
    v = int(p.read_text()); require(0 < v <= receipt['clock_request_hz'] * 1.001 + 2000, 'PL clock exceeds the timing-qualified limit'); return v

def preflight(V, build_dir):
    require(os.geteuid() == 0, 'Run with sudo on the KV260.')
    require(platform.machine() in ('aarch64', 'arm64'), 'Board runner requires the KV260 ARM Linux system.')
    receipt = readjson(build_dir / f'build_receipt_{V}.json')
    require(receipt.get('completed') and 'TIMING MET' in receipt['timing_status'], 'Build/timing gate failed')
    for filename, field in [(f'firmware_{V}/crisp_{V}.bit.bin', 'bitstream_sha256'),
                            (f'registers_{V}.json', 'regmap_sha256'),
                            (f'firmware_{V}/pl.dtsi', 'overlay_source_sha256')]:
        require(sha(build_dir / filename) == receipt[field], filename + ' hash mismatch')
    require(Path(os.sep).joinpath('sys/class/fpga_manager/fpga0/state').read_text().strip() == 'operating', 'FPGA manager is not operating')
    firmware_node = Path(os.sep).joinpath('proc/device-tree/fpga-full/firmware-name')
    loaded = firmware_node.read_bytes().rstrip(b'\0').decode(errors='replace') if firmware_node.exists() else '(none)'
    require(loaded == f'crisp_{V}.bit.bin', f'PL has {loaded!r} loaded. Install the matching firmware and overlay before using the AXI interface.')
    return receipt, clock_hz(receipt)

def check_identity(m, pkg, variant):
    m.call(CMD_NOP); r = m.results()
    require(r[R['PACKAGE']] == int(pkg.params['key'], 16), f'bitstream package 0x{r[R["PACKAGE"]]:08x} != package {pkg.params["key"]}')
    require(r[R['VARIANT']] == variant, f'bitstream variant {r[R["VARIANT"]]} != requested {variant}')

# ─────────────────────────────────────────── verification ───────────────────────────────────────────
def verify_all(m, pkg, variant, n, out):
    ids = list(range(n)); batches = pkg.plan_batches(ids)
    rec = {'started_utc': utc(), 'variant': variant, 'package': pkg.params['key'], 'n': n, 'batches': len(batches), 'completed': False, 'failures': []}
    arr = {k: np.full(n, -2, np.int64) for k in ('n1', 'seq', 'used', 'fix', 'ref', 'mf')}; ok = np.zeros(n, bool); dts = []
    t0 = time.time(); last = 0; done = 0
    print(f'[VERIFY] {n} input, {len(batches)} batch (events bellegi {pkg.params["fmt"]["M"]} draw x 2 akis / input)', flush=True)
    for bi, batch in enumerate(batches):
        nev = m.load_inputs(pkg, batch)
        for slot, i in enumerate(batch):
            err, got, dt, _ = m.verify_slot(pkg, slot, i, variant, check_z1=True); dts.append(dt)
            ok[i] = not err
            for k in got:
                if k in arr: arr[k][i] = got[k]
            if err:
                rec['failures'].append({'input': int(i), 'errors': err, 'got': {k: v for k, v in got.items() if k != 'logits'}})
                print(f'[FAIL] input {i}: {err}', flush=True)
                if len(rec['failures']) > 20: raise RuntimeError('too many mismatches; stopping')
            done += 1; now = time.time()
            if now - last >= 20 or done == n:
                last = now; rate = done / (now - t0); y = pkg.labels[:done]
                line = f'[VERIFY] {done}/{n} | bit-exact {int(ok[:done].sum())} | batch {bi + 1}/{len(batches)} ({nev} events) | {dt * 1e3:.0f} ms/input | remaining ~{fmt_dur((n - done) / max(rate, 1e-9))}'
                if variant == 1:
                    s = arr['seq'][:done]; line += f" | n=1 {np.mean(arr['n1'][:done] == y):.4f} | sequential {np.mean(s == y):.4f} abstention {np.mean(s < 0):.3f} ort {np.mean(arr['used'][:done]):.1f}"
                else: line += f" | MF {np.mean(arr['mf'][:done] == y):.4f}"
                print(line, flush=True)
    y = pkg.labels[:n]; summ = {'bit_exact': int(ok.sum()), 'n': n, 'mean_verify_s': float(np.mean(dts))}
    if variant == 1:
        seq, used, fix, ref, n1 = arr['seq'], arr['used'], arr['fix'], arr['ref'], arr['n1']; res = ref >= 0
        k_seq = int(((seq >= 0) & res & (seq != ref)).sum()); k_fix = int(((fix >= 0) & res & (fix != ref)).sum())
        summ.update(n1_acc=float((n1 == y).mean()), seq_acc=float((seq == y).mean()), seq_abstain=float((seq < 0).mean()), seq_mean_draws=float(used.mean()),
                    seq_wrong_among_certified=float((seq != y)[seq >= 0].mean()) if (seq >= 0).any() else None,
                    fix_acc=float((fix == y).mean()), fix_abstain=float((fix < 0).mean()), n_resolved=int(res.sum()),
                    seq_vs_ref_disagree=k_seq, seq_vs_ref_rate_unconditional=k_seq / n, fix_vs_ref_disagree=k_fix,
                    bound=len(pkg.looks) / pkg.params['fmt']['DENOM_SEQ'] + 1 / pkg.params['fmt']['DENOM_FIX'])
    else:
        summ.update(mf_acc=float((arr['mf'] == y).mean()), mf_agree_with_oracle=float((arr['mf'] == pkg.exp['mf'][:n]).mean()))
    rec.update(completed=True, finished_utc=utc(), summary=summ, decisions={k: v.tolist() for k, v in arr.items() if (v != -2).any()}, bit_exact=ok.tolist())
    save(out / 'board_verification.json', rec)
    print('[VERIFY] SUMMARY ' + json.dumps(summ), flush=True)
    return rec

# ─────────────────────────────────────────── power ───────────────────────────────────────────
CTX = {}
def find_sensor():
    found = []
    for d in Path(os.sep).joinpath('sys/class/hwmon').glob('hwmon*'):
        try:
            if (d / 'name').read_text().strip() == 'ina260_u14' and (d / 'power1_input').exists(): found.append(d / 'power1_input')
        except OSError: pass
    require(len(found) == 1, 'Expected exactly one ina260_u14 SOM input-power sensor'); return found[0]
def temperatures():
    out = {}
    for d in Path(os.sep).joinpath('sys/class/thermal').glob('thermal_zone*'):
        try: out[d.name + ':' + (d / 'type').read_text().strip()] = float((d / 'temp').read_text()) / 1000
        except (OSError, ValueError): pass
    return out
def read_governors(): return {f: Path(f).read_text().strip() for f in glob.glob(str(Path(os.sep) / 'sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'))}
def sample_power(sensor, seconds, m, running, hz=10):
    samples = []; start = time.perf_counter(); k = 0; step = min(10.0, max(2.0, seconds / 3)); nxt = step; i0 = m.items_done() if running else 0
    while True:
        now = time.perf_counter(); deadline = start + k / hz
        if now < deadline: time.sleep(deadline - now)
        now = time.perf_counter()
        if now - start >= seconds: break
        require(m.idle() != running, 'IP stopped during a power block or was active during idle reference')
        p = float(sensor.read_text()) / 1000; require(math.isfinite(p) and p > 0, 'Invalid sensor power')
        row = {'t_s': now - start, 'power_mW': p}
        if k % 10 == 0: row['temperature_C'] = temperatures()
        samples.append(row); k += 1
        if now - start >= nxt:
            nxt += step; pm = [r['power_mW'] for r in samples]
            line = f"[{'RUN' if running else 'IDLE'}] {CTX.get('rep', '?')} | {CTX.get('mode', 'idle reference')} | blok {now - start:5.0f}/{seconds:.0f} s | current {pm[-1]:7.1f} mW, mean {np.mean(pm):7.1f} mW"
            if running:
                line += f" | {(m.items_done() - i0) / (now - start):7.1f} decision/s"
                if CTX.get('last_idle') is not None: line += f" | above idle {np.mean(pm) - CTX['last_idle']:+6.1f} mW"
            t = [r.get('temperature_C') for r in samples if r.get('temperature_C')]
            if t: line += f" | {max(t[-1].values()):4.1f} C"
            print(line, flush=True)
    require(len(samples) >= int(seconds * hz * .9), 'Insufficient power samples')
    return {'duration_s': time.perf_counter() - start, 'samples': samples, 'mean_mW': float(np.mean([s['power_mW'] for s in samples]))}

T975 = [None, 12.7062, 4.3027, 3.1825, 2.7765, 2.5706, 2.4469, 2.3646, 2.3061, 2.2622, 2.2282, 2.2010, 2.1788, 2.1604, 2.1448, 2.1315]
def interval(values):
    x = np.asarray(values, dtype=float); n = len(x); mean = float(x.mean())
    if n < 2: return {'n': n, 'mean': mean, 'sd': None, 'ci95': None}
    sd = float(x.std(ddof=1)); half = T975[min(n - 1, 15)] * sd / math.sqrt(n); return {'n': n, 'mean': mean, 'sd': sd, 'ci95': [mean - half, mean + half]}

def measure_mode(m, sensor, mode, n_slots, a):
    """calibrate the pass time, then run enough passes to cover warm-up + block, sample power during the block."""
    t_pass = m.call(CMD_RUN, 1, 0, mode); r = m.results(); per_item = t_pass / max(1, r[R['ITEMS']])
    passes = max(1, math.ceil((a.block + a.warmup + 5) / t_pass))
    m.start(CMD_RUN, passes, 0, mode); time.sleep(a.warmup)
    i0 = m.items_done(); blk = sample_power(sensor, a.block, m, True); i1 = m.items_done()
    dt = m.wait(timeout=passes * t_pass * 2 + 60); r = m.results()
    items = int(r[R['ITEMS']]); require(items == passes * n_slots, 'item counter mismatch')
    rate = (i1 - i0) / blk['duration_s']
    blk.update(mode=mode, passes=passes, elapsed_s=dt, items=items, decisions_per_s=rate, s_per_decision=1 / rate if rate > 0 else None,
               calib_s_per_item=per_item, draws_total=int(r[R['DRAWS']]), draws_per_item=r[R['DRAWS']] / items, events_per_item=r[R['EV']] / items,
               spikes1_per_item=r[R['SPK1']] / items, spikes2_per_item=r[R['SPK2']] / items, abstain=int(r[R['ABSTAIN']]), correct=int(r[R['CORRECT']]),
               accuracy=r[R['CORRECT']] / items)
    return blk

def analyze(data):
    out = {'scope': 'SOM input power (INA260 U14), PS idle during blocks; energy per evaluated input, including abstentions = (P_mode - idle bracket) / decisions per second; '
                    'D0 (replay/control loop without compute) subtracted separately', 'units': 'mW / (decisions/s) = mJ per decision -> reported in uJ per evaluated input', 'modes': {}, 'paired_uJ': {}}
    reps = [r for r in data['repeats'] if r.get('complete')]
    if not reps: return out
    names = list(reps[0]['runs'].keys())
    per = {n: [] for n in names}; d0 = []
    for r in reps:
        d0_row = r['runs']['D0']; d0_uj = (d0_row['mean_mW'] - d0_row['idle_bracket_mW']) / d0_row['decisions_per_s'] * 1e3; d0.append(d0_uj)
        for n in names:
            row = r['runs'][n]; uj = (row['mean_mW'] - row['idle_bracket_mW']) / row['decisions_per_s'] * 1e3
            per[n].append({'idle_adjusted_uJ': uj, 'D0_subtracted_uJ': uj - d0_uj, 'ms_per_decision': 1e3 / row['decisions_per_s'], 'delta_mW': row['mean_mW'] - row['idle_bracket_mW'],
                           'draws_per_item': row['draws_per_item'], 'accuracy': row['accuracy']})
    for n in names:
        out['modes'][n] = {k: interval([x[k] for x in per[n]]) for k in per[n][0]}
    for a_, b_ in [('FIX', 'SEQ'), ('SEQ', 'N1'), ('N1', 'PREFIX'), ('MF', 'PREFIX'), ('SEQ', 'MF'), ('N1', 'MF')]:
        if a_ in per and b_ in per:
            out['paired_uJ'][f'{a_}_minus_{b_}'] = interval([x['idle_adjusted_uJ'] - y['idle_adjusted_uJ'] for x, y in zip(per[a_], per[b_])])
    return out

def power_run(m, pkg, variant, a, out, receipt, clk):
    sensor = find_sensor(); governors = read_governors(); require(governors, 'CPU governor information unavailable')
    modes = ['D0', 'PREFIX', 'N1', 'SEQ', 'FIX'] if variant == 1 else ['D0', 'PREFIX', 'MF']
    data = {'started_utc': utc(), 'complete': False, 'variant': variant, 'package': pkg.params['key'], 'settings': {k: v for k, v in vars(a).items() if k != 'out'},
            'environment': {'platform': platform.platform(), 'clock_hz': clk, 'sensor': str(sensor), 'governors_before': governors, 'temperatures_start_C': temperatures(),
                            'bitstream_sha256': receipt['bitstream_sha256']}, 'repeats': []}
    save(out / 'measurement.json', data); rng = random.Random(a.order_seed)
    all_batches = pkg.plan_batches(list(range(pkg.n)), max_slots=a.subset)
    if not a.quick:
        require(a.repeats == len(all_batches), f'Full-test acquisition requires exactly {len(all_batches)} batches for this input package')
    try:
        for p in governors: Path(p).write_text('performance')
        for rep in range(a.repeats):
            batch = all_batches[rep % len(all_batches)]; nev = m.load_inputs(pkg, batch); n_slots = len(batch)
            order = modes.copy(); rng.shuffle(order)
            record = {'repeat': rep, 'inputs': [int(i) for i in batch], 'n_events': int(nev), 'order': order, 'complete': False, 'runs': {}, 'idles': []}; data['repeats'].append(record)
            print(f'[REPEAT] {rep + 1}/{a.repeats}: inputler {batch[0]}..{batch[-1]} ({n_slots} slot, {nev} events), order {" -> ".join(order)}', flush=True)
            CTX.clear(); CTX.update(rep=f'repeat {rep + 1}/{a.repeats}', last_idle=None)
            record['idles'].append(sample_power(sensor, a.block, m, False)); CTX['last_idle'] = record['idles'][-1]['mean_mW']; save(out / 'measurement.json', data)
            for mode in order:
                CTX['mode'] = mode; row = measure_mode(m, sensor, MODE_NAMES[mode], n_slots, a); record['runs'][mode] = row
                idle = sample_power(sensor, a.block, m, False); record['idles'].append(idle); CTX['last_idle'] = idle['mean_mW']
                row['idle_bracket_mW'] = (record['idles'][-2]['mean_mW'] + idle['mean_mW']) / 2
                dmw = row['mean_mW'] - row['idle_bracket_mW']
                print(f"[POWER] {mode}: {row['mean_mW']:.1f} mW, above idle {dmw:+.1f} mW, {row['decisions_per_s']:.2f} decision/s -> {dmw / row['decisions_per_s'] * 1e3:.1f} uJ/decision "
                      f"({1e3 / row['decisions_per_s']:.2f} ms/decision, {row['draws_per_item']:.1f} draw/decision, accuracy {row['accuracy']:.3f})", flush=True)
                save(out / 'measurement.json', data)
            require(clock_hz(receipt) == clk, 'PL clock changed during measurement')
            record['complete'] = True; save(out / 'measurement.json', data); an = analyze(data); save(out / 'analysis.json', an)
            parts = [f"{n} {st['D0_subtracted_uJ']['mean']:.1f}" for n, st in an['modes'].items() if n != 'D0']
            print(f'[SUMMARY] {rep + 1}/{a.repeats} repeat | D0 subtracted uJ/decision: ' + ' | '.join(parts), flush=True)
        data.update(complete=True, finished_utc=utc()); save(out / 'measurement.json', data); save(out / 'analysis.json', analyze(data))
    finally:
        for p, v in governors.items():
            try: Path(p).write_text(v)
            except OSError: pass

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=['sampled', 'mf'], required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument('--verify', nargs='?', const=-1, type=int); g.add_argument('--power', action='store_true')
    ap.add_argument('--quick', action='store_true'); ap.add_argument('--block', type=float); ap.add_argument('--repeats', type=int); ap.add_argument('--warmup', type=float, default=5)
    ap.add_argument('--subset', type=int, default=SLOT_MAX, help='max inputs per replay load (event memory permitting)')
    ap.add_argument('--build-root', default='outputs/hardware/build');
    ap.add_argument('--package', default='data/reference/hardware/fixture'); ap.add_argument('--out'); ap.add_argument('--order-seed', type=int, default=2610026)
    a = ap.parse_args(); V = a.variant; variant = 1 if V == 'sampled' else 2
    a.block = a.block or (5 if a.quick else 30); a.repeats = a.repeats or (1 if a.quick else 28)
    if a.block <= 0 or a.repeats < 1 or a.warmup < 0 or not 1 <= a.subset <= SLOT_MAX:
        ap.error('Use positive block/repeat counts, non-negative warm-up and 1..512 input slots')
    if a.verify is not None and a.verify != -1 and a.verify < 1:
        ap.error('--verify requires a positive input count, or no count for all inputs')
    pkg_dir = repository_path(a.package)
    check_hashes(pkg_dir)
    pkg = Package(pkg_dir)
    out = repository_path(a.out or f'outputs/hardware/board_{V}'); out.mkdir(parents=True, exist_ok=False)
    build_dir = repository_path(a.build_root) / V
    receipt, clk = preflight(V, build_dir)
    m = Crisp(build_dir / f'registers_{V}.json')
    try:
        check_identity(m, pkg, variant)
        print(f'[LOAD] loading integer tables ({pkg_dir.name}, package {pkg.params["key"]}) ...', flush=True); m.load_tables(pkg)
        if a.verify is not None:
            n = pkg.n if a.verify < 0 else min(a.verify, pkg.n); rec = verify_all(m, pkg, variant, n, out)
            print('BOARD VERIFICATION', 'PASS' if rec['summary']['bit_exact'] == n else 'FAIL', out, flush=True)
            require(rec['summary']['bit_exact'] == n, f'Board verification failed: {rec["summary"]["bit_exact"]}/{n} inputs are bit-exact')
        else:
            require(pkg.n == 2264 or a.quick, 'Full measurement requires all 2264 test inputs; --quick is diagnostic only'); power_run(m, pkg, variant, a, out, receipt, clk)
            print('COMPLETED:', out, flush=True)
    except BaseException as exc:
        save(out / 'failure.json', {'utc': utc(), 'type': type(exc).__name__, 'error': str(exc)}); print('STOPPED:', str(exc), '\nEvidence:', out, flush=True); raise
    finally:
        m.close()
        with zipfile.ZipFile(out / f'board_{V}_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file() and p.suffix not in ('.zip', '.tmp'): z.write(p, p.relative_to(out))
            receipt_path = build_dir / f'build_receipt_{V}.json'
            z.write(receipt_path, 'provenance/' + receipt_path.name)
if __name__ == '__main__': main()
