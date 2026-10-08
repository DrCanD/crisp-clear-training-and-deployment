#!/usr/bin/env python3
"""PS-side driver for the CRISP HLS IP (AXI-Lite at 0xA000_0000) via /dev/mem; run as root (sudo).
Register offsets are provided in registers_<variant>.json.
The board must already contain the matching accelerator and device-tree overlay."""
import json, mmap, os, struct, time
from pathlib import Path
import numpy as np

BASE = 0xA0000000; SPAN = 0x10000
WIN_WORDS = 1024; RES_WORDS = 1024
CMD_NOP, CMD_LOAD_W, CMD_LOAD_EV, CMD_LOAD_DIR, CMD_RUN, CMD_VERIFY, CMD_READ_Z1 = range(7)
(RG_W1, RG_B1, RG_W2, RG_B2, RG_WRO, RG_BRO, RG_D1A, RG_D1C, RG_D1H, RG_D2A, RG_D2C, RG_D2H, RG_ROA, RG_ROC, RG_NOISE, RG_SIG, RG_THRS, RG_THRF) = range(18)
MODE_D0, MODE_PREFIX, MODE_N1, MODE_SEQ, MODE_FIX, MODE_MF = range(6)
MODE_NAMES = {'D0': MODE_D0, 'PREFIX': MODE_PREFIX, 'N1': MODE_N1, 'SEQ': MODE_SEQ, 'FIX': MODE_FIX, 'MF': MODE_MF}
R = dict(MAGIC=0, PACKAGE=1, VARIANT=2, BUILD=3, MODE=4, ITEMS=5, PASSES=6, DRAWS=7, EV=9, SPK1=11, SPK2=13, ABSTAIN=15, CORRECT=16,
         Z1_CHK=17, N1=18, SEQ=19, SEQ_USED=20, FIX=21, REF=22, MF=23, N_SLOTS=24, WIDTHS=25, MF_LOGIT=32)
RES_MAGIC = 0x43525331
EV_WORDS64 = 1 << 17; EV_MAX = EV_WORDS64 * 4; SLOT_MAX = 512

def s32(v): return v - (1 << 32) if v >= (1 << 31) else v

class Package:
    """An exported integer model and input fixture: tables packed into LOAD_W word streams, inputs as packed 16-bit event streams."""
    def __init__(self, d):
        self.dir = Path(d); self.params = json.loads((self.dir / 'parameters.json').read_text()); f = self.params['fmt']
        self.T, self.NIN, self.H, self.C, self.K, self.M = f['T'], f['N_IN'], f['H'], f['C'], f['K'], f['M']
        self.looks = [m for m in f['LOOKS'] if m <= self.M]
        rd = lambda n, dt: np.fromfile(self.dir / n, dtype=dt)
        H, NIN, C, K = self.H, self.NIN, self.C, self.K
        W1 = rd('layer1_weights.bin', '<i1').reshape(H, NIN); W2 = rd('layer2_weights.bin', '<i1').reshape(H, H); WRO = rd('readout_weights.bin', '<i1').reshape(C, H)
        self.regions = {
            RG_W1: self.pack8(np.ascontiguousarray(W1.T)), RG_W2: self.pack8(np.ascontiguousarray(W2.T)), RG_WRO: self.pack8(np.pad(np.ascontiguousarray(WRO.T), ((0, 0), (0, 32 - C)))),   # stride-32 rows
            RG_B1: rd('layer1_bias.bin', '<u4'), RG_B2: rd('layer2_bias.bin', '<u4'), RG_BRO: rd('readout_bias.bin', '<u4'),
            RG_D1A: rd('layer1_decay.bin', '<u4'), RG_D1C: rd('layer1_input_coefficient.bin', '<u4'), RG_D1H: rd('layer1_offset.bin', '<u4'),
            RG_D2A: rd('layer2_decay.bin', '<u4'), RG_D2C: rd('layer2_input_coefficient.bin', '<u4'), RG_D2H: rd('layer2_offset.bin', '<u4'),
            RG_ROA: rd('readout_decay.bin', '<u4'), RG_ROC: rd('readout_input_coefficient.bin', '<u4'),
            RG_NOISE: self.pack16(rd('logistic_noise.bin', '<u2')), RG_SIG: self.pack8(rd('sigmoid_probability.bin', '<u1')),
            RG_THRS: self.pack16(rd('sequential_threshold.bin', '<u2')), RG_THRF: self.pack16(rd('fixed_threshold.bin', '<u2'))}
        self.events = rd('test_events.bin', '<u4'); self.offsets = rd('test_offsets.bin', '<u4'); self.labels = rd('test_labels.bin', '<i2').astype(np.int64)
        self.n = len(self.labels)
        self.exp = {k: rd(n, dt) for k, n, dt in [('votes_c', 'expected_certificate_votes.bin', '<i1'), ('votes_r', 'expected_reference_votes.bin', '<i1'), ('seq', 'expected_sequential_decisions.bin', '<i2'),
                                                   ('fix', 'expected_fixed_decisions.bin', '<i2'), ('ref', 'expected_reference_decisions.bin', '<i2'), ('mf', 'expected_mean_field_decisions.bin', '<i2'), ('mf_logits', 'expected_mean_field_logits.bin', '<i8')]}
        self.exp['votes_c'] = self.exp['votes_c'].reshape(self.n, self.M); self.exp['votes_r'] = self.exp['votes_r'].reshape(self.n, self.M)
        self.exp['seq'] = self.exp['seq'].reshape(self.n, 2); self.exp['mf_logits'] = self.exp['mf_logits'].reshape(self.n, self.C)
        z1 = self.dir / 'expected_layer1_logits.bin'; self.exp_z1 = np.fromfile(z1, '<i2').reshape(self.n, self.T, H) if z1.exists() else None
        self._streams = {}
    @staticmethod
    def pack8(a):
        b = np.ascontiguousarray(a).reshape(-1).view(np.uint8); pad = (-len(b)) % 4
        return np.frombuffer(np.concatenate([b, np.zeros(pad, np.uint8)]).tobytes(), '<u4')
    @staticmethod
    def pack16(a):
        b = np.ascontiguousarray(a).astype('<u2').reshape(-1); pad = (-len(b)) % 2
        return np.frombuffer(np.concatenate([b, np.zeros(pad, '<u2')]).tobytes(), '<u4')
    def stream(self, i):
        """16-bit on-chip event stream of input i (per time step: ch | count<<10 words with count <= 63, then 0xFFFF)."""
        if i in self._streams: return self._streams[i]
        e = self.events[self.offsets[i]:self.offsets[i + 1]].astype(np.int64)
        t = e & 0x7F; ch = (e >> 7) & 0x3FF; c = (e >> 17) & 0xFF
        order = np.argsort(t, kind='stable'); t, ch, c = t[order], ch[order], c[order]
        out = []
        for step in range(self.T):
            sel = np.nonzero(t == step)[0]
            for k in sel:
                cc = int(c[k]); chk = int(ch[k])
                while cc > 0:
                    q = min(cc, 63); out.append(chk | (q << 10)); cc -= q
            out.append(0xFFFF)
        s = np.array(out, dtype=np.uint16); self._streams[i] = s; return s
    def batch(self, ids):
        """concatenated stream + slot starts for a list of input indices; raises if the on-chip memory would overflow."""
        parts = [self.stream(i) for i in ids]; starts = np.cumsum([0] + [len(p) for p in parts[:-1]]).astype(np.int64)
        total = sum(len(p) for p in parts)
        if total > EV_MAX or len(ids) > SLOT_MAX: raise ValueError(f'batch too large: {total} events / {len(ids)} slots')
        return np.concatenate(parts), starts
    def plan_batches(self, ids, max_slots=SLOT_MAX):
        """split input indices into consecutive batches that fit the event memory and the slot table."""
        batches, cur, size = [], [], 0
        for i in ids:
            n = len(self.stream(i))
            if cur and (size + n > EV_MAX or len(cur) >= max_slots): batches.append(cur); cur, size = [], 0
            cur.append(i); size += n
        if cur: batches.append(cur)
        return batches

class Crisp:
    def __init__(self, regmap, base=BASE):
        rm = json.load(open(regmap)); self.o = {k: int(v, 0) if isinstance(v, str) else v for k, v in rm.items()}
        for k in ('AP_CTRL', 'CMD_DATA', 'ARG0_DATA', 'ARG1_DATA', 'MODE_DATA', 'WINDOW_BASE', 'RES_BASE'):
            assert k in self.o, f'regmap lacks {k}'
        self.fd = os.open(Path(os.sep) / 'dev' / 'mem', os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, SPAN, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=base)
        self.mv = memoryview(self.mm).cast('I')
    def close(self):
        self.mv.release(); self.mm.close(); os.close(self.fd)
    def rd(self, off): return self.mv[off // 4]
    def wr(self, off, v): self.mv[off // 4] = v & 0xFFFFFFFF
    def rd_block(self, off, n): i = off // 4; return [self.mv[i + k] for k in range(n)]
    def wr_window(self, vals):
        i = self.o['WINDOW_BASE'] // 4
        for k, v in enumerate(vals): self.mv[i + k] = int(v) & 0xFFFFFFFF
    def idle(self): return bool(self.rd(self.o['AP_CTRL']) & 0x4)
    def start(self, cmd, arg0=0, arg1=0, mode=0):
        assert self.idle(), 'IP busy'
        self.wr(self.o['CMD_DATA'], cmd); self.wr(self.o['ARG0_DATA'], arg0); self.wr(self.o['ARG1_DATA'], arg1); self.wr(self.o['MODE_DATA'], mode)
        self.wr(self.o['AP_CTRL'], 0x1); self.t_start = time.perf_counter()
    def wait(self, timeout=3600.0):
        t0 = time.perf_counter()
        while not self.idle():
            if time.perf_counter() - t0 > timeout: raise TimeoutError('IP did not finish')
            time.sleep(0.001)
        return time.perf_counter() - self.t_start
    def call(self, *a, **k): self.start(*a, **k); return self.wait()
    def results(self):
        r = self.rd_block(self.o['RES_BASE'], RES_WORDS); assert r[R['MAGIC']] == RES_MAGIC, 'bad result magic (IP not programmed?)'; return r
    def items_done(self): return self.rd(self.o['RES_BASE'] + 4 * R['ITEMS'])
    def passes_done(self): return self.rd(self.o['RES_BASE'] + 4 * R['PASSES'])
    # ── loading ──
    def load_words(self, region, words):
        words = np.asarray(words, dtype=np.uint32)
        for c in range(0, max(1, (len(words) + WIN_WORDS - 1) // WIN_WORDS)):
            chunk = np.zeros(WIN_WORDS, np.uint32); seg = words[c * WIN_WORDS:(c + 1) * WIN_WORDS]; chunk[:len(seg)] = seg
            self.wr_window(chunk); self.call(CMD_LOAD_W, c, region)
    def load_tables(self, pkg):
        for region, words in pkg.regions.items(): self.load_words(region, words)
    def load_inputs(self, pkg, ids):
        stream, starts = pkg.batch(ids)
        w = Package.pack16(stream)
        for c in range((len(w) + WIN_WORDS - 1) // WIN_WORDS):
            chunk = np.zeros(WIN_WORDS, np.uint32); seg = w[c * WIN_WORDS:(c + 1) * WIN_WORDS]; chunk[:len(seg)] = seg
            self.wr_window(chunk); self.call(CMD_LOAD_EV, c, 0)
        d = np.zeros(WIN_WORDS, np.uint32)
        for s, i in enumerate(ids): d[2 * s] = starts[s]; d[2 * s + 1] = ((int(i) & 0xFFFF) << 8) | (int(pkg.labels[i]) & 0xFF)
        self.wr_window(d); self.call(CMD_LOAD_DIR, len(ids), 0)
        assert self.results()[R['N_SLOTS']] == len(ids)
        return len(stream)
    # ── verification of one loaded slot against the package expectations ──
    def verify_slot(self, pkg, slot, i, variant, check_z1=True):
        dt = self.call(CMD_VERIFY, slot, 0); r = np.array(self.results(), dtype=np.uint32); err = []
        if r[R['PACKAGE']] != int(pkg.params['key'], 16): err.append('package id')
        if check_z1 and pkg.exp_z1 is not None:
            chk = int(pkg.exp_z1[i].astype(np.int64).sum() & 0xFFFFFFFF)
            if int(r[R['Z1_CHK']]) != chk: err.append('z1 checksum')
        if variant == 1:
            M = pkg.M; vc = np.zeros(M, np.int64); vr = np.zeros(M, np.int64)
            for d in range(M):
                vc[d] = (r[128 + d // 4] >> (8 * (d % 4))) & 0xFF; vr[d] = (r[128 + M // 4 + d // 4] >> (8 * (d % 4))) & 0xFF
            if not np.array_equal(vc, pkg.exp['votes_c'][i]): err.append(f'votes_cert (first diff draw {int(np.nonzero(vc != pkg.exp["votes_c"][i])[0][0])})')
            if not np.array_equal(vr, pkg.exp['votes_r'][i]): err.append('votes_ref')
            got = dict(seq=s32(int(r[R['SEQ']])), used=int(r[R['SEQ_USED']]), fix=s32(int(r[R['FIX']])), ref=s32(int(r[R['REF']])), n1=s32(int(r[R['N1']])))
            exp = dict(seq=int(pkg.exp['seq'][i, 0]), used=int(pkg.exp['seq'][i, 1]), fix=int(pkg.exp['fix'][i]), ref=int(pkg.exp['ref'][i]), n1=int(pkg.exp['votes_c'][i, 0]))
            for k in got:
                if got[k] != exp[k]: err.append(f'{k} {got[k]}!={exp[k]}')
        else:
            got = dict(mf=s32(int(r[R['MF']])), logits=[int(r[R['MF_LOGIT'] + 2 * c]) | (s32(int(r[R['MF_LOGIT'] + 2 * c + 1])) << 32) for c in range(pkg.C)])
            got['logits'] = [int(np.int64(v)) for v in got['logits']]
            exp = dict(mf=int(pkg.exp['mf'][i]), logits=[int(v) for v in pkg.exp['mf_logits'][i]])
            if got['mf'] != exp['mf']: err.append(f'mf {got["mf"]}!={exp["mf"]}')
            if got['logits'] != exp['logits']: err.append('mf_logits')
        return err, got, dt, r
    def read_z1(self, pkg):
        n = pkg.T * pkg.H; out = np.zeros(n, np.int16)
        for c in range((n + 2 * WIN_WORDS - 1) // (2 * WIN_WORDS)):
            self.call(CMD_READ_Z1, c, 0)
            w = np.array(self.rd_block(self.o['WINDOW_BASE'], WIN_WORDS), dtype=np.uint32)
            seg = np.frombuffer(w.astype('<u4').tobytes(), '<i2'); k = c * 2 * WIN_WORDS; out[k:k + len(seg)] = seg[:max(0, min(len(seg), n - k))]
        return out.reshape(pkg.T, pkg.H)

