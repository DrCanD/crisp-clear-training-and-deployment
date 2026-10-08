# Integer arithmetic reference for CRISP KV260 inference.
# Fixed-point inference matches the archived tables; overflow diagnostics track
# every intermediate accumulator. Zero weights and empty event streams are supported.
#
# Network (SHD, T=100): x[t] (uint8 counts, 700) -> layer 1 -> layer 2 -> readout -> argmax over classes.
#   layer l:  u[t,j]   = sum_i Wq[j,i] * s_in[t,i] + bq[j]                       (integer, exact)
#             S[j,k]  <- (aq[j,k] * S[j,k]) >> FA  +  (cq[j,k] * u[t,j]) >> shc_l  (K=8 poles, zero initial state)
#             z[t,j]   = sat16( (sum_k S[j,k]) >> (FS - FZ) + hq[j] )            (logit units, beta folded in)
#             sampled: s[t,j] = 1 if z[t,j] > NOISE[U]  (U = top NB bits of the lane RNG word)   -> binary spike
#             mean-field: p8[t,j] = SIG[ clamp((z >> (FZ-5)) + 2^(SB-1), 0, 2^SB-1) ]            -> 8-bit probability
#   readout:  r[t,c]   = sum_j Wro_q[c,j] * s2[t,j] + bro_q[c];  R[c] <- (aro_q[c]*R[c]) >> FA + (cro_q[c]*r[t,c]) >> shc_ro
#             L[c]    += R[c];  decision = first argmax_c L[c]   (division by T dropped: argmax-invariant)
#   mean-field path uses p8 (0..255) in place of spikes with the bias and c shifts scaled by 2^8 (shc + 8).
# Layer 1 (input driven) is the deterministic prefix shared by every draw: z1[t,j] is computed once per input.
# RNG: xoshiro128** per lane; lane = j mod P inside a layer, consumed in order t, layer 1 neurons, layer 2 neurons.
#      Each (input, stream, draw) is its own seed: seed64 = idx<<32 | stream<<24 | draw<<8 | lane, expanded by splitmix64.
#      stream 0 = certificate draws, stream 1 = independent reference draws.
import numpy as np, torch, math, json, hashlib
from pathlib import Path

ORACLE_VERSION = 1
MASK32 = (1 << 32) - 1

def floor_shift(x, s):
    """x * 2^-s with floor rounding (arithmetic shift) for s >= 0, exact left shift for s < 0; int64 torch tensors."""
    if s >= 0:
        return torch.div(x, 1 << s, rounding_mode='floor') if s else x
    return x * (1 << (-s))

def sat(x, bits):
    lim = 1 << (bits - 1)
    return torch.clamp(x, -lim, lim - 1)

# ---------------------------------------------------------------- folding (float64) ----------------------------------
def fold_params(sd, beta, eps=1e-5):
    """state_dict of the trained SHD CRISP net -> float64 folded parameters (weights in float, per-layer dendrite
    coefficients in logit units). z = sum_k S_k + h,  S_k <- a_k S_k + (g mix_k B_k) u,  g = beta*gamma/sqrt(var+eps),
    h = beta*(delta - theta) - g*mu;  readout R_c <- a_c R_c + (mix_c B_c) r_c."""
    g = lambda k: sd[k].double().cpu().numpy()
    F = {'beta': float(beta), 'layers': []}
    for l in range(2):
        p = f'layers.{l}.'
        lam = -np.exp(g(p + 'dend.raw_lam')); a = np.exp(lam); B = np.expm1(lam) / lam        # dt = 1
        gain = beta * g(p + 'norm.weight') / np.sqrt(g(p + 'norm.running_var') + eps)
        h = beta * (g(p + 'norm.bias') - g(p + 'theta')) - gain * g(p + 'norm.running_mean')
        F['layers'].append({'W': g(p + 'proj.weight'), 'b': g(p + 'proj.bias'), 'a': a,
                            'c': gain[:, None] * g(p + 'dend.mix') * B, 'h': h})
    lam = -np.exp(g('ro.raw_lam'))[:, 0]; a = np.exp(lam); B = np.expm1(lam) / lam
    F['ro'] = {'W': g('readout.weight'), 'b': g('readout.bias'), 'a': a, 'c': g('ro.mix')[:, 0] * B}
    return F

def folded_float_forward(F, X):
    """Float64 graded forward of the folded model (numpy). X [B,T,700] -> logits [B,C], z1 [B,T,H]."""
    X = X.astype(np.float64); Bn, T, _ = X.shape
    s = X; zs = []
    for L in F['layers']:
        u = s @ L['W'].T + L['b']                                     # [B,T,H]
        S = np.zeros((Bn,) + L['a'].shape); z = np.zeros(u.shape)
        for t in range(T):
            S = L['a'] * S + L['c'] * u[:, t, :, None]; z[:, t] = S.sum(-1) + L['h']
        zs.append(z); s = 1 / (1 + np.exp(-z))
    r = s @ F['ro']['W'].T + F['ro']['b']; R = np.zeros((Bn, r.shape[-1])); Lg = np.zeros_like(R)
    for t in range(T):
        R = F['ro']['a'] * R + F['ro']['c'] * r[:, t]; Lg += R
    return Lg / T, zs[0]

# ---------------------------------------------------------------- quantization --------------------------------------
def q_weights(W, bits):
    s = np.abs(W).max() / (2 ** (bits - 1) - 1)
    if s == 0: return np.zeros_like(W,dtype=np.int64), 1.0
    return np.rint(W / s).astype(np.int64), float(s)

def q_coeff(c, bits):
    """signed coefficient table -> (int64 table, fractional bits E) with the largest E such that |cq| < 2^(bits-1)."""
    m = np.abs(c).max(); E = int(math.floor(math.log2((2 ** (bits - 1) - 1) / m))) if m > 0 else 0
    E = max(min(E, 60), -8)
    cq = np.rint(c * 2.0 ** E).astype(np.int64); assert np.abs(cq).max() < 2 ** (bits - 1)
    return cq, E

def exact_threshold_table(denom, n_max):
    """thr[n] = smallest a (b = n - a) with 2*denom*sum_{i>=a} C(n,i) <= 2^n, i.e. two-sided p = min(1, 2*tail) <= 1/denom
    (Cohen et al. 2019 BinomPValue at p0 = 1/2); n_max+1 marks 'impossible'. Exact integers (same statement as the Lean
    `accepts`)."""
    thr = np.full(n_max + 1, n_max + 1, dtype=np.int64)
    for n in range(1, n_max + 1):
        tail = 0
        for a in range(n, -1, -1):
            tail += math.comb(n, a)
            if 2 * denom * tail > 2 ** n: break
            if 2 * a > n: thr[n] = a                                   # the two-sided test never accepts a tie
    return thr

def build_tables(fmt):
    NB, FZ, SB = fmt['NB'], fmt['FZ'], fmt['SB']
    u = (np.arange(2 ** NB) + 0.5) / 2 ** NB
    noise = np.rint(np.log(u / (1 - u)) * 2.0 ** FZ); noise = np.clip(noise, -32768, 32767).astype(np.int64)
    zz = (np.arange(2 ** SB) - 2 ** (SB - 1)) / 32.0                   # z resolution 1/32 (FZ-5 bits dropped)
    sig = np.clip(np.rint(256 / (1 + np.exp(-zz))), 0, 255).astype(np.int64)
    return {'NOISE': noise, 'SIG': sig, 'THR_SEQ': exact_threshold_table(fmt['DENOM_SEQ'], fmt['M']),
            'THR_FIX': exact_threshold_table(fmt['DENOM_FIX'], fmt['M'])}

def quantize(F, fmt):
    """Folded float params + format -> integer tables (int64 numpy) and the shifts the hardware applies."""
    Q = {'fmt': dict(fmt), 'layers': [], 'scales': {}}
    for l, L in enumerate(F['layers']):
        Wq, sw = q_weights(L['W'], fmt['WB']); bq = np.rint(L['b'] / sw).astype(np.int64)
        aq = np.clip(np.rint(L['a'] * 2.0 ** fmt['FA']), 0, 2 ** fmt['FA'] - 1).astype(np.int64)
        cq, E = q_coeff(L['c'] * sw, fmt['CB'])
        hq = np.rint(L['h'] * 2.0 ** fmt['FZ']).astype(np.int64)
        Q['layers'].append({'W': Wq, 'b': bq, 'a': aq, 'c': cq, 'h': hq, 'shc': E - fmt['FS']})
        Q['scales'][f'sw{l + 1}'] = sw; Q['scales'][f'EC{l + 1}'] = E
    Wq, sw = q_weights(F['ro']['W'], fmt['WB']); bq = np.rint(F['ro']['b'] / sw).astype(np.int64)
    aq = np.clip(np.rint(F['ro']['a'] * 2.0 ** fmt['FA']), 0, 2 ** fmt['FA'] - 1).astype(np.int64)
    cq, E = q_coeff(F['ro']['c'] * sw, fmt['CB'])
    Q['ro'] = {'W': Wq, 'b': bq, 'a': aq, 'c': cq, 'shc': E - fmt['FS']}
    Q['scales']['swro'] = sw; Q['scales']['ECro'] = E
    Q.update(build_tables(fmt))
    return Q

# ---------------------------------------------------------------- RNG (xoshiro128**, splitmix64 seeding) ------------
def splitmix64_np(x):
    """x uint64 numpy array -> next (state, output), per Vigna; wrap-around arithmetic in uint64."""
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        z = x
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return x, z

def rng_init(idx, stream, draw, lanes):
    """Seeds for all (input idx [n], draw [m]) pairs and P lanes -> state tensor [n*m, P, 4] (int64 holding uint32).
    idx: int64 numpy [n]; draw: numpy [m]; returns torch int64 CPU tensor."""
    idx = np.asarray(idx, dtype=np.uint64)[:, None, None]; draw = np.asarray(draw, dtype=np.uint64)[None, :, None]
    lane = np.arange(lanes, dtype=np.uint64)[None, None, :]
    seed = (idx << np.uint64(32)) | (np.uint64(stream) << np.uint64(24)) | (draw << np.uint64(8)) | lane
    x = seed; words = []
    for _ in range(4):
        x, z = splitmix64_np(x); words.append((z & np.uint64(MASK32)).astype(np.int64))
    st = np.stack(words, -1).reshape(-1, lanes, 4)
    return torch.from_numpy(st)

def rotl(x, k):
    return ((x << k) | (x >> (32 - k))) & MASK32

def xoshiro_step(st):
    """st [..., 4] int64 (uint32 words) -> (new state, output [...]) ; xoshiro128** (Blackman & Vigna)."""
    s0, s1, s2, s3 = st[..., 0], st[..., 1], st[..., 2], st[..., 3]
    out = (rotl((s1 * 5) & MASK32, 7) * 9) & MASK32
    t = (s1 << 9) & MASK32
    s2 = s2 ^ s0; s3 = s3 ^ s1; s1 = s1 ^ s2; s0 = s0 ^ s3; s2 = s2 ^ t; s3 = rotl(s3, 11)
    return torch.stack([s0, s1, s2, s3], -1), out

def uniforms(st, H, NB):
    """Draw H uniforms per stream in hardware order (round r serves neurons r*P .. r*P+P-1). st [n, P, 4] -> (st, U [n, H])."""
    P = st.shape[1]; outs = []
    for _ in range(H // P):
        st, o = xoshiro_step(st); outs.append(o >> (32 - NB))
    return st, torch.cat(outs, 1)

# ---------------------------------------------------------------- integer forward --------------------------------------
def events_to_dense(ev_words, T, N_IN):
    """event words -> uint8 counts [T, N_IN] (one input). word: count<<17 | ch<<7 | t."""
    X = np.zeros((T, N_IN), np.int64)
    t = ev_words & 0x7F; ch = (ev_words >> 7) & 0x3FF; c = (ev_words >> 17) & 0xFF
    np.add.at(X, (t, ch), c)
    return X

def dense_to_events(X):
    t, ch = np.nonzero(X); c = X[t, ch].astype(np.int64)
    assert (not c.size or (c.min() >= 0 and c.max() <= 255)) and X.shape[0] <= 128 and X.shape[1] <= 1024
    return ((c << 17) | (ch.astype(np.int64) << 7) | t.astype(np.int64)).astype(np.uint32)

def dend_step(S, u, L, fmt, extra_shift=0):
    """S [n,H,K], u [n,H] int64 -> new S, z [n,H] (saturated to 16 bits)."""
    a = L['a_t']; c = L['c_t']
    S = floor_shift(a * S, fmt['FA']) + floor_shift(c * u[:, :, None], L['shc'] + extra_shift)
    z = sat(floor_shift(S.sum(-1), fmt['FS'] - fmt['FZ']) + L['h_t'], 16)
    return S, z

def to_dev(Q, dev):
    """attach torch copies of the tables (idempotent): int64 tables, plus float weight matrices for exact matmuls."""
    for L in Q['layers'] + [Q['ro']]:
        for k in ('W', 'b', 'a', 'c', 'h'):
            if k in L: L[k + '_t'] = torch.as_tensor(L[k], dtype=torch.int64, device=dev)
        L['W_f32'] = L['W_t'].to(torch.float32).T.contiguous(); L['W_f64'] = L['W_t'].to(torch.float64).T.contiguous()
        L['W_abs_rowsum'] = int(L['W_t'].abs().sum(1).max())        # for the exactness bound of float matmuls
    for k in ('NOISE', 'SIG', 'THR_SEQ', 'THR_FIX'):
        Q[k + '_t'] = torch.as_tensor(Q[k], dtype=torch.int64, device=dev)
    return Q

def imatmul(A, L, amax):
    """exact integer product A[n,i] (|A| <= amax, int64) x W[j,i] -> int64 [n,j]: float32 when every partial sum is
    < 2^24 in magnitude (exactly representable), float64 (exact < 2^53) otherwise. Integer matmul is not available on GPU."""
    bound = amax * L['W_abs_rowsum']
    if bound < 2 ** 24:
        return (A.to(torch.float32) @ L['W_f32']).round().to(torch.int64)
    assert bound < 2 ** 53
    return (A.to(torch.float64) @ L['W_f64']).round().to(torch.int64)

def prefix_z1(Q, X, dev):
    """X int64 [n,T,700] (counts) -> z1 [n,T,H] int64 (int16 range), u1 max magnitude, S1 max magnitude."""
    fmt = Q['fmt']; L = Q['layers'][0]
    X = torch.as_tensor(X, dtype=torch.int64, device=dev); n, T, _ = X.shape
    u = imatmul(X.reshape(n * T, -1), L, int(X.max())).view(n, T, -1) + L['b_t']
    S = torch.zeros(n, u.shape[-1], fmt['K'], dtype=torch.int64, device=dev); zs = []; smax = 0
    for t in range(T):
        S, z = dend_step(S, u[:, t], L, fmt); zs.append(z); smax = max(smax, int(S.abs().max()))
    return torch.stack(zs, 1), int(u.abs().max()), smax

def sampled_draws(Q, z1, idx, stream, draws, dev):
    """z1 [n,T,H] int64 (prefix of n inputs), idx [n] input indices, draws [m] draw indices ->
    votes [n,m] int8 class decisions and bookkeeping maxima. Streams = n*m (input-major)."""
    fmt = Q['fmt']; T, H, K, P, NB = fmt['T'], fmt['H'], fmt['K'], fmt['P'], fmt['NB']
    n, m = z1.shape[0], len(draws); ns = n * m
    L2, RO = Q['layers'][1], Q['ro']; NOISE = Q['NOISE_t']
    st = rng_init(idx, stream, draws, P).to(dev)
    S = torch.zeros(ns, H, K, dtype=torch.int64, device=dev)
    R = torch.zeros(ns, RO['W'].shape[0], dtype=torch.int64, device=dev); Lg = torch.zeros_like(R)
    mx = {'u2': 0, 'S2': 0, 'r': 0, 'R': 0, 'L': 0}; spikes = [0, 0]
    for t in range(T):
        z1t = z1[:, t].repeat_interleave(m, 0)                       # [ns, H] (input-major streams)
        st, U = uniforms(st, H, NB); s1 = (z1t > NOISE[U]).to(torch.int64)
        u2 = imatmul(s1, L2, 1) + L2['b_t']
        S, z2 = dend_step(S, u2, L2, fmt)
        st, U = uniforms(st, H, NB); s2 = (z2 > NOISE[U]).to(torch.int64)
        r = imatmul(s2, RO, 1) + RO['b_t']
        R = floor_shift(RO['a_t'] * R, fmt['FA']) + floor_shift(RO['c_t'] * r, RO['shc']); Lg = Lg + R
        spikes[0] += int(s1.sum()); spikes[1] += int(s2.sum())
        mx['u2'] = max(mx['u2'], int(u2.abs().max())); mx['S2'] = max(mx['S2'], int(S.abs().max()))
        mx['r'] = max(mx['r'], int(r.abs().max())); mx['R'] = max(mx['R'], int(R.abs().max()))
        mx['L'] = max(mx['L'],int(Lg.abs().max()))
    votes = Lg.argmax(1).view(n, m).to(torch.int8)                   # torch.argmax returns the first maximal index
    return votes, mx, [s / (ns * T * H) for s in spikes]

def mf_forward(Q, z1, dev):
    """deterministic mean-field path on the prefix: z1 [n,T,H] -> decisions [n] int, logits L [n,C] int64, maxima."""
    fmt = Q['fmt']; T, H, K = fmt['T'], fmt['H'], fmt['K']; L2, RO, SIG = Q['layers'][1], Q['ro'], Q['SIG_t']
    n = z1.shape[0]; half = 1 << (fmt['SB'] - 1)
    S = torch.zeros(n, H, K, dtype=torch.int64, device=dev)
    R = torch.zeros(n, RO['W'].shape[0], dtype=torch.int64, device=dev); Lg = torch.zeros_like(R); mx = {'u2': 0, 'S2': 0, 'r': 0, 'R': 0, 'L': 0}
    lut = lambda z: SIG[torch.clamp(floor_shift(z, fmt['FZ'] - 5) + half, 0, 2 * half - 1)]
    for t in range(T):
        p1 = lut(z1[:, t]); u2 = imatmul(p1, L2, 255) + L2['b_t'] * 256
        S, z2 = dend_step(S, u2, L2, fmt, extra_shift=8)
        p2 = lut(z2); r = imatmul(p2, RO, 255) + RO['b_t'] * 256
        R = floor_shift(RO['a_t'] * R, fmt['FA']) + floor_shift(RO['c_t'] * r, RO['shc'] + 8); Lg = Lg + R
        mx['u2'] = max(mx['u2'], int(u2.abs().max())); mx['S2'] = max(mx['S2'], int(S.abs().max())); mx['r'] = max(mx['r'], int(r.abs().max()))
        mx['R'] = max(mx['R'], int(R.abs().max())); mx['L'] = max(mx['L'], int(Lg.abs().max()))
    return Lg.argmax(1), Lg, mx

# ---------------------------------------------------------------- certificate on vote streams -------------------------
def counts_top2(votes_prefix, C):
    """votes [n, m] -> (cA, nA, nB) with stable ordering (lowest class index wins ties, as np.argsort(kind='stable'))."""
    n = votes_prefix.shape[0]
    cnt = np.zeros((n, C), np.int64); np.add.at(cnt, (np.repeat(np.arange(n), votes_prefix.shape[1]), votes_prefix.ravel()), 1)
    o = np.argsort(-cnt, 1, kind='stable'); cA = o[:, 0]; nA = cnt[np.arange(n), cA]; nB = cnt[np.arange(n), o[:, 1]]
    return cA, nA, nB, cnt

def predict_table(votes_prefix, C, thr):
    cA, nA, nB, _ = counts_top2(votes_prefix, C)
    ok = nA >= thr[nA + nB]
    return np.where(ok, cA, -1)

def sequential(votes, C, looks, thr_seq):
    """votes [n, M] -> (decision or -1, draws used): first look whose two-sided test passes at the per-look level."""
    n = votes.shape[0]; ret = np.full(n, -1); used = np.full(n, looks[-1]); open_ = np.ones(n, bool)
    for m in looks:
        r = predict_table(votes[:, :m], C, thr_seq); hit = open_ & (r >= 0)
        ret[hit] = r[hit]; used[hit] = m; open_ &= ~hit
    return ret, used
