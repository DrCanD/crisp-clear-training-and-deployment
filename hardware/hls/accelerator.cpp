// accelerator.cpp — CRISP SHD fixed-point model on the KV260: deterministic layer-1 prefix, sampled deployment with the
// logistic-noise comparator (VARIANT_SAMPLED) or the deterministic mean-field pass (VARIANT_MF), on-chip sequential /
// fixed-budget PREDICT certificate (two-sided integer threshold tables), replay of up to SLOT_MAX test inputs from
// URAM for PS-idle power measurement. Arithmetic is the one defined in the package's integer_reference.py (floor shifts, int16
// saturation of z, xoshiro128** lanes seeded per (input, stream, draw, lane) by splitmix64).
#include "accelerator.hpp"
#include "build_identity.hpp"

// ─────────────────────────────────────────── arithmetic helpers ───────────────────────────────────────────
template <int NA, int NB> static inline sint<NA + NB> mul(sint<NA> a, sint<NB> b) {
#pragma HLS INLINE
#if CRISP_AP_INT
    return a * b;                                    // ap_int widens the product to NA + NB bits
#else
    return (sint<NA + NB>)a * (sint<NA + NB>)b;      // plain integers: widen first
#endif
}
template <typename OUT, int W> static inline OUT shift_c(sint<W> v, const int sh) {
#pragma HLS INLINE
    if (sh >= 0) return (OUT)(v >> sh);
    return (OUT)((sint<W + 24>)v << (-sh));
}
static inline z_t sat16(sint<HW_WS + 8> v) {
#pragma HLS INLINE
    if (v > 32767) return (z_t)32767;
    if (v < -32768) return (z_t)(-32768);
    return (z_t)v;
}
static inline u32 rotl32(u32 x, int k) {
#pragma HLS INLINE
    return (u32)((x << k) | (x >> (32 - k)));
}
static inline u64 splitmix_step(u64 &x) {
#pragma HLS INLINE
    x = x + (u64)0x9E3779B97F4A7C15ULL;
    u64 z = x;
    z = (z ^ (z >> 30)) * (u64)0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * (u64)0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
// xoshiro128** (Blackman & Vigna): state 4 x uint32, output uint32
static inline u32 xoshiro(u32 s[4]) {
#pragma HLS INLINE
    const u32 out = (u32)(rotl32((u32)(s[1] * 5), 7) * 9);
    const u32 t = (u32)(s[1] << 9);
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl32(s[3], 11);
    return out;
}

// ─────────────────────────────────────────── on-chip tables (loaded by LOAD_W) ───────────────────────────
static w8_t W1[N_IN * NCHUNK][WCOL];     // column-major: W1[ch*8 + j/32][j%32] = W1_q[j][ch]
static w8_t W2[H * NCHUNK][WCOL];        // W2[i*8 + j/32][j%32] = W2_q[j][i]
static w8_t WRO[H][NC];                  // WRO[j][c] = Wro_q[c][j]
static sint<32> B1[H], B2[H], BRO[NC], H1[H], H2[H];
static a_t A1[K][H], A2[K][H], ARO[NC];
static c_t C1[K][H], C2[K][H], CRO[NC];
static z_t NOISE[P][1 << HW_NB];         // one copy per lane
static cnt_t SIG[P][1 << HW_SB];         // one copy per lane (mean-field variant)
static uint_<16> THRS[M_DRAWS + 1], THRF[M_DRAWS + 1];
static u64 EV[EV_WORDS];                 // packed 16-bit events, 4 per word (event e at word e>>2, half e&3)
static u32 SLOT_START[SLOT_MAX]; static uint_<16> SLOT_IDX[SLOT_MAX]; static uint_<8> SLOT_LBL[SLOT_MAX];
static z_t Z1[T_STEPS][H];               // layer-1 prefix of the current input
// working state
static s_t S1[K][H], S2[K][H];
static u1_t U1[H]; static u2_t U2[H]; static u2mf_t U2MF[H];
static cnt_t P1[H];
static uint_<5> ACTL[P][NROUND], ACTL2[P][NROUND];   // per-lane lists of spiking rounds (one write port per lane)
static int NL[P], NL2[P], OFF[P + 1], OFF2[P + 1];
static rs_t RS[NC]; static l_t LG[NC];
static u32 LANE[P][4];

static inline ev_t ev_read(u32 e) {
#pragma HLS INLINE
    const u64 w = EV[e >> 2];
    const int h = (int)(e & 3);
    return (ev_t)((w >> (16 * h)) & (u64)0xFFFF);
}

// ─────────────────────────────────────────── layer-1 prefix (input driven, once per input) ───────────────
// returns the event offset after the input; counts events; writes Z1[t][j]
static u32 prefix(u32 off, bool compute, u32 &n_events) {
#pragma HLS INLINE off
    u32 e = off; u32 nev = 0;
    if (compute) {
    ZS1: for (int j = 0; j < H; j++) {
#pragma HLS UNROLL factor=WCOL
            U1[j] = 0;
        K1: for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
                S1[k][j] = 0;
            }
        }
    }
STEPS: for (int t = 0; t < T_STEPS; t++) {
        if (compute) {
        ZU: for (int j = 0; j < H; j++) {
#pragma HLS UNROLL factor=WCOL
                U1[j] = 0;
            }
        }
    EVENTS: while (true) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=200
            const ev_t w = ev_read(e); e++;
            if (w == 0xFFFF) break;
            nev++;
            if (compute) {
                const int ch = (int)(w & 0x3FF); const cnt_t cnt = (cnt_t)((w >> 10) & 0x3F);
            COL1: for (int c = 0; c < NCHUNK; c++) {
#pragma HLS PIPELINE II=1
                    const int row = ch * NCHUNK + c;
                LANES1: for (int k = 0; k < WCOL; k++) {
#pragma HLS UNROLL
                        U1[c * WCOL + k] = (u1_t)(U1[c * WCOL + k] + mul<8, 8>(W1[row][k], (sint<8>)cnt));
                    }
                }
            }
        }
        if (compute) {
        DEND1: for (int r = 0; r < NROUND; r++) {
#pragma HLS PIPELINE II=1
            NEUR1: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                    const int j = r * P + l;
                    const u1_t u = (u1_t)(U1[j] + B1[j]);
                    sint<HW_WS + 4> sum = 0;
                POLE1: for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
                        const sint<HW_FA + 1 + HW_WS> pa = mul<HW_FA + 1, HW_WS>((sint<HW_FA + 1>)A1[k][j], S1[k][j]);
                        const sint<HW_CB + HW_WU1> pc = mul<HW_CB, HW_WU1>(C1[k][j], u);
                        const s_t s = (s_t)((s_t)(pa >> HW_FA) + shift_c<s_t, HW_CB + HW_WU1>(pc, HW_SHC1));
                        S1[k][j] = s; sum += s;
                    }
                    Z1[t][j] = sat16((sint<HW_WS + 8>)(sum >> (HW_FS - HW_FZ)) + H1[j]);
                }
            }
        }
    }
    n_events += nev;
    return e;
}

// ─────────────────────────────────────────── one sampled draw (VARIANT_SAMPLED) ───────────────────────────
static void seed_lanes(u32 idx, int stream, int draw) {
#pragma HLS INLINE off
SEED: for (int l = 0; l < P; l++) {
        u64 x = ((u64)idx << 32) | ((u64)stream << 24) | ((u64)draw << 8) | (u64)l;
    SW: for (int w = 0; w < 4; w++) {
            LANE[l][w] = (u32)(splitmix_step(x) & 0xFFFFFFFFULL);
        }
    }
}

static int argmax_first() {
#pragma HLS INLINE off
    int best = 0; l_t bv = LG[0];
AM: for (int c = 1; c < NC; c++) {
#pragma HLS PIPELINE II=1
        if (LG[c] > bv) { bv = LG[c]; best = c; }
    }
    return best;
}

#if CRISP_VARIANT == VARIANT_SAMPLED
static int draw_once(u32 idx, int stream, int draw, u32 &spk1, u32 &spk2) {
#pragma HLS INLINE off
    seed_lanes(idx, stream, draw);
ZS2: for (int j = 0; j < H; j++) {
#pragma HLS UNROLL factor=P
    K2: for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
            S2[k][j] = 0;
        }
    }
ZR: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
        RS[c] = 0; LG[c] = 0;
    }
    u32 n1 = 0, n2 = 0;
STEPS: for (int t = 0; t < T_STEPS; t++) {
        // layer-1 sampling from the prefix -> active list
    ZN1: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
            NL[l] = 0;
        }
    SAMP1: for (int r = 0; r < NROUND; r++) {
#pragma HLS PIPELINE II=1
        L1: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                const int j = r * P + l;
                const u32 o = xoshiro(LANE[l]);
                const int u = (int)(o >> (32 - HW_NB));
                const bool bit = Z1[t][j] > NOISE[l][u];
                if (bit) { ACTL[l][NL[l]] = (uint_<5>)r; NL[l]++; }
            }
        }
        OFF[0] = 0;
    OF1: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
            OFF[l + 1] = OFF[l] + NL[l];
        }
        const int na = OFF[P];
        n1 += na;
        // layer-2 synapses: column adds over active presynaptic neurons
    ZU2: for (int j = 0; j < H; j++) {
#pragma HLS UNROLL factor=WCOL
            U2[j] = 0;
        }
    COL2: for (int ec = 0; ec < na * NCHUNK; ec++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=2048
            const int e = ec / NCHUNK, c = ec % NCHUNK;
            int lane = 0;
        LSEL: for (int l = 1; l < P; l++) {
#pragma HLS UNROLL
                if (OFF[l] <= e) lane = l;
            }
            const int r = (int)ACTL[lane][e - OFF[lane]];
            const int row = (r * P + lane) * NCHUNK + c;
        LANES2: for (int k = 0; k < WCOL; k++) {
#pragma HLS UNROLL
                U2[c * WCOL + k] = (u2_t)(U2[c * WCOL + k] + W2[row][k]);
            }
        }
        // layer-2 dendrite + sampling -> active list 2
    ZN2: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
            NL2[l] = 0;
        }
    DEND2: for (int r = 0; r < NROUND; r++) {
#pragma HLS PIPELINE II=1
        NEUR2: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                const int j = r * P + l;
                const u2_t u = (u2_t)(U2[j] + B2[j]);
                sint<HW_WS + 4> sum = 0;
            POLE2: for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
                    const sint<HW_FA + 1 + HW_WS> pa = mul<HW_FA + 1, HW_WS>((sint<HW_FA + 1>)A2[k][j], S2[k][j]);
                    const sint<HW_CB + HW_WU2> pc = mul<HW_CB, HW_WU2>(C2[k][j], u);
                    const s_t s = (s_t)((s_t)(pa >> HW_FA) + shift_c<s_t, HW_CB + HW_WU2>(pc, HW_SHC2));
                    S2[k][j] = s; sum += s;
                }
                const z_t z = sat16((sint<HW_WS + 8>)(sum >> (HW_FS - HW_FZ)) + H2[j]);
                const u32 o = xoshiro(LANE[l]);
                const int ui = (int)(o >> (32 - HW_NB));
                const bool bit = z > NOISE[l][ui];
                if (bit) { ACTL2[l][NL2[l]] = (uint_<5>)r; NL2[l]++; }
            }
        }
        OFF2[0] = 0;
    OF2: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
            OFF2[l + 1] = OFF2[l] + NL2[l];
        }
        const int nb = OFF2[P];
        n2 += nb;
        // readout: column adds over active layer-2 neurons, then the per-class filter and accumulation
        r_t rr[NC];
#pragma HLS ARRAY_PARTITION variable=rr complete
    ZRR: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
            rr[c] = (r_t)BRO[c];
        }
    RO: for (int e = 0; e < nb; e++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=0 max=256
            int lane = 0;
        LSEL2: for (int l = 1; l < P; l++) {
#pragma HLS UNROLL
                if (OFF2[l] <= e) lane = l;
            }
            const int i = (int)ACTL2[lane][e - OFF2[lane]] * P + lane;
        ROC: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
                rr[c] = (r_t)(rr[c] + WRO[i][c]);
            }
        }
    RSU: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
            const sint<HW_FA + 1 + HW_WRS> pa = mul<HW_FA + 1, HW_WRS>((sint<HW_FA + 1>)ARO[c], RS[c]);
            const sint<HW_CB + HW_WR> pc = mul<HW_CB, HW_WR>(CRO[c], rr[c]);
            const rs_t v = (rs_t)((rs_t)(pa >> HW_FA) + shift_c<rs_t, HW_CB + HW_WR>(pc, HW_SHCRO));
            RS[c] = v; LG[c] = (l_t)(LG[c] + v);
        }
    }
    spk1 += n1; spk2 += n2;
    return argmax_first();
}
#endif

// ─────────────────────────────────────────── mean-field pass (VARIANT_MF) ────────────────────────────────
#if CRISP_VARIANT == VARIANT_MF
static inline cnt_t sig_lut(int lane, z_t z) {
#pragma HLS INLINE
    int idx = (int)(z >> (HW_FZ - 5)) + (1 << (HW_SB - 1));
    if (idx < 0) idx = 0;
    if (idx > (1 << HW_SB) - 1) idx = (1 << HW_SB) - 1;
    return SIG[lane][idx];
}
static int mf_pass() {
#pragma HLS INLINE off
    static cnt_t P2[H];
ZS2: for (int j = 0; j < H; j++) {
#pragma HLS UNROLL factor=P
    K2: for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
            S2[k][j] = 0;
        }
    }
ZR: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
        RS[c] = 0; LG[c] = 0;
    }
STEPS: for (int t = 0; t < T_STEPS; t++) {
    PR1: for (int r = 0; r < NROUND; r++) {
#pragma HLS PIPELINE II=1
        L1: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                P1[r * P + l] = sig_lut(l, Z1[t][r * P + l]);
            }
        }
    ZU2: for (int j = 0; j < H; j++) {
#pragma HLS UNROLL factor=WCOL
            U2MF[j] = 0;
        }
    DENSE2: for (int ic = 0; ic < H * NCHUNK; ic++) {
#pragma HLS PIPELINE II=1
            const int i = ic / NCHUNK, c = ic % NCHUNK;
            const cnt_t pi = P1[i];
        LANES2: for (int k = 0; k < WCOL; k++) {
#pragma HLS UNROLL
                U2MF[c * WCOL + k] = (u2mf_t)(U2MF[c * WCOL + k] + mul<8, 9>(W2[ic][k], (sint<9>)pi));
            }
        }
    DEND2: for (int r = 0; r < NROUND; r++) {
#pragma HLS PIPELINE II=1
        NEUR2: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                const int j = r * P + l;
                const u2mf_t u = (u2mf_t)(U2MF[j] + ((sint<40>)B2[j] << 8));
                sint<HW_WS + 4> sum = 0;
            POLE2: for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
                    const sint<HW_FA + 1 + HW_WS> pa = mul<HW_FA + 1, HW_WS>((sint<HW_FA + 1>)A2[k][j], S2[k][j]);
                    const sint<HW_CB + HW_WU2_MF> pc = mul<HW_CB, HW_WU2_MF>(C2[k][j], u);
                    const s_t s = (s_t)((s_t)(pa >> HW_FA) + shift_c<s_t, HW_CB + HW_WU2_MF>(pc, HW_SHC2 + 8));
                    S2[k][j] = s; sum += s;
                }
                const z_t z = sat16((sint<HW_WS + 8>)(sum >> (HW_FS - HW_FZ)) + H2[j]);
                P2[j] = sig_lut(l, z);
            }
        }
        rmf_t rr[NC];
#pragma HLS ARRAY_PARTITION variable=rr complete
    ZRR: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
            rr[c] = (rmf_t)((sint<40>)BRO[c] << 8);
        }
    ROD: for (int i = 0; i < H; i++) {
#pragma HLS PIPELINE II=1
            const cnt_t pi = P2[i];
        ROC: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
                rr[c] = (rmf_t)(rr[c] + mul<8, 9>(WRO[i][c], (sint<9>)pi));
            }
        }
    RSU: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
            const sint<HW_FA + 1 + HW_WRS> pa = mul<HW_FA + 1, HW_WRS>((sint<HW_FA + 1>)ARO[c], RS[c]);
            const sint<HW_CB + HW_WR_MF> pc = mul<HW_CB, HW_WR_MF>(CRO[c], rr[c]);
            const rs_t v = (rs_t)((rs_t)(pa >> HW_FA) + shift_c<rs_t, HW_CB + HW_WR_MF>(pc, HW_SHCRO + 8));
            RS[c] = v; LG[c] = (l_t)(LG[c] + v);
        }
    }
    return argmax_first();
}
#endif

// ─────────────────────────────────────────── certificate bookkeeping ─────────────────────────────────────
static void top2(const uint_<16> cnt[NC], int &cA, int &nA, int &nB) {
#pragma HLS INLINE off
    cA = 0; nA = 0; nB = 0;
T2: for (int c = 0; c < NC; c++) {
#pragma HLS PIPELINE II=1
        const int v = (int)cnt[c];
        if (v > nA) { nB = nA; nA = v; cA = c; }
        else if (v > nB) { nB = v; }
    }
}
static inline bool accepts(const uint_<16> thr[M_DRAWS + 1], int nA, int nB) {
#pragma HLS INLINE
    return nA >= (int)thr[nA + nB];
}
static inline bool is_look(int m) {
#pragma HLS INLINE
    const int looks[HW_N_LOOKS] = HW_LOOKS;
    bool f = false;
LK: for (int i = 0; i < HW_N_LOOKS; i++) {
#pragma HLS UNROLL
        if (looks[i] == m) f = true;
    }
    return f;
}

// processes one replay slot in `mode`; writes verify outputs when `verify`
struct Outcome { int n1, seq, used, fix, ref, mf; u32 draws; bool abstain; int decision; };
static Outcome process_slot(int slot, int mode, bool verify, unsigned int res[RES_WORDS], u32 &n_events, u32 &spk1, u32 &spk2) {
#pragma HLS INLINE off
    Outcome o; o.n1 = -1; o.seq = -1; o.used = 0; o.fix = -1; o.ref = -1; o.mf = -1; o.draws = 0; o.abstain = false; o.decision = -1;
    const u32 idx = (u32)SLOT_IDX[slot];
    prefix(SLOT_START[slot], mode != MODE_D0, n_events);
    if (mode == MODE_D0 || mode == MODE_PREFIX) return o;
#if CRISP_VARIANT == VARIANT_SAMPLED
    uint_<16> cnt[NC];
#pragma HLS ARRAY_PARTITION variable=cnt complete
ZC: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
        cnt[c] = 0;
    }
    const int n_draw = (mode == MODE_N1) ? 1 : M_DRAWS;
    bool decided = false;
CERT: for (int d = 0; d < n_draw; d++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=M_DRAWS
        const int v = draw_once(idx, 0, d, spk1, spk2);
        o.draws++;
        if (d == 0) o.n1 = v;
        cnt[v] = (uint_<16>)(cnt[v] + 1);
        if (verify) {
            const int w = R_VOTES_C + d / 4, sh = 8 * (d % 4);
            res[w] = (res[w] & ~(0xFFu << sh)) | ((unsigned int)v << sh);
        }
        if (mode != MODE_N1 && !decided && is_look(d + 1)) {
            int cA, nA, nB; top2(cnt, cA, nA, nB);
            if (accepts(THRS, nA, nB)) { decided = true; o.seq = cA; o.used = d + 1; }
            if (mode == MODE_SEQ && (decided || d + 1 == M_DRAWS)) {
                if (!decided) o.used = M_DRAWS;
                break;
            }
        }
    }
    if (mode != MODE_N1 && !decided) o.used = M_DRAWS;
    if (mode == MODE_FIX || verify) {
        int cA, nA, nB; top2(cnt, cA, nA, nB);
        if (accepts(THRF, nA, nB)) o.fix = cA;
    }
    if (verify) {
    WC: for (int c = 0; c < NC; c++) res[R_CNT_C + c] = (unsigned int)cnt[c];
        uint_<16> cr[NC];
#pragma HLS ARRAY_PARTITION variable=cr complete
    ZCR: for (int c = 0; c < NC; c++) {
#pragma HLS UNROLL
            cr[c] = 0;
        }
    REF: for (int d = 0; d < M_DRAWS; d++) {
            const int v = draw_once(idx, 1, d, spk1, spk2);
            cr[v] = (uint_<16>)(cr[v] + 1);
            const int w = R_VOTES_R + d / 4, sh = 8 * (d % 4);
            res[w] = (res[w] & ~(0xFFu << sh)) | ((unsigned int)v << sh);
        }
        int cA, nA, nB; top2(cr, cA, nA, nB);
        if (accepts(THRF, nA, nB)) o.ref = cA;
    WR: for (int c = 0; c < NC; c++) res[R_CNT_R + c] = (unsigned int)cr[c];
    }
    o.decision = (mode == MODE_N1) ? o.n1 : (mode == MODE_SEQ) ? o.seq : o.fix;
    o.abstain = (mode == MODE_SEQ || mode == MODE_FIX) && o.decision < 0;
#else
    o.mf = mf_pass(); o.decision = o.mf;
    if (verify) {
    WL: for (int c = 0; c < NC; c++) {
            const sint<64> v = (sint<64>)LG[c];
            res[R_MF_LOGIT + 2 * c] = (unsigned int)(v & 0xFFFFFFFF); res[R_MF_LOGIT + 2 * c + 1] = (unsigned int)((v >> 32) & 0xFFFFFFFF);
        }
    }
#endif
    return o;
}

// ─────────────────────────────────────────── top ──────────────────────────────────────────────────────────
void crisp_top(int cmd, int arg0, int arg1, int mode, int window[WIN_WORDS], unsigned int res[RES_WORDS]) {
#pragma HLS INTERFACE s_axilite port=cmd    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=arg0   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=arg1   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=mode   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=window bundle=ctrl
#pragma HLS INTERFACE s_axilite port=res    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return bundle=ctrl
    // all global array layouts are declared here (top scope), once
    #pragma HLS ARRAY_RESHAPE variable=W1 complete dim=2
    #pragma HLS BIND_STORAGE variable=W1 type=ram_1p impl=uram
    #pragma HLS ARRAY_PARTITION variable=U1 cyclic factor=WCOL
    #pragma HLS ARRAY_PARTITION variable=S1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=S1 cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=A1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A1 cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=C1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C1 cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=B1 cyclic factor=P
    #pragma HLS ARRAY_PARTITION variable=H1 cyclic factor=P
    #pragma HLS ARRAY_PARTITION variable=Z1 cyclic factor=P dim=2
    #pragma HLS ARRAY_RESHAPE variable=W2 complete dim=2
    #pragma HLS BIND_STORAGE variable=W2 type=ram_1p impl=uram
    #pragma HLS ARRAY_RESHAPE variable=WRO complete dim=2
#if CRISP_VARIANT == VARIANT_SAMPLED
    #pragma HLS ARRAY_PARTITION variable=U2 cyclic factor=WCOL
#endif
    #pragma HLS ARRAY_PARTITION variable=S2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=S2 cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=A2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=A2 cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=C2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C2 cyclic factor=P dim=2
    #pragma HLS ARRAY_PARTITION variable=B2 cyclic factor=P
    #pragma HLS ARRAY_PARTITION variable=H2 cyclic factor=P
#if CRISP_VARIANT == VARIANT_SAMPLED
    #pragma HLS ARRAY_PARTITION variable=NOISE complete dim=1
#endif
    #pragma HLS ARRAY_PARTITION variable=LANE complete dim=0
#if CRISP_VARIANT == VARIANT_SAMPLED
    #pragma HLS ARRAY_PARTITION variable=ACTL complete dim=1
    #pragma HLS ARRAY_PARTITION variable=ACTL2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=NL complete
    #pragma HLS ARRAY_PARTITION variable=NL2 complete
    #pragma HLS ARRAY_PARTITION variable=OFF complete
    #pragma HLS ARRAY_PARTITION variable=OFF2 complete
#endif
    #pragma HLS ARRAY_PARTITION variable=RS complete
    #pragma HLS ARRAY_PARTITION variable=LG complete
    #pragma HLS ARRAY_PARTITION variable=ARO complete
    #pragma HLS ARRAY_PARTITION variable=CRO complete
    #pragma HLS ARRAY_PARTITION variable=BRO complete
#if CRISP_VARIANT == VARIANT_MF
    #pragma HLS ARRAY_PARTITION variable=U2MF cyclic factor=WCOL
#endif
#if CRISP_VARIANT == VARIANT_MF
    #pragma HLS ARRAY_PARTITION variable=SIG complete dim=1
#endif
#if CRISP_VARIANT == VARIANT_MF
    #pragma HLS ARRAY_PARTITION variable=P1 cyclic factor=P
#endif
    #pragma HLS BIND_STORAGE variable=EV type=ram_1p impl=uram
    static int n_slots = 0;
    res[R_MAGIC] = RES_MAGIC; res[R_PACKAGE] = HW_PACKAGE_ID; res[R_VARIANT] = CRISP_VARIANT; res[R_BUILD] = CRISP_BUILD_ID;
    res[R_MODE] = (unsigned int)mode; res[R_N_SLOTS] = (unsigned int)n_slots;
    res[R_WIDTHS] = (unsigned int)((HW_WU1 << 24) | (HW_WS << 16) | (HW_WU2 << 8) | HW_WRS);
    if (cmd == CMD_LOAD_W) {
        const int base = arg0 * WIN_WORDS;
    LW: for (int j = 0; j < WIN_WORDS; j++) {
#pragma HLS PIPELINE II=2
            const int g = base + j; const uint32_t w = (uint32_t)window[j];          // plain integer extraction
            const w8_t b0 = (w8_t)(int8_t)(w & 0xFF), b1 = (w8_t)(int8_t)((w >> 8) & 0xFF),
                       b2 = (w8_t)(int8_t)((w >> 16) & 0xFF), b3 = (w8_t)(int8_t)((w >> 24) & 0xFF);
            const z_t h0 = (z_t)(int16_t)(w & 0xFFFF), h1 = (z_t)(int16_t)((w >> 16) & 0xFFFF);
            switch (arg1) {
            case RG_W1:  if (g < N_IN * H / 4) { const int e = 4 * g; W1[e / WCOL][e % WCOL] = b0; W1[(e + 1) / WCOL][(e + 1) % WCOL] = b1; W1[(e + 2) / WCOL][(e + 2) % WCOL] = b2; W1[(e + 3) / WCOL][(e + 3) % WCOL] = b3; } break;
            case RG_W2:  if (g < H * H / 4)    { const int e = 4 * g; W2[e / WCOL][e % WCOL] = b0; W2[(e + 1) / WCOL][(e + 1) % WCOL] = b1; W2[(e + 2) / WCOL][(e + 2) % WCOL] = b2; W2[(e + 3) / WCOL][(e + 3) % WCOL] = b3; } break;
            case RG_WRO: if (g < H * WRO_STRIDE / 4) {           // stride-32 packing: row = e >> 5, column = e & 31 (no divider)
                const int e = 4 * g, row = e >> 5, c0 = e & (WRO_STRIDE - 1);
                if (c0 < NC) WRO[row][c0] = b0; if (c0 + 1 < NC) WRO[row][c0 + 1] = b1; if (c0 + 2 < NC) WRO[row][c0 + 2] = b2; if (c0 + 3 < NC) WRO[row][c0 + 3] = b3; } break;
            case RG_B1:  if (g < H) B1[g] = (int32_t)w; break;
            case RG_B2:  if (g < H) B2[g] = (int32_t)w; break;
            case RG_BRO: if (g < NC) BRO[g] = (int32_t)w; break;
            case RG_D1A: if (g < H * K) A1[g % K][g / K] = (a_t)w; break;
            case RG_D1C: if (g < H * K) C1[g % K][g / K] = (c_t)(int32_t)w; break;
            case RG_D1H: if (g < H) H1[g] = (int32_t)w; break;
            case RG_D2A: if (g < H * K) A2[g % K][g / K] = (a_t)w; break;
            case RG_D2C: if (g < H * K) C2[g % K][g / K] = (c_t)(int32_t)w; break;
            case RG_D2H: if (g < H) H2[g] = (int32_t)w; break;
            case RG_ROA: if (g < NC) ARO[g] = (a_t)w; break;
            case RG_ROC: if (g < NC) CRO[g] = (c_t)(int32_t)w; break;
            case RG_NOISE: if (g < (1 << HW_NB) / 2) {
                NL: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                        NOISE[l][2 * g] = h0; NOISE[l][2 * g + 1] = h1;
                    }
                } break;
            case RG_SIG: if (g < (1 << HW_SB) / 4) {
                SL: for (int l = 0; l < P; l++) {
#pragma HLS UNROLL
                        SIG[l][4 * g] = (cnt_t)(w & 0xFF); SIG[l][4 * g + 1] = (cnt_t)((w >> 8) & 0xFF);
                        SIG[l][4 * g + 2] = (cnt_t)((w >> 16) & 0xFF); SIG[l][4 * g + 3] = (cnt_t)((w >> 24) & 0xFF);
                    }
                } break;
            case RG_THRS: if (2 * g < M_DRAWS + 1) { THRS[2 * g] = (uint_<16>)(w & 0xFFFF); if (2 * g + 1 < M_DRAWS + 1) THRS[2 * g + 1] = (uint_<16>)((w >> 16) & 0xFFFF); } break;
            case RG_THRF: if (2 * g < M_DRAWS + 1) { THRF[2 * g] = (uint_<16>)(w & 0xFFFF); if (2 * g + 1 < M_DRAWS + 1) THRF[2 * g + 1] = (uint_<16>)((w >> 16) & 0xFFFF); } break;
            default: break;
            }
        }
    } else if (cmd == CMD_LOAD_EV) {
        const u32 base = (u32)arg0 * WIN_WORDS;      // 32-bit words; word g holds events 2g (low) and 2g+1 (high)
    LE: for (int j = 0; j < WIN_WORDS; j += 2) {
#pragma HLS PIPELINE II=2
            const u32 g = base + j;                   // even -> events 2g..2g+3 = one 64-bit EV word
            if ((g >> 1) < EV_WORDS) EV[g >> 1] = ((u64)(u32)window[j + 1] << 32) | (u64)(u32)window[j];
        }
    } else if (cmd == CMD_LOAD_DIR) {
        n_slots = (arg0 > SLOT_MAX) ? SLOT_MAX : arg0;
    LD: for (int s = 0; s < SLOT_MAX; s++) {
#pragma HLS PIPELINE II=1
            if (s < n_slots) {
                SLOT_START[s] = (u32)window[2 * s];
                SLOT_IDX[s] = (uint_<16>)(((u32)window[2 * s + 1] >> 8) & 0xFFFF);
                SLOT_LBL[s] = (uint_<8>)((u32)window[2 * s + 1] & 0xFF);
            }
        }
        res[R_N_SLOTS] = (unsigned int)n_slots;
    } else if (cmd == CMD_READ_Z1) {
        const int base = arg0 * WIN_WORDS;            // word g holds Z1 elements 2g, 2g+1 (t-major: element = t*H + j)
    RZ: for (int j = 0; j < WIN_WORDS; j++) {
#pragma HLS PIPELINE II=1
            const int e = 2 * (base + j);
            unsigned int v = 0;
            if (e + 1 < T_STEPS * H) { const int z0 = (int)Z1[e / H][e % H], z1v = (int)Z1[(e + 1) / H][(e + 1) % H]; v = ((unsigned int)(uint16_t)z1v << 16) | (unsigned int)(uint16_t)z0; }
            window[j] = (int)v;
        }
    } else if (cmd == CMD_RUN || cmd == CMD_VERIFY) {
        u32 n_events = 0, spk1 = 0, spk2 = 0, draws = 0, items = 0, abst = 0, correct = 0;
        res[R_ITEMS] = 0; res[R_PASSES] = 0;
        if (cmd == CMD_VERIFY) {
            const int slot = (arg0 < n_slots) ? arg0 : 0;
        CLR: for (int w = R_VOTES_C; w < R_CNT_R + NC; w++) res[w] = 0;
            // z1 checksum (sum of int16 values as uint32)
            Outcome o = process_slot(slot, (CRISP_VARIANT == VARIANT_SAMPLED) ? MODE_FIX : MODE_MF, true, res, n_events, spk1, spk2);
            u32 chk = 0;
        CK: for (int t = 0; t < T_STEPS; t++) {
            CKJ: for (int j = 0; j < H; j++) {
#pragma HLS PIPELINE II=1
                    chk += (u32)(sint<32>)Z1[t][j];
                }
            }
            res[R_Z1_CHK] = (unsigned int)chk; res[R_N1] = (unsigned int)o.n1; res[R_SEQ] = (unsigned int)o.seq; res[R_SEQ_USED] = (unsigned int)o.used;
            res[R_FIX] = (unsigned int)o.fix; res[R_REF] = (unsigned int)o.ref; res[R_MF] = (unsigned int)o.mf; draws = o.draws; items = 1;
        } else {
        PASSES: for (int p = 0; p < arg0; p++) {
            SLOTS: for (int s = 0; s < n_slots; s++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=SLOT_MAX
                    Outcome o = process_slot(s, mode, false, res, n_events, spk1, spk2);
                    draws += o.draws; items++;
                    if (o.abstain) abst++;
                    if (o.decision >= 0 && o.decision == (int)SLOT_LBL[s]) correct++;
                    res[R_ITEMS] = items; res[R_DRAWS_LO] = draws;
                }
                res[R_PASSES] = (unsigned int)(p + 1);
            }
        }
        res[R_ITEMS] = items; res[R_DRAWS_LO] = draws; res[R_DRAWS_HI] = 0; res[R_EV_LO] = n_events; res[R_EV_HI] = 0;
        res[R_SPK1_LO] = spk1; res[R_SPK1_HI] = 0; res[R_SPK2_LO] = spk2; res[R_SPK2_HI] = 0; res[R_ABSTAIN] = abst; res[R_CORRECT] = correct;
    }
}
