// types.hpp — word-length abstraction (ap_int under Vitis HLS / -DUSE_AP_INT, plain C integers otherwise) and the
// AXI-Lite protocol of the CRISP KV260 measurement IP. The same source compiles with g++ for the bit-exact comparison
// against the Python oracle (integer_reference.py in the package); both builds agree as long as no ap_int<N> overflows, which the
// width checks and C simulation test on the supplied inputs. This is not a universal overflow guarantee.
#ifndef CRISP_TYPES_H
#define CRISP_TYPES_H
#include <cstdint>
#include "parameters.hpp"

#if defined(__VITIS_HLS__) || defined(USE_AP_INT)
  #include "ap_int.h"
  template <int N> using sint  = ap_int<N>;
  template <int N> using uint_ = ap_uint<N>;
  #define CRISP_AP_INT 1
#else
  #include <type_traits>
  template <int N> using sint  = typename std::conditional<(N <= 32), int32_t,  typename std::conditional<(N <= 64), int64_t,  __int128>::type>::type;
  template <int N> using uint_ = typename std::conditional<(N <= 32), uint32_t, typename std::conditional<(N <= 64), uint64_t, unsigned __int128>::type>::type;
  #define CRISP_AP_INT 0
#endif

#ifndef CRISP_VARIANT
#define CRISP_VARIANT 1          // 1 = sampled (spike-driven column adds + logistic-noise comparator), 2 = deterministic mean-field (8-bit p MACs + sigmoid LUT)
#endif
#define VARIANT_SAMPLED 1
#define VARIANT_MF 2

// ── geometry (parameters.hpp) ──
#define T_STEPS   HW_T
#define N_IN      HW_N_IN
#define H         HW_H
#define NC        HW_C
#define K         HW_K
#define P         HW_P            // RNG lanes = neurons per cycle in the dendrite loops
#define WCOL      32              // weights per cycle in a column add (256-bit memory words)
#define NCHUNK    (H / WCOL)      // chunks per weight column (8)
#define NROUND    (H / P)         // dendrite rounds per step (32)
#define M_DRAWS   HW_M

// ── widths ──
typedef sint<8>            w8_t;          // int8 weight
typedef uint_<8>           cnt_t;         // input count / 8-bit probability
typedef uint_<HW_FA>       a_t;           // pole decay, unsigned Q0.FA (a < 1)
typedef sint<HW_CB>        c_t;           // c coefficient (signed, HW_CB bits)
typedef sint<HW_WU1>       u1_t;          // layer-1 synaptic sum
typedef sint<HW_WU2>       u2_t;          // layer-2 synaptic sum (sampled)
typedef sint<HW_WU2_MF>    u2mf_t;        // layer-2 synaptic sum (mean-field, x256)
typedef sint<HW_WS>        s_t;           // dendritic state (FS fractional bits)
typedef sint<16>           z_t;           // saturated logit-unit potential (FZ fractional bits)
typedef sint<HW_WR>        r_t;           // readout synaptic sum (sampled)
typedef sint<HW_WR_MF>     rmf_t;         // readout synaptic sum (mean-field)
typedef sint<HW_WRS>       rs_t;          // readout filter state
typedef sint<HW_WL>        l_t;           // accumulated logit
typedef uint_<16>          ev_t;          // packed on-chip event (ch | count<<10), 0xFFFF = end of time step
typedef uint_<32>          u32;
typedef uint_<64>          u64;

// ── on-chip memories (sizes) ──
#define EV_WORDS   (1 << 17)     // 131072 x 64-bit event memory words (4 events each) in URAM (32 of 64) -> 524,288 events
#define EV_MAX     (EV_WORDS * 4)
#define SLOT_MAX   512           // replay slots (inputs) per load
#define WRO_STRIDE 32            // host packs the readout column of neuron j as 32 bytes (NC used, rest zero): no division by 20
#define WIN_WORDS  1024
#define RES_WORDS  1024

// ── commands ──
enum { CMD_NOP = 0, CMD_LOAD_W = 1, CMD_LOAD_EV = 2, CMD_LOAD_DIR = 3, CMD_RUN = 4, CMD_VERIFY = 5, CMD_READ_Z1 = 6 };
// LOAD_W regions (arg1); each window chunk (arg0) carries WIN_WORDS 32-bit words, little-endian packing of the package files
enum { RG_W1 = 0, RG_B1 = 1, RG_W2 = 2, RG_B2 = 3, RG_WRO = 4, RG_BRO = 5, RG_D1A = 6, RG_D1C = 7, RG_D1H = 8, RG_D2A = 9, RG_D2C = 10,
       RG_D2H = 11, RG_ROA = 12, RG_ROC = 13, RG_NOISE = 14, RG_SIG = 15, RG_THRS = 16, RG_THRF = 17 };
// modes (RUN / VERIFY)
enum { MODE_D0 = 0, MODE_PREFIX = 1, MODE_N1 = 2, MODE_SEQ = 3, MODE_FIX = 4, MODE_MF = 5 };

// ── result words ──
enum {
    R_MAGIC = 0, R_PACKAGE = 1, R_VARIANT = 2, R_BUILD = 3, R_MODE = 4, R_ITEMS = 5, R_PASSES = 6, R_DRAWS_LO = 7, R_DRAWS_HI = 8,
    R_EV_LO = 9, R_EV_HI = 10, R_SPK1_LO = 11, R_SPK1_HI = 12, R_SPK2_LO = 13, R_SPK2_HI = 14, R_ABSTAIN = 15, R_CORRECT = 16,
    R_Z1_CHK = 17, R_N1 = 18, R_SEQ = 19, R_SEQ_USED = 20, R_FIX = 21, R_REF = 22, R_MF = 23, R_N_SLOTS = 24, R_WIDTHS = 25,
    R_MF_LOGIT = 32,                 // 2 words (lo, hi) per class -> 32 .. 32 + 2*NC - 1
    R_VOTES_C = 128,                 // M_DRAWS votes, 4 per word (byte-packed, draw 0 in the low byte)
    R_VOTES_R = 128 + M_DRAWS / 4,
    R_CNT_C = 128 + M_DRAWS / 2,     // NC counts after the certificate draws
    R_CNT_R = 128 + M_DRAWS / 2 + NC
};
#define RES_MAGIC 0x43525331u        // 'CRS1'
#endif
