// verify_accelerator.cpp — bit-exact check of crisp_top against the package written by the integer exporter (integer_reference.py).
//   verify_accelerator <package_dir> [n_inputs]
// Loads the tables through the same LOAD_W / LOAD_EV / LOAD_DIR windows the board driver uses, runs VERIFY on every
// input and compares z1 (checksum + full read-back when expected_layer1_logits.bin exists), the vote streams, the sequential / fixed /
// reference decisions (sampled variant) or the mean-field decision and logits (MF variant); then one RUN pass per mode
// and checks the counters against the expected decisions. Exit 0 = PASS.
//
// Packing spec (shared with hardware/board_interface.py):
//   RG_W1 : bytes e = ch*H + j  -> W1_q[j][ch]        RG_W2 : e = i*H + j -> W2_q[j][i]      RG_WRO: e = j*32 + c -> Wro_q[c][j] (c < NC, rest 0)
//   4 bytes per 32-bit word (little-endian, element 4g in the low byte)
//   RG_B*, RG_D*H, RG_RO*: one int32/uint32 per word in file order;  RG_D*A/C: file order (j-major, k-minor)
//   RG_NOISE: int16 pairs (low half first);  RG_SIG: 4 bytes per word;  RG_THR*: uint16 pairs
//   events: per input, per time step t: words (ch | count<<10) with count <= 63 (larger counts split), then 0xFFFF;
//           2 events per 32-bit window word (low half first); slot table: start offset in events, idx<<8 | label
#include "accelerator.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <algorithm>

static int window_data[WIN_WORDS];
static unsigned int results_data[RES_WORDS];

template <class T> static std::vector<T> read_file(const std::string &p, bool required = true) {
    std::ifstream f(p.c_str(), std::ios::binary);
    if (!f) { if (required) { std::cerr << "cannot open " << p << std::endl; std::exit(2); } return std::vector<T>(); }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    if (n % sizeof(T)) { std::cerr << "size mismatch " << p << std::endl; std::exit(3); }
    std::vector<T> v(n / sizeof(T)); if (n) f.read((char *)v.data(), n); return v;
}
static void call(int cmd, int arg0, int arg1, int mode) { crisp_top(cmd, arg0, arg1, mode, window_data, results_data); }
static void load_words(int region, const std::vector<uint32_t> &w) {
    for (size_t c = 0; c * WIN_WORDS < w.size(); c++) {
        for (int j = 0; j < WIN_WORDS; j++) { size_t g = c * WIN_WORDS + j; window_data[j] = g < w.size() ? (int)w[g] : 0; }
        call(CMD_LOAD_W, (int)c, region, 0);
    }
}
static std::vector<uint32_t> pack_bytes(const std::vector<int8_t> &b) {
    std::vector<uint32_t> w((b.size() + 3) / 4, 0);
    for (size_t e = 0; e < b.size(); e++) w[e / 4] |= (uint32_t)(uint8_t)b[e] << (8 * (e % 4));
    return w;
}
static std::vector<uint32_t> pack_u8(const std::vector<uint8_t> &b) {
    std::vector<uint32_t> w((b.size() + 3) / 4, 0);
    for (size_t e = 0; e < b.size(); e++) w[e / 4] |= (uint32_t)b[e] << (8 * (e % 4));
    return w;
}
static std::vector<uint32_t> pack_i16(const std::vector<int16_t> &h) {
    std::vector<uint32_t> w((h.size() + 1) / 2, 0);
    for (size_t e = 0; e < h.size(); e++) w[e / 2] |= (uint32_t)(uint16_t)h[e] << (16 * (e % 2));
    return w;
}
static std::vector<uint32_t> pack_u16(const std::vector<uint16_t> &h) {
    std::vector<uint32_t> w((h.size() + 1) / 2, 0);
    for (size_t e = 0; e < h.size(); e++) w[e / 2] |= (uint32_t)h[e] << (16 * (e % 2));
    return w;
}
static std::vector<uint32_t> as_u32(const std::vector<int32_t> &v) { return std::vector<uint32_t>(v.begin(), v.end()); }

int main(int argc, char **argv) {
    if (argc < 2) { std::cerr << "verify_accelerator <package_dir> [n_inputs]" << std::endl; return 2; }
    const std::string dir = argv[1];
    int n_req = argc > 2 ? std::atoi(argv[2]) : -1;
    // ── tables ──
    auto W1 = read_file<int8_t>(dir + "/layer1_weights.bin"); auto W2 = read_file<int8_t>(dir + "/layer2_weights.bin"); auto WRO = read_file<int8_t>(dir + "/readout_weights.bin");
    if (W1.size() != (size_t)H * N_IN || W2.size() != (size_t)H * H || WRO.size() != (size_t)NC * H) { std::cerr << "weight sizes" << std::endl; return 3; }
    std::vector<int8_t> w1c((size_t)N_IN * H), w2c((size_t)H * H), wroc((size_t)H * WRO_STRIDE, 0);
    for (int j = 0; j < H; j++) for (int ch = 0; ch < N_IN; ch++) w1c[(size_t)ch * H + j] = W1[(size_t)j * N_IN + ch];
    for (int j = 0; j < H; j++) for (int i = 0; i < H; i++) w2c[(size_t)i * H + j] = W2[(size_t)j * H + i];
    for (int c = 0; c < NC; c++) for (int j = 0; j < H; j++) wroc[(size_t)j * WRO_STRIDE + c] = WRO[(size_t)c * H + j];   // stride-32 rows
    load_words(RG_W1, pack_bytes(w1c)); load_words(RG_W2, pack_bytes(w2c)); load_words(RG_WRO, pack_bytes(wroc));
    load_words(RG_B1, as_u32(read_file<int32_t>(dir + "/layer1_bias.bin"))); load_words(RG_B2, as_u32(read_file<int32_t>(dir + "/layer2_bias.bin")));
    load_words(RG_BRO, as_u32(read_file<int32_t>(dir + "/readout_bias.bin")));
    load_words(RG_D1A, read_file<uint32_t>(dir + "/layer1_decay.bin")); load_words(RG_D1C, as_u32(read_file<int32_t>(dir + "/layer1_input_coefficient.bin")));
    load_words(RG_D1H, as_u32(read_file<int32_t>(dir + "/layer1_offset.bin")));
    load_words(RG_D2A, read_file<uint32_t>(dir + "/layer2_decay.bin")); load_words(RG_D2C, as_u32(read_file<int32_t>(dir + "/layer2_input_coefficient.bin")));
    load_words(RG_D2H, as_u32(read_file<int32_t>(dir + "/layer2_offset.bin")));
    load_words(RG_ROA, read_file<uint32_t>(dir + "/readout_decay.bin")); load_words(RG_ROC, as_u32(read_file<int32_t>(dir + "/readout_input_coefficient.bin")));
    load_words(RG_NOISE, pack_i16(read_file<int16_t>(dir + "/logistic_noise.bin"))); load_words(RG_SIG, pack_u8(read_file<uint8_t>(dir + "/sigmoid_probability.bin")));
    load_words(RG_THRS, pack_u16(read_file<uint16_t>(dir + "/sequential_threshold.bin"))); load_words(RG_THRF, pack_u16(read_file<uint16_t>(dir + "/fixed_threshold.bin")));
    // ── inputs -> on-chip event stream ──
    auto ev = read_file<uint32_t>(dir + "/test_events.bin"); auto off = read_file<uint32_t>(dir + "/test_offsets.bin"); auto lbl = read_file<int16_t>(dir + "/test_labels.bin");
    int n_in = (int)lbl.size(); if (n_req > 0 && n_req < n_in) n_in = n_req; if (n_in > SLOT_MAX) n_in = SLOT_MAX;
    std::vector<uint16_t> stream; std::vector<uint32_t> starts;
    for (int i = 0; i < n_in; i++) {
        starts.push_back((uint32_t)stream.size());
        std::vector<uint32_t> e(ev.begin() + off[i], ev.begin() + off[i + 1]);
        std::stable_sort(e.begin(), e.end(), [](uint32_t a, uint32_t b) { return (a & 0x7F) < (b & 0x7F); });
        size_t k = 0;
        for (int t = 0; t < T_STEPS; t++) {
            while (k < e.size() && (int)(e[k] & 0x7F) == t) {
                int ch = (e[k] >> 7) & 0x3FF, c = (e[k] >> 17) & 0xFF;
                while (c > 0) { int cc = c > 63 ? 63 : c; stream.push_back((uint16_t)(ch | (cc << 10))); c -= cc; }
                k++;
            }
            stream.push_back(0xFFFF);
        }
        if (k != e.size()) { std::cerr << "events of input " << i << " not t-sorted within range" << std::endl; return 3; }
    }
    if (stream.size() > (size_t)EV_MAX) { std::cerr << "event memory too small: " << stream.size() << " > " << EV_MAX << std::endl; return 3; }
    {
        std::vector<uint32_t> w((stream.size() + 1) / 2, 0);
        for (size_t e = 0; e < stream.size(); e++) w[e / 2] |= (uint32_t)stream[e] << (16 * (e % 2));
        for (size_t c = 0; c * WIN_WORDS < w.size(); c++) {
            for (int j = 0; j < WIN_WORDS; j++) { size_t g = c * WIN_WORDS + j; window_data[j] = g < w.size() ? (int)w[g] : 0; }
            call(CMD_LOAD_EV, (int)c, 0, 0);
        }
        for (int s = 0; s < WIN_WORDS; s++) window_data[s] = 0;
        for (int i = 0; i < n_in; i++) { window_data[2 * i] = (int)starts[i]; window_data[2 * i + 1] = (int)(((uint32_t)i << 8) | (uint32_t)(uint8_t)lbl[i]); }
        call(CMD_LOAD_DIR, n_in, 0, 0);
    }
    std::cout << "[TB] package " << dir << ": " << n_in << " inputs, " << stream.size() << " event words, variant " << CRISP_VARIANT << std::endl;
    // ── expected ──
    auto exp_z1 = read_file<int16_t>(dir + "/expected_layer1_logits.bin", false);
    auto vc = read_file<int8_t>(dir + "/expected_certificate_votes.bin"); auto vr = read_file<int8_t>(dir + "/expected_reference_votes.bin");
    auto eseq = read_file<int16_t>(dir + "/expected_sequential_decisions.bin"); auto efix = read_file<int16_t>(dir + "/expected_fixed_decisions.bin"); auto eref = read_file<int16_t>(dir + "/expected_reference_decisions.bin");
    auto emf = read_file<int16_t>(dir + "/expected_mean_field_decisions.bin"); auto emfl = read_file<int64_t>(dir + "/expected_mean_field_logits.bin");
    int fails = 0;
    for (int i = 0; i < n_in; i++) {
        call(CMD_VERIFY, i, 0, 0);
        if (results_data[R_MAGIC] != RES_MAGIC || results_data[R_PACKAGE] != HW_PACKAGE_ID) { std::cerr << "bad magic/package" << std::endl; return 4; }
        std::string err;
        if (!exp_z1.empty()) {
            uint32_t chk = 0; for (int e = 0; e < T_STEPS * H; e++) chk += (uint32_t)(int32_t)exp_z1[(size_t)i * T_STEPS * H + e];
            if (results_data[R_Z1_CHK] != chk) err += " z1-checksum";
            int bad = -1;
            for (int c = 0; c * WIN_WORDS * 2 < T_STEPS * H && bad < 0; c++) {
                call(CMD_READ_Z1, c, 0, 0);
                for (int j = 0; j < WIN_WORDS && bad < 0; j++) {
                    for (int h = 0; h < 2; h++) {
                        int e = 2 * (c * WIN_WORDS + j) + h; if (e >= T_STEPS * H) break;
                        int16_t got = (int16_t)(((uint32_t)window_data[j] >> (16 * h)) & 0xFFFF);
                        if (got != exp_z1[(size_t)i * T_STEPS * H + e]) { bad = e; break; }
                    }
                }
            }
            if (bad >= 0) err += " z1[t=" + std::to_string(bad / H) + ",j=" + std::to_string(bad % H) + "]";
        }
#if CRISP_VARIANT == VARIANT_SAMPLED
        for (int d = 0; d < M_DRAWS; d++) {
            int gc = (results_data[R_VOTES_C + d / 4] >> (8 * (d % 4))) & 0xFF, gr = (results_data[R_VOTES_R + d / 4] >> (8 * (d % 4))) & 0xFF;
            if (gc != vc[(size_t)i * M_DRAWS + d]) { err += " vote_cert[" + std::to_string(d) + "]=" + std::to_string(gc) + "!=" + std::to_string((int)vc[(size_t)i * M_DRAWS + d]); break; }
            if (gr != vr[(size_t)i * M_DRAWS + d]) { err += " vote_ref[" + std::to_string(d) + "]"; break; }
        }
        if ((int)results_data[R_SEQ] != eseq[2 * i] || (int)results_data[R_SEQ_USED] != eseq[2 * i + 1]) err += " seq(" + std::to_string((int)results_data[R_SEQ]) + "," + std::to_string((int)results_data[R_SEQ_USED]) + ")!=(" + std::to_string(eseq[2 * i]) + "," + std::to_string(eseq[2 * i + 1]) + ")";
        if ((int)results_data[R_FIX] != efix[i]) err += " fix"; if ((int)results_data[R_REF] != eref[i]) err += " ref";
        if ((int)results_data[R_N1] != vc[(size_t)i * M_DRAWS]) err += " n1";
#else
        if ((int)results_data[R_MF] != emf[i]) err += " mf_dec(" + std::to_string((int)results_data[R_MF]) + "!=" + std::to_string(emf[i]) + ")";
        for (int c = 0; c < NC; c++) {
            int64_t got = (int64_t)(((uint64_t)results_data[R_MF_LOGIT + 2 * c + 1] << 32) | results_data[R_MF_LOGIT + 2 * c]);
            if (got != emfl[(size_t)i * NC + c]) { err += " mf_logit[" + std::to_string(c) + "]=" + std::to_string(got) + "!=" + std::to_string(emfl[(size_t)i * NC + c]); break; }
        }
#endif
        if (!err.empty()) { fails++; std::cout << "[FAIL] input " << i << ":" << err << std::endl; if (fails > 5) break; }
    }
    std::cout << "[TB] verify: " << (n_in - fails) << "/" << n_in << " inputs bit-exact" << std::endl;
    if (fails) return 4;
    // ── RUN passes: counters vs expected ──
#if CRISP_VARIANT == VARIANT_SAMPLED
    const int modes[] = {MODE_D0, MODE_PREFIX, MODE_N1, MODE_SEQ, MODE_FIX}; const char *names[] = {"D0", "PREFIX", "N1", "SEQ", "FIX"};
#else
    const int modes[] = {MODE_D0, MODE_PREFIX, MODE_MF}; const char *names[] = {"D0", "PREFIX", "MF"};
#endif
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        call(CMD_RUN, 1, 0, modes[m]);
        uint32_t items = results_data[R_ITEMS], draws = results_data[R_DRAWS_LO], abst = results_data[R_ABSTAIN], corr = results_data[R_CORRECT];
        uint32_t ex_corr = 0, ex_abst = 0, ex_draws = 0;
        for (int i = 0; i < n_in; i++) {
            int dec = -1;
            if (modes[m] == MODE_N1) { dec = vc[(size_t)i * M_DRAWS]; ex_draws += 1; }
            else if (modes[m] == MODE_SEQ) { dec = eseq[2 * i]; ex_draws += eseq[2 * i + 1]; if (dec < 0) ex_abst++; }
            else if (modes[m] == MODE_FIX) { dec = efix[i]; ex_draws += M_DRAWS; if (dec < 0) ex_abst++; }
            else if (modes[m] == MODE_MF) { dec = emf[i]; }
            if (dec >= 0 && dec == lbl[i]) ex_corr++;
        }
        bool ok = items == (uint32_t)n_in && draws == ex_draws && abst == ex_abst && corr == ex_corr;
        std::cout << "[RUN] " << names[m] << ": items " << items << " draws " << draws << " (exp " << ex_draws << ") abstain " << abst << " (exp " << ex_abst
                  << ") correct " << corr << " (exp " << ex_corr << ") events " << results_data[R_EV_LO] << " spikes " << results_data[R_SPK1_LO] << "/" << results_data[R_SPK2_LO]
                  << (ok ? "  OK" : "  MISMATCH") << std::endl;
        if (!ok) return 5;
    }
    std::cout << "PASS" << std::endl;
    return 0;
}
