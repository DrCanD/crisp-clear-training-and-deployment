#ifndef CRISP_TOP_H
#define CRISP_TOP_H
#include "types.hpp"
// Single HLS IP, AXI-Lite controlled (bundle 'ctrl'); see types.hpp for the command / region / mode / result layout.
//   cmd      LOAD_W (window chunk arg0 -> table region arg1), LOAD_EV (window chunk arg0 -> packed event memory),
//            LOAD_DIR (window -> replay slot table: 2 words per slot: start event offset, idx<<8 | label; arg0 = n_slots),
//            RUN (arg0 passes over the n_slots replay inputs in `mode`), VERIFY (slot arg0: full outputs),
//            READ_Z1 (chunk arg0 of the prefix buffer -> window, 2 x int16 per word)
//   window   WIN_WORDS x 32-bit transfer window;  res  RES_WORDS x 32-bit results
void crisp_top(int cmd, int arg0, int arg1, int mode, int window[WIN_WORDS], unsigned int res[RES_WORDS]);
#endif
