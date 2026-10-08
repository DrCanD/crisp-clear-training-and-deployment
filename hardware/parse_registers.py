#!/usr/bin/env python3
"""HLS driver header (…/impl/ip/drivers/crisp_top_v1_0/src/xcrisp_top_hw.h) -> regmap_<variant>.json for crisp_ctl.py
   python3 parse_regmap.py <path/to/xcrisp_top_hw.h> [regmap.json]"""
import re, sys, json
src = open(sys.argv[1]).read(); out = sys.argv[2] if len(sys.argv) > 2 else 'regmap.json'
m = dict(re.findall(r'#define\s+XCRISP_TOP_CTRL_ADDR_(\w+)\s+(0x[0-9A-Fa-f]+)', src))
need = ['AP_CTRL', 'CMD_DATA', 'ARG0_DATA', 'ARG1_DATA', 'MODE_DATA', 'WINDOW_BASE', 'WINDOW_HIGH', 'RES_BASE', 'RES_HIGH']
missing = [k for k in need if k not in m]
if missing: sys.exit(f'missing in header: {missing}; found {sorted(m)}')
json.dump({k: m[k] for k in need}, open(out, 'w'), indent=1)
print({k: m[k] for k in need}); print('->', out)
