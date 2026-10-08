#!/usr/bin/env bash
# Install one already-built KV260 variant. Project paths are repository-relative.
set -euo pipefail
variant="${1:-sampled}"
build_relative="${2:-outputs/hardware/build}"
case "$variant" in sampled|mf) ;; *) echo 'Variant must be sampled or mf.'; exit 2;; esac
case "$build_relative" in /*|*..*|*:\\*) echo 'Build directory must be repository-relative.'; exit 2;; esac
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd -- "$script_directory/.." && pwd)"
cd "$repository_directory/$build_relative/$variant"
if [ "$(id -u)" -ne 0 ]; then echo 'Run with sudo on the KV260.'; exit 1; fi
command -v dtc >/dev/null || { echo 'device-tree-compiler (dtc) is required.'; exit 1; }
command -v xmutil >/dev/null || { echo 'The Kria Ubuntu xmutil loader is required.'; exit 1; }
python3 - "$variant" <<'PY'
import hashlib, json, sys
from pathlib import Path
variant = sys.argv[1]
record = json.loads(Path(f'build_receipt_{variant}.json').read_text())
if not record['completed'] or 'TIMING MET' not in record['timing_status']:
    raise RuntimeError('Build receipt is incomplete or timing did not close')
for filename, field in [(f'firmware_{variant}/crisp_{variant}.bit.bin', 'bitstream_sha256'),
                        (f'registers_{variant}.json', 'regmap_sha256'),
                        (f'firmware_{variant}/pl.dtsi', 'overlay_source_sha256')]:
    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != record[field]:
        raise RuntimeError(filename + ' hash mismatch')
print('Build receipt and firmware hashes match:', variant, record['package'])
PY
dtc -@ -O dtb -o "firmware_$variant/crisp_$variant.dtbo" "firmware_$variant/pl.dtsi"
system_root="$(python3 -c 'import os; print(os.path.abspath(os.sep))')"
firmware_target="${system_root%/}/lib/firmware/xilinx/crisp_$variant"
debug_directory="${system_root%/}/sys/kernel/debug"
mkdir -p "$firmware_target"
cp "firmware_$variant/crisp_$variant.bit.bin" "firmware_$variant/crisp_$variant.dtbo" "firmware_$variant/shell.json" "$firmware_target/"
if ! xmutil unloadapp; then echo 'No active application was unloaded; continuing to the requested application.'; fi
xmutil loadapp "crisp_$variant"
mountpoint -q "$debug_directory" || mount -t debugfs none "$debug_directory"
python3 - "$variant" <<'PY'
import json, os, sys
from pathlib import Path
variant = sys.argv[1]
record = json.loads(Path(f'build_receipt_{variant}.json').read_text())
root = Path(os.sep)
actual = int((root / 'sys/kernel/debug/clk/pl0_ref/clk_rate').read_text())
if not 0 < actual <= record['clock_request_hz'] * 1.001 + 2000:
    raise RuntimeError('Actual clock exceeds the timing-qualified clock')
if (root / 'sys/class/fpga_manager/fpga0/state').read_text().strip() != 'operating':
    raise RuntimeError('FPGA manager is not operating')
Path(f'firmware_{variant}/pl_clk_actual_hz.txt').write_text(str(actual) + '\n')
print(f'Loaded crisp_{variant}; PL clock {actual} Hz')
PY
