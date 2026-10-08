# Integer inference and KV260 measurements

This directory reproduces the fixed-point CRISP model used for the SHD hardware
experiment: seed 999, two layers of 256 neurons, eight poles, 100 time bins, and
256 samples per certificate stream. The recorded hardware package identifier is
`344b5f19`. This identifier is part of the accelerator interface, not a folder name.

## What is included

| File or directory | Purpose |
|---|---|
| `build.py`, `hls/`, `tcl/` | Portable C++ simulation, HLS synthesis and Vivado implementation |
| `install_firmware.sh` | Install a timing-qualified build on the KV260 |
| `analyze_power.py` | Recompute full-test sampled and mean-field energy and processing time from raw sensor records |
| `integer_reference.py` | Fixed-point arithmetic, lookup tables, pseudorandom streams and certificate rules |
| `float_model.py` | Floating-point network required to load and fold the hardware checkpoint |
| `quantize.py` | Validation-only precision selection and frozen-format test evaluation |
| `verify_integer.py` | CPU inference checks and comparison with the original board records |
| `prepare_full_test.py` | Rebuild the original full-test event package from the prepared SHD cache |
| `board_interface.py` | AXI register access and exact table/event packing |
| `run_board.py` | Board verification and INA260 U14 power acquisition |
| `registers_sampled.json`, `registers_mf.json` | Register offsets whose hashes match the original build receipts |
| `../data/checkpoints/hardware/` | Seed-999 checkpoint with unchanged tensor values and a provenance manifest |
| `../data/reference/hardware/fixture/` | 128 real test inputs, integer tables and expected integer outputs |
| `../data/reference/hardware/full_test/` | Expected outputs for all 2,264 test inputs and input-event hashes |
| `../data/reference/hardware/board_verification_*.json` | Original per-input board verification records |
| `../data/reference/hardware/build_receipt_*.json` | Original numerical build identities with local machine paths removed |
| `../data/reference/hardware/power_sampled.json` | Original 28-batch sampled power measurement, including raw sensor samples |
| `../data/reference/hardware/power_mean_field.json` | Original 28-batch mean-field power measurement, including raw sensor samples |
| `../data/manifests/power_measurements.json` | Source hashes, acquisition settings and the scope of both measurements |

All command arguments naming files or directories are relative to the repository
root. Scripts find that root from their own location. Generated files are written
to `outputs/hardware/`. Linux device interfaces are resolved from the operating
system root; no user-specific path is required.

## CPU verification

From the repository root:

```bash
python hardware/verify_integer.py
```

The default run checks four inputs, both 256-draw streams, every integer
mean-field logit, the deterministic first-layer prefix, the complete 128-input
fixture certificate outputs, and all 2,264 recorded board decisions. It writes
`outputs/hardware/integer_verification.json`.

To rerun sampled inference on all fixture inputs:

```bash
python hardware/verify_integer.py --inputs 128 --draws 256
```

The full-test record check recomputes certificates from stored vote streams. It
is separate from new full-test inference and from a new physical board run.

## Rebuild the full input package

Prepare the SHD cache as described in the main README. It must contain
`Xtr`, `ytr`, `Xte` and `yte`; the test input shape is `(2264, 100, 700)` and
contains integer counts in `[0, 255]`.

```bash
python hardware/prepare_full_test.py --dataset data/processed/shd.npz
```

The command regenerates event words and checks their archived SHA-256 identity
before writing `outputs/hardware/full_test/`. A different time-binning rule or
input order causes a failure. No checkpoint or format is selected in this step.
The included 128-input fixture remains usable without the full SHD download.

## Validation-only quantization

The original format selection used the test set. The subsequent validation-only
reevaluation does not turn that historical test set into an untouched test set.
The code preserves this distinction and freezes precision before reading test
arrays. The checkpoint is fixed throughout.

```bash
python hardware/quantize.py --stage calibrate --dataset data/processed/shd.npz --device cuda
```

Calibration prints the repository-relative path of `format_freeze.json`. Pass
that exact path to evaluation:

```bash
python hardware/quantize.py --stage evaluate --freeze outputs/hardware/quantization/validation_HASH/format_freeze.json --device cuda
```

`HASH` denotes the directory printed by calibration. Source identities, the
checkpoint identity, validation indices and integer tables are checked before
the test arrays are loaded. A width violation stops evaluation; test results do
not widen the format. CPU execution is supported with `--device cpu`, but the
complete 256-draw evaluation is expensive.

The distributed checkpoint was repacked to remove private run labels. Its 24
state tensors are unchanged. `manifest.json` records both checkpoint file hashes
and a deterministic tensor-state hash. Re-exporting the checkpoint reproduces
every integer table in the original fixture exactly. New calibration freezes
refer to the current source and checkpoint hashes; older freezes are not silently
relabelled as executions of this cleaned source.

## Accelerator build

The recovered synthesis-source hash matches both original measured build receipts
exactly. `data/reference/hardware/source_identity.json` records this identity and
the file-name/comment cleanup. The design sources are included in `hls/`, with a sampled and a
mean-field build selected by `CRISP_VARIANT`. Both retain the original arithmetic:
floor shifts, an eight-lane xoshiro128** sampler, a 4,096-entry logistic-noise
table and a separate 2,048-entry mean-field sigmoid table scaled by 1/256.

The same source has a portable C++ mode. Check both variants before using the
vendor tools:

```bash
python hardware/build.py --variant sampled --stage verify
python hardware/build.py --variant mf --stage verify
```

Portable verification checks the prefix, sampled streams or mean-field logits,
and replay counters against the fixture. Plain C++ arithmetic does not simulate
all reduced-width `ap_int` effects. The subsequent vendor C simulation is a
separate required gate before synthesis and packaging.

Open an AMD Vitis HLS/Vivado 2025.2 environment with the KV260 board definitions
and `bootgen` on PATH, then run:

```bash
python hardware/build.py --variant sampled --stage all
python hardware/build.py --variant mf --stage all
```

The build copies sources into `outputs/hardware/build/`, verifies the fixture,
runs HLS C simulation, synthesis, IP packaging, placement/routing and timing
checks, and produces firmware plus a content-hashed build receipt. `--cosim`
adds RTL co-simulation. Individual stages can be resumed with `--stage`; their
prerequisites and source identities are checked. Changed inputs require a new
relative `--output` directory, so an earlier build is preserved.

The default fixture check uses eight inputs. `--inputs` is part of the build
identity and must remain the same when resuming stages. Vivado/Vitis execution
and a new physical-board run were not performed while preparing this repository.
Portable C++ checks were run for four inputs on each variant.

## Board verification and power acquisition

Copy the repository and its generated build outputs to the KV260. Install one
variant at a time and verify it:

```bash
sudo bash hardware/install_firmware.sh sampled
sudo python3 hardware/run_board.py --variant sampled --verify --package outputs/hardware/full_test --out outputs/hardware/board_sampled_verification
sudo bash hardware/install_firmware.sh mf
sudo python3 hardware/run_board.py --variant mf --verify --package outputs/hardware/full_test --out outputs/hardware/board_mean_field_verification
```

The board runner requires Python and NumPy; PyTorch is not needed on the board.
The installer verifies timing and firmware hashes before loading the overlay.
The runner checks the input package's SHA-256 manifest before loading any tables.
Before any AXI access, the runner checks the loaded firmware name, supplied
bitstream, register map, overlay and measured PL clock against that build's
receipt. Original build identities and per-input board records are retained
separately under `data/reference/hardware/`. New builds have their own receipts.
A verification mismatch returns a nonzero process status and preserves the
per-input record, failure description and result archive.

Power acquisition uses 30-second blocks, 5-second warm-up, 10 Hz sensor sampling,
28 input batches and randomized mode order. Inputs are assigned to batches once;
each loaded batch is replayed repeatedly during its timed block.

```bash
sudo bash hardware/install_firmware.sh sampled
sudo python3 hardware/run_board.py --variant sampled --power --package outputs/hardware/full_test --repeats 28 --block 30 --out outputs/hardware/power_sampled
sudo bash hardware/install_firmware.sh mf
sudo python3 hardware/run_board.py --variant mf --power --package outputs/hardware/full_test --repeats 28 --block 30 --out outputs/hardware/power_mean_field
```

The raw file records power, item counters, input indices, temperatures and idle
brackets. Energy is incremental SOM input energy per evaluated input, including
abstentions. Inverse throughput is mean processing time per evaluated input, not
single-request latency. The runner's `analysis.json` is a diagnostic repeat
summary. The full analysis below uses input weighting and batch residuals.

## Recompute the recorded power results

```bash
python hardware/analyze_power.py
```

This command checks both raw recordings and disjoint coverage of all 2,264
inputs in each variant, then produces `outputs/hardware/power_analysis/summary.json`
and `batches.csv`. It reproduces sampled and mean-field energy, inverse throughput,
standard errors, 86.1% sequential-versus-fixed saving, the sequential-to-mean-field
energy ratio, and the 112-observation sampled energy/time fits. The summary's
`modes_by_variant` contains all eight variant/mode combinations, including
idle-adjusted D0 control energies before D0 subtraction. The existing `modes`
keys retain the five main results for compatibility. Both files distinguish the
two recordings' PREFIX and D0 rows. No plotted values are used as input.

| Mean-field result | Recomputed value |
|---|---:|
| Evaluated inputs / disjoint batches | 2,264 / 28 |
| Incremental energy per input | 563.9394 microjoules |
| Energy standard error | 1.6000 microjoules |
| Approximate energy 95% interval | 560.6505–567.2283 microjoules |
| Processing time per input (inverse throughput) | 3.3089 ms |
| Sequential sampled / mean-field energy | 4.6172 |

Energy means are raw batch estimates weighted by input count. With normalized
weights `w_b`, the standard error is `sqrt(s2 * sum(w_b^2))`, where `s2` is the
residual variance of an ordinary least-squares model in events per input and,
when it varies, draw count. This estimates uncertainty of the reported raw mean;
the covariance of a fitted mean estimates a different quantity. The calculation
assumes independent, equal-variance batch measurement errors conditional on
workload. Approximate Student-t 95% intervals use 25 or 26 residual degrees of
freedom. The summary includes the residual variance and squared-weight sum.
The across-mode draw-cost fits are unweighted ordinary least squares. Sensor
calibration uncertainty and training-seed variation are not estimated by these
intervals.

Both recordings were acquired on 5 October 2026 in separate consecutive sessions,
using the same input package and identical input groups. Their source hashes,
firmware identities and normalization are recorded in
[the measurement manifest](../data/manifests/power_measurements.json). The mean-field
record contains 58,800 sensor samples across 84 active and 112 idle windows. Sensor
and CPU-governor paths are stored as device-relative labels; all measurement values
are unchanged.

The board's original `analysis.json` gives equal-weight batch diagnostics. The
results above weight each batch by its input count and use the residual model
described above, so the two summaries need not have identical means or intervals.
Throughput uses item-counter increments during the power-sampling window;
whole-run item counts divided by elapsed time must not replace that rate.

Use `--sampled` or `--mean-field` with repository-relative paths to analyze new
full-test recordings. Pass `--mean-field ""` for sampled-only analysis. New
physical measurements still require the corresponding firmware and KV260 board.
