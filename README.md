# CRISP and CLEAR: training and deployment

Reproduction code for **Sampler sharpening reduces the deployment gap in stochastic spiking networks**.

CRISP means *continuous-time reset-free independent-sampling perceptron*. CLEAR means *closed-form local eligibility and adjoint rule*. The experiments connect mean-field training to sampled inference, statistical decisions, and measured FPGA decision cost. They also compare training rules, sampling locations, time-step transfer, local learning, and runtime implementations.

## Install

Use Python 3.11–3.12 and Git. Run the installation commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,state-space]'
python -m reproduce fetch-state-space
```

This installs the CPU build for implementation checks and numerical analysis. For GPU training, install the corresponding PyTorch 2.11 and torchvision 0.26 CUDA builds using the [official installation instructions](https://pytorch.org/get-started/previous-versions/) before installing the package. Training configurations request CUDA; `--device cpu` overrides the device for a check or a small run. Complete training and long-sequence benchmarks are expensive on a CPU.

The P-SpikeSSM checkout is pinned to `8f7a954852ae29f4ef961bac341321f7f9330c51`; its commit and 25 source-file hashes are checked before use. External source is stored under `external_code/` and retains its upstream license.

All configuration paths are relative to this repository. After editable installation, commands resolve those paths from the package location, so the working directory does not change the selected input or output. New results and checkpoints are written under `outputs/`.

## Check the implementation

```bash
python -m reproduce test
python -m reproduce check-gradients
python -m reproduce analyse-reference
python hardware/verify_integer.py
```

The tests cover local derivatives, decision thresholds, recurrence arithmetic, operation counts, and numerical aggregation. `check-gradients` compares every parameter with autograd over 36 float64 settings. `analyse-reference` recalculates CSV summaries from stored numerical evidence and verifies source-table hashes. It does not run new training or hardware measurements. `verify_integer.py` checks genuine integer fixtures and recorded board decisions; see [hardware/README.md](hardware/README.md) for its exact scope.

## Prepare data

```bash
python -m reproduce prepare-data shd
python -m reproduce prepare-data ssc
```

Audio preparation downloads the official [SHD and SSC datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/), verifies source checksums, bins events into 100 time steps and 700 channels, and checks the registered array identity. SHD uses a stratified 10% validation split with seed 0; SSC uses its official validation split. SHD preprocessing has been rerun from the official HDF5 files and reproduces the archived array fingerprint. SSC preprocessing retains the original arithmetic and checks its archived fingerprint before publishing a cache.

For DVS128 Gesture, obtain and extract the official recordings under `data/raw/dvs_gesture/`, including `trials_to_train.txt` and `trials_to_test.txt`. Dataset source information is recorded in [data/manifests/datasets.json](data/manifests/datasets.json). Then run:

```bash
python -m reproduce prepare-data dvs_gesture --split-by time
python -m reproduce prepare-data dvs_gesture --split-by number
```

These commands prepare fixed-time and event-count frames, respectively. Both require the archived fingerprint. Raw datasets and prepared caches are generated locally and are excluded from Git.

The archived fixed-time integrator has an edge case: if its final interval is empty before the maximum timestamp, the final frame can include earlier events again. This behavior is retained for reproduction; gesture preprocessing has not been rerun on the full raw dataset.

## Run experiments

Each YAML file states its seeds, training recipe, model settings, checkpoint inputs, and output directory. Invoke it with:

```bash
python -m reproduce run configs/deployment_gap_shd.yaml
```

Run deployment-gap and sampler-sharpening training for each audio dataset before its checkpoint-based diagnostics and certification. Sampling variance uses checkpoints from both training workflows. Gesture certification uses checkpoints from `dvs_gesture.yaml`. The configurations use three seeds unless an explicitly named historical comparison used fewer. New runs preserve separate checkpoints for each model, training rule, and seed. Resume rejects a different protocol in the same output directory.

Moment diagnostics use the three CRISP mean-field checkpoints from deployment-gap training. Conditional-variance diagnostics use the baseline and strong-penalty CRISP/LIF checkpoints from the SHD sparsity sweep.

| Question | Configuration | Stored reference data |
|---|---|---|
| Architecture and training rule | `deployment_gap_shd.yaml`, `deployment_gap_ssc.yaml` | `architecture_training.csv` |
| Sampler sharpening | `sampler_sharpening_shd.yaml`, `sampler_sharpening_ssc.yaml` | `sampler_sharpening.csv` |
| Sampling location at fixed weights | `sampling_locations_shd.yaml`, `sampling_locations_ssc.yaml` | `sampling_locations.csv` |
| Matched fixed/annealed slope | `slope_schedule_shd.yaml` | `matched_slope_schedule.csv` |
| Local sampling variance | `sampling_variance_shd.yaml`, `sampling_variance_ssc.yaml` | `sampling_variance.csv` |
| Propagated moments and mean bias | `moment_diagnostics_shd.yaml`, `moment_diagnostics_ssc.yaml` | `moment_diagnostics_reported.csv` |
| Conditional variance with and without reset | `conditional_variance_shd.yaml` | Fresh per-seed diagnostic outputs |
| Coarse-to-fine time-step transfer | `time_step_transfer_shd.yaml`, `time_step_transfer_ssc.yaml` | `time_step_transfer.csv` |
| Sparsity and operation cost | `sparsity_shd.yaml` | `sparsity.csv` |
| CLEAR at a fixed epoch budget | `local_learning_shd.yaml`, `local_learning_ssc.yaml` | `local_learning.csv` |
| CLEAR/BPTT computation and peak memory | `local_learning_cost.yaml` | `local_learning_memory_reported.csv` |
| Earlier stopping comparison | `local_learning_shd_early_stopping.yaml`, `local_learning_ssc_early_stopping.yaml` | `local_learning.csv` |
| Gesture binning, transfer and decisions | `dvs_gesture.yaml`, `dvs_gesture_event_count.yaml` | `gesture_binning.csv` |
| Statistical decision audit | `certification_shd.yaml`, `certification_ssc.yaml`, `certification_dvs_gesture.yaml` | `certification_audit.csv` |
| Runtime and memory | `runtime_benchmark.yaml` | `runtime_reported.csv` |

Configuration names in the table are relative to `configs/`; reference tables are relative to `data/reference/`. Tables whose names end in `_reported.csv` contain archived aggregate observations. Their metadata distinguish those observations from per-seed results. Their underlying raw observations are not fabricated or inferred from reported means.

The matched slope control uses 100 epochs, validates both arms at slope 5, and fixes every validation-selected checkpoint before any test evaluation. Local-learning experiments calibrate batch normalization on training batches, freeze its statistics, and evaluate both the validation-selected model and the final-epoch model. CLEAR's within-layer derivative is exact given the incoming learning signal. Truncated and random cross-layer feedback are approximate; the reverse-time adjoint is an offline calculation.

The certification workflows use two-sided top-two binomial decisions, abstention, fixed budgets, and a prespecified sequential schedule. Reference votes use a separate draw stream. These are statistical statements about the sampled decision under the stated independent-draw model. They are not perturbation-robustness certificates. The FPGA pseudorandom generator is checked for implementation consistency; that check does not establish the independence assumption mathematically.

Recorded numerical draw seeds are retained for P-SpikeSSM and Gesture. The original CRISP audio stream-identity source was not recovered, so its fresh certificate runs use explicit deterministic stream identifiers. Those runs reproduce the statistical procedure rather than the historical bit sequence.

Moment diagnostics calculate variance, mean bias, tail frequencies and approximate stopping curves from newly sampled checkpoints. Gaussian and moment-based acceptance curves are diagnostics; they do not replace the vote-based statistical certificate. The conditional-variance comparison adds a noisy threshold to LIF for that diagnostic and retains its reset-induced correlations.

## Runtime benchmark

```bash
python -m reproduce fetch-spsn
python -m reproduce run configs/runtime_benchmark.yaml
```

The benchmark retains scan, chunked, FFT, Toeplitz, matrix, fused, compiled, and XLA variants where supported. Each implementation passes a forward/gradient gate before timing. Timing includes synchronization, warm-up, precision settings and device metadata. Unavailable implementations and projected resource skips are recorded separately from measurements. CUDA/Triton and TPU/XLA require their corresponding runtimes; CPU execution cannot reproduce GPU/TPU throughput or peak-memory observations.

## Integer inference and FPGA

[hardware/README.md](hardware/README.md) describes checkpoint quantization, integer replay, HLS synthesis, board verification, raw power acquisition, and power analysis. The seed-999 hardware checkpoint is included. Its tensor values are unchanged; its manifest records the original artifact hash and the repacked tensor identity. The 128-input integer fixture and all 2,264 recorded board decisions are included.

The sampled full-test raw power recording is included and can be reanalysed. The final 28-batch mean-field raw power recording was not recovered; the older 404-input recording is not substituted for it. A new synthesis or physical board measurement requires the appropriate AMD tools and KV260 hardware.

## Formal identities

Install [Lean via elan](https://lean-lang.org/install/) and run:

```bash
python -m reproduce check-proofs --setup
python -m reproduce check-proofs
```

The project pins Lean 4.19.0, mathlib, and its transitive dependencies. Verification kernel-checks 30 theorems, audits their axiom dependencies, and checks rejection of invalid or unfinished proofs. This formal scope is distinct from numerical gradient checks and empirical training results.

## Contents and evidence

| Path | Role |
|---|---|
| `src/reproduce/` | Data preparation, models, training, diagnostics and numerical analysis |
| `configs/` | Explicit experiment protocols |
| `data/reference/` | Numerical observations, integer fixtures and measurement records |
| `data/checkpoints/` | The checkpoint needed for the hardware reproduction |
| `data/manifests/` | Dataset identities and reference provenance |
| `hardware/` | Fixed-point implementation, synthesis and board workflows |
| `proofs/` | Lean source and verification runner |
| `tests/` | Implementation checks |
| `licenses/` | Required notices for incorporated third-party code |

A fresh training run is a new stochastic experiment; hardware and software differences can change its exact trajectory. Stored results remain unchanged when new experiments run. Full historical training was not repeated during repository preparation. The numerical code was checked against the original implementation, and the SHD preprocessing, local derivatives, decision rules, integer exports, and sampled power analysis were verified independently.

SHD and SSC are distributed under CC BY 4.0; the dataset manifest records their source and citation DOI. The archived DVS preprocessor attributes its parser and frame integrator to SpikingJelly, but records no upstream revision or license version; its historical license provenance is unverified. Separately, the pinned dependency `spikingjelly==0.0.0.0.14` ships the Open-Intelligence Open Source License V1.0, retained verbatim in [English](licenses/SpikingJelly-0.0.0.0.14-LICENSE.txt) and [Chinese](licenses/SpikingJelly-0.0.0.0.14-LICENSE-CN.txt). This dependency pin does not establish the archived parser's upstream revision. P-SpikeSSM and SPSN are fetched at pinned revisions and retain their own licenses.
