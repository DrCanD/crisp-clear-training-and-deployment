# CRISP and CLEAR: training and deployment

Code and numerical evidence for the manuscript **Sampler sharpening reduces the deployment gap in stochastic spiking networks**.
[![DOI](https://zenodo.org/badge/1410025622.svg)](https://doi.org/10.5281/zenodo.23239552)


CRISP means *continuous-time reset-free independent-sampling perceptron*. CLEAR means *closed-form local eligibility and adjoint rule*. The experiments connect mean-field training to sampled inference, statistical decisions, and measured FPGA decision cost. They also compare training rules, sampling locations, time-step transfer, local learning, and runtime implementations.

Start with the [CPU demo](#cpu-demo) to recalculate the stored numerical evidence without downloading a dataset. The [experiment map](#run-experiments) gives the configurations for fresh training and inference. [Hardware instructions](hardware/README.md) cover integer replay and physical measurements.

## Install

Use Python 3.11–3.12 and Git. Clone the repository and create an environment:

```bash
git clone https://github.com/DrCanD/crisp-clear-training-and-deployment.git
cd crisp-clear-training-and-deployment
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead. Install the CPU dependencies from the repository root:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

This installs the CPU build for implementation checks and numerical analysis. For GPU training, install the corresponding PyTorch 2.11 and torchvision 0.26 CUDA builds using the [official installation instructions](https://pytorch.org/get-started/previous-versions/) before installing the package. Training configurations request CUDA; `--device cpu` overrides the device for a check or a small run. Complete training and long-sequence benchmarks are expensive on a CPU.

For the P-SpikeSSM comparisons, also install and verify the optional baseline:

```bash
python -m pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,state-space]"
python -m reproduce fetch-state-space
```

The P-SpikeSSM checkout is pinned to `8f7a954852ae29f4ef961bac341321f7f9330c51`; its commit and 25 source-file hashes are checked before use. External source is stored under `external_code/` and retains its upstream license. The CPU demo and implementation tests do not require this checkout.

All configuration paths are relative to this repository. After editable installation, commands resolve those paths from the package location, so the working directory does not change the selected input or output. New results and checkpoints are written under `outputs/`.

This is a source-checkout workflow: retain the repository and use the editable install above. The configurations, reference data, hardware sources and proofs live alongside the Python package. A standalone wheel does not contain the full reproduction material.

## CPU demo

After installation, run:

```bash
python -m reproduce analyse-reference
```

This command needs no dataset download, GPU, FPGA or external model checkout. It verifies the hashes of 14 reference tables, writes numerical summaries to `outputs/reference_analysis/`, and prints the input/output manifest. Expected results include:

| Output | Expected result |
|---|---|
| `sampler_sharpening_paired_summary.csv` | P-SpikeSSM deployment-gap recovery: 20.524146 percentage points on SHD and 30.836032 on SSC; three seeds each |
| `certification_implementation_audit.csv` | 19,921,282 reference comparisons and 9 accepted-decision/reference disagreements |
| `analysis_metadata.json` | Verified source hashes, output paths and the scope of the reference analysis |

The disagreements are comparisons with a finite sampled reference, not a count of proven violations of the statistical guarantee. The demo recalculates archived evidence; use the experiment configurations below to generate fresh observations.

## Check the implementation

```bash
python -m reproduce test
python -m reproduce check-gradients
python -m reproduce analyse-reference
python hardware/verify_integer.py
```

The tests cover local derivatives, decision thresholds, recurrence arithmetic, operation counts, and numerical aggregation. `check-gradients` compares every parameter with autograd over 36 float64 settings. `analyse-reference` recalculates CSV summaries from stored numerical evidence and verifies source-table hashes. It does not run new training or hardware measurements. `verify_integer.py` checks genuine integer fixtures and recorded board decisions; see [hardware/README.md](hardware/README.md) for its exact scope.

The [CPU workflow](.github/workflows/reproduce.yml) runs on Python 3.11 and 3.12. It checks implementation tests, the pinned DVS source adapter, local gradients, stored numerical evidence, sampled and mean-field power analysis and one real integer input in each portable C++ variant. A separate job checks the Lean proofs. GPU training, vendor synthesis and physical measurements require their respective environments.

### Tested environment and timing

The 8 October 2026 audit used Linux x86_64 (kernel 6.18.44, glibc 2.39), Python 3.12.14, PyTorch 2.11.0 CPU, and an AMD EPYC 9V74 with nine logical CPUs visible. A clean environment passed dependency and implementation checks, including source provenance, cache compatibility and board failure gates. Full-test power checks also verify both raw recordings and their comparison. Python 3.11 is covered by the CPU workflow; Windows and macOS were not exercised in this audit.

| Check | Observed elapsed time |
|---|---:|
| PyTorch/torchvision installation from already-downloaded wheels | 27 s |
| Editable installation and remaining dependencies | 61 s |
| Reference-analysis demo | 0.07 s |
| 36 local-gradient checks | 2.75 s |
| Complete implementation test suite | 14.83 s |

Installation used a mixture of cached and newly downloaded dependencies; these times exclude the initial PyTorch download. The largest relative gradient error was 3.59 × 10⁻¹⁵ in float64. Full training duration and GPU memory depend on the configuration and device; the short CPU checks do not estimate those costs.

## Prepare data

SSC preparation uses disk-backed arrays, and the audio loader memory-maps modern Torch caches while retaining support for the older file format. The SSC uint8 inputs occupy about 6.9 GiB on disk. Preparing SSC from scratch used about 20.2 GB of peak disk space for compressed sources, expanded HDF5 files, temporary arrays and the final cache; allow at least 22 GB free, in addition to the software environment and experiment outputs. Training tensors and checkpoints need further resources. A small CPU check does not establish that a machine can run the complete training protocol.

```bash
python -m reproduce prepare-data shd
python -m reproduce prepare-data ssc
```

Audio preparation downloads the official [SHD and SSC datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/), verifies source checksums, bins events into 100 time steps and 700 channels, and checks the registered array identity. SHD uses a stratified 10% validation split with seed 0; SSC uses its official validation split. Both datasets were rebuilt from the official HDF5 sources and reproduce their archived array fingerprints. The [SSC verification record](data/manifests/ssc_preprocessing_verification.json) includes all source checksums, full-array SHA256 values and a successful reload through the public CLI. SSC binning and cache creation took 90.5 seconds in the audit environment, excluding download and decompression.

For DVS128 Gesture, obtain and extract the official recordings under `data/raw/dvs_gesture/`, including `trials_to_train.txt` and `trials_to_test.txt`. Dataset source information is recorded in [data/manifests/datasets.json](data/manifests/datasets.json). Then run:

```bash
python -m reproduce prepare-data dvs_gesture --split-by time
python -m reproduce prepare-data dvs_gesture --split-by number
```

These commands prepare fixed-time and event-count frames, respectively. Both require the archived fingerprint. Raw datasets and prepared caches are generated locally and are excluded from Git.

Both Gesture preprocessing modes were rerun from all 122 recordings in the checksummed official archive. The resulting training/validation/test splits reproduce the archived fingerprints: `3c87c257bb2d` for fixed-time frames and `56155d24937b` for event-count frames. The [verification record](data/manifests/dvs_gesture_verification.json) includes full-array SHA256 values, source and cache identities, and checks through the training data loader.

The archived fixed-time integrator has an edge case: if its final interval is empty before the maximum timestamp, the final frame can include earlier events again. This behavior is retained for reproduction and covered by a regression test. Full raw-data verification does not change the preprocessing used by the reported experiments.

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

The sampled and mean-field full-test raw power recordings are included. Each covers all 2,264 inputs in the same 28 disjoint batches. Run `python hardware/analyze_power.py` to recompute both variants and their energy comparison. Mean-field incremental energy is 563.94 microjoules per evaluated input, with a 95% interval of 560.66–567.22 microjoules; sequential sampled inference uses 4.617 times that energy. These quantities subtract bracketed idle power and the D0 replay/control cost. Source hashes and acquisition details are recorded in [the power measurement manifest](data/manifests/power_measurements.json). A new synthesis or physical board measurement requires the appropriate AMD tools and KV260 hardware.

## Formal identities

Install [Lean via elan](https://lean-lang.org/install/) and run:

```bash
python -m reproduce check-proofs --setup
python -m reproduce check-proofs
```

The project pins Lean 4.19.0, mathlib, and its transitive dependencies. Verification kernel-checks 30 theorems, audits their axiom dependencies, and checks rejection of invalid or unfinished proofs. This formal scope is distinct from numerical gradient checks and empirical training results.

A fresh Ubuntu CI run passed all 30 theorem checks both with dependency setup and on a repeat run. All three negative controls detected the intended invalid proof or disallowed axiom. The [verification record](data/manifests/proof_verification.json) contains the exact source hash, dependency revisions, axiom inventory and the originating CI run.

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

A fresh training run is a new stochastic experiment; hardware and software differences can change its exact trajectory. Stored results remain unchanged when new experiments run. Full historical training was not repeated during repository preparation. The numerical code was checked against the original implementation. Full SHD, SSC and both DVS preprocessing paths, local derivatives, decision rules, integer exports, sampled and mean-field power analysis and the Lean identities were verified within their documented scopes.

## Citation and questions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23239553.svg)](https://doi.org/10.5281/zenodo.23239553)

Version 1.0.0 is archived on Zenodo at [doi:10.5281/zenodo.23239553](https://doi.org/10.5281/zenodo.23239553), corresponding to GitHub tag [`v1.0.0`](https://github.com/DrCanD/crisp-clear-training-and-deployment/releases/tag/v1.0.0) and commit `eafc2ca666d869995eeab4f91f976a724604ccca`. Software citation metadata are provided in [CITATION.cff](CITATION.cff). Cite this version DOI and record the full commit used for each result with `git rev-parse HEAD`, together with the configuration and output metadata.

For a reproducibility question, open a [GitHub issue](https://github.com/DrCanD/crisp-clear-training-and-deployment/issues) with the commit, command, configuration, software/device versions and the relevant error output.

## License and third-party sources

Original project code and documentation are available under the [MIT License](LICENSE). Third-party code and datasets retain their respective terms; the MIT grant does not replace those terms. Component-specific sources, changes and exceptions are recorded in [third-party notices](licenses/THIRD_PARTY_NOTICES.md).

SHD, SSC and DVS128 Gesture are distributed under CC BY 4.0. The dataset manifest records their sources; the [original DVS dataset notice](licenses/dvs128-gesture-dataset.txt) is included.

The DVS parser and frame integration are adapted from SpikingJelly. Their [source manifest](data/manifests/dvs_preprocessing_source.json) pins a verified comparison release, records source and license hashes, and explains each local change. Source-comparison tests check this relationship against `spikingjelly==0.0.0.0.14`. Its Open-Intelligence Open Source License V1.0 is retained in [English](licenses/SpikingJelly-0.0.0.0.14-LICENSE.txt) and [Chinese](licenses/SpikingJelly-0.0.0.0.14-LICENSE-CN.txt). This verified comparison does not recover the original notebook's unrecorded upstream revision or relicense its upstream portions under MIT.

Optional external checkouts are fetched at pinned revisions and are outside the project MIT grant. P-SpikeSSM includes an MIT license. The pinned SPSN checkout contains no explicit reuse license; its source and checkpoints are fetched separately and are not redistributed here.
