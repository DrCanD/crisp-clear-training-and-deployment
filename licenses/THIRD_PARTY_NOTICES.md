# Third-party sources

The root MIT license covers original project code and documentation. It does
not replace licenses for third-party code, datasets, dependencies or tools.

## DVS128 Gesture preprocessing

The event parser and frame-integration routines in
[`src/reproduce/vision_preparation.py`](../src/reproduce/vision_preparation.py)
are adapted from SpikingJelly's `spikingjelly/datasets/__init__.py`.
SpikingJelly identifies its developers as PKU MLG, PCL and other contributors.
Its parser also credits [iniVation dv-python](https://gitlab.com/inivation/dv/dv-python)
as a reference; that attribution is retained here.

The distributed adapter was checked against the official PyPI release
[`spikingjelly==0.0.0.0.14`](https://pypi.org/project/spikingjelly/0.0.0.0.14/).
The wheel, source file and license texts were verified by SHA256. The
[source manifest](../data/manifests/dvs_preprocessing_source.json) records
the immutable distribution URL, hashes, function mapping and local changes.
The archived notebook did not record its original upstream revision. This
verification establishes a specific compatibility reference for the adapter;
it does not reconstruct that missing historical information.

The upstream portions retain the **Open-Intelligence Open Source License
V1.0**, provided verbatim in [English](SpikingJelly-0.0.0.0.14-LICENSE.txt)
and [Chinese](SpikingJelly-0.0.0.0.14-LICENSE-CN.txt). In particular, this
component is not covered solely by the root MIT license. The upstream
license includes commercial-use conditions in Section V. Original project
modifications are covered by MIT without removing the upstream conditions.

The local changes include vectorized little-endian packet decoding,
float32 frame counts, guards for empty intervals, coordinate downsampling
and dataset assembly. Archived time-bin behavior is retained, including the
empty-final-interval case. Source-comparison tests use the exact hashed
upstream functions on supported packets and nonempty boundaries; separate
tests document the archived edge case.

## Optional baseline checkouts

- **P-SpikeSSM** is fetched separately at commit
  `8f7a954852ae29f4ef961bac341321f7f9330c51`. Its `LICENSE` is MIT,
  copyright 2025 Neuromorphic Computing Lab - PSU. Its checkout retains
  that file and any notices of incorporated components.
- **SPSN** is fetched separately from
  [NECOTIS/Stochastic-Parallelizable-Spiking-Neuron-SPSN](https://github.com/NECOTIS/Stochastic-Parallelizable-Spiking-Neuron-SPSN)
  at commit `c46d16931b1736208ced87974587e0d4c9a76634`. That pinned
  checkout contains no explicit license file. This repository does not
  redistribute its source or checkpoints or grant rights to them.

Both checkouts live in the ignored `external_code/` directory. Their
availability for optional comparisons does not make them part of the
project's MIT grant. Installed Python packages, Lean/mathlib and FPGA
vendor tools retain their own terms.

## Datasets

Dataset use is separate from the code license. SHD and SSC use CC BY 4.0;
their attribution and source records are in the
[dataset manifest](../data/manifests/datasets.json). DVS128 Gesture also uses CC BY 4.0. Its
[provider notice](dvs128-gesture-dataset.txt) requires attribution to
IBM Research; the dataset manifest records the verified source and hashes.
The project MIT license grants no additional rights to those datasets.
