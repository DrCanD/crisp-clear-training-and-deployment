"""Compare optimized kernels with the archived recurrence and its gradients."""
import copy
import json

import pytest
import torch

from reproduce import benchmarks as benchmark


@pytest.fixture(autouse=True)
def small_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    benchmark.configure({'device': 'cpu', 'model': {'n_in': 8, 'n_hidden': 12, 'n_classes': 3,
                                                   'k_modes': 4},
                         'benchmark': {'gate_timesteps': 25, 'warmup': 1, 'repeats': 1}})
    yield
    benchmark.configure()
    torch.set_num_threads(threads)


@pytest.mark.parametrize('implementation', ['crisp_hillis', 'crisp_chunked', 'crisp_fft',
                                            'crisp_toeplitz', 'crisp_chunkmm'])
def test_optimized_network_forward_and_gradient(implementation):
    gate = benchmark.gate_crisp(implementation)
    assert gate['pass'], gate
    assert gate['fp64_max_rel_err'] < 1e-9


def test_fused_chunk_algebra_on_cpu():
    # Checks the mathematical fallback for the GPU kernel, including final
    # partial chunks; this does not claim that a Triton kernel ran on CPU.
    torch.manual_seed(91)
    reference = benchmark.DendRef(3, 8, zoh_b=True, ct_exact=True).double()
    candidate = copy.deepcopy(reference)
    candidate.__class__ = benchmark.FusedDendriticSSM
    reference.dt = candidate.dt = 0.5
    x = torch.randn(5, 77, 3, dtype=torch.float64, requires_grad=True)
    xc = x.detach().clone().requires_grad_(True)
    incoming = torch.randn_like(x)
    expected = reference(x)
    actual = candidate(xc)
    expected_gradients = torch.autograd.grad((expected * incoming).sum(),
                                            (x, reference.raw_lam, reference.mix))
    actual_gradients = torch.autograd.grad((actual * incoming).sum(),
                                          (xc, candidate.raw_lam, candidate.mix))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    for value, expected_value in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(value, expected_value, rtol=1e-10, atol=1e-12)


def test_lif_manual_adjoint_and_straight_through_math():
    gate = benchmark.gate_lif_math()
    assert gate['pass'], gate
    assert all(value == 0 for value in gate['fwd_rel_err'].values())


def test_optional_devices_are_reported_unavailable():
    benchmark.load_triton()
    available, reasons = benchmark.available_impls(['crisp_fused', 'lif_fused', 'lif_fused2',
                                                   'lif_xla_scan', 'lif_compiled'])
    assert available == []
    assert set(reasons) == {'crisp_fused', 'lif_fused', 'lif_fused2', 'lif_xla_scan', 'lif_compiled'}


def test_matched_timing_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'resolve_path', lambda path: tmp_path / path)
    config = {'device': 'cpu', 'output': 'runtime',
              'model': {'n_in': 8, 'n_hidden': 12, 'n_classes': 3, 'k_modes': 4},
              'benchmark': {'implementations': ['crisp_chunkmm', 'lif_loop'],
                            'timesteps': [9], 'batch_sizes': [2], 'gate_timesteps': 25,
                            'warmup': 1, 'repeats': 2}}
    result = benchmark.run(config)
    assert result['status'] == 'complete'
    assert len(result['units']) == 2
    for row in result['units'].values():
        assert row['status'] == 'ok'
        for mode in ('train', 'infer'):
            assert row[mode]['n_timed'] == 2
            assert row[mode]['median_ms'] > 0
            assert row[mode]['mem'] == {}  # No fabricated CPU/CUDA memory comparison.
    stored = json.loads((tmp_path / 'runtime' / 'benchmark.json').read_text())
    second = benchmark.run(config)
    assert second['units'] == stored['units']


def test_runtime_projections_are_not_measurements():
    done = {'small': {'impl': 'crisp_fft', 'T': 100, 'B': 64, 'status': 'ok',
                      'train': {'median_ms': 10.0, 'first_ms': 40.0}}}
    projected, first = benchmark.project({'name': 'crisp_fft', 'T': 200, 'B': 64}, done)
    assert projected == pytest.approx(0.020)
    assert first == pytest.approx(0.080)
    assert len(done) == 1


def test_archived_device_warmups_and_legacy_override():
    from reproduce.cli import read_config

    config = read_config('configs/runtime_benchmark.yaml')
    config['device'] = 'cpu'
    benchmark.configure(config)
    assert benchmark.cfg.n_warm_gpu == 3
    assert benchmark.cfg.n_warm_xla == 5
    assert benchmark.XLA_PREC_REQ == 'highest'

    tpu_config = read_config('configs/runtime_benchmark_tpu.yaml')
    assert tpu_config['model'] == config['model']
    assert tpu_config['output'] != config['output']
    tpu_config['device'] = 'cpu'
    benchmark.configure(tpu_config)
    assert benchmark.cfg.n_warm_xla == 5
    assert benchmark.XLA_PREC_REQ == 'highest'

    benchmark.configure({'device': 'cpu', 'benchmark': {'warmup': 2}})
    assert benchmark.cfg.n_warm_gpu == benchmark.cfg.n_warm_xla == 2

    benchmark.configure({'device': 'cpu', 'benchmark': {'warmup': 2, 'warmup_xla': 5}})
    assert benchmark.cfg.n_warm_gpu == 2
    assert benchmark.cfg.n_warm_xla == 5
