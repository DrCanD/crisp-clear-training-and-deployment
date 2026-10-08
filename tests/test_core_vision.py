"""Small independent arithmetic and data-boundary checks, not experiment results."""
import itertools

import h5py
import numpy as np
import pytest
import torch

from reproduce import paths, training
from reproduce.data_preparation import bin_h5
from reproduce.models.crisp import Config, analytic_variance, build_model
from reproduce.vision import LinearConvStem128, VisArch, build_crisp, _certificate_votes
from reproduce.vision_preparation import integrate_to_frames


@pytest.fixture(autouse=True)
def small_cpu_workload():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_logit_variance_matches_enumerated_bernoulli_distribution():
    torch.manual_seed(15)
    model = build_model(Config(n_in=2, n_hidden=2, n_layers=1, k_modes=2,
                               n_classes=2, hidden_dropout=0., norm_type='none')).double().eval()
    inputs = torch.tensor([[[.2, .1], [.4, .7]]], dtype=torch.float64)
    with torch.no_grad():
        probability = model.layers[0](inputs, 'graded', 2.)[0].flatten()
        outcomes = torch.tensor(list(itertools.product((0., 1.), repeat=4)), dtype=torch.float64)
        mass = torch.where(outcomes.bool(), probability, 1 - probability).prod(1)
        logits = model.pool(model.ro(model.readout(outcomes.reshape(-1, 2, 2))))
        mean = (mass[:, None] * logits).sum(0)
        expected = (mass[:, None] * (logits - mean).square()).sum(0)
    observed = analytic_variance(model, [(inputs, torch.zeros(1, dtype=torch.long))],
                                 beta=2., subset=1, M=16)
    assert observed['n_points'] == 2
    np.testing.assert_allclose(observed['pred_sample'], expected.numpy(), rtol=1e-12, atol=1e-14)


def test_audio_binning_edges_empty_samples_and_count_saturation(tmp_path):
    filename = tmp_path / 'events.h5'
    with h5py.File(filename, 'w') as stream:
        spikes = stream.create_group('spikes')
        times = spikes.create_dataset('times', (2,), dtype=h5py.vlen_dtype(np.dtype('float64')))
        units = spikes.create_dataset('units', (2,), dtype=h5py.vlen_dtype(np.dtype('int64')))
        times[0] = np.concatenate([np.array([0., .249, .25, .5, 1.]), np.full(260, .75)])
        units[0] = np.concatenate([np.array([0, 1, 0, 1, 0]), np.ones(260, dtype=np.int64)])
        times[1], units[1] = np.array([], dtype=float), np.array([], dtype=np.int64)
        stream['labels'] = np.array([2, 1])
    inputs, labels, maximum = bin_h5(filename, 4, 2, max_time=1.)
    np.testing.assert_array_equal(inputs[0], [[1, 1], [1, 0], [0, 1], [1, 255]])
    assert inputs.dtype == np.uint8 and not inputs[1].any()
    assert maximum == 1. and labels.tolist() == [2, 1]


def test_gesture_time_and_count_bins_preserve_distinct_boundaries():
    events = {'t': np.array([0, 1, 2, 3, 4, 5, 7, 14]),
              'x': np.zeros(8, dtype=int), 'y': np.ones(8, dtype=int),
              'p': np.array([0, 1] * 4)}
    time_frames = integrate_to_frames(events, 'time', 2, 2, 2)
    count_frames = integrate_to_frames(events, 'number', 2, 2, 2)
    np.testing.assert_array_equal(time_frames[:, :, 1, 0], [[3, 3], [1, 1]])
    np.testing.assert_array_equal(count_frames[:, :, 1, 0], [[2, 2], [2, 2]])
    assert time_frames.sum() == count_frames.sum() == 8


def test_vision_stem_commutes_with_time_aggregation():
    torch.manual_seed(8)
    stem = LinearConvStem128(2, 3, 1, dropout=.2).double().eval()
    inputs = torch.randn(2, 4, 2, 16, 16, dtype=torch.float64)
    torch.testing.assert_close(stem(inputs).sum(1), stem(inputs.sum(1, keepdim=True))[:, 0],
                               rtol=1e-12, atol=1e-12)


def test_archived_empty_final_time_interval_behavior_is_explicit():
    # The archived parser repeats earlier events in this edge case. Preserve
    # reported-data identity; a corrected preprocessing study needs new data.
    events = {'t': np.array([0, 1, 2, 3, 4, 5, 6, 14]),
              'x': np.zeros(8, dtype=int), 'y': np.zeros(8, dtype=int),
              'p': np.array([0, 1] * 4)}
    frames = integrate_to_frames(events, 'time', 2, 1, 1)
    np.testing.assert_array_equal(frames[:, :, 0, 0], [[4, 3], [4, 4]])


def test_cached_gesture_draw_matches_full_sampled_forward():
    torch.manual_seed(23)
    model = build_crisp(VisArch(stem_ch=2, stem_out_hw=1, stem_dropout=0.,
                                n_hidden=4, n_layers=2, k_modes=2,
                                hidden_dropout=0., norm_type='none'), 'cpu').eval()
    inputs = torch.rand(2, 3, 2, 16, 16)
    with torch.no_grad():
        torch.manual_seed(29)
        expected_certificate = model(inputs, 'sample', 2.)[0].argmax(1)
        expected_reference = model(inputs, 'sample', 2.)[0].argmax(1)
    certificate, reference = _certificate_votes(model, [(inputs, None)], 2., 29,
                                                draws=1, chunk=1)
    np.testing.assert_array_equal(certificate[:, 0], expected_certificate.numpy())
    np.testing.assert_array_equal(reference[:, 0], expected_reference.numpy())


def test_training_selects_on_validation_and_resumes_completed_model(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, 'ROOT', tmp_path)
    torch.manual_seed(5)
    inputs = torch.rand(8, 3, 2)
    labels = torch.tensor([0, 1] * 4)
    # Opaque objects ensure training does not inspect test inputs or labels.
    data = ((inputs, labels), (inputs[:4], labels[:4]), (object(), object()))
    config = Config(n_in=2, n_hidden=3, n_layers=1, k_modes=2, n_classes=2,
                    hidden_dropout=0., norm_type='none', beta_start=1., beta_end=2.)
    recipe = training.TrainConfig(n_epochs=1, batch_size=4, device='cpu', validation_beta=5.,
                                  aug={'aug_jitter': 0, 'aug_chdrop': 0., 'aug_tmask': 0})
    validation_slopes = []
    evaluate = training.eval_mode

    def record_validation(model, loader, mode, beta, n_samples=1):
        validation_slopes.append(beta)
        return evaluate(model, loader, mode, beta, n_samples)

    monkeypatch.setattr(training, 'eval_mode', record_validation)
    info, model = training.train_one(7, data, 'meanfield', config, recipe, 'outputs/checkpoint')
    resumed_info, resumed = training.train_one(7, data, 'meanfield', config, recipe, 'outputs/checkpoint')
    assert validation_slopes == [5.] and info == resumed_info
    assert info['n_epochs_run'] == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name])
