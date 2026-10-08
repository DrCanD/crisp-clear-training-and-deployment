"""Independent arithmetic checks; these fixtures are not scientific results."""
import copy
import torch
from reproduce.models.lif import LIFLayer, install_zoh_hooks, build_model
from reproduce.diagnostics import counter_selftest, sampling_statistics
from reproduce.models.crisp import Config, build_model as build_crisp


def test_lif_hard_reset_recurrence():
    layer = LIFLayer(2, 2, 0., 'none', 5., False, .7, .3).eval()
    with torch.no_grad():
        layer.proj.weight.copy_(torch.eye(2))
        layer.proj.bias.zero_()
    inputs = torch.tensor([[[.5, .1], [.1, .3], [.4, .2], [.0, .0]]])
    alpha = layer.alpha()
    membrane = torch.zeros(1, 2)
    previous = torch.zeros(1, 2)
    expected = []
    for current in inputs.unbind(1):
        membrane = alpha * membrane * (1 - previous) + current
        previous = (membrane > layer.theta).float()
        expected.append(previous)
    observed, _ = layer(inputs)
    assert torch.equal(observed, torch.stack(expected, 1))


def test_zoh_gain_matches_held_input_coarse_step():
    model = build_model({'n_in': 2, 'n_hidden': 2, 'n_layers': 1,
                         'hidden_dropout': 0., 'norm_type': 'none'}, n_classes=2)
    factor = 4
    layer = model.layers[0]
    tau = torch.nn.functional.softplus(layer.raw_tau.detach()) + 1e-3
    coarse_alpha, fine_alpha = torch.exp(-1 / tau), torch.exp(-1 / factor / tau)
    gain = (1 - fine_alpha) / (1 - coarse_alpha)
    fine_state = torch.zeros(2)
    for _ in range(factor):
        fine_state = fine_alpha * fine_state + gain
    assert torch.allclose(fine_state, torch.ones(2), atol=2e-6)


def test_operation_accounting_has_independent_integer_totals():
    assert counter_selftest() == {'AC_event': 60, 'MAC_dense': 294, 'dend_MAC': 120}


def test_sampling_moments_are_weighted_by_unit_steps():
    torch.manual_seed(11)
    config = Config(n_in=2, n_hidden=3, n_layers=1, k_modes=2, n_classes=2,
                    hidden_dropout=0., norm_type='none')
    model = build_crisp(config, 'cpu').eval()
    x = torch.randn(3, 5, 2)
    y = torch.zeros(3, dtype=torch.long)
    result = sampling_statistics(model, [(x[:2], y[:2]), (x[2:], y[2:])], beta=2.)[0]
    with torch.no_grad():
        probability = model.layers[0](x, 'graded', 2.)[0].double()
    assert result['unit_steps'] == probability.numel()
    assert abs(result['mean_probability'] - float(probability.mean())) < 1e-7
    assert abs(result['mean_bernoulli_variance'] - float((probability * (1 - probability)).mean())) < 1e-7


def test_cached_first_layer_preserves_sampled_logits():
    from reproduce.diagnostics import SomaProbe, iter_sampled_logits
    torch.manual_seed(17)
    model = build_crisp(Config(n_in=2, n_hidden=4, n_layers=2, k_modes=2,
                               n_classes=3, hidden_dropout=0., norm_type='none'), 'cpu').eval()
    inputs = torch.rand(2, 5, 2)
    probe = SomaProbe(model)
    with torch.no_grad():
        model(inputs, 'graded', 3.)
    first_probability = probe.S[0].clamp(1e-6, 1 - 1e-6)
    probe.remove()
    with torch.no_grad():
        torch.manual_seed(23)
        expected = model(inputs, 'sample', 3.)[0]
        torch.manual_seed(23)
        actual = next(iter_sampled_logits(model, first_probability, 1, 1, 3.))[0][0]
    assert torch.equal(actual, expected)
