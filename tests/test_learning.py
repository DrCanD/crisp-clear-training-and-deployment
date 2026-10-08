"""Numerical checks against autograd and the original network forward."""
import pytest
import torch

from reproduce.learning import ClearNet, LearningConfig, bptt_grads, dend_forward, dend_grads
from reproduce.models.crisp import Config, DendStochAudioNet


def make_network(modes=4, dt=0.5):
    torch.manual_seed(42)
    model = DendStochAudioNet(Config(n_in=4, n_hidden=6, n_classes=3,
                                    k_modes=modes, hidden_dropout=0.2)).double()
    model.set_dt(dt)
    with torch.no_grad():
        for layer in model.layers:
            layer.norm.running_mean.normal_(0, 0.5)
            layer.norm.running_var.uniform_(0.5, 2.0)
            layer.norm.weight.uniform_(0.5, 1.5)
            layer.norm.bias.normal_(0, 0.2)
    return model.eval()


@pytest.mark.parametrize("modes,dt,length", [(1, 0.25, 7), (4, 0.5, 25), (8, 1.0, 37)])
def test_complete_gradients_match_autograd(modes, dt, length):
    model = make_network(modes, dt)
    clear = ClearNet(model)
    x = torch.rand(3, length, 4, dtype=torch.float64)
    target = torch.tensor([0, 1, 2])
    masks = clear.make_masks(x, 3.0)
    reference, logits, caches, readout_cache = bptt_grads(clear, x, target, 3.0, masks)
    for rule in ("exact", f"clear_sym_k{length}"):
        actual = clear.local_grads(logits, target, caches, readout_cache, 3.0, rule)
        assert set(actual) == set(reference)
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=1e-10, atol=1e-12)


def test_explicit_forward_matches_original_model():
    model = make_network()
    x = torch.rand(3, 19, 4, dtype=torch.float64)
    with torch.no_grad():
        expected, _, _ = model(x, "graded", 2.0)
        actual, _, _ = ClearNet(model).forward_explicit(x, 2.0)
    assert torch.equal(actual, expected)


def test_dendrite_input_and_parameter_derivatives():
    model = make_network()
    dendrite = model.layers[0].dend
    x = torch.rand(2, 17, 6, dtype=torch.float64, requires_grad=True)
    y, cache = dend_forward(dendrite, x)
    incoming = torch.randn_like(y)
    reference = torch.autograd.grad((incoming * y).sum(), (dendrite.raw_lam, dendrite.mix, x))
    actual = dend_grads(incoming, cache)
    for measured, expected in zip(actual, reference):
        torch.testing.assert_close(measured, expected, rtol=1e-10, atol=1e-12)


def test_truncated_rule_is_not_claimed_exact():
    model = make_network()
    clear = ClearNet(model, learning_config=LearningConfig(reg_coef=0))
    x = torch.rand(3, 25, 4, dtype=torch.float64)
    target = torch.tensor([0, 1, 2])
    reference, logits, caches, readout_cache = bptt_grads(clear, x, target, 3.0, None)
    approximate = clear.local_grads(logits, target, caches, readout_cache, 3.0, "clear_sym")
    assert not torch.allclose(approximate["layers.0.proj.weight"], reference["layers.0.proj.weight"],
                              rtol=1e-4, atol=1e-10)
    torch.testing.assert_close(approximate["readout.weight"], reference["readout.weight"])


def test_unknown_rule_is_rejected():
    model = make_network()
    clear = ClearNet(model)
    x = torch.rand(2, 7, 4, dtype=torch.float64)
    logits, caches, readout_cache = clear.forward_explicit(x, 1.0)
    with pytest.raises(ValueError):
        clear.local_grads(logits, torch.tensor([0, 1]), caches, readout_cache, 1.0, "typo")


def test_exact_supplied_learning_signals_match_within_layer_gradients():
    model = make_network()
    clear = ClearNet(model, learning_config=LearningConfig(reg_coef=0))
    x = torch.rand(3, 13, 4, dtype=torch.float64)
    target = torch.tensor([0, 1, 2])
    masks = clear.make_masks(x, 2.0)
    probes = [torch.zeros(3, 13, layer.proj.out_features, dtype=torch.float64,
                          requires_grad=True) for layer in model.layers]
    logits, caches, readout_cache = clear.forward_explicit(x, 2.0, masks=masks, probes=probes)
    loss = torch.nn.functional.cross_entropy(logits, target)
    parameters = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, list(parameters.values()) + probes)
    reference = dict(zip(parameters, gradients[:len(parameters)]))
    signals = dict(enumerate(gradients[len(parameters):]))
    actual = clear.local_grads(logits.detach(), target, caches, readout_cache, 2.0,
                               "clear_sym", Lsig=signals)
    for name in reference:
        torch.testing.assert_close(actual[name], reference[name], rtol=1e-10, atol=1e-12)
