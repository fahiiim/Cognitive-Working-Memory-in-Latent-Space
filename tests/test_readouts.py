import math

import torch

from glwm2.eval.metrics import first_stable_correct
from glwm2.models.halting_units import act_loss, geometric_prior, halting_distribution, ponder_loss, _check_ponder_formula
from glwm2.readouts.rules import all_rule_outcomes, em_at, first_cross, select_threshold, threshold_steps
from glwm2.train.objectives import done_targets


def test_first_cross_and_default_to_tmax():
    cond = torch.tensor([[[False, True], [True, False], [True, False]]])  # [K=1, T=3, B=2]
    assert first_cross(cond).tolist() == [[2, 1]]
    cond = torch.zeros(1, 4, 3, dtype=torch.bool)
    assert first_cross(cond).tolist() == [[4, 4, 4]]


def test_threshold_steps_per_sample_masks():
    stat = torch.tensor([[0.1, 0.9], [0.95, 0.2], [0.99, 0.99]])  # [T=3, B=2]
    steps = threshold_steps(stat, ">", [0.5, 0.98])
    assert steps.tolist() == [[2, 1], [3, 3]]


def test_em_at_gathers_per_instance():
    em = torch.tensor([[True, False], [False, True], [True, True]])
    steps = torch.tensor([[1, 2], [2, 1]])
    assert em_at(em, steps).tolist() == [[True, True], [False, False]]


def test_select_threshold_prefers_cheapest_within_tolerance():
    steps = torch.tensor([[1.0] * 4, [2.0] * 4, [5.0] * 4])
    correct = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]).bool()
    assert select_threshold(steps, correct) == 1


def test_done_targets_and_first_stable():
    pc = torch.tensor([[1, 0], [0, 1], [1, 1], [1, 1]]).bool()  # [T=4, B=2] (one position each)
    assert done_targets(pc).int().tolist() == [[0, 0], [0, 1], [1, 1], [1, 1]]
    assert first_stable_correct(pc).tolist() == [3, 2]


def test_ponder_distribution_matches_published_formula():
    assert _check_ponder_formula() < 1e-6
    g = torch.randn(6, 5)
    p = halting_distribution(g)
    lam = torch.sigmoid(g)
    lam[-1] = 1
    for t in range(6):
        expected = lam[t] * torch.prod(1 - lam[:t], 0)
        assert torch.allclose(p[t], expected, atol=1e-6)
    assert torch.allclose(p.sum(0), torch.ones(5), atol=1e-6)
    prior = geometric_prior(6, 0.2, "cpu")
    assert math.isclose(float(prior.sum()), 1.0, rel_tol=1e-6)
    loss, parts = ponder_loss(g, torch.rand(6, 5), beta=0.01, lambda_p=0.2)
    assert parts["kl"] >= 0


def test_act_weights_sum_to_one_and_remainder():
    g = torch.full((5, 3), -1.0)  # lambda = 0.269 per step
    logp = torch.zeros(5, 3, 2)
    mix, ponder = act_loss(g, logp, tau=0.01)
    # with log p = 0 everywhere the mixture log-likelihood is log(sum of weights) = 0
    assert torch.allclose(mix, torch.zeros(3, 2), atol=1e-5)
    lam = torch.sigmoid(torch.tensor(-1.0))
    n = math.ceil((1 - 0.01) / lam)  # first step at which the cumulative sum reaches 1 - eps
    assert torch.allclose(ponder, torch.full((3,), n + (1 - (n - 1) * lam)), atol=1e-5)


def test_rule_outcomes_reference_oracles():
    em = torch.tensor([[False, True], [True, False], [True, True]])
    traj = {"em": em, "logconf_sum": torch.zeros(3, 2)}
    out = all_rule_outcomes(traj, torch.tensor([2, 1]))
    assert out["oracle_iter"]["steps"].tolist() == [[2, 1]]
    assert out["oracle_stable"]["steps"].tolist() == [[2, 3]]
    assert out["oracle_lb"]["correct"].tolist() == [[True, True]]
