import torch

from lcp_physics.lcp.lcp import lcp_function
from lcp_physics.physics.bodies import Circle
from lcp_physics.physics.engines import PdipmEngine
from lcp_physics.physics.world import World


def _one_variable_problem(*, p_requires_grad=False):
    Q = torch.tensor([[[1.0]]], dtype=torch.double)
    p = torch.tensor([[-1.0]], dtype=torch.double, requires_grad=p_requires_grad)
    G = torch.tensor([[[-1.0]]], dtype=torch.double)
    h = torch.tensor([[0.0]], dtype=torch.double)
    A = torch.empty(1, 0, 1, dtype=torch.double)
    b = torch.empty(1, 0, dtype=torch.double)
    F = torch.tensor([[[0.0]]], dtype=torch.double)
    return Q, p, G, h, A, b, F


def test_one_variable_convex_lcp_has_finite_expected_solution():
    problem = _one_variable_problem()

    solution = lcp_function(max_iter=20)(*problem)

    assert torch.isfinite(solution).all()
    torch.testing.assert_close(solution, torch.ones_like(solution), atol=1e-8, rtol=1e-8)


def test_one_variable_lcp_passes_gradcheck_for_interior_p():
    Q, p, G, h, A, b, F = _one_variable_problem(p_requires_grad=True)
    solve = lcp_function(max_iter=20)

    assert torch.autograd.gradcheck(
        lambda candidate_p: solve(Q, candidate_p, G, h, A, b, F),
        (p,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_one_variable_lcp_backward_matches_known_p_gradient():
    problem = _one_variable_problem(p_requires_grad=True)
    p = problem[1]

    solution = lcp_function(max_iter=20)(*problem)
    (gradient,) = torch.autograd.grad(solution.sum(), p)

    torch.testing.assert_close(gradient, -torch.ones_like(p), atol=1e-8, rtol=1e-8)


def test_lcp_backward_supports_positive_equality_count():
    Q = torch.tensor([[[1.0]]], dtype=torch.double)
    p = torch.tensor([[-1.0]], dtype=torch.double)
    G = torch.tensor([[[-1.0]]], dtype=torch.double)
    h = torch.tensor([[0.0]], dtype=torch.double)
    A = torch.tensor([[[1.0]]], dtype=torch.double)
    b = torch.tensor([[0.5]], dtype=torch.double, requires_grad=True)
    F = torch.tensor([[[0.0]]], dtype=torch.double)

    solution = lcp_function(max_iter=20)(Q, p, G, h, A, b, F)
    (gradient,) = torch.autograd.grad(solution.sum(), b)

    torch.testing.assert_close(solution, b, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(gradient, torch.ones_like(b), atol=1e-8, rtol=1e-8)


def test_pdipm_engine_solves_world_contact_scene():
    world = World(
        [Circle([0.0, 0.0], 1.0), Circle([1.9, 0.0], 1.0)],
        engine=PdipmEngine,
        strict_no_penetration=False,
    )

    new_velocity = world.engine.solve_dynamics(world, world.dt)

    assert world.contacts
    assert new_velocity.shape == world.get_v().shape
    assert torch.isfinite(new_velocity).all()
