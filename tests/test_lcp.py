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


def _strictly_interior_problem(*, with_equality):
    Q = torch.tensor(
        [[[2.0, 0.3], [0.3, 1.5]]],
        dtype=torch.double,
        requires_grad=True,
    )
    p = torch.tensor([[-1.0, -0.5]], dtype=torch.double, requires_grad=True)
    G = torch.tensor(
        [[[-1.0, 0.0], [0.0, -1.0], [0.4, 0.6]]],
        dtype=torch.double,
        requires_grad=True,
    )
    h = torch.tensor([[0.0, 0.0, 1.2]], dtype=torch.double, requires_grad=True)
    F = torch.tensor(
        [[[0.2, 0.01, 0.0], [0.01, 0.3, 0.02], [0.0, 0.02, 0.25]]],
        dtype=torch.double,
        requires_grad=True,
    )
    if with_equality:
        A = torch.tensor([[[1.0, 1.0]]], dtype=torch.double, requires_grad=True)
        b = torch.tensor([[0.8]], dtype=torch.double, requires_grad=True)
    else:
        A = torch.empty(1, 0, 2, dtype=torch.double)
        b = torch.empty(1, 0, dtype=torch.double)
    return Q, p, G, h, A, b, F


def _symmetrize(Q):
    return 0.5 * (Q + Q.transpose(1, 2))


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


def test_lcp_gradcheck_covers_all_inputs_without_equalities():
    Q, p, G, h, A, b, F = _strictly_interior_problem(with_equality=False)
    solve = lcp_function(max_iter=30)

    # The unconstrained optimum has x > 0 and Gx < h with finite margins, so
    # gradcheck perturbations remain on one smooth, strictly inactive branch.
    solution = solve(_symmetrize(Q), p, G, h, A, b, F)
    slack = h - G.bmm(solution.unsqueeze(2)).squeeze(2)
    assert torch.all(slack > 0.1)
    assert torch.autograd.gradcheck(
        lambda candidate_Q, candidate_p, candidate_G, candidate_h, candidate_F:
            solve(_symmetrize(candidate_Q), candidate_p, candidate_G,
                  candidate_h, A, b, candidate_F),
        (Q, p, G, h, F),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_lcp_gradcheck_covers_all_inputs_with_equalities():
    problem = _strictly_interior_problem(with_equality=True)
    solve = lcp_function(max_iter=30)

    # A has full row rank, and its constrained optimum also has strict slack
    # in every inequality, avoiding complementarity active-set kinks.
    Q, p, G, h, A, b, F = problem
    solution = solve(_symmetrize(Q), p, G, h, A, b, F)
    slack = h - G.bmm(solution.unsqueeze(2)).squeeze(2)
    assert torch.all(slack > 0.1)
    assert torch.autograd.gradcheck(
        lambda candidate_Q, candidate_p, candidate_G, candidate_h,
               candidate_A, candidate_b, candidate_F:
            solve(_symmetrize(candidate_Q), candidate_p, candidate_G,
                  candidate_h, candidate_A, candidate_b, candidate_F),
        problem,
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_active_inequality_gradcheck_covers_G_h_and_F():
    Q = torch.tensor([[[1.0]]], dtype=torch.double, requires_grad=True)
    p = torch.tensor([[-2.0]], dtype=torch.double, requires_grad=True)
    G = torch.tensor([[[1.0]]], dtype=torch.double, requires_grad=True)
    h = torch.tensor([[1.0]], dtype=torch.double, requires_grad=True)
    A = torch.empty(1, 0, 1, dtype=torch.double)
    b = torch.empty(1, 0, dtype=torch.double)
    F = torch.tensor([[[0.0]]], dtype=torch.double, requires_grad=True)
    solve = lcp_function(max_iter=30)

    # The unconstrained x=2 violates x<=1. At the smooth active solution,
    # x=1 and lambda=1, so strict complementarity is bounded away from zero.
    solution = solve(Q, p, G, h, A, b, F)
    dual = -(Q.bmm(solution.unsqueeze(2)).squeeze(2) + p) / G.squeeze(2)
    gradients = torch.autograd.grad(solution.sum(), (Q, p, G, h, F))

    torch.testing.assert_close(solution, torch.ones_like(solution), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(dual, torch.ones_like(dual), atol=1e-8, rtol=1e-8)
    for gradient in gradients[2:]:
        assert torch.all(gradient.abs() > 0.5)
    torch.testing.assert_close(gradients[2], -torch.ones_like(G), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(gradients[3], torch.ones_like(h), atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(gradients[4], torch.ones_like(F), atol=1e-8, rtol=1e-8)
    assert torch.autograd.gradcheck(
        lambda candidate_Q, candidate_p, candidate_G, candidate_h, candidate_F:
            solve(candidate_Q, candidate_p, candidate_G, candidate_h,
                  A, b, candidate_F),
        (Q, p, G, h, F),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


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


def test_lcp_solver_state_uses_saved_tensor_hooks_for_repeated_backward():
    problem = _one_variable_problem(p_requires_grad=True)
    p = problem[1]
    packed_signatures = []
    unpacked_signatures = []

    def pack_hook(tensor):
        signature = (tensor.dtype, tuple(tensor.shape))
        packed_signatures.append(signature)
        return signature, tensor.detach().clone()

    def unpack_hook(packed):
        signature, tensor = packed
        unpacked_signatures.append(signature)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        solution = lcp_function(max_iter=20)(*problem)

    first_gradient = torch.autograd.grad(
        solution.sum(), p, retain_graph=True)[0]
    second_gradient = torch.autograd.grad(solution.sum(), p)[0]

    integral_solver_state = [
        signature for signature in packed_signatures
        if not signature[0].is_floating_point and signature[0] != torch.bool
    ]
    expected_categories = {
        (torch.double, (1, 1, 1)),
        (torch.double, (1, 1)),
        (torch.double, (1, 0, 1)),
        (torch.double, (1, 0)),
        (torch.double, (0,)),
    }
    assert expected_categories.issubset(set(packed_signatures))
    assert packed_signatures.count((torch.double, (1, 1, 1))) >= 5
    assert packed_signatures.count((torch.double, (1, 1))) >= 5
    assert integral_solver_state, "LU pivots bypassed saved_tensors_hooks"
    assert expected_categories.issubset(set(unpacked_signatures))
    assert all(
        unpacked_signatures.count(signature) >= 2
        for signature in integral_solver_state + [(torch.double, (0,))]
    )
    torch.testing.assert_close(first_gradient, -torch.ones_like(p))
    torch.testing.assert_close(second_gradient, first_gradient)


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
