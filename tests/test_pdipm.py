import pytest
import torch

from lcp_physics.lcp.solvers import pdipm
from lcp_physics.lcp.util import bdiag, efficient_btriunpack


@pytest.mark.parametrize("rhs_columns", [None, 3])
def test_lu_solve_matches_torch_linalg_solve_and_preserves_rhs_shape(rhs_columns):
    matrix = torch.tensor(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[3.0, -1.0], [1.0, 2.0]],
        ],
        dtype=torch.double,
    )
    if rhs_columns is None:
        rhs = torch.tensor([[1.0, 5.0], [-2.0, 4.0]], dtype=torch.double)
        expected = torch.linalg.solve(matrix, rhs.unsqueeze(-1)).squeeze(-1)
    else:
        rhs = torch.tensor(
            [
                [[1.0, 0.0, 2.0], [5.0, -1.0, 3.0]],
                [[-2.0, 3.0, 1.0], [4.0, 0.5, -3.0]],
            ],
            dtype=torch.double,
        )
        expected = torch.linalg.solve(matrix, rhs)

    factorization = pdipm.lu_factor(matrix)
    actual = pdipm.lu_solve(factorization, rhs)

    assert actual.shape == rhs.shape
    torch.testing.assert_close(actual, expected)


def test_bdiag_builds_batched_diagonal_matrix():
    diagonal = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.double)

    actual = bdiag(diagonal)

    torch.testing.assert_close(actual, torch.diag_embed(diagonal))


@pytest.mark.filterwarnings("error:where received a uint8 condition tensor")
def test_efficient_btriunpack_reconstructs_batched_matrix():
    matrix = torch.tensor(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[0.0, 2.0], [3.0, 1.0]],
        ],
        dtype=torch.double,
    )

    P, L, U = efficient_btriunpack(*pdipm.lu_factor(matrix))

    torch.testing.assert_close(P.bmm(L).bmm(U), matrix)


def _solve_full_kkt(Q, d, G, F, A, rx, rs, rz, ry):
    batch_size, nz, _ = Q.shape
    nineq = G.shape[1]
    neq = A.shape[1]
    solutions = []

    for batch in range(batch_size):
        size = nz + 2 * nineq + neq
        matrix = Q.new_zeros(size, size)
        x = slice(0, nz)
        s = slice(nz, nz + nineq)
        z = slice(nz + nineq, nz + 2 * nineq)
        y = slice(nz + 2 * nineq, size)

        matrix[x, x] = Q[batch]
        matrix[x, z] = G[batch].transpose(0, 1)
        matrix[s, s] = torch.diag(d[batch])
        matrix[s, z] = torch.eye(nineq, dtype=Q.dtype)
        matrix[z, x] = G[batch]
        matrix[z, s] = torch.eye(nineq, dtype=Q.dtype)
        matrix[z, z] = -F[batch]
        if neq:
            matrix[x, y] = A[batch].transpose(0, 1)
            matrix[y, x] = A[batch]

        rhs = -torch.cat((rx[batch], rs[batch], rz[batch], ry[batch]))
        solutions.append(torch.linalg.solve(matrix, rhs))

    solution = torch.stack(solutions)
    dx = solution[:, :nz]
    ds = solution[:, nz:nz + nineq]
    dz = solution[:, nz + nineq:nz + 2 * nineq]
    dy = solution[:, -neq:] if neq else None
    return dx, ds, dz, dy


@pytest.mark.parametrize("neq", [0, 1])
def test_pre_factor_factor_and_solve_kkt_match_full_system(neq):
    Q = torch.tensor(
        [
            [[4.0, 0.5], [0.5, 3.0]],
            [[3.0, -0.25], [-0.25, 2.0]],
        ],
        dtype=torch.double,
    )
    G = torch.tensor(
        [
            [[1.0, -0.5], [0.25, 1.0]],
            [[-0.75, 0.5], [1.0, 0.25]],
        ],
        dtype=torch.double,
    )
    F = torch.tensor(
        [
            [[0.4, 0.05], [0.05, 0.3]],
            [[0.25, -0.02], [-0.02, 0.35]],
        ],
        dtype=torch.double,
    )
    if neq:
        A = torch.tensor([[[1.0, 1.0]], [[0.5, -1.0]]], dtype=torch.double)
        ry = torch.tensor([[0.2], [-0.1]], dtype=torch.double)
    else:
        A = torch.empty(2, 0, 2, dtype=torch.double)
        ry = torch.empty(2, 0, dtype=torch.double)

    rx = torch.tensor([[0.5, -1.0], [1.5, 0.25]], dtype=torch.double)
    rs = torch.tensor([[0.2, -0.3], [-0.5, 0.4]], dtype=torch.double)
    rz = torch.tensor([[-0.7, 0.1], [0.3, -0.8]], dtype=torch.double)
    d_values = (
        torch.tensor([[1.2, 0.8], [0.6, 1.4]], dtype=torch.double),
        torch.tensor([[0.7, 1.5], [1.1, 0.9]], dtype=torch.double),
    )

    Q_LU, S_LU, R = pdipm.pre_factor_kkt(Q, G, F, A)
    for d in d_values:
        pdipm.factor_kkt(S_LU, R, d)
        actual = pdipm.solve_kkt(Q_LU, d, G, A, S_LU, rx, rs, rz, ry)
        expected = _solve_full_kkt(Q, d, G, F, A, rx, rs, rz, ry)

        for actual_part, expected_part in zip(actual, expected):
            if expected_part is None:
                assert actual_part is None
            else:
                torch.testing.assert_close(actual_part, expected_part)
