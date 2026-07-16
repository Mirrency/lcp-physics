import torch
from torch.autograd import Function

from .solvers import pdipm
from .util import bger, extract_batch_size


class LCPFunction(Function):
    """A differentiable LCP solver, uses the primal dual interior point method
       implemented in pdipm.
    """
    # @profile
    @staticmethod
    def forward(ctx, Q, p, G, h, A, b, F, eps, verbose,
                not_improved_lim, max_iter):
        _, nineq, _ = G.size()
        neq = A.size(1) if A.ndimension() > 1 else 0
        assert(neq > 0 or nineq > 0)
        ctx.neq, ctx.nineq = neq, nineq

        Q_LU, S_LU, R = pdipm.pre_factor_kkt(Q, G, F, A)
        zhats, nus, lams, slacks = pdipm.forward(
            Q, p, G, h, A, b, F, Q_LU, S_LU, R,
            eps=eps, max_iter=max_iter, verbose=verbose,
            not_improved_lim=not_improved_lim)

        saved_nus = nus if nus is not None else Q.new_empty(0)
        ctx.save_for_backward(
            zhats, Q, p, G, h, A, b, F,
            Q_LU[0], Q_LU[1], R, lams, slacks, saved_nus)
        return zhats

    @staticmethod
    def backward(ctx, dl_dzhat):
        (zhats, Q, p, G, h, A, b, F,
         Q_LU_data, Q_LU_pivots, R, lams, slacks,
         saved_nus) = ctx.saved_tensors
        batch_size = extract_batch_size(Q, p, G, h, A, b)

        neq, nineq = ctx.neq, ctx.nineq
        Q_LU = (Q_LU_data, Q_LU_pivots)
        S_LU = [None, None]
        nus = saved_nus if neq > 0 else None

        d = lams / slacks

        # The backward solve is an adjoint system; generalized F may be
        # non-symmetric, so its cached Schur base must be transposed.
        pdipm.factor_kkt(S_LU, R.transpose(1, 2), d)
        dx, _, dlam, dnu = pdipm.solve_kkt(
            Q_LU, d, G, A, S_LU,
            dl_dzhat, G.new_zeros(batch_size, nineq),
            G.new_zeros(batch_size, nineq),
            G.new_zeros(batch_size, neq))

        dps = dx
        dGs = bger(dlam, zhats) + bger(lams, dx)
        dFs = -bger(dlam, lams)
        dhs = -dlam
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(nus, dx)
            dbs = -dnu
        else:
            dAs, dbs = None, None
        dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))

        return dQs, dps, dGs, dhs, dAs, dbs, dFs, None, None, None, None


def lcp_function(eps=1e-12, verbose=-1, not_improved_lim=3, max_iter=10):
    """Return a configured differentiable LCP solver callable."""
    def solve(Q, p, G, h, A, b, F):
        return LCPFunction.apply(
            Q, p, G, h, A, b, F,
            eps, verbose, not_improved_lim, max_iter)

    return solve
