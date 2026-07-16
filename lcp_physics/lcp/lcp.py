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
        _, nineq, nz = G.size()
        neq = A.size(1) if A.ndimension() > 1 else 0
        assert(neq > 0 or nineq > 0)
        ctx.neq, ctx.nineq, ctx.nz = neq, nineq, nz

        ctx.Q_LU, ctx.S_LU, ctx.R = pdipm.pre_factor_kkt(Q, G, F, A)
        zhats, ctx.nus, ctx.lams, ctx.slacks = pdipm.forward(
            Q, p, G, h, A, b, F, ctx.Q_LU, ctx.S_LU, ctx.R,
            eps=eps, max_iter=max_iter, verbose=verbose,
            not_improved_lim=not_improved_lim)

        ctx.save_for_backward(zhats, Q, p, G, h, A, b, F)
        return zhats

    @staticmethod
    def backward(ctx, dl_dzhat):
        zhats, Q, p, G, h, A, b, F = ctx.saved_tensors
        batch_size = extract_batch_size(Q, p, G, h, A, b)

        neq, nineq, nz = ctx.neq, ctx.nineq, ctx.nz

        # D = torch.diag((ctx.lams / ctx.slacks).squeeze(0)).unsqueeze(0)
        d = ctx.lams / ctx.slacks

        pdipm.factor_kkt(ctx.S_LU, ctx.R, d)
        dx, _, dlam, dnu = pdipm.solve_kkt(
            ctx.Q_LU, d, G, A, ctx.S_LU,
            dl_dzhat, G.new_zeros(batch_size, nineq),
            G.new_zeros(batch_size, nineq),
            G.new_zeros(batch_size, neq))

        dps = dx
        dGs = bger(dlam, zhats) + bger(ctx.lams, dx)
        dFs = -bger(dlam, ctx.lams)
        dhs = -dlam
        if neq > 0:
            dAs = bger(dnu, zhats) + bger(ctx.nus, dx)
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
