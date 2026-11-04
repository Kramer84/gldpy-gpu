"""The batched CDF solver: safeguarded Newton on logit(y)."""
from typing import Optional

import torch
import torch.nn.functional as F

from ._torch_core import _log_density_from_logs, _quantile_from_logs


_START_TABLE_SIZE = 512
_SMALL_STEP_FACTOR = 1000.0


def _table_start(x: torch.Tensor, param: torch.Tensor, param_type: str, bound: torch.Tensor) -> torch.Tensor:
    """Starting t for each x, by linear interpolation in a small table of Q(t).

    Q is monotone, so a binary search finds the cell. The start is then close enough that about
    three Newton steps finish, instead of the 6 to 13 it takes from t = 0.
    """
    limit = min(40.0, bound.item())
    grid = torch.linspace(-limit, limit, _START_TABLE_SIZE, dtype=x.dtype, device=x.device)
    table = _quantile_from_logs(F.logsigmoid(grid), F.logsigmoid(-grid), param, param_type)
    above = torch.searchsorted(table, x).clamp(1, _START_TABLE_SIZE - 1)
    below = above - 1
    weight = torch.clamp((x - table[below]) / (table[above] - table[below]), 0.0, 1.0)
    start = grid[below] + weight * (grid[above] - grid[below])
    return torch.where(torch.isfinite(start), start, torch.zeros_like(start))


def solve_logit_cdf(
    x: torch.Tensor,
    param: torch.Tensor,
    param_type: str,
    xtol: float = 1e-6,
    max_iter: int = 64,
    t0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve Q(sigmoid(t)) = x element-wise with a safeguarded Newton iteration.

    Working on t = logit(y) makes `xtol` a relative tolerance on the smaller of y and 1 - y, so
    both tails converge alike and F, 1 - F and their logs all come out of t without cancellation.
    `xtol` has a floor of 32 eps. Each element stops on its own, and finished elements leave the arrays, so the cost of an
    iteration follows the number of elements still unresolved.

    Returns -inf/+inf at or beyond the support, NaN for NaN input, and saturates at t = +-log(tiny),
    i.e. tail probabilities below ~1e-38 (float32) or ~1e-308 (float64) report as that floor.
    `t0` is an optional starting point (a warm start); by default a table supplies one.
    Not differentiable: `fit_ML` shows how to get gradients from the implicit function theorem.
    """
    finfo = torch.finfo(x.dtype)
    bound = -torch.log(torch.as_tensor(finfo.tiny, dtype=x.dtype, device=x.device))
    tol = max(float(xtol), 32.0 * finfo.eps)  # below this, steps are rounding noise and never settle
    zero = torch.zeros(1, dtype=x.dtype, device=x.device)
    inf = torch.full_like(zero, float("inf"))
    q_min = _quantile_from_logs(-inf, zero, param, param_type)
    q_max = _quantile_from_logs(zero, -inf, param, param_type)
    x, q_min, q_max = torch.broadcast_tensors(x, q_min, q_max)

    below, above, missing = x <= q_min, x >= q_max, torch.isnan(x)
    result = torch.where(below, -inf, torch.where(above, inf, x))  # x itself is only kept where it is NaN
    open_elements = ~(below | above | missing)
    positions = torch.nonzero(open_elements, as_tuple=True)
    x_a = x[positions]
    if not x_a.numel():
        return result

    # A single parameter set stays one vector, and a batch is indexed along with the elements.
    single = param.dim() == 1
    p_a = param if single else param.expand(*x.shape, 4)[positions]
    if t0 is not None:
        start = torch.broadcast_to(t0, x.shape)[positions]
        t = torch.where(torch.isfinite(start), start, torch.zeros_like(start)).clamp(-bound, bound)
    elif single:
        t = _table_start(x_a, param, param_type, bound)
    else:
        t = torch.zeros_like(x_a)
    low, high = torch.full_like(x_a, -bound), torch.full_like(x_a, bound)
    prev_step = torch.full_like(x_a, 2.0 * bound)
    done = torch.zeros_like(x_a, dtype=torch.bool)

    def retire(mask: torch.Tensor) -> None:
        result[tuple(index[mask] for index in positions)] = t[mask]

    for _ in range(max_iter):
        log_y, log_s = F.logsigmoid(t), F.logsigmoid(-t)
        q = _quantile_from_logs(log_y, log_s, p_a, param_type)
        err = q - x_a
        low = torch.where(err < 0.0, t, low)
        high = torch.where(err >= 0.0, t, high)

        # dt/dx = f(y) / (y (1 - y))
        log_density = _log_density_from_logs(log_y, log_s, p_a, param_type, interior=True)
        newton = -err * torch.exp(log_density - log_y - log_s)
        candidate = t + newton
        use_newton = (
            torch.isfinite(candidate)
            & (candidate > low)
            & (candidate < high)
            # A step this small cannot diverge, and noise in it would otherwise trigger a bisection
            # of a bracket that is still wide, losing an answer that had already converged.
            & ((newton.abs() <= 0.5 * prev_step.abs()) | (newton.abs() <= _SMALL_STEP_FACTOR * tol))
        )
        t_next = torch.where(use_newton, candidate, 0.5 * (low + high))
        step = t_next - t
        # A root met to rounding error keeps t: `low` or `high` now equals t, and a further bisection would leave it.
        at_root = err.abs() <= 2.0 * finfo.eps * (x_a.abs() + q.abs())
        t = torch.where(at_root, t, t_next)
        prev_step = step
        done = at_root | (step.abs() <= tol) | (high - low <= tol)

        finished = int(done.sum())
        if finished == done.numel():
            break
        if 4 * finished >= done.numel():  # compact only when it pays for the indexing
            retire(done)
            keep = ~done
            positions = tuple(index[keep] for index in positions)
            x_a, t, low, high, prev_step = x_a[keep], t[keep], low[keep], high[keep], prev_step[keep]
            if not single:
                p_a = p_a[keep]

    retire(torch.ones_like(t, dtype=torch.bool))
    return result


def invert_quantile_newton_bisect(
    x: torch.Tensor,
    param: torch.Tensor,
    param_type: str,
    xtol: float = 1e-6,
    max_iter: int = 64,
) -> torch.Tensor:
    """Vectorized F(x), the inverse of Q (see `solve_logit_cdf`)."""
    return torch.sigmoid(solve_logit_cdf(x, param, param_type, xtol=xtol, max_iter=max_iter))
