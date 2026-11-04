"""Fit solvers: multistart Levenberg-Marquardt on the shape parameters, closed forms for VSL, roots chosen by Q-Q fit."""
import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ._torch_core import (
    _box_cox,
    FORMULAS,
    _plotting_logs,
    _pow_minus_one,
    _quantile_from_logs,
    validate_gld_params,
)


def _with_support_covering(param: torch.Tensor, z_min: torch.Tensor, z_max: torch.Tensor, param_type: str) -> torch.Tensor:
    """Smallest location/scale change that puts [z_min, z_max] strictly inside the support.

    Q is affine in location and scale, so this is exact, and the shape parameters do not change.
    An infinite end is never violated, and a single finite end only needs a shift.
    """
    zero = torch.zeros_like(param[0])
    inf = torch.full_like(zero, float("inf"))
    lo = _quantile_from_logs(-inf, zero, param, param_type)
    hi = _quantile_from_logs(zero, -inf, param, param_type)
    margin = 1e-6 * (z_max - z_min)
    new_lo = torch.minimum(lo, z_min - margin)
    new_hi = torch.maximum(hi, z_max + margin)
    if torch.isfinite(lo) and torch.isfinite(hi):
        k = (new_hi - new_lo) / (hi - lo)
        scale = param[1] * k if param_type == "VSL" else param[1] / k
        return torch.stack([new_lo - (lo - param[0]) * k, scale, param[2], param[3]])
    shift = new_lo - lo if torch.isfinite(lo) else (new_hi - hi if torch.isfinite(hi) else zero)
    return torch.stack([param[0] + shift, param[1], param[2], param[3]])


_GOF_MAX_POINTS = 100_000
_LM_MAX_IDLE_STEPS = 4  # with no accepted step this long, the damping has grown by 4**4 and nothing moves


def _root_loss(dtype: torch.dtype) -> float:
    """Squared residual below which a shape solution counts as an exact root."""
    return max(1e-10, (2000.0 * torch.finfo(dtype).eps) ** 2)


def _quantile_sample(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sorted data, log y, log(1 - y)), thinned and moved to the CPU, in the data's precision.

    The shape solves are small batches of scalar formulas: on a GPU, launch overhead makes them
    slower than on the CPU. Only the sort of the full sample benefits from the device.
    """
    x, _ = torch.sort(data.reshape(-1))
    n = x.numel()
    m = min(n, _GOF_MAX_POINTS)
    idx = (torch.arange(m, device=x.device) * (n - 1)) // max(m - 1, 1)
    log_y, log_s = _plotting_logs(idx, n, x.dtype)
    return x[idx].cpu(), log_y.cpu(), log_s.cpu()


def _gof_per_candidate(candidates: torch.Tensor, sample: Tuple[torch.Tensor, ...], param_type: str) -> torch.Tensor:
    """Q-Q discrepancy of each candidate (K, 4) on the sample, invalid ones -> inf."""
    x, log_y, log_s = sample
    q = _quantile_from_logs(log_y.unsqueeze(0), log_s.unsqueeze(0), candidates.unsqueeze(1), param_type)
    mse = torch.mean((x.unsqueeze(0) - q) ** 2, dim=-1)
    valid = validate_gld_params(candidates, param_type) & torch.isfinite(mse)
    return torch.where(valid, mse, torch.full_like(mse, float("inf")))


def solve_vsl_lmm_closed_form(
    sample: Tuple[torch.Tensor, ...],
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    a4: torch.Tensor,
    initial_guess: Optional[float] = None,
) -> torch.Tensor:
    """Exact analytical L-moment estimator for the VSL parameterization."""
    disc = a4 * a4 + 98.0 * a4 + 1.0
    min_a4 = (-98.0 + (98.0**2 - 4.0) ** 0.5) / 2.0 + 1e-10
    a4_safe = torch.where(disc < 0.0, torch.full_like(a4, min_a4), a4)
    sqrt_disc = torch.sqrt(torch.clamp(a4_safe * a4_safe + 98.0 * a4_safe + 1.0, min=0.0))

    p4_pair = torch.stack(
        [
            (3.0 + 7.0 * a4_safe + sqrt_disc) / (2.0 * (1.0 - a4_safe)),
            (3.0 + 7.0 * a4_safe - sqrt_disc) / (2.0 * (1.0 - a4_safe)),
        ]
    )
    p3_pair = torch.where(
        p4_pair == 1.0,
        torch.full_like(p4_pair, 0.5),
        0.5 * (1.0 - a3 * (p4_pair + 3.0) / (p4_pair - 1.0)),
    )
    p3_pair = torch.clamp(p3_pair, 0.0, 1.0)
    p2_pair = a2 * (p4_pair + 1.0) * (p4_pair + 2.0)
    p1_pair = a1 + p2_pair * (1.0 - 2.0 * p3_pair) / (p4_pair + 1.0)

    candidates = torch.stack([p1_pair, p2_pair, p3_pair, p4_pair], dim=-1)  # (2, 4)
    if initial_guess is not None:
        return candidates[torch.argmin(torch.abs(p4_pair - float(initial_guess)))]
    return candidates[torch.argmin(_gof_per_candidate(candidates, sample, "VSL"))]


def _levenberg_marquardt(residual_fn, is_valid, l3: torch.Tensor, l4: torch.Tensor, steps: int) -> Tuple[torch.Tensor, ...]:
    """Batched Levenberg-Marquardt on a 2-equation, 2-unknown system.

    The Jacobian comes from autograd, because a difference step (eps**(1/3) ~ 5e-3 in float32) can
    be larger than the root itself: logistic data have their RS root at lambda ~ 1e-4. Each batch
    element depends only on its own unknowns, so the gradient of a sum gives per-element derivatives.
    """
    stop = (64.0 * torch.finfo(l3.dtype).eps) ** 2
    l3, l4 = l3.detach(), l4.detach()
    with torch.no_grad():
        r1, r2 = residual_fn(l3, l4)
    loss = r1 * r1 + r2 * r2
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, float("inf")))
    mu = torch.full_like(l3, 1e-3)
    idle = 0
    for _ in range(steps):
        if loss.max() < stop or idle >= _LM_MAX_IDLE_STEPS:
            break
        x3, x4 = l3.clone().requires_grad_(True), l4.clone().requires_grad_(True)
        g1_r, g2_r = residual_fn(x3, x4)
        j11, j12 = torch.autograd.grad(g1_r.sum(), (x3, x4), retain_graph=True)
        j21, j22 = torch.autograd.grad(g2_r.sum(), (x3, x4))

        a11 = j11 * j11 + j21 * j21 + mu
        a12 = j11 * j12 + j21 * j22
        a22 = j12 * j12 + j22 * j22 + mu
        g1 = j11 * r1 + j21 * r2
        g2 = j12 * r1 + j22 * r2
        det = a11 * a22 - a12 * a12
        d3 = torch.clamp((a12 * g2 - a22 * g1) / det, -1.0, 1.0)
        d4 = torch.clamp((a12 * g1 - a11 * g2) / det, -1.0, 1.0)

        new3, new4 = l3 + d3, l4 + d4
        with torch.no_grad():
            nr1, nr2 = residual_fn(new3, new4)
        new_loss = nr1 * nr1 + nr2 * nr2
        accept = torch.isfinite(new_loss) & (new_loss < loss) & is_valid(new3, new4)
        idle = 0 if accept.any() else idle + 1
        l3, l4 = torch.where(accept, new3, l3), torch.where(accept, new4, l4)
        r1, r2 = torch.where(accept, nr1, r1), torch.where(accept, nr2, r2)
        loss = torch.where(accept, new_loss, loss)
        mu = torch.clamp(torch.where(accept, mu * 0.3, mu * 4.0), 1e-14, 1e8)
    return l3, l4, loss


def solve_shape_multistart(
    residual_fn,
    param_fn,
    sample: Tuple[torch.Tensor, ...],
    param_type: str,
    grid_min: float,
    grid_max: float,
    n_grid: int = 128,
    refine_steps: int = 40,
    max_starts: int = 16,
    initial_guess=None,
) -> torch.Tensor:
    """Solve the two shape equations of a fit and assemble the (4,) parameter vector.

    With `initial_guess`, this refines from there, as the CPU class does. Otherwise it refines
    every local minimum of a parallel grid. The equations often have several exact roots
    (e.g. a far root at lambda ~ 3.5 for data drawn with lambda ~ 0.2), so among the roots it
    keeps the one with the smallest Q-Q discrepancy on the data.
    """
    device, dtype = sample[0].device, sample[0].dtype

    def is_valid(l3, l4):
        p = param_fn(l3, l4)
        return validate_gld_params(p, param_type) & torch.isfinite(p).all(dim=-1)

    if initial_guess is not None:
        guess = torch.as_tensor(initial_guess, dtype=dtype, device=device).reshape(-1)
        if guess.numel() != 2:
            raise ValueError("Specify initial guess for two parameters.")
        l3, l4, loss = _levenberg_marquardt(residual_fn, is_valid, guess[:1], guess[1:], max(refine_steps, 100))
        if loss.item() <= _root_loss(dtype) and is_valid(l3, l4).all():
            return param_fn(l3, l4)[0]
        warnings.warn("Refining from initial_guess failed; falling back to a grid search.", UserWarning, stacklevel=3)

    g = torch.linspace(grid_min, grid_max, n_grid, device=device, dtype=dtype)
    l3_grid, l4_grid = torch.meshgrid(g, g, indexing="ij")
    r1, r2 = residual_fn(l3_grid, l4_grid)
    loss = r1 * r1 + r2 * r2
    usable = torch.isfinite(loss) & is_valid(l3_grid, l4_grid)
    loss = torch.where(usable, loss, torch.full_like(loss, float("inf")))
    if not usable.any():
        raise ValueError("No valid GLD parameters found on the search grid.")

    neighbor_min = -F.max_pool2d(-loss[None, None], 3, stride=1, padding=1)[0, 0]
    starts = torch.nonzero((loss == neighbor_min) & usable)
    starts = starts[torch.argsort(loss[starts[:, 0], starts[:, 1]])[:max_starts]]

    l3, l4, final = _levenberg_marquardt(residual_fn, is_valid, l3_grid[starts[:, 0], starts[:, 1]], l4_grid[starts[:, 0], starts[:, 1]], refine_steps)
    roots = final <= max(_root_loss(dtype), 4.0 * final.min().item())
    candidates = param_fn(l3[roots], l4[roots])
    return candidates[torch.argmin(_gof_per_candidate(candidates, sample, param_type))]


def _lmm_equations(param_type: str, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, a4: torch.Tensor):
    """Residuals and parameter assembly of the L-moment equations for RS and FMKL.

    Uses simplified closed forms of the upstream sums, for example
    (-1/(1+l) + 2/(2+l)) / l = 1/((1+l)(2+l)). FMKL's 1/l singularity cancels out that way.
    """
    rs = param_type == "RS"

    def terms(l):
        g2, g3, g4 = FORMULAS.fmkl_lm_terms(l)
        return (l * g2, l * g3, l * g4) if rs else (g2, g3, g4)

    def residual_fn(l3, l4):
        # L-moments exist only for lambda > -1; the formulas alone would not notice.
        exists = (l3 > -1.0) & (l4 > -1.0)
        g2_3, g3_3, g4_3 = terms(l3)
        g2_4, g3_4, g4_4 = terms(l4)
        scale = g2_3 + g2_4
        r1 = (g3_3 - g3_4) / scale - a3
        r2 = (g4_3 + g4_4) / scale - a4
        nan = torch.full_like(r1, float("nan"))
        return torch.where(exists, r1, nan), torch.where(exists, r2, nan)

    def param_fn(l3, l4):
        l2 = (terms(l3)[0] + terms(l4)[0]) / a2
        shift = 1.0 / (1.0 + l4) - 1.0 / (1.0 + l3)
        l1 = a1 + (shift if rs else -shift) / l2
        return torch.stack([l1, l2, l3, l4], dim=-1)

    return residual_fn, param_fn


def _pm_equations(param_type: str, u: float, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, a4: torch.Tensor):
    """Residuals and parameter assembly of the percentile equations for RS and FMKL."""
    term = _pow_minus_one if param_type == "RS" else _box_cox
    dtype, device = a1.dtype, a1.device
    log_of = lambda v: torch.log(torch.as_tensor(v, dtype=dtype, device=device))  # noqa: E731
    lu, lhalf, lq1, lq3, l1mu = log_of(u), log_of(0.5), log_of(0.25), log_of(0.75), log_of(1.0 - u)

    def spread(hi, lo, l3, l4):
        """[term(hi, l3) - term(lo, l3)] + [term(1 - lo, l4) - term(1 - hi, l4)], i.e. Q(hi) - Q(lo) up to scale."""
        return term(hi, l3) - term(lo, l3) + term(_mirror[lo], l4) - term(_mirror[hi], l4)

    _mirror = {lu: l1mu, l1mu: lu, lhalf: lhalf, lq1: lq3, lq3: lq1}

    def residual_fn(l3, l4):
        r1 = spread(lhalf, lu, l3, l4) / spread(l1mu, lhalf, l3, l4) - a3
        r2 = spread(lq3, lq1, l3, l4) / spread(l1mu, lu, l3, l4) - a4
        return r1, r2

    def param_fn(l3, l4):
        l2 = spread(l1mu, lu, l3, l4) / a2
        l1 = a1 - (term(lhalf, l3) - term(lhalf, l4)) / l2
        return torch.stack([l1, l2, l3, l4], dim=-1)

    return residual_fn, param_fn


def _scalar_guess(initial_guess) -> float:
    return float(np.asarray(initial_guess, dtype=float).reshape(-1)[0])


def _solve_vsl_pm(sample: Tuple[torch.Tensor, ...], u: float, stats: Tuple[torch.Tensor, ...], n_grid: int, initial_guess=None) -> torch.Tensor:
    """Percentile estimator for VSL: every root of the one-dimensional lambda equation, then closed forms."""
    a1, a2, a3, a4 = stats
    dtype, device = a1.dtype, a1.device
    log_of = lambda v: torch.log(torch.as_tensor(v, dtype=dtype, device=device))  # noqa: E731
    lu, lhalf, lq1, lq3, l1mu = log_of(u), log_of(0.5), log_of(0.25), log_of(0.75), log_of(1.0 - u)

    def excess(l):
        return (_box_cox(lq3, l) - _box_cox(lq1, l)) / (_box_cox(l1mu, l) - _box_cox(lu, l)) - a4

    g = torch.linspace(-2.0, 4.0, n_grid * 8, device=device, dtype=dtype)
    values = excess(g)
    crossing = torch.nonzero((values[:-1] * values[1:] <= 0.0) & torch.isfinite(values[:-1] * values[1:])).reshape(-1)
    if crossing.numel():
        low, high = g[crossing], g[crossing + 1]
        for _ in range(80):
            mid = 0.5 * (low + high)
            same_side = excess(mid) * excess(low) > 0.0
            low, high = torch.where(same_side, mid, low), torch.where(same_side, high, mid)
        lam = 0.5 * (low + high)
    else:
        lam = g[torch.argmin(torch.where(torch.isfinite(values), values.abs(), torch.full_like(values, float("inf"))))].reshape(1)

    b_up, b_half, b_lo = _box_cox(l1mu, lam), _box_cox(lhalf, lam), _box_cox(lu, lam)
    delta = (a3 * (b_up - b_half) - b_half + b_lo) / ((a3 + 1.0) * (b_up - 2.0 * b_half + b_lo))
    delta = torch.where(torch.isfinite(delta), torch.clamp(delta, 0.0, 1.0), torch.full_like(delta, 0.5))
    beta = a2 / (b_up - b_lo)
    alpha = a1 - beta * b_half * (1.0 - 2.0 * delta)
    candidates = torch.stack([alpha, beta, delta, lam], dim=-1)
    if initial_guess is not None:
        return candidates[torch.argmin((lam - _scalar_guess(initial_guess)).abs())]
    return candidates[torch.argmin(_gof_per_candidate(candidates, sample, "VSL"))]
