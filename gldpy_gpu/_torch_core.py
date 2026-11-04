"""The torch side of Q and the density: parameter validity, the `Formulas` adapter, sampling."""
import math
from typing import Tuple

import torch

from ._formulas import Formulas


def validate_gld_params(param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Vectorized validity check for GLD parameters. Supports shape (..., 4)."""
    if param.shape[-1] != 4:
        raise ValueError("GLD requires 4 parameters along the last dimension.")
    finite_mask = torch.isfinite(param).all(dim=-1)
    p1, p2, p3, p4 = param.unbind(dim=-1)

    if param_type == "FMKL":
        return finite_mask & (p2 > 0)
    if param_type == "VSL":
        return finite_mask & (p2 > 0) & (p3 >= 0.0) & (p3 <= 1.0)
    if param_type == "RS":
        r1 = (p2 < 0) & (p3 <= -1) & (p4 >= 1)
        r2 = (p2 < 0) & (p3 >= 1) & (p4 <= -1)
        r3 = (p2 > 0) & (p3 >= 0) & (p4 >= 0) & ((p3 != 0) | (p4 != 0))
        r4 = (p2 < 0) & (p3 <= 0) & (p4 <= 0) & ((p3 != 0) | (p4 != 0))

        # Compared in log space: the powers themselves overflow float32 for lambda ~ 30.
        def lhs_le_rhs(neg, pos):
            neg_c = torch.clamp(neg, min=-1.0, max=0.0)
            pos_c = torch.clamp(pos, min=1.0)
            xlogy = torch.special.xlogy
            log_lhs = (
                xlogy(1.0 - neg_c, 1.0 - neg_c)
                + xlogy(pos_c - 1.0, pos_c - 1.0)
                - xlogy(pos_c - neg_c, pos_c - neg_c)
            )
            return log_lhs <= torch.log(-neg_c / pos_c)

        r5 = (p2 < 0) & (p3 <= 0) & (p3 >= -1) & (p4 >= 1) & lhs_le_rhs(p3, p4)
        r6 = (p2 < 0) & (p3 >= 1) & (p4 >= -1) & (p4 <= 0) & lhs_le_rhs(p4, p3)

        return finite_mask & (r1 | r2 | r3 | r4 | r5 | r6)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


# The arithmetic of Q and the density is shared with the NumPy class, in _formulas.py.
FORMULAS = Formulas(
    torch,
    unpack=lambda param: param.unbind(dim=-1),
    detach=lambda x: x.detach(),
    tracks_gradient=lambda x: torch.is_grad_enabled() and x.requires_grad,
)
_pow_minus_one = FORMULAS.pow_minus_one
_box_cox = FORMULAS.box_cox
_quantile_from_logs = FORMULAS.quantile_from_logs
_log_density_from_logs = FORMULAS.log_density_from_logs


def compute_quantile(y: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Evaluate quantile function Q(y) with full broadcasting over y and param."""
    return _quantile_from_logs(torch.log(y), torch.log1p(-y), param, param_type)


def compute_pdf_quantile(y: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Evaluate density quantile function f_Q(y) = 1 / Q'(y). Underflows to 0 far out; see `log_PDF_num`."""
    return FORMULAS.density_from_y(y, param, param_type)


def _plotting_logs(k: torch.Tensor, n: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """log and log1p(-) of (k + 0.5) / n for integer k.

    y and 1 - y come from k and n - k separately, so the upper positions keep their relative
    accuracy in any precision, instead of being the difference of two numbers close to 1.
    """
    log_n = math.log(n)
    return torch.log(k.to(dtype) + 0.5) - log_n, torch.log((n - k).to(dtype) - 0.5) - log_n


def _sample_log_uniform(size: Tuple[int, ...], dtype: torch.dtype, device: torch.device, generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """log(y) and log(1 - y) for y uniform on (0, 1), with relative accuracy in both tails.

    A plain uniform has a fixed step (2**-24 in float32), so it can be exactly 0, which gives
    Q = -inf, and it stops the upper tail at Q(1 - 2**-24). Here the distance to the nearer end is
    refined instead: a value below `cut` is uniform on (0, cut) given that it is below `cut`, so
    it is multiplied by a fresh uniform, as many times as it takes. Each round touches a
    fraction `cut` of the previous one, so this costs almost nothing.
    """
    u = torch.rand(size, generator=generator, dtype=dtype, device=device)
    upper = u >= 0.5
    tail = torch.where(upper, 1.0 - u, u)
    cut = 2.0**-12
    log_cut = math.log(cut)
    log_tail = torch.log(torch.clamp(tail, min=cut))
    log_other = torch.log1p(-tail)

    deep = torch.nonzero(tail < cut, as_tuple=True)
    pending = deep
    while pending[0].numel():
        v = torch.rand(pending[0].numel(), generator=generator, dtype=dtype, device=device)
        again = v < cut
        log_tail[pending] += torch.where(again, torch.full_like(v, log_cut), torch.log(torch.clamp(v, min=cut)))
        pending = tuple(index[again] for index in pending)
    if deep[0].numel():
        log_other[deep] = torch.log1p(-torch.exp(log_tail[deep]))

    return torch.where(upper, log_other, log_tail), torch.where(upper, log_tail, log_other)
