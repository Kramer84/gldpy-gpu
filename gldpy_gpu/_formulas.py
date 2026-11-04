"""The GLD's formulas, written once for both array libraries.

`Formulas` holds the arithmetic of the quantile function and the density, in terms of log(y) and
log(1 - y) (see docs/numerics.md). It uses only functions that NumPy and PyTorch share, through the
namespace `xp`, and takes the two things they do not share as arguments. This module must not
import torch, so that the package works without it. `_torch_backend.py` builds the torch adapter.
"""
import math

import numpy as np


class Formulas:
    """The formulas over one array library.

    Parameters
    ----------
    xp : module
        `numpy` or `torch`.
    detach : callable
        Removes an array from automatic differentiation, if the library has it.
    tracks_gradient : callable
        Whether a gradient with respect to this array is being recorded.
    unpack : callable
        The four parameters along the last dimension. Each library has its own fast way, which
        matters because root finding and small batches evaluate these formulas thousands of times.
    """

    def __init__(self, xp, unpack, detach=lambda x: x, tracks_gradient=lambda x: False):
        self.xp = xp
        self.unpack = unpack
        self.detach = detach
        self.tracks_gradient = tracks_gradient

    @staticmethod
    def _no_zero(values) -> bool:
        # Hot on scalars, where root finding spends its time and .all() would cost a ufunc reduction.
        nonzero = values != 0
        return bool(nonzero if nonzero.ndim == 0 else nonzero.all())

    def pow_from_log(self, log_u, lam):
        """u**lam from log(u), with 0**0 = 1 as in NumPy."""
        xp = self.xp
        if self._no_zero(lam):
            return xp.exp(lam * log_u)
        return xp.where(lam == 0, xp.ones_like(lam * 1.0), xp.exp(lam * log_u))

    def pow_minus_one(self, log_u, lam):
        """u**lam - 1 from log(u). RS uses y**a - s**b = (y**a - 1) - (s**b - 1), which avoids subtracting two numbers near 1."""
        xp = self.xp
        if self._no_zero(lam):
            return xp.expm1(lam * log_u)
        return xp.where(lam == 0, xp.zeros_like(lam * 1.0), xp.expm1(lam * log_u))

    def box_cox(self, log_u, lam):
        """(u**lam - 1) / lam from log(u). No cancellation as lam -> 0, and autograd-safe in lam."""
        xp = self.xp
        if not self.tracks_gradient(lam) and self._no_zero(lam):
            return xp.expm1(lam * log_u) / lam  # expm1 keeps every digit; only d/dlam needs the series
        z = lam * log_u
        small = (lam == 0) | (xp.abs(z) < 1e-3)
        z_series = lam * xp.where(xp.isinf(log_u), xp.zeros_like(log_u), log_u)
        series = log_u * (1.0 + z_series / 2.0 + z_series * z_series / 6.0)
        safe_lam = xp.where(lam == 0, xp.ones_like(lam), lam)
        return xp.where(small, series, xp.expm1(xp.where(lam == 0, xp.zeros_like(z), z)) / safe_lam)

    def quantile_from_logs(self, log_y, log_s, param, param_type: str):
        """Q from log(y) and log(s), where s = 1 - y, so neither tail ever forms `1 - y`."""
        xp = self.xp
        p1, p2, p3, p4 = self.unpack(param)
        if param_type == "RS":
            return p1 + (self.pow_minus_one(log_y, p3) - self.pow_minus_one(log_s, p4)) / p2
        if param_type == "FMKL":
            return p1 + (self.box_cox(log_y, p3) - self.box_cox(log_s, p4)) / p2
        if param_type == "VSL":
            lower, upper = (1.0 - p3) * self.box_cox(log_y, p4), p3 * self.box_cox(log_s, p4)
            if not (self._no_zero(p3) and self._no_zero(1.0 - p3)):
                # delta = 0 or 1 removes a term entirely, also where its box-cox value is infinite.
                lower = xp.where(p3 == 1, xp.zeros_like(lower), lower)
                upper = xp.where(p3 == 0, xp.zeros_like(upper), upper)
            return p1 + p2 * (lower - upper)
        raise ValueError(f"Unknown parameterization '{param_type}'.")

    def _density_terms(self, param, param_type: str):
        """(c3, c4, e3, e4, log_scale) such that f(y) = exp(log_scale) / |c3 y**e3 + c4 (1 - y)**e4|."""
        xp = self.xp
        p1, p2, p3, p4 = self.unpack(param)
        if param_type == "RS":
            return p3, p4, p3 - 1.0, p4 - 1.0, xp.log(xp.abs(p2))
        if param_type == "FMKL":
            return xp.ones_like(p3), xp.ones_like(p4), p3 - 1.0, p4 - 1.0, xp.log(p2)
        if param_type == "VSL":
            return 1.0 - p3, p3, p4 - 1.0, p4 - 1.0, -xp.log(p2)
        raise ValueError(f"Unknown parameterization '{param_type}'.")

    def log_density_from_logs(self, log_y, log_s, param, param_type: str, interior: bool = False):
        """log f(y) = -log Q'(y). Q' = c3 y**e3 + c4 s**e4 is summed after factoring out its larger term.

        `interior=True` promises finite log_y and log_s, which skips the handling of the endpoints.
        """
        xp = self.xp
        c3, c4, e3, e4, log_scale = self._density_terms(param, param_type)
        if interior and self._no_zero(c3) and self._no_zero(c4):
            term3, term4 = e3 * log_y, e4 * log_s
            log_c3, log_c4 = xp.log(xp.abs(c3)), xp.log(xp.abs(c4))
            m = self.detach(xp.maximum(log_c3 + term3, log_c4 + term4))
            return log_scale - m - xp.log(xp.abs(c3 * xp.exp(term3 - m) + c4 * xp.exp(term4 - m)))

        # 0 * inf terms are 0: a term with zero exponent is constant even at y = 0.
        term3 = xp.where(e3 == 0, xp.zeros_like(e3 * 1.0), e3 * log_y)
        term4 = xp.where(e4 == 0, xp.zeros_like(e4 * 1.0), e4 * log_s)
        # `m` is only a scaling constant, so it is detached. log|c| has an infinite gradient at c = 0
        # (VSL delta on its bounds), which would turn into NaN if it were differentiated through max().
        neg_inf = xp.full_like(log_scale, -float("inf"))
        m = self.detach(
            xp.maximum(
                xp.where(c3 == 0, neg_inf, xp.log(xp.abs(c3)) + term3),
                xp.where(c4 == 0, neg_inf, xp.log(xp.abs(c4)) + term4),
            )
        )
        ok = xp.isfinite(m)  # at y = 0 or 1 one term is infinite and decides alone
        m0 = xp.where(ok, m, xp.zeros_like(m))
        limit = 0.5 * math.log(xp.finfo(m.dtype).max)
        exp3 = xp.exp(self._at_most(xp.where(ok, term3 - m0, xp.zeros_like(m0)), limit))
        exp4 = xp.exp(self._at_most(xp.where(ok, term4 - m0, xp.zeros_like(m0)), limit))
        inner = xp.log(xp.abs(c3 * exp3 + c4 * exp4))
        return log_scale - m - xp.where(ok, inner, xp.zeros_like(inner))

    def _at_most(self, values, limit):
        return self.xp.where(values > limit, self.xp.full_like(values, limit), values)

    def density_from_y(self, y, param, param_type: str):
        """f(y) = 1 / Q'(y), directly. Underflows to 0 far out, where the log form does not."""
        xp = self.xp
        c3, c4, e3, e4, log_scale = self._density_terms(param, param_type)
        term3 = c3 * self.pow_from_log(xp.log(y), e3)
        term4 = c4 * self.pow_from_log(xp.log1p(-y), e4)
        if not (self._no_zero(c3) and self._no_zero(c4)):
            # A term with a zero coefficient is left out, also where its power is infinite.
            term3 = xp.where(c3 == 0, xp.zeros_like(term3), term3)
            term4 = xp.where(c4 == 0, xp.zeros_like(term4), term4)
        return xp.exp(log_scale) / xp.abs(term3 + term4)

    @staticmethod
    def fmkl_lm_terms(lam):
        """FMKL L-moment terms in closed form: (-1/(1+l) + 2/(2+l)) / l = 1/((1+l)(2+l)), and so on.

        The 1/l of the textbook sums cancels, so these stay finite and accurate at l = 0. Multiplied
        by l, they are the RS terms.
        """
        g2 = 1.0 / ((1.0 + lam) * (2.0 + lam))
        g3 = (lam - 1.0) * g2 / (3.0 + lam)
        g4 = (lam - 1.0) * (lam - 2.0) * g2 / ((3.0 + lam) * (4.0 + lam))
        return g2, g3, g4


def _unpack_numpy(param):
    if param.ndim == 1:
        return param  # four scalars, by plain iteration
    return param[..., 0], param[..., 1], param[..., 2], param[..., 3]


NUMPY = Formulas(np, unpack=_unpack_numpy)
