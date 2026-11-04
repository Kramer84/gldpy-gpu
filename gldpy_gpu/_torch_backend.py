"""The torch `GLD` class, device selection and the dtype rule.

The numerics are in `_torch_core`, `_torch_solve`, `_torch_stats` and `_torch_fit`. Their helpers stay
importable from here, because `gpu.py` forwards unknown names to this module.
"""
import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ._torch_core import (  # noqa: F401
    _box_cox,
    compute_pdf_quantile,
    compute_quantile,
    FORMULAS,
    _log_density_from_logs,
    _plotting_logs,
    _pow_minus_one,
    _quantile_from_logs,
    _sample_log_uniform,
    validate_gld_params,
)
from ._torch_solve import invert_quantile_newton_bisect, solve_logit_cdf  # noqa: F401
from ._torch_stats import compute_sample_lmoments, compute_sample_percentiles, _require_fit_data  # noqa: F401
from ._torch_fit import (  # noqa: F401
    _lmm_equations,
    _pm_equations,
    _quantile_sample,
    _scalar_guess,
    solve_shape_multistart,
    solve_vsl_lmm_closed_form,
    _solve_vsl_pm,
    _with_support_covering,
)


def _cuda_usable() -> bool:
    """True if CUDA is present AND the installed torch build ships kernels for this GPU.

    A cubin `sm_XY` runs on devices with the same major version and minor >= Y;
    PTX `compute_XY` can be JIT-compiled for any device with capability >= XY.
    """
    if not torch.cuda.is_available():
        return False
    cap = tuple(torch.cuda.get_device_capability())
    for arch in torch.cuda.get_arch_list():
        kind, _, num = arch.partition("_")
        if not num.isdigit():
            continue
        a = (int(num) // 10, int(num) % 10)
        if kind == "sm" and a[0] == cap[0] and a[1] <= cap[1]:
            return True
        if kind == "compute" and a <= cap:
            return True
    return False


def _default_device() -> torch.device:
    if _cuda_usable():
        return torch.device("cuda")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        warnings.warn(
            f"CUDA device with compute capability {cap[0]}.{cap[1]} found, but this PyTorch build "
            f"({torch.__version__}) has no kernels for it (supports {torch.cuda.get_arch_list()}). "
            "Falling back to CPU. For Pascal GPUs (GTX 10xx) install a CUDA 12.6 build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu126",
            UserWarning,
            stacklevel=3,
        )
    return torch.device("cpu")


def _as_dtype(dtype: Union[None, str, torch.dtype]) -> Optional[torch.dtype]:
    """torch.float32/float64 or their names ("float32", "float64"), which also work without torch."""
    if dtype is None:
        return None
    known = {"float32": torch.float32, "float64": torch.float64, torch.float32: torch.float32, torch.float64: torch.float64}
    if dtype not in known:
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}.")
    return known[dtype]


class GLD:
    """PyTorch Generalized Lambda Distribution ('RS', 'FMKL', 'VSL') for CPU and CUDA.

    Every method computes in a single float dtype, chosen in this order: the method's `dtype`
    argument, the constructor's `dtype`, the widest float dtype of the tensor or NumPy inputs, and
    finally float32 for plain Python numbers and lists. `dtype` is torch.float32, torch.float64,
    or their names. Results have that dtype, and nothing is computed in another precision.
    """

    def __init__(
        self,
        param_type: str = "FMKL",
        device: Optional[torch.device] = None,
        dtype: Union[None, str, torch.dtype] = None,
    ):
        if param_type not in ("RS", "FMKL", "VSL"):
            raise ValueError(f"Unknown parameterization '{param_type}'. Use 'RS', 'FMKL', or 'VSL'.")
        self.param_type = param_type
        if device is None:
            device = _default_device()
        self.device = torch.device(device)
        self.dtype = _as_dtype(dtype)

    def _resolve_dtype(self, xs, dtype=None) -> torch.dtype:
        chosen = _as_dtype(dtype)
        if chosen is None:
            chosen = self.dtype
        if chosen is not None:
            return chosen
        found = None
        for x in xs:
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                dt = x.dtype
            elif getattr(getattr(x, "dtype", None), "kind", "") == "f":
                dt = torch.float32 if x.dtype.itemsize <= 4 else torch.float64  # NumPy array or scalar
            else:
                continue
            found = dt if found is None else torch.promote_types(found, dt)
        return found if found is not None else torch.float32

    def _tensors(self, *xs, dtype=None) -> Tuple[torch.Tensor, ...]:
        """Convert inputs to tensors of one common dtype on this device."""
        chosen = self._resolve_dtype(xs, dtype)
        return tuple(torch.as_tensor(x, dtype=chosen, device=self.device) for x in xs)

    def _checked(self, p: torch.Tensor) -> torch.Tensor:
        if not bool(validate_gld_params(p, self.param_type).all().item()):
            raise ValueError("Parameters are not valid.")
        return p

    def check_param(self, param, dtype=None) -> bool:
        p = self._tensors(param, dtype=dtype)[0]
        return bool(validate_gld_params(p, self.param_type).all().item())

    def _probability_and_param(self, y, param, name: str = "y", dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        y_t, p = self._tensors(y, param, dtype=dtype)
        if torch.any((y_t < 0.0) | (y_t > 1.0)):
            raise ValueError(f"{name} must be in range [0, 1].")
        return y_t, self._checked(p)

    def Q(self, y, param, dtype=None) -> torch.Tensor:
        """Quantile function at lower-tail probability `y`."""
        y_t, p = self._probability_and_param(y, param, dtype=dtype)
        return compute_quantile(y_t, p, self.param_type)

    def ISF(self, s, param, dtype=None) -> torch.Tensor:
        """Q(1 - s): the quantile at upper-tail probability `s`, exact where `1 - s` would round to 1."""
        s_t, p = self._probability_and_param(s, param, name="s", dtype=dtype)
        return _quantile_from_logs(torch.log1p(-s_t), torch.log(s_t), p, self.param_type)

    def PDF_Q(self, y, param, dtype=None) -> torch.Tensor:
        y_t, p = self._probability_and_param(y, param, dtype=dtype)
        return compute_pdf_quantile(y_t, p, self.param_type)

    def supp(self, param, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self._checked(self._tensors(param, dtype=dtype)[0])
        zero = torch.zeros(1, dtype=p.dtype, device=p.device)
        inf = torch.full_like(zero, float("inf"))
        lo = _quantile_from_logs(-inf, zero, p, self.param_type)
        hi = _quantile_from_logs(zero, -inf, p, self.param_type)
        return (lo.squeeze(0), hi.squeeze(0)) if p.dim() == 1 else (lo, hi)

    def _logit(self, x, param, xtol: float, max_iter: int, dtype=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t, p = self._tensors(x, param, dtype=dtype)
        self._checked(p)
        return solve_logit_cdf(x_t, p, self.param_type, xtol=xtol, max_iter=max_iter), x_t, p

    def CDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        """P(X <= x). `xtol` is the relative accuracy of the smaller tail probability."""
        return torch.sigmoid(self._logit(x, param, xtol, max_iter, dtype)[0])

    def SF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        """P(X > x), computed directly. Use it for upper tails: `1 - CDF_num` loses everything below eps."""
        return torch.sigmoid(-self._logit(x, param, xtol, max_iter, dtype)[0])

    def log_PDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        """log density at `x`, -inf outside the support. Valid where PDF_num itself underflows."""
        t, x_t, p = self._logit(x, param, xtol, max_iter, dtype)
        out = _log_density_from_logs(F.logsigmoid(t), F.logsigmoid(-t), p, self.param_type)
        lo, hi = self.supp(p)
        outside = (x_t < lo) | (x_t > hi)
        return torch.where(outside, torch.full_like(out, -float("inf")), out)

    def PDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        return torch.exp(self.log_PDF_num(x, param, xtol=xtol, max_iter=max_iter, dtype=dtype))

    def rand(
        self,
        param,
        size: Tuple[int, ...] = (1,),
        generator: Optional[torch.Generator] = None,
        dtype=None,
    ) -> torch.Tensor:
        """Inverse-transform sampling, with both tails resolved in any precision (`_sample_log_uniform`)."""
        p = self._checked(self._tensors(param, dtype=dtype)[0])
        if isinstance(size, int):
            size = (size,)
        log_y, log_s = _sample_log_uniform(tuple(size), p.dtype, p.device, generator)
        return _quantile_from_logs(log_y, log_s, p, self.param_type)

    def GoF_Q_metric(self, data, param, dtype=None) -> torch.Tensor:
        d, p = self._tensors(data, param, dtype=dtype)
        self._checked(p)
        x_sorted, _ = torch.sort(d.reshape(-1))
        n = x_sorted.numel()
        log_y, log_s = _plotting_logs(torch.arange(n, device=p.device), n, p.dtype)
        return torch.mean((x_sorted - _quantile_from_logs(log_y, log_s, p, self.param_type)) ** 2)

    def fit_LMM(self, data, initial_guess=None, n_grid: int = 128, refine_steps: int = 40, dtype=None) -> torch.Tensor:
        """L-moment estimation: closed form for VSL, multistart solve for RS/FMKL.

        Without `initial_guess`, the root closest to the data wins (see `solve_shape_multistart`).
        """
        d = _require_fit_data(self._tensors(data, dtype=dtype)[0])
        stats = tuple(v.cpu() for v in compute_sample_lmoments(d))
        sample = _quantile_sample(d)
        if self.param_type == "VSL":
            guess = None if initial_guess is None else _scalar_guess(initial_guess)
            return solve_vsl_lmm_closed_form(sample, *stats, initial_guess=guess).to(d.device)
        residual_fn, param_fn = _lmm_equations(self.param_type, *stats)
        return solve_shape_multistart(
            residual_fn, param_fn, sample, self.param_type, grid_min=-0.95, grid_max=4.0,
            n_grid=n_grid, refine_steps=refine_steps, initial_guess=initial_guess,
        ).to(d.device)

    def fit_PM(self, data, initial_guess=None, u: float = 0.1, n_grid: int = 128, refine_steps: int = 40, dtype=None) -> torch.Tensor:
        """Percentile-method estimation. Without `initial_guess`, the root closest to the data wins."""
        d = _require_fit_data(self._tensors(data, dtype=dtype)[0])
        stats = tuple(v.cpu() for v in compute_sample_percentiles(d, u))
        sample = _quantile_sample(d)
        if self.param_type == "VSL":
            return _solve_vsl_pm(sample, u, stats, n_grid, initial_guess=initial_guess).to(d.device)
        residual_fn, param_fn = _pm_equations(self.param_type, u, *stats)
        return solve_shape_multistart(
            residual_fn, param_fn, sample, self.param_type, grid_min=-2.0, grid_max=4.0,
            n_grid=n_grid, refine_steps=refine_steps, initial_guess=initial_guess,
        ).to(d.device)

    def correct_supp(self, data, param, eps: float = 1e-4, dtype=None) -> torch.Tensor:
        """Shift and rescale `param` so Q(eps) and Q(1 - eps) cover the data's range.

        Q is affine in location and scale, so the two end conditions have an exact solution.
        Returns new parameters and leaves `param` untouched.
        """
        d, p = self._tensors(data, param, dtype=dtype)
        self._checked(p)
        d = d.reshape(-1)
        log_low = torch.as_tensor([math.log(eps), math.log1p(-eps)], dtype=p.dtype, device=p.device)
        log_high = log_low.flip(0)
        q_lo, q_hi = _quantile_from_logs(log_low, log_high, p, self.param_type).unbind(0)
        lo = torch.minimum(d.min(), q_lo)
        hi = torch.maximum(d.max(), q_hi)
        k = (hi - lo) / (q_hi - q_lo)
        new_scale = p[1] * k if self.param_type == "VSL" else p[1] / k
        return torch.stack([lo - (q_lo - p[0]) * k, new_scale, p[2], p[3]])

    def fit_ML(self, data, init_method: str = "LMM", steps: int = 300, lr: float = 0.02, xtol: float = 1e-6, initial_guess=None, dtype=None) -> torch.Tensor:
        """Maximum likelihood: Adam with implicit differentiation through the CDF solve.

        Runs on standardized data, so `lr` means the same in any unit and a large offset does not
        eat float32 precision. A point outside the support has zero likelihood, so after each step
        location and scale move just enough to cover the data (`_with_support_covering`), which
        leaves the shape parameters free to keep moving. A step that makes the parameters invalid
        is undone and `lr` halved.
        """
        if init_method not in ("LMM", "PM"):
            raise ValueError("init_method must be 'LMM' or 'PM'.")
        d = _require_fit_data(self._tensors(data, dtype=dtype)[0])
        ptype = self.param_type
        init = self.fit_LMM(d, initial_guess) if init_method == "LMM" else self.fit_PM(d, initial_guess)

        center, spread = d.median(), d.std()
        z = (d - center) / spread

        def to_standard(p):
            scale = p[1] / spread if ptype == "VSL" else p[1] * spread
            return torch.stack([(p[0] - center) / spread, scale, p[2], p[3]])

        def from_standard(p):
            scale = p[1] * spread if ptype == "VSL" else p[1] / spread
            return torch.stack([p[0] * spread + center, scale, p[2], p[3]])

        z_min, z_max = z.min(), z.max()
        start = _with_support_covering(to_standard(init), z_min, z_max, ptype)

        param = start.clone().requires_grad_(True)
        best_param, best_nll = start.clone(), float("inf")
        optimizer = torch.optim.Adam([param], lr=lr)
        t_prev = None

        for _ in range(steps):
            optimizer.zero_grad()
            nll = None
            if validate_gld_params(param.detach(), ptype):
                with torch.no_grad():
                    t = solve_logit_cdf(z, param.detach(), ptype, xtol=xtol, t0=t_prev)
                if torch.isfinite(t).all():
                    t_prev = t
                    log_y, log_s = F.logsigmoid(t), F.logsigmoid(-t)
                    # Implicit function theorem: dt/dtheta = -(dQ/dtheta) * dt/dx
                    dt_dx = torch.exp(_log_density_from_logs(log_y, log_s, param.detach(), ptype, interior=True) - log_y - log_s)
                    t_diff = t - (_quantile_from_logs(log_y, log_s, param, ptype) - z) * dt_dx
                    nll = -_log_density_from_logs(F.logsigmoid(t_diff), F.logsigmoid(-t_diff), param, ptype, interior=True).mean()

            if nll is None or not torch.isfinite(nll):
                lr *= 0.5
                if lr < 1e-5:
                    break
                param = best_param.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([param], lr=lr)
                continue

            if nll.item() < best_nll:
                best_nll, best_param = nll.item(), param.detach().clone()
            nll.backward()
            optimizer.step()
            with torch.no_grad():
                if ptype == "VSL":
                    param[2].clamp_(0.0, 1.0)
                if validate_gld_params(param, ptype):
                    param.copy_(_with_support_covering(param, z_min, z_max, ptype))

        return from_standard(best_param)
