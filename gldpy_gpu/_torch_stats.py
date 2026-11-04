"""Sample statistics the fits start from: L-moments and percentiles."""
from typing import Optional, Tuple

import torch


def compute_sample_lmoments(data: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, ...]:
    """Sample (l1, l2, t3 = l3/l2, t4 = l4/l2) in O(N log N), computed in `dtype` (default: the data's).

    The unbiased weights are built as products of ratios, each in [0, 1], so nothing overflows, and
    applied to centered data, which cancels the mean because the weights of l2, l3 and l4 sum to 0.
    """
    if dtype is None:
        dtype = data.dtype if data.is_floating_point() else torch.float32
    x, _ = torch.sort(data.reshape(-1).to(dtype))
    n = x.numel()
    if n < 4:
        raise ValueError("At least 4 data points are required to compute 4 L-moments.")

    k = torch.arange(n, dtype=dtype, device=x.device)
    m = (n - 1) - k
    # k(k-1)/(n(n-1)), k m/(n(n-1)) and m(m-1)/(n(n-1))
    kk = (k / n) * ((k - 1.0) / (n - 1))
    km = (k / n) * (m / (n - 1))
    mm = (m / n) * ((m - 1.0) / (n - 1))

    l1 = torch.mean(x)
    c = x - l1
    l2 = torch.sum((k - m) / n / (n - 1) * c)
    l3 = torch.sum((kk - 4.0 * km + mm) / (n - 2) * c)
    w4 = kk * ((k - 2.0 - 9.0 * m) / (n - 2)) + km * (9.0 * (m - 1.0) / (n - 2)) - mm * ((m - 2.0) / (n - 2))
    l4 = torch.sum(w4 / (n - 3) * c)
    return l1, l2, l3 / l2, l4 / l2


def compute_sample_percentiles(data: torch.Tensor, u: float = 0.1, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, ...]:
    """Sample percentile statistics, interpolated like numpy.quantile, in `dtype` (default: the data's).

    Written out because torch.quantile refuses inputs above 2**24 elements.
    """
    if not (0.0 < u < 0.25):
        raise ValueError("u must be in the open interval (0, 0.25).")
    if dtype is None:
        dtype = data.dtype if data.is_floating_point() else torch.float32
    x, _ = torch.sort(data.reshape(-1).to(dtype))
    last = x.numel() - 1

    def at(q: float) -> torch.Tensor:
        position = q * last  # a Python float: exact as an index in every precision
        below = int(position)
        above = min(below + 1, last)
        return x[below] + (x[above] - x[below]) * (position - below)

    qu, q25, q50, q75, q1mu = at(u), at(0.25), at(0.5), at(0.75), at(1.0 - u)
    a2 = q1mu - qu
    return q50, a2, (q50 - qu) / (q1mu - q50), (q75 - q25) / a2


def _require_fit_data(data: torch.Tensor) -> torch.Tensor:
    d = data.reshape(-1)
    if d.numel() < 4:
        raise ValueError("At least 4 data points are required.")
    if not torch.isfinite(d).all():
        raise ValueError("Data contains NaN or infinite values.")
    if d.min() == d.max():
        raise ValueError("All data points are equal: a GLD cannot be fitted.")
    return d
