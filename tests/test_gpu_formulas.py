"""The shared formulas: the NumPy and torch adapters must give the same numbers from the same code."""
import numpy as np
import pytest
import torch

from gldpy_gpu import _torch_backend as backend
from gldpy_gpu import _torch_core as core
from gldpy_gpu import _torch_fit as fit
from gldpy_gpu import _torch_solve as solve
from gldpy_gpu import _torch_stats as stats
from gldpy_gpu._formulas import NUMPY

TORCH = backend.FORMULAS
CASES = [
    ("FMKL", [0.0, 1.0, 0.2, 0.3]),
    ("FMKL", [1.0, 2.0, -0.2, -0.1]),
    ("FMKL", [0.0, 1.0, 0.0, 0.0]),
    ("FMKL", [0.0, 1.0, 0.0, 0.5]),
    ("FMKL", [0.0, 1.0, 1e-9, 1e-9]),
    ("FMKL", [0.0, 1.0, 2.0, 2.0]),
    ("RS", [0.0, 0.2, 0.1, 0.15]),
    ("RS", [0.0, -0.2, -0.1, -0.15]),
    ("RS", [0.0, -1.0, -0.5, 2.0]),
    ("RS", [0.0, 1.0, 0.0, 1.0]),
    ("RS", [0.0, 1.0, 1.0, 0.0]),
    ("VSL", [0.0, 1.0, 0.4, 0.3]),
    ("VSL", [0.0, 1.0, 0.4, -0.2]),
    ("VSL", [0.0, 1.0, 0.4, 0.0]),
    ("VSL", [0.0, 1.0, 1.0, 0.0]),
    ("VSL", [0.0, 1.0, 1.0, -1.0]),
    ("VSL", [0.0, 1.0, 0.0, 0.5]),
]
IDS = [f"{t}{p[1:]}" for t, p in CASES]
Y = np.r_[0.0, np.logspace(-300, -1, 40), np.linspace(0.11, 0.89, 40), 1.0 - np.logspace(-1, -15, 20), 1.0]


def logs():
    with np.errstate(divide="ignore"):
        return np.log(Y), np.log1p(-Y)


def compare(np_out, torch_out):
    np.testing.assert_allclose(np.asarray(np_out, dtype=float), torch_out.numpy(), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
def test_quantile_agrees(ptype, p):
    log_y, log_s = logs()
    with np.errstate(all="ignore"):
        out = NUMPY.quantile_from_logs(log_y, log_s, np.array(p), ptype)
    compare(out, TORCH.quantile_from_logs(torch.as_tensor(log_y), torch.as_tensor(log_s), torch.as_tensor(p, dtype=torch.float64), ptype))


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
def test_log_density_agrees(ptype, p):
    log_y, log_s = logs()
    with np.errstate(all="ignore"):
        out = NUMPY.log_density_from_logs(log_y, log_s, np.array(p), ptype)
    compare(out, TORCH.log_density_from_logs(torch.as_tensor(log_y), torch.as_tensor(log_s), torch.as_tensor(p, dtype=torch.float64), ptype))


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
def test_density_from_y_agrees(ptype, p):
    with np.errstate(all="ignore"):
        out = NUMPY.density_from_y(Y, np.array(p), ptype)
    compare(out, TORCH.density_from_y(torch.as_tensor(Y), torch.as_tensor(p, dtype=torch.float64), ptype))


@pytest.mark.parametrize("lam", [0.0, 1e-12, -1e-6, 0.3, -0.7])
def test_box_cox_agrees_and_has_the_right_value(lam):
    log_u = np.log(np.array([1e-300, 0.25, 0.75, 1.0]))
    out = NUMPY.box_cox(log_u, np.float64(lam))
    compare(out, TORCH.box_cox(torch.as_tensor(log_u), torch.tensor(lam, dtype=torch.float64)))
    truth = log_u if lam == 0 else np.expm1(lam * log_u) / lam
    np.testing.assert_allclose(out, truth, rtol=1e-14)


def test_lm_terms_agree_and_are_regular_at_zero():
    for lam in (0.0, 1e-9, 0.2, -0.5, 3.0):
        compare(NUMPY.fmkl_lm_terms(np.float64(lam)), torch.stack(TORCH.fmkl_lm_terms(torch.tensor(lam, dtype=torch.float64))))
    assert all(np.isfinite(v) for v in NUMPY.fmkl_lm_terms(np.float64(0.0)))


def test_scalar_inputs_give_scalars_in_the_numpy_adapter():
    out = NUMPY.quantile_from_logs(np.log(0.5), np.log(0.5), np.array([0.0, 1.0, 0.2, 0.3]), "FMKL")
    assert np.ndim(out) == 0


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError):
        NUMPY.quantile_from_logs(np.log(0.5), np.log(0.5), np.array([0.0, 1.0, 0.2, 0.3]), "nope")
    with pytest.raises(ValueError):
        NUMPY.log_density_from_logs(np.log(0.5), np.log(0.5), np.array([0.0, 1.0, 0.2, 0.3]), "nope")


def test_each_formula_exists_once():
    """The point of the module: neither class keeps its own copy of the arithmetic."""
    import inspect

    import gldpy_gpu._gld_module as np_module

    modules = (np_module, core, solve, stats, fit, backend)
    for source in map(inspect.getsource, modules):
        assert "def _fmkl_lm_terms" not in source and "def _density_terms" not in source
        assert "expm1" not in source.split("class GLD")[0].replace("math.expm1", "")
