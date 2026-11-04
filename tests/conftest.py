import importlib.util

import numpy as np
import pytest

# Without torch, only the NumPy-side tests can run (see test_optional_torch.py).
HAS_TORCH = importlib.util.find_spec("torch") is not None
collect_ignore_glob = [] if HAS_TORCH else ["test_gpu_*.py", "test_speed.py", "test_analytic.py"]
torch = None
if HAS_TORCH:
    import torch

# Valid, representative parameters for each parameterization.
PARAMS = {
    "FMKL": [0.0, 1.0, 0.2, 0.3],
    "RS": [0.0, 0.2, 0.1, 0.15],
    "VSL": [0.0, 1.0, 0.4, 0.3],
}


# testmon measures this process only, so it cannot see what these subprocess tests depend on.
ALWAYS_WITH_CHANGED = ("test_optional_torch.py", "test_optional_matplotlib.py")


def pytest_addoption(parser):
    parser.addoption("--changed", action="store_true", help="run only the tests the edits since the last run can affect (pytest-testmon)")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not config.getoption("--changed"):
        return
    if not config.pluginmanager.hasplugin("pytest-testmon"):
        raise pytest.UsageError("--changed needs pytest-testmon (Python 3.10+, in the dev group: `uv sync`)")
    # forceselect, because testmon stops selecting when it sees -m, which addopts always passes.
    config.option.testmon = config.option.testmon_forceselect = True


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    if config.getoption("--changed") and collection_path.name in ALWAYS_WITH_CHANGED:
        return False  # testmon would otherwise skip the file whole
    return None


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(config, items):
    always = [item for item in items if item.path.name in ALWAYS_WITH_CHANGED] if config.getoption("--changed") else []
    result = yield
    items.extend(item for item in always if item not in items)
    return result


def _devices():
    devs = ["cpu"]
    if HAS_TORCH and torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.fixture(params=_devices())
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=sorted(PARAMS))
def ptype(request):
    return request.param


@pytest.fixture
def param(ptype):
    return np.array(PARAMS[ptype])


@pytest.fixture
def cpu_gld(ptype):
    from gldpy_gpu import GLD

    return GLD(ptype)


@pytest.fixture
def gpu_gld(ptype, device):
    from gldpy_gpu import gpu

    return gpu.GLD(ptype, device=device)


@pytest.fixture
def sample(cpu_gld, param):
    """1000 reproducible draws from the reference CPU implementation."""
    return cpu_gld.rand(param, size=1000, random_state=123)
