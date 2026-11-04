"""matplotlib is optional: only plots need it, and a fit that would plot still returns its result."""
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PARAM = [0.0, 1.0, 0.2, 0.3]

BLOCK_MATPLOTLIB = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
sys.meta_path.insert(0, _Block())
"""


def run_without_matplotlib(code: str) -> str:
    """Run `code` in a fresh interpreter where `import matplotlib` fails, return stdout."""
    script = BLOCK_MATPLOTLIB + textwrap.dedent(code)
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def test_matplotlib_is_an_extra_not_a_requirement():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any(d.startswith("matplotlib") for d in meta["dependencies"])
    assert any(d.startswith("matplotlib") for d in meta["optional-dependencies"]["plot"])


def test_import_does_not_load_matplotlib():
    code = "import sys, gldpy_gpu; assert not any(m.split('.')[0] == 'matplotlib' for m in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_computing_needs_no_matplotlib_and_does_not_warn():
    out = run_without_matplotlib(f"""
        import warnings
        warnings.simplefilter("error", UserWarning)
        import gldpy_gpu
        g = gldpy_gpu.GLD("FMKL")
        x = g.rand({PARAM}, size=200, random_state=1)
        print(len(g.fit_LMM(x, [0.2, 0.3], disp_optimizer=False, disp_fit=False, test_gof=False)))
    """)
    assert out == "4"


def test_default_fit_warns_and_still_returns_the_parameters():
    """disp_fit=True is the default, so this is what a plain `fit_LMM(data)` does."""
    out = run_without_matplotlib(f"""
        import warnings
        import gldpy_gpu
        g = gldpy_gpu.GLD("FMKL")
        x = g.rand({PARAM}, size=200, random_state=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = g.fit_LMM(x, [0.2, 0.3], disp_optimizer=False)
        assert any("gldpy-gpu[plot]" in str(i.message) for i in w), [str(i.message) for i in w]
        print("ok", g.check_param(fit))
    """)
    assert "Parameters:" in out  # the printed report does not need matplotlib
    assert out.splitlines()[-1] == "ok True"


def test_default_fit_curve_warns_and_finishes():
    out = run_without_matplotlib(f"""
        import warnings
        import numpy as np
        import gldpy_gpu
        g = gldpy_gpu.GLD("FMKL")
        x = np.linspace(-1, 1, 30)
        y = g.PDF_num(x, {PARAM})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g.fit_curve(x, y, [0.2, 0.3], optimization_phase=False, random_state=1, disp_optimizer=False)
        assert any("gldpy-gpu[plot]" in str(i.message) for i in w), [str(i.message) for i in w]
        print("ok")
    """)
    assert out.splitlines()[-1] == "ok"


@pytest.mark.parametrize("call", [f"plot_cdf({PARAM})", f"plot_pdf({PARAM})", f"plot_fitting(np.arange(9.0), {PARAM})"])
def test_asking_for_a_plot_says_how_to_get_matplotlib(call):
    run_without_matplotlib(f"""
        import numpy as np
        import gldpy_gpu
        try:
            gldpy_gpu.GLD("FMKL").{call}
        except ImportError as error:
            assert "gldpy-gpu[plot]" in str(error), error
        else:
            raise AssertionError("no ImportError")
    """)


# With matplotlib: the plots show the numbers they are given.


@pytest.fixture
def plt():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    yield pyplot
    pyplot.close("all")


@pytest.fixture
def gld():
    from gldpy_gpu import GLD

    return GLD("FMKL")


def test_plot_cdf_draws_each_distribution_and_the_data(plt, gld):
    data = gld.rand(PARAM, size=50, random_state=1)
    gld.plot_cdf([PARAM, [0.0, 2.0, 0.2, 0.3]], data=data, n_points=11)
    ax = plt.gca()
    assert ax.get_title() == "CDF"
    empirical, first, second = ax.get_lines()
    np.testing.assert_array_equal(empirical.get_xdata(), np.sort(data))
    y = np.linspace(0.01, 0.99, 11)
    np.testing.assert_allclose(first.get_xdata(), gld.Q(y, PARAM))
    np.testing.assert_allclose(second.get_ydata(), y)


def test_plot_pdf_draws_the_density_on_the_quantile_grid(plt, gld):
    gld.plot_pdf(PARAM, n_points=11)
    ax = plt.gca()
    assert ax.get_title() == "PDF"
    line = ax.get_lines()[0]
    y = np.linspace(0.01, 0.99, 11)
    np.testing.assert_allclose(line.get_xdata(), gld.Q(y, PARAM))
    np.testing.assert_allclose(line.get_ydata(), gld.PDF_Q(y, PARAM))


def test_plot_fitting_has_pdf_cdf_and_qq_panels(plt, gld):
    data = gld.rand(PARAM, size=50, random_state=1)
    gld.plot_fitting(data, PARAM)
    axes = plt.gcf().axes
    assert [a.get_title() for a in axes] == ["PDF", "CDF", "Q-Q-plot"]
    np.testing.assert_array_equal(axes[2].get_lines()[0].get_ydata(), np.sort(data))


def test_default_fit_draws_the_fit_panels(plt, gld, capsys):
    data = gld.rand(PARAM, size=200, random_state=1)
    gld.fit_LMM(data, [0.2, 0.3], disp_optimizer=False)
    assert [a.get_title() for a in plt.gcf().axes] == ["PDF", "CDF", "Q-Q-plot"]


def test_fit_curve_draws_both_panels(plt, gld, capsys):
    x = np.linspace(-1, 1, 30)
    gld.fit_curve(x, gld.PDF_num(x, PARAM), [0.2, 0.3], optimization_phase=False, random_state=1, disp_optimizer=False)
    assert [a.get_title() for a in plt.gcf().axes] == ["Simulation phase", "Result of curve fitting"]
