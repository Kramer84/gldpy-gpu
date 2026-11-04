"""`pytest --changed` runs what the edit can affect, and what testmon cannot see."""
import pathlib
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("testmon")  # dev tool, for Python 3.10+

ROOT = pathlib.Path(__file__).resolve().parent.parent

SOURCE = "def one():\n    return 1\n\ndef two():\n    return 2\n"
TEST_ONE = "from pkg import one\n\ndef test_one():\n    assert one() == 1\n"
TEST_TWO = "from pkg import two\n\ndef test_two():\n    assert two() == 2\n"
# Stands for the real file: the code it runs is in a subprocess, out of testmon's sight.
TEST_SUBPROCESS = "def test_in_subprocess():\n    pass\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pkg.py").write_text(SOURCE)
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\naddopts = -m 'not gpu_speed'\nmarkers = gpu_speed: x\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    shutil.copy(ROOT / "tests" / "conftest.py", tests)
    (tests / "test_one.py").write_text(TEST_ONE)
    (tests / "test_two.py").write_text(TEST_TWO)
    (tests / "test_optional_torch.py").write_text(TEST_SUBPROCESS)
    return tmp_path


def changed(project) -> set:
    """Run `pytest --changed` on the project, return the names of the tests that ran."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "--changed", "-v", "-p", "no:cacheprovider"],
        cwd=project, capture_output=True, text=True, env={"PYTHONPATH": str(project), "PATH": ""},
    )
    assert res.returncode == 0, res.stdout + res.stderr
    return {line.split("::")[1].split()[0] for line in res.stdout.splitlines() if "PASSED" in line}


def test_only_tests_of_the_edited_function_run(project):
    assert changed(project) == {"test_one", "test_two", "test_in_subprocess"}
    (project / "pkg.py").write_text(SOURCE.replace("return 2", "return 1 + 1"))
    assert changed(project) == {"test_two", "test_in_subprocess"}


def test_without_an_edit_only_the_subprocess_tests_run(project):
    changed(project)
    assert changed(project) == {"test_in_subprocess"}
