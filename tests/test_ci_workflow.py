"""Sanity checks for the CI workflow file (no YAML parser needed)."""
import pathlib

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "tests.yml"


def _text():
    assert WORKFLOW.is_file(), "CI workflow is missing"
    return WORKFLOW.read_text()


def test_runs_on_every_push():
    text = _text()
    assert "\non:\n" in text
    assert "\n  push:\n" in text


def test_runs_pytest():
    assert "python -m pytest" in _text()


def test_uses_cpu_only_torch_wheels():
    text = _text()
    assert "whl/cpu" in text
    assert "cuda" not in text.lower()


def test_no_tabs():
    assert "\t" not in _text()
