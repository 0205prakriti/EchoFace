"""Keep the README honest: the files it points to must exist."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _backticked_paths():
    found = set(re.findall(r"`((?:src|notes|notebooks|samples|scripts|tests|\.github)/[\w./-]+)`", README))
    return sorted(p for p in found if not p.endswith("/"))


def test_referenced_files_exist():
    paths = _backticked_paths()
    assert paths, "expected the README to reference repo files"
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"README mentions files that do not exist: {missing}"


def test_markdown_links_to_local_files_exist():
    links = re.findall(r"\]\((?!https?://|#)([^)]+)\)", README)
    missing = [l for l in links if not (ROOT / l).exists()]
    assert not missing, missing


def test_runner_command_in_quick_start_matches_module():
    assert "python -m src.inference.run" in README
    assert (ROOT / "src" / "inference" / "run.py").is_file()
    for flag in ("--photo", "--audio", "--out", "--smooth", "--dry-run"):
        assert flag in (ROOT / "src" / "inference" / "run.py").read_text()


def test_no_unfinished_placeholder_text():
    assert "add yours here" not in README
