from pathlib import Path

from chat_template_audit.audit import scan_tree

FIXTURES = Path(__file__).parent / "fixtures"


def _verdicts(name):
    return [f.verdict for f in scan_tree(FIXTURES / name)]


def test_plain_environment_on_untrusted_template_is_flagged():
    verdicts = _verdicts("vulnerable_lib")
    assert "REVIEW" in verdicts, "a plain Environment rendering chat_template must be flagged REVIEW"


def test_sandboxed_render_is_not_flagged():
    verdicts = _verdicts("safe_lib")
    assert verdicts, "the render site should still be reported"
    assert "REVIEW" not in verdicts, "a sandboxed render of chat_template must not be flagged REVIEW"
    assert set(verdicts) == {"sandbox"}


def test_single_file_scan():
    findings = scan_tree(FIXTURES / "vulnerable_lib" / "model.py")
    assert any(f.verdict == "REVIEW" for f in findings)
    assert all(f.path == "model.py" for f in findings)
