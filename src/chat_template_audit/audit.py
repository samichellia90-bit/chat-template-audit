"""Static auditor for the "untrusted chat template rendered with a non-sandboxed
Jinja2 environment" code smell.

Modern chat models ship a ``chat_template`` (a Jinja2 string) inside their
``tokenizer_config.json`` or GGUF metadata. A program that renders that template
is compiling attacker-controlled Jinja source whenever the model comes from an
untrusted source, so it MUST use a Jinja sandbox. A plain ``Environment`` or
``Template`` is server-side template injection: ``{{ cycler.__init__.__globals__
}}`` reaches ``os`` and executes code. This is why jinja2 ships
``SandboxedEnvironment`` / ``ImmutableSandboxedEnvironment`` and why libraries
such as transformers pin ``jinja2>=3.1.6`` (the release that closes the known
sandbox escapes).

This tool is a heuristic linter, not a proof of exploitability. It reports, for
each Jinja render site, whether its file builds a Jinja sandbox and whether the
file also reads a model-supplied template. A render site that is NOT sandboxed in
a file that reads a ``chat_template`` is marked ``REVIEW`` for a human to confirm.

Confirm every finding by hand: reachability decides it. For example guidance
renders ``chat_template`` with a plain ``Environment`` yet is not exploitable,
because it drops any untrusted template string that is not a known built-in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# A Jinja render site: `.from_string(`, a bare `Environment(` or `Template(`
# (not when preceded by an identifier char, to skip `class Environment(Enum)` et al.
# only partially -- false positives on such lines are classified, not silenced).
_RENDER = re.compile(r"\.from_string\(|(?<![A-Za-z0-9_.])(?:Environment|Template)\(")
_SANDBOX = re.compile(r"ImmutableSandboxedEnvironment|SandboxedEnvironment")
_UNTRUSTED = re.compile(r"chat_template|tokenizer_config|template_str|\.metadata\[")
_SKIP_PATH = re.compile(r"(?:^|[\\/])(?:tests?|examples?|docs?)[\\/]|_test\.py$|(?:^|[\\/])test_[^\\/]*\.py$")


@dataclass(frozen=True)
class Finding:
    verdict: str  # "REVIEW" | "sandbox" | "render"
    path: str
    line: int
    code: str
    file_has_sandbox: bool
    file_reads_untrusted_template: bool


def scan_file(path: Path, root: Path) -> list[Finding]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    file_has_sandbox = any(_SANDBOX.search(ln) for ln in lines)
    file_reads_untrusted = any(_UNTRUSTED.search(ln) for ln in lines)
    out: list[Finding] = []
    for i, ln in enumerate(lines, start=1):
        if not _RENDER.search(ln) or _SANDBOX.search(ln):
            continue
        if file_has_sandbox:
            verdict = "sandbox"
        elif file_reads_untrusted:
            verdict = "REVIEW"
        else:
            verdict = "render"
        out.append(
            Finding(
                verdict=verdict,
                path=str(path.relative_to(root)).replace("\\", "/"),
                line=i,
                code=ln.strip()[:160],
                file_has_sandbox=file_has_sandbox,
                file_reads_untrusted_template=file_reads_untrusted,
            )
        )
    return out


def scan_tree(root: Path) -> list[Finding]:
    """Scan a directory tree; a single .py file is scanned directly."""
    if root.is_file():
        return scan_file(root, root.parent)
    findings: list[Finding] = []
    for f in sorted(root.rglob("*.py")):
        if _SKIP_PATH.search(str(f.relative_to(root))):
            continue
        findings.extend(scan_file(f, root))
    return findings


def _format(findings: list[Finding]) -> str:
    order = {"REVIEW": 0, "sandbox": 1, "render": 2}
    rows = []
    for f in sorted(findings, key=lambda x: (order.get(x.verdict, 3), x.path, x.line)):
        tags = []
        if f.file_has_sandbox:
            tags.append("sandboxed")
        if f.file_reads_untrusted_template:
            tags.append("untrusted-template-in-file")
        rows.append(f"  [{f.verdict}] {f.path}:{f.line}  {f.code}   {{{','.join(tags)}}}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="repository directories or .py files to scan")
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = parser.parse_args(argv)

    all_findings: list[Finding] = []
    sections: list[tuple[str, list[Finding]]] = []
    for p in args.paths:
        root = Path(p)
        if not root.exists():
            print(f"skip (not found): {p}", file=sys.stderr)
            continue
        found = scan_tree(root)
        sections.append((p, found))
        all_findings.extend(found)

    if args.json:
        print(json.dumps([asdict(f) for f in all_findings], indent=2))
    else:
        for p, found in sections:
            print(f"\n=== {p} ===")
            body = _format(found)
            print(body if body else "  (no Jinja render sites found)")

    # exit 1 if anything needs manual review, so CI / pre-commit can gate on it
    return 1 if any(f.verdict == "REVIEW" for f in all_findings) else 0


if __name__ == "__main__":
    sys.exit(main())
