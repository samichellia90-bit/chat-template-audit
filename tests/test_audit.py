import json
import os
import textwrap
from pathlib import Path

import pytest

from chat_template_audit import CHECK, PIN, REVIEW, main, render_sarif, scan

FIXTURES = Path(__file__).parent / "fixtures"


def project(root: Path, files: dict[str, str]) -> Path:
    for rel, src in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(src).lstrip("\n"), encoding="utf-8")
    return root


def lines(report, verdict):
    return [(Path(f.path).name, f.line) for f in report.findings if f.verdict == verdict]


# ---------------------------------------------------------------- what must be flagged


def test_plain_environment_on_tokenizer_template(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2 import Environment

        def render(tokenizer, messages):
            return Environment().from_string(tokenizer.chat_template).render(messages=messages)
    """
            },
        )
    )
    assert lines(r, REVIEW) == [("m.py", 4)]


def test_module_qualified_template_class(tmp_path):
    # v0.1's regex missed `jinja2.Template(` because of the `jinja2.` prefix
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        import jinja2

        def render(cfg):
            return jinja2.Template(cfg["chat_template"]).render()
    """
            },
        )
    )
    assert lines(r, REVIEW) == [("m.py", 4)]


def test_aliases_star_import_and_native_environment(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "a.py": """
            import jinja2 as j
            def f(chat_template):
                return j.Environment().from_string(chat_template)
        """,
                "b.py": """
            from jinja2 import Environment as E
            def f(chat_template):
                return E(autoescape=False).from_string(chat_template)
        """,
                "c.py": """
            from jinja2 import *
            def f(chat_template):
                return Template(chat_template)
        """,
                "d.py": """
            from jinja2.nativetypes import NativeEnvironment
            def f(chat_template):
                return NativeEnvironment().from_string(chat_template)
        """,
            },
        )
    )
    assert sorted(lines(r, REVIEW)) == [("a.py", 3), ("b.py", 3), ("c.py", 3), ("d.py", 3)]


def test_multiline_call_reports_its_first_line(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        import jinja2

        def build(tokenizer):
            env = jinja2.Environment(
                trim_blocks=True,
                lstrip_blocks=True,
            )
            return env.from_string(
                tokenizer.chat_template
            )
    """
            },
        )
    )
    assert lines(r, REVIEW) == [("m.py", 8)]


def test_plain_environment_next_to_a_sandbox_in_the_same_file(tmp_path):
    # v0.1 treated the whole file as sandboxed as soon as it built one sandbox
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2 import Environment
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        SAFE = ImmutableSandboxedEnvironment()

        def preview(chat_template):
            return SAFE.from_string(chat_template)

        def render(chat_template):
            return Environment().from_string(chat_template)
    """
            },
        )
    )
    assert lines(r, REVIEW) == [("m.py", 10)]
    assert lines(r, "SANDBOX") == [("m.py", 7)]


def test_gguf_metadata_flows_through_a_class_in_another_module(tmp_path):
    # the shape of CVE-2024-34359: the template crosses a call and a `self` attribute
    r = scan(
        project(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/formatter.py": """
            import jinja2

            class ChatFormatter:
                def __init__(self, template, eos_token):
                    self.template = template
                    self._environment = jinja2.Environment(
                        loader=jinja2.BaseLoader()
                    ).from_string(self.template)
        """,
                "pkg/model.py": """
            from . import formatter

            class Model:
                def __init__(self, metadata):
                    self.metadata = metadata
                    template = self.metadata["tokenizer.chat_template"]
                    self.handler = formatter.ChatFormatter(template=template, eos_token="</s>")
        """,
            },
        )
    )
    [f] = [f for f in r.findings if f.verdict == REVIEW]
    assert (Path(f.path).name, f.line) == ("formatter.py", 6)
    assert Path(f.source_path).name == "model.py" and f.source_line == 6
    assert "tokenizer.chat_template" in f.source_code


def test_template_read_from_a_chat_template_file(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        import os
        from jinja2 import Template

        def load(model_dir):
            with open(os.path.join(model_dir, "chat_template.jinja")) as fh:
                source = fh.read()
            return Template(source)
    """
            },
        )
    )
    [f] = [f for f in r.findings if f.verdict == REVIEW]
    assert f.line == 7 and f.source_line == 5


def test_taint_through_a_return_value(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2 import Environment

        def template_of(tok):
            return tok.chat_template

        def render(tok):
            src = template_of(tok)
            return Environment().from_string(src)
    """
            },
        )
    )
    assert lines(r, REVIEW) == [("m.py", 8)]


def test_environment_subclass_and_compile_method(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "envs.py": """
            import jinja2
            class FastEnv(jinja2.Environment):
                pass
        """,
                "use.py": """
            from envs import FastEnv
            def render(chat_template):
                return FastEnv().compile(chat_template)
        """,
            },
        )
    )
    assert lines(r, REVIEW) == [("use.py", 3)]


def test_unresolved_environment_is_check(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        import jinja2

        def render(env, tokenizer):
            return env.from_string(tokenizer.chat_template)
    """
            },
        )
    )
    assert lines(r, CHECK) == [("m.py", 4)]
    assert not lines(r, REVIEW)


def test_environment_sandboxed_on_one_branch_only_is_check(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2 import Environment
        from jinja2.sandbox import SandboxedEnvironment

        def render(chat_template, safe):
            env = SandboxedEnvironment() if safe else Environment()
            return env.from_string(chat_template)
    """
            },
        )
    )
    assert lines(r, CHECK) == [("m.py", 6)]


# ---------------------------------------------------------------- what must not be flagged


def test_sandboxed_render_of_model_template(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def render(tokenizer):
            env = ImmutableSandboxedEnvironment(trim_blocks=True)
            return env.from_string(tokenizer.chat_template)
    """
            },
        )
    )
    assert lines(r, "SANDBOX") == [("m.py", 5)]
    assert not lines(r, REVIEW) and not lines(r, CHECK)


def test_sandbox_via_module_attribute_chain_and_self_attribute(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        import jinja2.sandbox

        class Renderer:
            def __init__(self):
                self._env = None

            def env(self):
                if self._env is None:
                    self._env = jinja2.sandbox.ImmutableSandboxedEnvironment()
                return self._env

            def render(self, tokenizer):
                return self._env.from_string(tokenizer.chat_template)
    """
            },
        )
    )
    assert lines(r, "SANDBOX") == [("m.py", 13)]
    assert not lines(r, REVIEW) and not lines(r, CHECK)


def test_sandbox_factory_and_subclass_across_modules(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "pkg/__init__.py": "from .envs import make_env, SafeEnv\n",
                "pkg/envs.py": """
            from jinja2.sandbox import SandboxedEnvironment

            class SafeEnv(SandboxedEnvironment):
                pass

            def make_env():
                return SafeEnv(trim_blocks=True)
        """,
                "app.py": """
            from pkg import make_env, SafeEnv

            def a(chat_template):
                return make_env().from_string(chat_template)

            def b(chat_template):
                return SafeEnv().from_string(chat_template)
        """,
            },
        )
    )
    assert sorted(lines(r, "SANDBOX")) == [("app.py", 4), ("app.py", 7)]
    assert not lines(r, REVIEW) and not lines(r, CHECK)


def test_developer_template_is_not_traced_to_a_model(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2 import Environment

        chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
        PAGE = "<h1>{{ title }}</h1>"

        def render(messages):
            return Environment().from_string(chat_template).render(messages=messages)

        def page(title):
            return Environment().from_string(PAGE).render(title=title)
    """
            },
        )
    )
    assert not lines(r, REVIEW)
    assert sorted(lines(r, "PLAIN")) == [("m.py", 7), ("m.py", 10)]


def test_comments_strings_lookalikes_and_other_template_engines(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        import enum
        import re
        from string import Template

        # never call Environment().from_string(chat_template) here
        DOC = "Environment().from_string(chat_template) is unsafe"

        class Environment(enum.Enum):
            DEV = 1

        def f(chat_template):
            re.compile(chat_template)
            return Template(chat_template).substitute()
    """
            },
        )
    )
    assert r.findings == []


def test_inline_ignore_comment(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "m.py": """
        from jinja2 import Environment

        def render(chat_template):
            return Environment().from_string(chat_template)  # chat-template-audit: ignore
    """
            },
        )
    )
    [f] = r.findings
    assert f.verdict == REVIEW and f.suppressed
    assert main([str(tmp_path)]) == 0


def test_tests_are_skipped_unless_asked(tmp_path):
    root = project(
        tmp_path,
        {
            "tests/test_x.py": """
        from jinja2 import Environment
        def test_it(chat_template):
            Environment().from_string(chat_template)
    """
        },
    )
    assert scan(root).findings == []
    assert lines(scan(root, include_tests=True), REVIEW) == [("test_x.py", 3)]


def test_syntax_errors_are_skipped_not_fatal(tmp_path):
    r = scan(
        project(
            tmp_path,
            {
                "broken.py": "import jinja2\ndef f(:\n",
                "ok.py": """
            from jinja2 import Environment
            def f(chat_template):
                return Environment().from_string(chat_template)
        """,
            },
        )
    )
    assert lines(r, REVIEW) == [("ok.py", 3)]
    assert [Path(p).name for p, _ in r.skipped] == ["broken.py"]


# ---------------------------------------------------------------- jinja2 version pins

SANDBOXED = """
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    def f(chat_template):
        return ImmutableSandboxedEnvironment().from_string(chat_template)
"""


@pytest.mark.parametrize(
    ("filename", "content", "flagged"),
    [
        ("pyproject.toml", '[project]\nname = "x"\ndependencies = ["jinja2>=3.0"]\n', True),
        ("pyproject.toml", '[project]\nname = "x"\ndependencies = ["Jinja2>=3.1.6,<4"]\n', False),
        ("pyproject.toml", '[tool.poetry.dependencies]\njinja2 = "^3.1.2"\n', True),
        ("pyproject.toml", '[tool.ruff.lint.isort]\nknown-third-party = ["jinja2"]\n', False),
        ("requirements.txt", "jinja2==3.1.4  # pinned\n", True),
        ("requirements.txt", "jinja2-time\njinja2>=3.1.6\n", False),
        ("setup.py", 'setup(install_requires=["jinja2"])\n', True),
    ],
)
def test_pin_check(tmp_path, filename, content, flagged):
    root = project(tmp_path, {"m.py": SANDBOXED, filename: content})
    assert bool(lines(scan(root), PIN)) is flagged


def test_no_pin_finding_without_a_sandbox(tmp_path):
    root = project(
        tmp_path,
        {
            "m.py": "import jinja2\nTEMPLATE = jinja2.Template('{{ x }}')\n",
            "requirements.txt": "jinja2==2.11\n",
        },
    )
    assert not lines(scan(root), PIN)


# ---------------------------------------------------------------- CLI and formats


def test_cli_exit_codes_and_strict(tmp_path, capsys):
    vulnerable = project(
        tmp_path / "v",
        {
            "m.py": """
        from jinja2 import Environment
        def f(chat_template):
            return Environment().from_string(chat_template)
    """
        },
    )
    unresolved = project(
        tmp_path / "u",
        {
            "m.py": """
        import jinja2
        def f(env, chat_template):
            return env.from_string(chat_template)
    """
        },
    )
    assert main([str(vulnerable)]) == 1
    assert main([str(unresolved)]) == 0
    assert main(["--strict", str(unresolved)]) == 1
    assert main([str(FIXTURES / "safe_lib")]) == 0
    assert main([str(tmp_path / "missing")]) == 2
    capsys.readouterr()


def test_json_and_sarif(tmp_path, capsys):
    root = project(
        tmp_path,
        {
            "m.py": """
        from jinja2 import Environment
        def f(tokenizer):
            return Environment().from_string(tokenizer.chat_template)
    """
        },
    )
    main(["--format", "json", str(root)])
    data = json.loads(capsys.readouterr().out)
    assert [f["verdict"] for f in data["reports"][0]["findings"]] == [REVIEW]

    sarif = json.loads(render_sarif([scan(root)]))
    assert sarif["version"] == "2.1.0"
    [result] = sarif["runs"][0]["results"]
    assert result["ruleId"] == "CTA001" and result["level"] == "error"
    assert result["locations"][0]["physicalLocation"]["region"]["startLine"] == 3
    assert {r["id"] for r in sarif["runs"][0]["tool"]["driver"]["rules"]} == {"CTA001", "CTA002", "CTA003"}


def test_fixtures():
    assert [f.verdict for f in scan(FIXTURES / "vulnerable_lib").findings] == [REVIEW]
    assert [f.verdict for f in scan(FIXTURES / "safe_lib").findings] == ["SANDBOX"]


# ---------------------------------------------------------------- the bug class is real


def test_payload_reaches_os_in_a_plain_environment_and_not_in_a_sandbox():
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.exceptions import SecurityError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    payload = "{{ cycler.__init__.__globals__.os.sep }}"
    assert jinja2.Environment().from_string(payload).render() == os.sep
    with pytest.raises(SecurityError):
        ImmutableSandboxedEnvironment().from_string(payload).render()
