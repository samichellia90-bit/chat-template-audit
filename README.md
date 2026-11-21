<div align="center">

# chat-template-audit

**A static check for one specific remote-code-execution pattern: an LLM's
`chat_template` rendered with a non-sandboxed Jinja2 environment.**

[![tests](https://github.com/samichellia90-bit/chat-template-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/samichellia90-bit/chat-template-audit/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![deps](https://img.shields.io/badge/dependencies-none-brightgreen)

</div>

## The bug class

A modern chat model ships a **`chat_template`** — a Jinja2 string — inside its
`tokenizer_config.json` (or, for GGUF, in `tokenizer.chat_template` metadata). A
program that builds a prompt from it compiles that string:

```python
from jinja2 import Environment
prompt = Environment().from_string(tokenizer.chat_template).render(messages=msgs)
```

When the model comes from an untrusted source — any public hub — that template
is **attacker-controlled Jinja source**. With a plain `Environment`, a crafted
template reaches Python internals and executes code:

```jinja
{{ cycler.__init__.__globals__.os.popen('id').read() }}
```

So the render **must** use a Jinja sandbox: `SandboxedEnvironment` or
`ImmutableSandboxedEnvironment`, on a recent Jinja (the 3.1.5–3.1.6 line closed
the known sandbox escapes, e.g. CVE-2024-56326 and CVE-2025-27516). Hugging Face
`transformers`, for instance, both sandboxes the render and refuses to run on
`jinja2 < 3.1.6`.

`chat-template-audit` finds the render sites that do **not** take that
precaution.

## What it does

It scans a Python tree. For every Jinja render site (`.from_string(`,
`Environment(`, `Template(`) it reports a verdict:

| verdict | meaning |
|---|---|
| **`REVIEW`** | render site **not** sandboxed, in a file that also reads a model-supplied template (`chat_template` / `tokenizer_config` / GGUF `metadata[...]`). Review by hand. |
| `sandbox` | the file builds a Jinja sandbox; the render is most likely safe. |
| `render`  | a render site with no model-template source nearby. |

It is a **heuristic linter, not a proof of exploitability.** It points a human at
the lines worth reading; reachability decides whether a `REVIEW` is a real bug.

## Install & use

```bash
pip install -e .
chat-template-audit path/to/repo            # human-readable
chat-template-audit path/to/repo --json     # machine-readable
```

Exit code is `1` when any `REVIEW` finding exists, so it drops into CI or a
pre-commit hook. Zero third-party dependencies.

Example on the two fixtures in this repo:

```
=== tests/fixtures/vulnerable_lib ===
  [REVIEW] model.py:8  template = Environment(loader=BaseLoader).from_string(chat_template)   {untrusted-template-in-file}

=== tests/fixtures/safe_lib ===
  [sandbox] model.py:9  template = env.from_string(chat_template)   {sandboxed,untrusted-template-in-file}
```

## Why `REVIEW` still needs a human

The flag is a code smell, not a verdict. A render can be non-sandboxed **and**
read a model template, yet still be safe if the untrusted string never reaches
the compiler. A real example: `guidance` renders `chat_template` with a plain
`Environment`, but its loader keeps only template strings that match a built-in
it already trusts and drops everything else, so an attacker's template is never
compiled. The tool marks it `REVIEW`; a human reads the data flow and clears it.
That is the intended workflow.

## Scope and limitations

- Sandbox detection is **file-level**: a file that builds a sandbox anywhere is
  treated as sandboxed at every render site in it. This can hide a plain
  `Environment` that sits next to a sandboxed one; cross-check large files.
- It reasons about **where the template is compiled**, not where its string
  comes from. Confirm the untrusted source reaches the render before reporting
  anything.
- Python only. Templates compiled in other languages are out of scope.
- **For defensive use and responsible disclosure only.** If you confirm a real
  issue in someone else's project, report it privately to the maintainers.

## License

MIT — see [LICENSE](LICENSE).
