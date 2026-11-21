"""Find LLM chat templates compiled by a non-sandboxed Jinja2 environment.

A chat model ships its prompt format as a Jinja2 template: ``chat_template`` in
``tokenizer_config.json``, a ``chat_template.jinja`` file, or the
``tokenizer.chat_template`` key of GGUF metadata. Code that compiles that string
with a plain ``jinja2.Environment`` or ``jinja2.Template`` executes template code
chosen by whoever published the model. That is how CVE-2024-34359
(llama-cpp-python) and CVE-2024-2952 (LiteLLM) worked. The fix is a Jinja
sandbox (``ImmutableSandboxedEnvironment``) on jinja2 >= 3.1.6.

The analysis is static: the scanned code is parsed with :mod:`ast`, never
imported or executed.

* jinja2 imports, aliases, ``jinja2.sandbox`` attribute chains, subclasses,
  module-level environments, ``self`` attributes and factory functions are
  resolved, also across the modules of the scanned tree;
* a template is *model-supplied* when it derives from a name, attribute, key or
  file name containing ``chat_template`` or ``tokenizer_config``. The taint
  follows assignments, ``self`` attributes, return values and calls between the
  functions of the scanned tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, Iterator, Optional

import tomllib

__version__ = "0.2.0"

PLAIN, SANDBOX, MIXED = "plain", "sandbox", "mixed"
REVIEW, CHECK, PIN = "REVIEW", "CHECK", "PIN"
SANDBOXED, UNSANDBOXED, UNRESOLVED = "SANDBOX", "PLAIN", "UNKNOWN"
ACTIONABLE = (REVIEW, CHECK, PIN)
RULE_IDS = {REVIEW: "CTA001", CHECK: "CTA002", PIN: "CTA003"}
SAFE_JINJA = (3, 1, 6)
REPO_URL = "https://github.com/samichellia90-bit/chat-template-audit"

_CTOR_KINDS = {
    "jinja2.Environment": PLAIN,
    "jinja2.environment.Environment": PLAIN,
    "jinja2.nativetypes.NativeEnvironment": PLAIN,
    "jinja2.sandbox.SandboxedEnvironment": SANDBOX,
    "jinja2.sandbox.ImmutableSandboxedEnvironment": SANDBOX,
}
_TEMPLATE_CTORS = frozenset({"jinja2.Template", "jinja2.environment.Template", "jinja2.nativetypes.NativeTemplate"})
_JINJA_MODULES = frozenset({"jinja2", "jinja2.sandbox", "jinja2.environment", "jinja2.nativetypes"})
_STAR_NAMES = {
    "jinja2": ("Environment", "Template"),
    "jinja2.environment": ("Environment", "Template"),
    "jinja2.sandbox": ("SandboxedEnvironment", "ImmutableSandboxedEnvironment"),
    "jinja2.nativetypes": ("NativeEnvironment", "NativeTemplate"),
}
_COMPILE_METHODS = frozenset({"from_string", "compile"})
_TAINT = re.compile(r"chat_template|tokenizer_config", re.IGNORECASE)
_PREFILTER = re.compile(r"jinja|from_string|Template\(|Environment\(|chat_template|tokenizer_config", re.IGNORECASE)
_IGNORE_MARK = "chat-template-audit: ignore"
_MAX_BYTES = 10_000_000
_MAX_DEPTH = 40

EXCLUDED_DIRS = frozenset({"node_modules", "__pycache__", "build", "dist", "site-packages", "venv", "env", "htmlcov"})
TEST_DIRS = frozenset({"test", "tests"})
_REQ_NAME = re.compile(
    r"^(?:requirements.*\.(?:txt|in)|constraints.*\.txt|pyproject\.toml|setup\.cfg|setup\.py)$", re.I
)


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class Finding:
    """One render site (or, for ``PIN``, one jinja2 requirement)."""

    verdict: str
    path: str
    line: int
    column: int
    message: str
    code: str
    source_path: Optional[str] = None
    source_line: Optional[int] = None
    source_code: Optional[str] = None
    suppressed: bool = False


@dataclass
class Report:
    path: str
    python_files: int
    analysed: int
    skipped: list[tuple[str, str]]
    findings: list[Finding]

    def count(self, verdict: str) -> int:
        return sum(1 for f in self.findings if f.verdict == verdict and not f.suppressed)


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class _Val:
    """Abstract value of an expression: what it is, as far as jinja2 is concerned."""

    cat: str  # module | pmodule | ext | envclass | tplclass | env | envmethod | class | instance | func | const
    kind: Optional[str] = None  # PLAIN / SANDBOX / MIXED for environments
    label: str = ""
    ref: object = None
    bound: bool = False


_CONST = _Val("const")
_MISSING = _Val("missing")


@dataclass(eq=False)
class _Module:
    name: str
    package: str
    rel: str
    lines: list[str]
    has_jinja: bool
    tree: ast.Module
    scope: Optional[_Scope] = None


@dataclass(eq=False)
class _Scope:
    module: _Module
    node: ast.AST
    parent: Optional[_Scope]
    kind: str  # module | function | class
    key: tuple
    cls: Optional[_Class] = None
    self_name: Optional[str] = None
    skip_first: bool = False
    is_classmethod: bool = False
    pos_params: list[str] = field(default_factory=list)
    all_params: set[str] = field(default_factory=set)
    bindings: dict[str, list[tuple]] = field(default_factory=dict)
    stars: list[Optional[str]] = field(default_factory=list)
    returns: list[ast.expr] = field(default_factory=list)
    calls: list[ast.Call] = field(default_factory=list)


@dataclass(eq=False)
class _Class:
    key: tuple
    node: ast.ClassDef
    outer: _Scope
    body: _Scope
    methods: dict[str, _Scope] = field(default_factory=dict)
    self_assigns: dict[str, list[tuple[ast.expr, _Scope, bool]]] = field(default_factory=dict)


def _short(qualname: str) -> str:
    return qualname.replace("jinja2.environment.", "jinja2.")


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


class _Project:
    """Every module of one scanned tree, with name resolution and taint tracking."""

    def __init__(self) -> None:
        self.modules: dict[str, _Module] = {}
        self._suffix: dict[str, set[str]] = {}
        self.scopes: list[_Scope] = []
        self.functions: dict[tuple, _Scope] = {}
        self.classes: dict[tuple, _Class] = {}
        self.tainted_params: dict[tuple, tuple[_Scope, ast.expr]] = {}
        self.tainted_returns: set[tuple] = set()
        self._resolved: dict[tuple, Optional[_Val]] = {}
        self._taint: dict[tuple, bool] = {}
        self._class_kinds: dict[tuple, Optional[str]] = {}
        self._returns: dict[tuple, Optional[_Val]] = {}

    # ---------------------------------------------------------------- building

    def add_module(self, name: str, package: str, rel: str, text: str, tree: ast.Module) -> None:
        if name in self.modules:  # two files mapping to one dotted name: keep both, resolve the first
            name = f"{name}#{len(self.modules)}"
        mod = _Module(name, package, rel, text.splitlines(), "jinja" in text.lower(), tree)
        self.modules[name] = mod
        parts = name.split(".")
        for i in range(len(parts)):
            self._suffix.setdefault(".".join(parts[i:]), set()).add(name)

    def build(self) -> None:
        for mod in list(self.modules.values()):
            mod.scope = _Scope(mod, mod.tree, None, "module", (mod.name, "<module>", 0))
            self.scopes.append(mod.scope)
            self._collect(mod.scope, mod.tree.body)

    def _find_module(self, name: Optional[str]) -> Optional[str]:
        if not name:
            return None
        if name in self.modules:
            return name
        hits = self._suffix.get(name)
        return next(iter(hits)) if hits and len(hits) == 1 else None

    @staticmethod
    def _absolute(mod: _Module, name: Optional[str], level: int) -> Optional[str]:
        if level == 0:
            return name
        base = mod.package.split(".") if mod.package else []
        if level - 1 > len(base):
            return None
        base = base[: len(base) - (level - 1)]
        return ".".join(p for p in (*base, name or "") if p) or None

    def _collect(self, scope: _Scope, nodes: Iterable[ast.AST]) -> None:
        todo = list(nodes)
        todo.reverse()
        while todo:
            n = todo.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(scope, n)
                outer = [*n.decorator_list, *n.args.defaults, *(d for d in n.args.kw_defaults if d is not None)]
                todo.extend(reversed(outer))
                continue
            if isinstance(n, ast.ClassDef):
                self._class(scope, n)
                todo.extend(reversed([*n.decorator_list, *n.bases, *(k.value for k in n.keywords)]))
                continue
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    self._bind(scope, target, n.value)
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
                if n.value is not None:
                    self._bind(scope, n.target, n.value)
            elif isinstance(n, ast.NamedExpr):
                self._bind(scope, n.target, n.value)
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                self._bind(scope, n.target, n.iter, iterate=True)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars is not None:
                        self._bind(scope, item.optional_vars, item.context_expr)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                scope.bindings.setdefault(n.name, []).append(("unknown",))
            elif isinstance(n, ast.Import):
                for alias in n.names:
                    if alias.asname:
                        scope.bindings.setdefault(alias.asname, []).append(("import", alias.name))
                    else:
                        top = alias.name.partition(".")[0]
                        scope.bindings.setdefault(top, []).append(("import", top))
            elif isinstance(n, ast.ImportFrom):
                source = self._absolute(scope.module, n.module, n.level)
                for alias in n.names:
                    if alias.name == "*":
                        scope.stars.append(source)
                    else:
                        scope.bindings.setdefault(alias.asname or alias.name, []).append(("from", source, alias.name))
            elif isinstance(n, ast.Return) and n.value is not None:
                scope.returns.append(n.value)
            elif isinstance(n, ast.Call):
                scope.calls.append(n)
            todo.extend(reversed(list(ast.iter_child_nodes(n))))

    def _bind(self, scope: _Scope, target: ast.AST, value: ast.expr, iterate: bool = False) -> None:
        if isinstance(target, ast.Name):
            scope.bindings.setdefault(target.id, []).append(("iter" if iterate else "expr", value))
        elif isinstance(target, ast.Attribute):
            if (
                scope.cls is not None
                and scope.self_name is not None
                and isinstance(target.value, ast.Name)
                and target.value.id == scope.self_name
            ):
                scope.cls.self_assigns.setdefault(target.attr, []).append((value, scope, iterate))
        elif isinstance(target, (ast.Tuple, ast.List)):
            pairwise = (
                not iterate
                and isinstance(value, (ast.Tuple, ast.List))
                and len(value.elts) == len(target.elts)
                and not any(isinstance(e, ast.Starred) for e in target.elts)
            )
            if pairwise:
                for t, v in zip(target.elts, value.elts):
                    self._bind(scope, t, v)
            else:
                for t in target.elts:
                    self._bind(scope, t, value, iterate=True)
        elif isinstance(target, ast.Starred):
            self._bind(scope, target.value, value, iterate=True)

    def _function(self, scope: _Scope, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        owner = scope.cls if scope.kind == "class" else None
        parent = scope.parent if scope.kind == "class" else scope
        qual = node.name if scope.kind == "module" else f"{scope.key[1]}.{node.name}"
        key = (scope.module.name, qual, node.lineno)
        fs = _Scope(scope.module, node, parent, "function", key)
        args = node.args
        fs.pos_params = [a.arg for a in (*args.posonlyargs, *args.args)]
        names = [*fs.pos_params, *(a.arg for a in args.kwonlyargs)]
        names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        fs.all_params = set(names)
        for name in names:
            fs.bindings.setdefault(name, []).append(("param",))
        decorators = {_decorator_name(d) for d in node.decorator_list}
        if owner is not None:
            fs.cls = owner
            if "staticmethod" not in decorators and fs.pos_params:
                fs.self_name, fs.skip_first = fs.pos_params[0], True
                fs.is_classmethod = "classmethod" in decorators
            owner.methods[node.name] = fs
        elif scope.kind == "function" and scope.self_name and scope.self_name not in fs.all_params:
            fs.cls, fs.self_name = scope.cls, scope.self_name  # closure over `self`
        self.functions[key] = fs
        self.scopes.append(fs)
        scope.bindings.setdefault(node.name, []).append(("def", key))
        self._collect(fs, node.body)

    def _class(self, scope: _Scope, node: ast.ClassDef) -> None:
        parent = scope.parent if scope.kind == "class" else scope
        qual = node.name if scope.kind == "module" else f"{scope.key[1]}.{node.name}"
        key = (scope.module.name, qual, node.lineno)
        body = _Scope(scope.module, node, parent, "class", key)
        info = _Class(key, node, scope, body)
        body.cls = info
        self.classes[key] = info
        self.scopes.append(body)
        scope.bindings.setdefault(node.name, []).append(("class", key))
        self._collect(body, node.body)

    # ---------------------------------------------------------------- name resolution

    def _lookup(self, name: str, scope: _Scope) -> tuple[Optional[_Scope], Optional[list[tuple]]]:
        s: Optional[_Scope] = scope
        while s is not None:
            found = s.bindings.get(name)
            if found:
                return s, found
            for star in s.stars:
                v = self._star(star, name, 0)
                if v is not _MISSING:
                    return s, [("val", v)]
            s = s.parent
        return None, None

    def _star(self, module: Optional[str], name: str, depth: int) -> _Val:
        if module is None or depth > _MAX_DEPTH:
            return _MISSING
        if module in _STAR_NAMES:
            return self._jinja_attr(module, name) if name in _STAR_NAMES[module] else _MISSING
        found = self._find_module(module)
        if found is not None and name in self.modules[found].scope.bindings:
            return self._export(found, name, depth + 1)
        return _MISSING

    def _export(self, module: str, name: str, depth: int) -> _Val:
        if depth > _MAX_DEPTH:
            return _MISSING
        mod = self.modules.get(module)
        if mod is None:
            return _MISSING
        found = mod.scope.bindings.get(name)
        if found:
            v = self._merge([self._binding_value(mod.scope, b, depth + 1) for b in found])
            return _MISSING if v is None else v
        for star in mod.scope.stars:
            v = self._star(star, name, depth + 1)
            if v is not _MISSING:
                return v
        sub = f"{module}.{name}" if module else name
        return _Val("pmodule", ref=sub) if sub in self.modules else _MISSING

    def _binding_value(self, scope: _Scope, binding: tuple, depth: int) -> Optional[_Val]:
        tag = binding[0]
        if tag == "expr":
            return self.resolve(binding[1], scope, depth + 1)
        if tag == "import":
            return self._module_value(binding[1])
        if tag == "from":
            return self._from_value(binding[1], binding[2], depth + 1)
        if tag == "def":
            return _Val("func", ref=binding[1])
        if tag == "class":
            return _Val("class", ref=binding[1])
        if tag == "val":
            return binding[1]
        return None  # param, iter, unknown

    def _module_value(self, name: str) -> _Val:
        if name in _JINJA_MODULES:
            return _Val("module", label=name)
        found = self._find_module(name)
        return _Val("pmodule", ref=found) if found is not None else _Val("ext", label=name)

    def _from_value(self, module: Optional[str], name: str, depth: int) -> Optional[_Val]:
        if module is None:
            return None
        if module == "jinja2" or module.startswith("jinja2."):
            return self._jinja_attr(module, name)
        found = self._find_module(module)
        if found is not None:
            v = self._export(found, name, depth + 1)
            return None if v is _MISSING else v
        return _Val("ext", label=f"{module}.{name}")

    @staticmethod
    def _jinja_attr(base: str, attr: str) -> _Val:
        q = f"{base}.{attr}"
        if q in _CTOR_KINDS:
            return _Val("envclass", _CTOR_KINDS[q], _short(q))
        if q in _TEMPLATE_CTORS:
            return _Val("tplclass", PLAIN, _short(q))
        if q in _JINJA_MODULES:
            return _Val("module", label=q)
        return _Val("ext", label=q)

    @staticmethod
    def _merge(values: list[Optional[_Val]]) -> Optional[_Val]:
        real = [v for v in values if v is not _CONST and v is not _MISSING]
        if not real:
            return _CONST if any(v is _CONST for v in values) else None
        envs = [v for v in real if v is not None and v.cat == "env"]
        if envs:
            kinds = {v.kind for v in envs}
            if MIXED in kinds or {PLAIN, SANDBOX} <= kinds:
                kind = MIXED
            elif PLAIN in kinds:
                kind = PLAIN
            elif len(envs) != len(real):
                return None  # sandboxed on some paths, unknown on others
            else:
                kind = SANDBOX
            labels = {v.label for v in envs}
            return _Val("env", kind, labels.pop() if len(labels) == 1 else "")
        first = real[0]
        return first if all(v == first for v in real) else None

    def resolve(self, expr: ast.expr, scope: _Scope, depth: int = 0) -> Optional[_Val]:
        if depth > _MAX_DEPTH:
            return None
        key = (id(expr), id(scope))
        if key in self._resolved:
            return self._resolved[key]
        self._resolved[key] = None  # cycle guard
        value = self._resolve(expr, scope, depth)
        self._resolved[key] = value
        return value

    def _resolve(self, e: ast.expr, scope: _Scope, depth: int) -> Optional[_Val]:
        if isinstance(e, ast.Constant):
            return _CONST
        if isinstance(e, ast.Name):
            s, found = self._lookup(e.id, scope)
            if not found:
                return None
            return self._merge([self._binding_value(s, b, depth + 1) for b in found])
        if isinstance(e, ast.Attribute):
            if self._is_self(e.value, scope):
                return self._class_attr(scope.cls, e.attr, depth + 1)
            return self._attr(self.resolve(e.value, scope, depth + 1), e.attr, depth + 1)
        if isinstance(e, ast.Call):
            return self._call(self.resolve(e.func, scope, depth + 1), depth + 1)
        if isinstance(e, ast.IfExp):
            return self._merge([self.resolve(e.body, scope, depth + 1), self.resolve(e.orelse, scope, depth + 1)])
        if isinstance(e, ast.BoolOp):
            return self._merge([self.resolve(v, scope, depth + 1) for v in e.values])
        if isinstance(e, ast.NamedExpr):
            return self.resolve(e.value, scope, depth + 1)
        return None

    @staticmethod
    def _is_self(node: ast.expr, scope: _Scope) -> bool:
        return scope.cls is not None and isinstance(node, ast.Name) and node.id == scope.self_name

    def _attr(self, base: Optional[_Val], attr: str, depth: int) -> Optional[_Val]:
        if base is None or base is _CONST or base is _MISSING:
            return None
        if base.cat == "module":
            return self._jinja_attr(base.label, attr)
        if base.cat == "pmodule":
            v = self._export(base.ref, attr, depth + 1)
            return None if v is _MISSING else v
        if base.cat == "ext":
            return _Val("ext", label=f"{base.label}.{attr}")
        if base.cat in ("class", "instance"):
            return self._class_attr(self.classes[base.ref], attr, depth + 1, bound=base.cat == "instance")
        if base.cat == "env" and attr == "overlay":
            return _Val("envmethod", base.kind, base.label)
        return None

    def _bases(self, info: _Class) -> Iterator[_Class]:
        for b in info.node.bases:
            v = self.resolve(b, info.outer)
            if v is not None and v.cat == "class":
                yield self.classes[v.ref]

    def _mro(self, info: _Class) -> list[_Class]:
        out, todo, seen = [], [info], set()
        while todo:
            c = todo.pop(0)
            if c.key in seen:
                continue
            seen.add(c.key)
            out.append(c)
            todo.extend(self._bases(c))
        return out

    def _class_attr(self, info: _Class, attr: str, depth: int, bound: bool = True) -> Optional[_Val]:
        if depth > _MAX_DEPTH:
            return None
        for c in self._mro(info):
            candidates: list[Optional[_Val]] = []
            for expr, sc, iterate in c.self_assigns.get(attr, ()):
                candidates.append(None if iterate else self.resolve(expr, sc, depth + 1))
            for b in c.body.bindings.get(attr, ()):
                if b[0] == "def":
                    candidates.append(_Val("func", ref=b[1], bound=bound))
                else:
                    candidates.append(self._binding_value(c.body, b, depth + 1))
            if candidates:
                return self._merge(candidates)
        return None

    def _class_kind(self, info: _Class) -> Optional[str]:
        if info.key in self._class_kinds:
            return self._class_kinds[info.key]
        self._class_kinds[info.key] = None
        kind = None
        for b in info.node.bases:
            v = self.resolve(b, info.outer)
            if v is None or v is _CONST:
                continue
            if v.cat == "envclass":
                kind = v.kind
            elif v.cat == "class":
                kind = self._class_kind(self.classes[v.ref])
            if kind:
                break
        self._class_kinds[info.key] = kind
        return kind

    def _call(self, callee: Optional[_Val], depth: int) -> Optional[_Val]:
        if callee is None or callee is _CONST or callee is _MISSING:
            return None
        if callee.cat in ("envclass", "envmethod"):
            return _Val("env", callee.kind, callee.label)
        if callee.cat == "class":
            info = self.classes[callee.ref]
            kind = self._class_kind(info)
            return _Val("env", kind, info.node.name) if kind else _Val("instance", ref=callee.ref)
        if callee.cat == "func":
            return self._function_result(self.functions[callee.ref], depth + 1)
        if callee.cat == "ext":
            return _Val("ext", label=f"{callee.label}()")
        return None

    def _function_result(self, fs: _Scope, depth: int) -> Optional[_Val]:
        if fs.key in self._returns:
            return self._returns[fs.key]
        self._returns[fs.key] = None
        value = self._merge([self.resolve(r, fs, depth + 1) for r in fs.returns]) if fs.returns else None
        if value is not None and value.cat not in ("env", "instance"):
            value = None
        self._returns[fs.key] = value
        return value

    # ---------------------------------------------------------------- taint

    def tainted(self, expr: ast.expr, scope: _Scope) -> bool:
        key = (id(expr), id(scope))
        if key in self._taint:
            return self._taint[key]
        self._taint[key] = False  # cycle guard
        result = self._tainted(expr, scope)
        self._taint[key] = result
        return result

    def _tainted(self, e: ast.expr, scope: _Scope) -> bool:
        if isinstance(e, ast.Constant):
            return isinstance(e.value, str) and _TAINT.search(e.value) is not None
        if isinstance(e, ast.Name):
            s, found = self._lookup(e.id, scope)
            if not found:
                return _TAINT.search(e.id) is not None
            return any(self._binding_tainted(s, e.id, b) for b in found)
        if isinstance(e, ast.Attribute):
            if self._is_self(e.value, scope):
                known = self._attr_tainted(scope.cls, e.attr)
                return _TAINT.search(e.attr) is not None if known is None else known
            return _TAINT.search(e.attr) is not None or self.tainted(e.value, scope)
        if isinstance(e, ast.Call):
            callee = self.resolve(e.func, scope)
            if callee is not None and callee.cat == "func" and callee.ref in self.tainted_returns:
                return True
            if isinstance(e.func, ast.Attribute):
                if self.tainted(e.func, scope):
                    return True
            elif isinstance(e.func, ast.Name) and _TAINT.search(e.func.id):
                return True
            args = [a.value if isinstance(a, ast.Starred) else a for a in e.args]
            return any(self.tainted(a, scope) for a in (*args, *(k.value for k in e.keywords)))
        for child in ast.iter_child_nodes(e):
            node = child.iter if isinstance(child, ast.comprehension) else child
            if isinstance(node, ast.expr) and self.tainted(node, scope):
                return True
        return False

    def _binding_tainted(self, scope: _Scope, name: str, binding: tuple) -> bool:
        tag = binding[0]
        if tag == "param":
            return (scope.key, name) in self.tainted_params or _TAINT.search(name) is not None
        if tag in ("expr", "iter"):
            return self.tainted(binding[1], scope)
        return False

    def _attr_tainted(self, info: _Class, attr: str) -> Optional[bool]:
        for c in self._mro(info):
            found = False
            for expr, sc, _ in c.self_assigns.get(attr, ()):
                found = True
                if self.tainted(expr, sc):
                    return True
            for b in c.body.bindings.get(attr, ()):
                found = True
                if self._binding_tainted(c.body, attr, b):
                    return True
            if found:
                return False
        return None

    def _call_target(self, callee: Optional[_Val]) -> tuple[Optional[_Scope], int]:
        if callee is None or callee is _CONST:
            return None, 0
        if callee.cat == "func":
            fs = self.functions[callee.ref]
            return fs, 1 if fs.skip_first and (callee.bound or fs.is_classmethod) else 0
        if callee.cat == "class":
            for c in self._mro(self.classes[callee.ref]):
                init = c.methods.get("__init__")
                if init is not None:
                    return init, 1
        return None, 0

    def propagate(self) -> None:
        """Fixed point: taint flows into the parameters of called functions and out of their returns."""
        for _ in range(25):
            self._taint.clear()
            changed = False
            for sc in self.scopes:
                try:
                    changed |= self._propagate_scope(sc)
                except RecursionError:
                    continue
            if not changed:
                break
        self._taint.clear()

    def _propagate_scope(self, sc: _Scope) -> bool:
        changed = False
        if sc.kind == "function" and sc.key not in self.tainted_returns:
            if any(self.tainted(r, sc) for r in sc.returns):
                self.tainted_returns.add(sc.key)
                changed = True
        for call in sc.calls:
            target, offset = self._call_target(self.resolve(call.func, sc))
            if target is None:
                continue
            for i, arg in enumerate(call.args):
                if isinstance(arg, ast.Starred):
                    break
                if i + offset < len(target.pos_params):
                    changed |= self._mark(target, target.pos_params[i + offset], arg, sc)
            for kw in call.keywords:
                if kw.arg is not None and kw.arg in target.all_params:
                    changed |= self._mark(target, kw.arg, kw.value, sc)
        return changed

    def _mark(self, target: _Scope, param: str, arg: ast.expr, caller: _Scope) -> bool:
        key = (target.key, param)
        if key in self.tainted_params or not self.tainted(arg, caller):
            return False
        self.tainted_params[key] = (caller, arg)
        return True

    def explain(self, e: ast.expr, scope: _Scope, depth: int = 0) -> Optional[tuple[_Module, int]]:
        """Where the model-supplied value enters: the file and line of the origin."""
        if depth > _MAX_DEPTH:
            return None
        if isinstance(e, ast.Name):
            s, found = self._lookup(e.id, scope)
            if not found:
                return scope.module, e.lineno
            # a local assignment explains the value better than whoever called the function
            for b in found:
                if b[0] in ("expr", "iter") and self.tainted(b[1], s):
                    return self.explain(b[1], s, depth + 1) or (s.module, b[1].lineno)
            if ("param",) in found:
                origin = self.tainted_params.get((s.key, e.id))
                if origin is not None:
                    caller, arg = origin
                    return self.explain(arg, caller, depth + 1) or (caller.module, arg.lineno)
                if _TAINT.search(e.id):
                    return s.module, s.node.lineno
            return None
        if isinstance(e, ast.Attribute):
            if self._is_self(e.value, scope):
                for c in self._mro(scope.cls):
                    for expr, sc, _ in c.self_assigns.get(e.attr, ()):
                        if self.tainted(expr, sc):
                            return self.explain(expr, sc, depth + 1) or (sc.module, expr.lineno)
                return scope.module, e.lineno
            if _TAINT.search(e.attr):
                return scope.module, e.lineno
            return self.explain(e.value, scope, depth + 1)
        if isinstance(e, ast.Call):
            callee = self.resolve(e.func, scope)
            if callee is not None and callee.cat == "func" and callee.ref in self.tainted_returns:
                fs = self.functions[callee.ref]
                for r in fs.returns:
                    if self.tainted(r, fs):
                        return self.explain(r, fs, depth + 1) or (fs.module, r.lineno)
            for sub in (e.func, *e.args, *(k.value for k in e.keywords)):
                sub = sub.value if isinstance(sub, ast.Starred) else sub
                if self.tainted(sub, scope):
                    return self.explain(sub, scope, depth + 1) or (scope.module, sub.lineno)
            return scope.module, e.lineno
        if isinstance(e, ast.Constant):
            return scope.module, e.lineno
        for child in ast.iter_child_nodes(e):
            node = child.iter if isinstance(child, ast.comprehension) else child
            if isinstance(node, ast.expr) and self.tainted(node, scope):
                return self.explain(node, scope, depth + 1)
        return None

    # ---------------------------------------------------------------- render sites

    def sites(self, scope: _Scope) -> Iterator[tuple[ast.Call, Optional[str], str, ast.expr]]:
        for call in scope.calls:
            source = call.args[0] if call.args and not isinstance(call.args[0], ast.Starred) else None
            if source is None:
                source = next((k.value for k in call.keywords if k.arg == "source"), None)
            f = call.func
            if isinstance(f, ast.Attribute) and f.attr in _COMPILE_METHODS:
                receiver = self.resolve(f.value, scope)
                if receiver is None:
                    if f.attr != "from_string" or not scope.module.has_jinja:
                        continue
                    kind, label = None, ""
                elif receiver.cat == "env":
                    kind, label = receiver.kind, receiver.label
                else:
                    continue
            else:
                callee = self.resolve(f, scope)
                if callee is None or callee.cat != "tplclass":
                    continue
                kind, label = PLAIN, callee.label
            if source is not None:
                yield call, kind, label, source

    def findings(self) -> list[Finding]:
        out = []
        for sc in self.scopes:
            try:
                for call, kind, label, source in self.sites(sc):
                    out.append(self._finding(sc, call, kind, label, source))
            except RecursionError:
                continue
        return out

    def _finding(self, sc: _Scope, call: ast.Call, kind: Optional[str], label: str, source: ast.expr) -> Finding:
        tainted = self.tainted(source, sc)
        what = label or "a jinja2 environment"
        if kind == SANDBOX:
            verdict, message = SANDBOXED, f"template compiled in a Jinja sandbox ({what})"
        elif kind == PLAIN and tainted:
            verdict, message = REVIEW, f"model-supplied template compiled by {what}, which is not sandboxed"
        elif kind == PLAIN:
            verdict, message = UNSANDBOXED, f"{what} is not sandboxed; the template is not traced to a model"
        elif tainted:
            how = "sandboxed on some code paths only" if kind == MIXED else "not resolved statically"
            verdict, message = CHECK, f"model-supplied template compiled by an environment {how}"
        else:
            verdict, message = UNRESOLVED, "environment not resolved statically; the template is not traced to a model"
        lines = sc.module.lines
        end = min(call.end_lineno or call.lineno, len(lines))
        suppressed = any(_IGNORE_MARK in lines[i - 1] for i in range(call.lineno, end + 1))
        source_path = source_line = source_code = None
        if verdict in (REVIEW, CHECK):
            origin = self.explain(source, sc)
            if origin is not None and (origin[0] is not sc.module or origin[1] != call.lineno):
                mod, line = origin
                source_path, source_line = mod.rel, line
                source_code = mod.lines[line - 1].strip() if 0 < line <= len(mod.lines) else ""
        code = lines[call.lineno - 1].strip() if 0 < call.lineno <= len(lines) else ""
        return Finding(
            verdict,
            sc.module.rel,
            call.lineno,
            call.col_offset + 1,
            message,
            code,
            source_path,
            source_line,
            source_code,
            suppressed,
        )


# --------------------------------------------------------------------------- jinja2 version pins


def _version_tuple(text: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    return tuple(int(g or 0) for g in m.groups()) if m else (0, 0, 0)  # type: ignore[return-value]


def _lowest_allowed(spec: str, exact_if_bare: bool = False) -> Optional[tuple[int, int, int]]:
    """Lowest jinja2 version a requirement admits, or None when it has no lower bound."""
    spec = spec.strip()
    lows = [
        _version_tuple(ver)
        for op, ver in re.findall(r"(===|==|>=|~=|<=|!=|\^|~|>|<)\s*([0-9][0-9A-Za-z.*+!-]*)", spec)
        if op in ("===", "==", ">=", "~=", ">", "^", "~")
    ]
    if not lows and exact_if_bare and re.match(r"[0-9]", spec):
        lows.append(_version_tuple(spec))
    return max(lows) if lows else None


def _jinja_requirements(path: Path, text: str) -> Iterator[tuple[int, str, bool]]:
    """Yield (line, specifier, is_poetry) for each jinja2 requirement declared in a file."""
    name = path.name.lower()
    pep508 = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*([^;#]*)")

    def is_jinja(dist: str) -> bool:
        return re.sub(r"[-_.]+", "-", dist).lower() == "jinja2"

    def line_of(needle: str) -> int:
        for i, ln in enumerate(text.splitlines(), 1):
            if needle.lower() in ln.lower():
                return i
        return 1

    if name == "pyproject.toml":
        try:
            data = tomllib.loads(text)
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            return
        project = data.get("project") or {}
        deps = list(project.get("dependencies") or [])
        for group in (project.get("optional-dependencies") or {}).values():
            deps += list(group or [])
        for group in (data.get("dependency-groups") or {}).values():
            deps += [d for d in group or [] if isinstance(d, str)]
        for dep in deps:
            m = pep508.match(dep) if isinstance(dep, str) else None
            if m and is_jinja(m.group(1)):
                yield line_of(dep), m.group(2).strip(), False
        poetry = (data.get("tool") or {}).get("poetry") or {}
        tables = [poetry.get("dependencies") or {}]
        tables += [(g or {}).get("dependencies") or {} for g in (poetry.get("group") or {}).values()]
        for table in tables:
            for dist, spec in table.items():
                if is_jinja(dist):
                    version = (
                        spec if isinstance(spec, str) else (spec.get("version", "") if isinstance(spec, dict) else "")
                    )
                    yield line_of(dist), version, True
        return
    for i, raw in enumerate(text.splitlines(), 1):
        candidates = re.findall(r"[\"']\s*([^\"']+)[\"']", raw) if name == "setup.py" else [raw]
        for cand in candidates:
            if cand.lstrip().startswith(("-", "#")):
                continue
            m = pep508.match(cand)
            if m and is_jinja(m.group(1)):
                yield i, m.group(2).strip().strip(",").strip(), False


def _pin_findings(files: list[tuple[Path, str]]) -> list[Finding]:
    out = []
    for path, display in files:
        try:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for line, spec, poetry in _jinja_requirements(path, text):
            low = _lowest_allowed(spec, exact_if_bare=poetry)
            if low is not None and low >= SAFE_JINJA:
                continue
            shown = spec or "no version constraint"
            message = (
                f"jinja2 requirement ({shown}) admits versions below 3.1.6, whose sandbox has known escapes "
                "(CVE-2024-56326, CVE-2025-27516)"
            )
            code = lines[line - 1].strip() if 0 < line <= len(lines) else ""
            out.append(Finding(PIN, display, line, 1, message, code))
    return out


# --------------------------------------------------------------------------- scanning


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _package_prefix(root: Path) -> list[str]:
    parts: list[str] = []
    p = root.resolve()
    while (p / "__init__.py").is_file():
        parts.insert(0, p.name)
        if p.parent == p:
            break
        p = p.parent
    return parts


def _module_name(path: Path, root: Path, prefix: list[str]) -> tuple[str, str]:
    if root.is_file():
        parts = [*prefix, path.stem] if path.name != "__init__.py" else list(prefix)
    else:
        parts = [*prefix, *path.relative_to(root).with_suffix("").parts]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
        return ".".join(parts), ".".join(parts)
    return ".".join(parts), ".".join(parts[:-1])


def _is_test_file(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def _walk(root: Path, include_tests: bool, exclude: tuple[str, ...]) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_dir = here.relative_to(root)

        def excluded(name: str) -> bool:
            rel = (rel_dir / name).as_posix()
            return any(fnmatch(rel, pat) or fnmatch(name, pat) for pat in exclude)

        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in EXCLUDED_DIRS
            and not d.endswith(".egg-info")
            and (include_tests or d not in TEST_DIRS)
            and not excluded(d)
        )
        for name in sorted(filenames):
            if excluded(name):
                continue
            if name.endswith(".py"):
                if include_tests or not _is_test_file(name):
                    yield here / name
            elif _REQ_NAME.match(name) or (here.name == "requirements" and name.endswith(".txt")):
                yield here / name


def scan(
    path: str | os.PathLike[str],
    *,
    include_tests: bool = False,
    exclude: Iterable[str] = (),
    deep: bool = False,
) -> Report:
    """Scan a directory tree (or one file) and return every render site found."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(path)
    prefix = _package_prefix(root if root.is_dir() else root.parent)
    project = _Project()
    python_files = analysed = 0
    skipped: list[tuple[str, str]] = []
    requirement_files: list[tuple[Path, str]] = []
    for f in _walk(root, include_tests, tuple(exclude)):
        display = _display(f)
        if f.suffix != ".py":
            requirement_files.append((f, display))
            continue
        python_files += 1
        if f.name == "setup.py":
            requirement_files.append((f, display))
        try:
            data = f.read_bytes()
        except OSError as exc:
            skipped.append((display, f"unreadable: {exc.strerror or exc}"))
            continue
        if len(data) > _MAX_BYTES:
            skipped.append((display, "larger than 10 MB"))
            continue
        text = data.decode("utf-8-sig", errors="replace")
        name, package = _module_name(f, root, prefix)
        if not (deep or f.name == "__init__.py" or _PREFILTER.search(text)):
            # never mentions jinja or a chat template: register the name only, for import resolution
            project.add_module(name, package, display, "", ast.Module(body=[], type_ignores=[]))
            continue
        try:
            tree = ast.parse(text, filename=display)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            reason = exc.msg if isinstance(exc, SyntaxError) else str(exc) or type(exc).__name__
            skipped.append((display, f"{type(exc).__name__}: {reason}"))
            continue
        project.add_module(name, package, display, text, tree)
        analysed += 1
    project.build()
    project.propagate()
    findings = project.findings()
    if any(f.verdict == SANDBOXED for f in findings):
        findings.extend(_pin_findings(requirement_files))
    order = {REVIEW: 0, CHECK: 1, PIN: 2, SANDBOXED: 3, UNSANDBOXED: 4, UNRESOLVED: 5}
    findings.sort(key=lambda x: (order[x.verdict], x.path, x.line, x.column))
    return Report(str(path), python_files, analysed, skipped, findings)


# --------------------------------------------------------------------------- output


_COLORS = {REVIEW: "1;31", CHECK: "1;33", PIN: "33", SANDBOXED: "32", UNSANDBOXED: "2", UNRESOLVED: "2"}


def _enable_ansi() -> bool:
    if os.name != "nt":
        return True
    try:  # turn on VT processing in legacy Windows consoles
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        return True
    except Exception:
        return False


def render_text(reports: list[Report], show_all: bool = False, color: bool = False) -> str:
    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    out: list[str] = []
    for r in reports:
        for f in r.findings:
            if not (show_all or f.verdict in ACTIONABLE):
                continue
            tag = paint(f.verdict.ljust(7), _COLORS[f.verdict])
            note = paint("  (ignored by comment)", "2") if f.suppressed else ""
            out.append(f"{tag} {f.path}:{f.line}:{f.column}{note}")
            out.append(f"        {f.code}")
            out.append(f"        {paint(f.message, '2')}")
            if f.source_path:
                out.append(f"        {paint('template from', '2')} {f.source_path}:{f.source_line}  {f.source_code}")
            out.append("")
        sandboxed = r.count(SANDBOXED)
        info = sum(r.count(v) for v in (SANDBOXED, UNSANDBOXED, UNRESOLVED))
        counts = ", ".join(f"{r.count(v)} {v}" for v in ACTIONABLE)
        tail = (
            f"{r.analysed} of {r.python_files} Python files analysed in {r.path}, {sandboxed} sandboxed render site(s)"
        )
        if info and not show_all:
            tail += "; --all lists every render site"
        out.append(f"{paint(counts, '1')}  ({tail})")
        for p, reason in r.skipped:
            out.append(paint(f"  skipped {p}: {reason}", "2"))
    return "\n".join(out)


def render_json(reports: list[Report]) -> str:
    payload = {
        "tool": "chat-template-audit",
        "version": __version__,
        "reports": [
            {
                "path": r.path,
                "python_files": r.python_files,
                "analysed": r.analysed,
                "skipped": [{"path": p, "reason": why} for p, why in r.skipped],
                "findings": [asdict(f) for f in r.findings],
            }
            for r in reports
        ],
    }
    return json.dumps(payload, indent=2)


_RULES = [
    {
        "id": "CTA001",
        "name": "UnsandboxedChatTemplate",
        "shortDescription": {"text": "Model-supplied chat template compiled by a non-sandboxed Jinja2 environment"},
        "fullDescription": {
            "text": "A template that comes from a model (tokenizer_config.json, chat_template.jinja, GGUF metadata) is "
            "compiled by jinja2.Environment or jinja2.Template. A crafted template then runs arbitrary Python. "
            "Use jinja2.sandbox.ImmutableSandboxedEnvironment with jinja2>=3.1.6."
        },
        "helpUri": f"{REPO_URL}#rules",
        "defaultConfiguration": {"level": "error"},
        "properties": {"tags": ["security", "external/cwe/cwe-1336"]},
    },
    {
        "id": "CTA002",
        "name": "UnresolvedChatTemplateEnvironment",
        "shortDescription": {"text": "Model-supplied chat template compiled by an environment that needs review"},
        "fullDescription": {
            "text": "The environment that compiles a model-supplied template could not be resolved statically, or is "
            "sandboxed on some code paths only. Check that every path uses a Jinja sandbox."
        },
        "helpUri": f"{REPO_URL}#rules",
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["security", "external/cwe/cwe-1336"]},
    },
    {
        "id": "CTA003",
        "name": "OutdatedJinjaSandbox",
        "shortDescription": {"text": "Jinja sandbox used with a jinja2 requirement that admits versions below 3.1.6"},
        "fullDescription": {
            "text": "The jinja2 sandbox has known escapes before 3.1.6 (CVE-2024-56326 fixed in 3.1.5, "
            "CVE-2025-27516 fixed in 3.1.6). Require jinja2>=3.1.6."
        },
        "helpUri": f"{REPO_URL}#rules",
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["security"]},
    },
]


def _uri(path: str) -> str:
    return Path(path).as_uri() if os.path.isabs(path) else path


def render_sarif(reports: list[Report]) -> str:
    results = []
    for r in reports:
        for f in r.findings:
            if f.verdict not in ACTIONABLE or f.suppressed:
                continue
            location = {
                "physicalLocation": {
                    "artifactLocation": {"uri": _uri(f.path)},
                    "region": {"startLine": f.line, "startColumn": f.column},
                }
            }
            result = {
                "ruleId": RULE_IDS[f.verdict],
                "level": "error" if f.verdict == REVIEW else "warning",
                "message": {
                    "text": f.message + (f". Template from {f.source_path}:{f.source_line}." if f.source_path else ".")
                },
                "locations": [location],
            }
            if f.source_path:
                result["relatedLocations"] = [
                    {
                        "id": 1,
                        "message": {"text": "origin of the model-supplied template"},
                        "physicalLocation": {
                            "artifactLocation": {"uri": _uri(f.source_path)},
                            "region": {"startLine": f.source_line},
                        },
                    }
                ]
            results.append(result)
    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "chat-template-audit",
                        "version": __version__,
                        "informationUri": REPO_URL,
                        "rules": _RULES,
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, indent=2)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chat-template-audit",
        description="Find LLM chat templates compiled by a non-sandboxed Jinja2 environment.",
    )
    parser.add_argument("paths", nargs="*", default=["."], help="directories or files to scan (default: .)")
    parser.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    parser.add_argument("--all", action="store_true", help="also list sandboxed and untraced render sites")
    parser.add_argument("--strict", action="store_true", help="exit 1 on CHECK and PIN findings too, not only REVIEW")
    parser.add_argument("--include-tests", action="store_true", help="scan test files and test directories too")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB", help="skip matching paths (repeatable)"
    )
    parser.add_argument("--deep", action="store_true", help="parse every Python file, not only those mentioning jinja")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    reports = []
    for p in args.paths:
        if not Path(p).exists():
            print(f"chat-template-audit: no such file or directory: {p}", file=sys.stderr)
            return 2
        reports.append(scan(p, include_tests=args.include_tests, exclude=args.exclude, deep=args.deep))

    if args.format == "json":
        print(render_json(reports))
    elif args.format == "sarif":
        print(render_sarif(reports))
    else:
        color = not args.no_color and not os.environ.get("NO_COLOR") and sys.stdout.isatty() and _enable_ansi()
        print(render_text(reports, show_all=args.all, color=color))

    failing = {REVIEW, CHECK, PIN} if args.strict else {REVIEW}
    return 1 if any(f.verdict in failing and not f.suppressed for r in reports for f in r.findings) else 0


if __name__ == "__main__":
    sys.exit(main())
