"""The frozen ``aaa.python.v0`` Python subset, enforced by AST validation.

Validation is an allow-list over AST node types, operators, builtins and
identifiers, plus declared size limits. Anything not explicitly allowed is
refused: imports, attribute access of any kind (which also blocks every dunder
reflection escape), ``while`` loops, lambdas, comprehensions, ``try``/``with``,
f-strings, exponentiation, true division, and calls to anything other than an
allowed builtin or a function defined *earlier* in the program (so there is no
recursion). ``for`` loops iterate only ``range`` over integer literals, so
every program terminates within a known bound.

The validator is the first containment layer. The sandboxed subprocess in
:mod:`.oracle` is the second; it does not rely on the first.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any

from . import spec as spec_module

_ALLOWED_STRUCTURAL = {
    "Module",
    "Load",
    "Store",
    "arguments",
    "arg",
    "keyword",
}


class SubsetError(ValueError):
    """The program is outside the frozen subset."""


def _rules(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    subset = (spec or spec_module.load())["subset"]
    return {
        "nodes": set(subset["statements"]) | set(subset["expressions"]) | set(subset["operators"]),
        "builtins": set(subset["builtins"]),
        "reserved": set(subset["reserved_names"]),
        "identifier": re.compile(subset["identifier_pattern"]),
        "limits": subset["limits"],
    }


def _loop_depth(node: ast.AST, depth: int = 0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        deepest = max(deepest, _loop_depth(child, depth + isinstance(child, ast.For)))
    return deepest


def validate(source: str, spec: Mapping[str, Any] | None = None) -> ast.Module:
    """Parse and validate ``source``; raise :class:`SubsetError` naming the first violation."""

    rules = _rules(spec)
    limits = rules["limits"]
    if not isinstance(source, str):
        raise SubsetError("source must be text")
    if len(source.encode("utf-8")) > limits["max_source_bytes"]:
        raise SubsetError("source exceeds the byte limit")
    if source.count("\n") + 1 > limits["max_lines"]:
        raise SubsetError("source exceeds the line limit")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise SubsetError(f"not valid Python: {error.msg}") from error
    nodes = list(ast.walk(tree))
    if len(nodes) > limits["max_ast_nodes"]:
        raise SubsetError("program exceeds the AST node limit")
    defined: list[str] = []
    for node in nodes:
        kind = type(node).__name__
        if kind not in rules["nodes"] and kind not in _ALLOWED_STRUCTURAL:
            raise SubsetError(f"line {getattr(node, 'lineno', '?')}: {kind} is not in the subset")
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, int):
                if abs(value) > limits["max_int_literal"]:
                    raise SubsetError(f"line {node.lineno}: integer literal out of range")
            elif isinstance(value, str):
                if len(value) > limits["max_string_literal"] or not value.isprintable():
                    raise SubsetError(f"line {node.lineno}: string literal too long or not printable")
            else:
                raise SubsetError(f"line {node.lineno}: literal type {type(value).__name__} is not allowed")
        elif isinstance(node, ast.List) and len(node.elts) > limits["max_list_literal"]:
            raise SubsetError(f"line {node.lineno}: list literal too long")
        elif isinstance(node, ast.Name):
            name = node.id
            if name in rules["builtins"]:
                if not isinstance(node.ctx, ast.Load):
                    raise SubsetError(f"line {node.lineno}: builtin {name} may not be rebound")
            elif name in rules["reserved"] or not rules["identifier"].match(name):
                raise SubsetError(f"line {node.lineno}: name {name!r} is not allowed")
        elif isinstance(node, ast.Subscript):
            if not isinstance(node.value, ast.Name | ast.List) or isinstance(node.slice, ast.Slice):
                raise SubsetError(f"line {node.lineno}: only name[index] or [literal list][index] subscripts")
        elif isinstance(node, ast.arguments):
            if node.posonlyargs or node.vararg or node.kwonlyargs or node.kwarg or node.defaults:
                raise SubsetError("functions take plain positional arguments only")
        elif isinstance(node, ast.arg):
            if node.annotation is not None:
                raise SubsetError("annotations are not allowed")
            if node.arg in rules["reserved"] or not rules["identifier"].match(node.arg):
                raise SubsetError(f"parameter name {node.arg!r} is not allowed")
        elif isinstance(node, ast.FunctionDef):
            if node.name in rules["reserved"] or not rules["identifier"].match(node.name):
                raise SubsetError(f"line {node.lineno}: function name {node.name!r} is not allowed")
        elif isinstance(node, ast.keyword):
            raise SubsetError("keyword arguments are not allowed")
    for statement in ast.walk(tree):
        if isinstance(statement, ast.FunctionDef):
            if statement.decorator_list or statement.returns is not None:
                raise SubsetError(f"line {statement.lineno}: decorators and annotations are not allowed")
            if getattr(statement, "type_params", None):
                raise SubsetError(f"line {statement.lineno}: type parameters are not allowed")
            if statement not in tree.body:
                raise SubsetError(f"line {statement.lineno}: functions must be defined at top level")
    for statement in tree.body:
        _check_calls(statement, defined, rules)
        if isinstance(statement, ast.FunctionDef):
            defined.append(statement.name)
    for node in nodes:
        if isinstance(node, ast.For):
            _check_for(node, limits)
        if isinstance(node, ast.AugAssign | ast.Assign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name | ast.Subscript):
                raise SubsetError(f"line {node.lineno}: assign to exactly one name or name[index]")
    if _loop_depth(tree) > limits["max_loop_depth"]:
        raise SubsetError("loops are nested too deeply")
    return tree


def _check_for(node: ast.For, limits: Mapping[str, int]) -> None:
    if node.orelse:
        raise SubsetError(f"line {node.lineno}: for-else is not allowed")
    if not isinstance(node.target, ast.Name):
        raise SubsetError(f"line {node.lineno}: loop target must be a name")
    call = node.iter
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"):
        raise SubsetError(f"line {node.lineno}: loops iterate range(...) only")
    if not 1 <= len(call.args) <= 3:
        raise SubsetError(f"line {node.lineno}: range takes one to three arguments")
    for argument in call.args:
        value = argument.value if isinstance(argument, ast.Constant) else None
        if isinstance(argument, ast.UnaryOp) and isinstance(argument.operand, ast.Constant):
            value = argument.operand.value
        if isinstance(value, bool) or not isinstance(value, int) or abs(value) > limits["max_range_bound"]:
            raise SubsetError(f"line {node.lineno}: range arguments must be small integer literals")


def _check_calls(statement: ast.stmt, defined: list[str], rules: Mapping[str, Any]) -> None:
    own = statement.name if isinstance(statement, ast.FunctionDef) else None
    for node in ast.walk(statement):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            raise SubsetError(f"line {node.lineno}: only direct calls by name are allowed")
        name = node.func.id
        if name == own:
            raise SubsetError(f"line {node.lineno}: recursion is not allowed")
        if name not in rules["builtins"] and name not in defined:
            raise SubsetError(
                f"line {node.lineno}: call to {name!r} is not an allowed builtin or earlier function"
            )
