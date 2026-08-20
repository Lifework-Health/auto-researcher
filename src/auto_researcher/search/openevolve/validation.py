"""Fail-closed AST validation for the PR 6 Python fixture representation."""

from __future__ import annotations

import ast

from auto_researcher.search.openevolve.identity import (
    component_interface_identity,
    source_hash,
)
from auto_researcher.search.openevolve.models import (
    CandidateValidationResult,
    CandidateValidationStatus,
    EvolvableComponentSpec,
    OpenEvolveCandidate,
)

FORBIDDEN_ROOTS = frozenset(
    {
        "asyncio",
        "auto_researcher",
        "builtins",
        "ctypes",
        "http",
        "importlib",
        "inspect",
        "multiprocessing",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "resource",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
    }
)
FORBIDDEN_CALLS = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "setattr",
        "vars",
    }
)
FORBIDDEN_ATTRIBUTE_CALLS = frozenset(
    {
        "system",
        "popen",
        "fork",
        "spawn",
        "connect",
        "bind",
        "request",
        "write_text",
        "write_bytes",
        "unlink",
        "chmod",
        "link",
        "hardlink_to",
        "symlink",
        "symlink_to",
        "mkfifo",
        "mknod",
    }
)


def candidate_static_validation_guidance(
    component: EvolvableComponentSpec,
) -> tuple[str, ...]:
    """Exact development-model guidance derived from the fail-closed validator."""

    return (
        f"Define exactly one synchronous `def {component.entry_point}(configuration)` entry point.",
        "Do not define classes or async functions, and do not use while, global, "
        "nonlocal, or delete statements.",
        "Never assign or augmented-assign through an attribute or subscript. Build "
        "and return a new dictionary instead of mutating `configuration` in place.",
        "Do not use names or attributes beginning with double underscores.",
        "Do not call the entry point recursively.",
        "Do not use these names or module roots: "
        + ", ".join(sorted(FORBIDDEN_ROOTS)),
        "Do not call these builtins: " + ", ".join(sorted(FORBIDDEN_CALLS)),
        "Do not call methods named: "
        + ", ".join(sorted(FORBIDDEN_ATTRIBUTE_CALLS)),
    )


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, component: EvolvableComponentSpec) -> None:
        self.component = component
        self.reasons: set[str] = set()
        self.entry_points: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in FORBIDDEN_ROOTS or root not in self.component.allowed_imports:
                self.reasons.add("candidate_forbidden_import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if (
            node.level
            or root in FORBIDDEN_ROOTS
            or root not in self.component.allowed_imports
        ):
            self.reasons.add("candidate_forbidden_import")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.component.entry_point:
            self.entry_points.append(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == self.component.entry_point:
            self.entry_points.append(node)
        self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.reasons.add("candidate_forbidden_operation")
        root = node.value
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in FORBIDDEN_ROOTS:
            self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in FORBIDDEN_ROOTS:
            self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if (
                node.func.id in FORBIDDEN_CALLS
                or node.func.id == self.component.entry_point
            ):
                self.reasons.add("candidate_forbidden_operation")
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_ATTRIBUTE_CALLS:
                self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.reasons.add("candidate_forbidden_operation")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.reasons.add("candidate_forbidden_operation")

    def visit_Delete(self, node: ast.Delete) -> None:
        self.reasons.add("candidate_forbidden_operation")

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(
            isinstance(target, (ast.Attribute, ast.Subscript))
            for target in node.targets
        ):
            self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, (ast.Attribute, ast.Subscript)):
            self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.reasons.add("candidate_forbidden_operation")
        self.generic_visit(node)


def validate_candidate(
    candidate: OpenEvolveCandidate,
    component: EvolvableComponentSpec,
) -> CandidateValidationResult:
    reasons: set[str] = set()
    payload = candidate.source_payload
    try:
        encoded = payload.encode("utf-8", errors="strict")
    except UnicodeError:
        encoded = b""
        reasons.add("candidate_patch_invalid")
    if "\x00" in payload or len(encoded) > component.maximum_source_bytes:
        reasons.add("candidate_patch_invalid")
    if candidate.mutable_file != component.mutable_file:
        reasons.add("candidate_patch_invalid")
    if source_hash(payload) != candidate.source_hash:
        reasons.add("candidate_patch_invalid")
    interface_hash = component_interface_identity(component)
    if candidate.component_interface_hash != interface_hash:
        reasons.add("candidate_interface_mismatch")
    try:
        tree = ast.parse(payload, filename=component.mutable_file, mode="exec")
    except SyntaxError:
        reasons.add("candidate_static_validation_failed")
    else:
        visitor = _SafetyVisitor(component)
        visitor.visit(tree)
        reasons.update(visitor.reasons)
        if len(visitor.entry_points) != 1:
            reasons.add("candidate_interface_mismatch")
        else:
            entry = visitor.entry_points[0]
            arguments = entry.args
            if (
                len(arguments.args) != 1
                or arguments.args[0].arg != "configuration"
                or arguments.posonlyargs
                or arguments.kwonlyargs
                or arguments.vararg is not None
                or arguments.kwarg is not None
                or arguments.defaults
            ):
                reasons.add("candidate_interface_mismatch")
    ordered = tuple(sorted(reasons))
    code = ordered[0] if ordered else None
    return CandidateValidationResult(
        candidate_id=candidate.candidate_id,
        status=(
            CandidateValidationStatus.INVALID
            if ordered
            else CandidateValidationStatus.VALID
        ),
        safe_error_code=code,
        reason_codes=ordered,
        source_hash=candidate.source_hash,
        interface_hash=interface_hash,
    )
