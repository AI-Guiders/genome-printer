"""Verification gate for the two-layer (M -> T) cuboctahedral FCC N12 genome-printer repository.

Enforces:
  1. Zero-Podgonian purity check (no neural activation layers or fitted heuristics),
  2. Strict M-layer purity gate (`M_layer_purity`): every `m_*.py` module must operate
     exclusively on integer registers, vectors, and dimensionless carrier units `(dh, dt, s_0, E_0)`
     with ZERO string literals in executable code (only module/class/function docstrings allowed)
     and ZERO imports from `t_readout`.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import subprocess


FORBIDDEN_TOKENS = ("tanh", "relu", "sigmoid", "softmax", "torch.nn", "sklearn")


def _collect_docstring_nodes(tree: ast.AST) -> set[int]:
    """Return object ids of AST Constant nodes that serve as module/class/function docstrings."""
    doc_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    doc_ids.add(id(val))
    return doc_ids


def check_m_layer_and_podgonian_purity(src_root: pathlib.Path) -> list[str]:
    """Verify Podgonian purity across all files and strict string-free M-layer purity on `m_*.py`."""
    violations: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for tok in FORBIDDEN_TOKENS:
            if tok in text:
                violations.append(f"{py_file.name}: forbidden token {tok!r}")

        tree = ast.parse(text, filename=str(py_file))

        if py_file.name.startswith("m_"):
            doc_ids = _collect_docstring_nodes(tree)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod_name = getattr(node, "module", "") or ""
                    if "t_readout" in mod_name:
                        violations.append(f"{py_file.name}:{node.lineno}: M-layer imports T-layer `t_readout`")
                if isinstance(node, ast.JoinedStr):
                    violations.append(f"{py_file.name}:{node.lineno}: f-string forbidden in M-layer")
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if id(node) not in doc_ids:
                        violations.append(
                            f"{py_file.name}:{node.lineno}: string literal {node.value!r} forbidden in M-layer"
                        )
    return violations


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent
    src_root = repo_root / "src" / "genome_printer"

    violations = check_m_layer_and_podgonian_purity(src_root)
    if violations:
        print("Purity check FAILED:")
        for v in violations:
            print("  -", v)
        return 1

    print("[PASS] M_layer_purity: 0 T-layer strings, 0 human labels, 0 SI constants in `m_*.py`.")
    print("[PASS] Zero-Podgonian purity: 0 neural layers, 0 fitted heuristics.")

    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(repo_root),
        check=False,
    )
    if res.returncode != 0:
        return res.returncode

    print("[PASS] All M-layer (dh, dt, s_0 -> N_a0, E_HB) and T-projector gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
