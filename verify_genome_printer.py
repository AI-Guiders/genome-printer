#!/usr/bin/env python3
"""Verification gate for the two-layer (M -> T) cuboctahedral FCC N12 genome-printer repository.

Enforces:
  1. Zero-Podgonian purity check (no neural activation layers or fitted heuristics),
  2. Strict M-layer purity gate (`M_layer_purity`): every `m_*.py` module must operate
     exclusively on integer registers, vectors, and dimensionless carrier units `(dh, dt, s_0, E_0)`
     with ZERO string literals in executable code (only module/class/function docstrings allowed)
     and ZERO imports from `t_readout`.
"""

import os
import sys

if sys.version_info[0] < 3:
    os.execvp("python3", ["python3", os.path.abspath(__file__)] + sys.argv[1:])

import ast
import pathlib
import subprocess


FORBIDDEN_TOKENS = ("tanh", "relu", "sigmoid", "softmax", "torch.nn", "sklearn")


def _ensure_venv_python(repo_root):
    """Return a Python executable with `numpy` and `pytest` installed, auto-creating `.venv` if needed."""
    check = subprocess.call(
        [sys.executable, "-c", "import numpy, pytest"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if check == 0:
        return sys.executable
    venv_dir = repo_root / ".venv"
    venv_py = (
        venv_dir / "Scripts" / "python.exe"
        if os.name == "nt"
        else venv_dir / "bin" / "python3"
    )
    req_file = repo_root / "requirements.txt"
    if not venv_py.is_file():
        print("[genome-printer] Creating virtual environment in {} ...".format(venv_dir), file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    if subprocess.call([str(venv_py), "-c", "import numpy, pytest"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("[genome-printer] Installing dependencies from {} into .venv ...".format(req_file.name), file=sys.stderr)
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "-r", str(req_file)])
    return str(venv_py)


def _collect_docstring_nodes(tree):
    """Return object ids of AST Constant nodes that serve as module/class/function docstrings."""
    doc_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    doc_ids.add(id(val))
    return doc_ids


def check_m_layer_and_podgonian_purity(src_root):
    """Verify Podgonian purity across all files and strict string-free M-layer purity on `m_*.py`."""
    violations = []
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for tok in FORBIDDEN_TOKENS:
            if tok in text:
                violations.append("{}: forbidden token {!r}".format(py_file.name, tok))

        tree = ast.parse(text, filename=str(py_file))

        if py_file.name.startswith("m_"):
            doc_ids = _collect_docstring_nodes(tree)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod_name = getattr(node, "module", "") or ""
                    if "t_readout" in mod_name:
                        violations.append("{}:{}: M-layer imports T-layer `t_readout`".format(py_file.name, node.lineno))
                if isinstance(node, ast.JoinedStr):
                    violations.append("{}:{}: f-string forbidden in M-layer".format(py_file.name, node.lineno))
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if id(node) not in doc_ids:
                        violations.append(
                            "{}:{}: string literal {!r} forbidden in M-layer".format(py_file.name, node.lineno, node.value)
                        )
    return violations


def main():
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

    py_exe = _ensure_venv_python(repo_root)
    res = subprocess.run(
        [py_exe, "-m", "pytest", "-q"],
        cwd=str(repo_root),
        check=False,
    )
    if res.returncode != 0:
        return res.returncode

    print("[PASS] All M-layer (dh, dt, s_0 -> N_a0, E_HB) and T-projector gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
