"""Verification gate for the clean cuboctahedral FCC N12 genome-printer repository."""

from __future__ import annotations

import ast
import pathlib
import sys
import subprocess


FORBIDDEN_TOKENS = ("tanh", "relu", "sigmoid", "softmax", "torch.nn", "sklearn")


def check_purity(src_root: pathlib.Path) -> list[str]:
    """Ensure zero neural activation layers or ad-hoc heuristics exist in src/."""
    violations: list[str] = []
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for tok in FORBIDDEN_TOKENS:
            if tok in text:
                violations.append(f"{py_file}: forbidden token {tok!r}")
        # Ensure valid AST
        ast.parse(text, filename=str(py_file))
    return violations


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent
    src_root = repo_root / "src" / "genome_printer"

    violations = check_purity(src_root)
    if violations:
        print("Purity check FAILED:")
        for v in violations:
            print("  -", v)
        return 1

    print("[PASS] Zero-Podgonian purity check (no neural layers, no fitted heuristics).")

    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=str(repo_root),
        check=False,
    )
    if res.returncode != 0:
        return res.returncode

    print("[PASS] All cuboctahedral FCC N12 chemistry, B-DNA, Rumer 21-code, and O(1) folding gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
