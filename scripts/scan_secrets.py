"""
Lightweight repository secret scanner.
Usage:
  python scripts/scan_secrets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PATTERNS = [
    ("OpenAI API key", re.compile(r"\bsk-(proj|live|test)?-[A-Za-z0-9_\-]{20,}\b")),
    ("Generic Bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9_\-\.=]{20,}\b")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]

ALLOW_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".db",
    ".zip",
    ".woff",
    ".woff2",
}


def tracked_files() -> list[Path]:
    repo = Path(__file__).resolve().parents[1]
    skip_dirs = {".git", "node_modules", "dist", "__pycache__", ".venv", "venv"}
    files: list[Path] = []
    for p in repo.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        name = p.name.lower()
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            continue
        files.append(p.relative_to(repo))
    return files


def main() -> int:
    findings = []
    for p in tracked_files():
        if p.suffix.lower() in ALLOW_EXT:
            continue
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pattern in PATTERNS:
            for m in pattern.finditer(txt):
                findings.append((str(p), label, m.group(0)[:64]))

    if findings:
        print("Secret scan failed. Potential secrets found:")
        for fp, label, sample in findings:
            print(f"- {fp}: {label} -> {sample}")
        return 1

    print("Secret scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
