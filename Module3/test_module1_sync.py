"""
test_module1_sync.py

Module 3 vendors three files verbatim from Module 1 (Prateek's perception
code): fashion_segmenter.py, attribute_analyzer.py, clip_extract_embeddings.py.
They are copied, not imported, so nothing stops them silently drifting if
Module 1 changes and the copies here don't.

This test enforces the "verbatim" claim: each vendored file must be identical
to its Module 1 original, comparing content only (line endings normalized, so
Module 3's LF vs Module 1's CRLF is not flagged).

Run directly:      python test_module1_sync.py
Run under pytest:  pytest test_module1_sync.py

Exit code 0 = in sync; 1 = drift detected (prints which files and where).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Files Module 3 vendors verbatim from Module 1.
VENDORED_FILES = [
    "fashion_segmenter.py",
    "attribute_analyzer.py",
    "clip_extract_embeddings.py",
]

_HERE = Path(__file__).resolve().parent          # .../Module3
_MODULE1 = _HERE.parent / "Module1"


def _normalized_lines(path: Path) -> list[str]:
    """Read a file as text with line endings normalized, so CRLF vs LF is not
    treated as a difference. Returns a list of lines for a readable first-diff."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")


def _first_difference(a: list[str], b: list[str]) -> str | None:
    """Human-readable description of the first differing (1-based) line, or None
    if the two line lists are identical."""
    for i, (la, lb) in enumerate(zip(a, b), start=1):
        if la != lb:
            return f"line {i}:\n    Module1: {la!r}\n    Module3: {lb!r}"
    if len(a) != len(b):
        longer = "Module1" if len(a) > len(b) else "Module3"
        return f"files have different line counts ({len(a)} vs {len(b)}); {longer} is longer"
    return None


def check_sync() -> list[str]:
    """Returns a list of human-readable problem strings (empty = all in sync)."""
    problems: list[str] = []
    for name in VENDORED_FILES:
        m1 = _MODULE1 / name
        m3 = _HERE / name
        if not m1.exists():
            problems.append(f"{name}: missing Module 1 original at {m1}")
            continue
        if not m3.exists():
            problems.append(f"{name}: missing Module 3 copy at {m3}")
            continue
        diff = _first_difference(_normalized_lines(m1), _normalized_lines(m3))
        if diff:
            problems.append(f"{name}: DRIFTED from Module 1 — {diff}")
    return problems


def test_vendored_files_match_module1():
    """pytest entry point."""
    problems = check_sync()
    assert not problems, (
        "Module 3's vendored copies have drifted from Module 1:\n  - "
        + "\n  - ".join(problems)
        + "\nRe-copy the changed file(s) from Module 1, or update both together."
    )


def main() -> int:
    problems = check_sync()
    if problems:
        print("OUT OF SYNC — Module 3 copies differ from Module 1 originals:")
        for p in problems:
            print(f"  - {p}")
        print("\nFix: re-copy the changed file(s) from Module1/ into Module3/,")
        print("or intentionally update both and note the divergence.")
        return 1
    print(f"IN SYNC — all {len(VENDORED_FILES)} vendored files match Module 1 "
          f"(content-identical, line endings ignored):")
    for name in VENDORED_FILES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
