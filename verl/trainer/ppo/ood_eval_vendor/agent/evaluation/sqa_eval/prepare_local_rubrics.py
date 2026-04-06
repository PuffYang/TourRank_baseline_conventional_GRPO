#!/usr/bin/env python3
"""Prepare and validate the local SQA rubric directory layout."""

from __future__ import annotations

from pathlib import Path


FILE_DIR = Path(__file__).resolve().parent
RUBRICS_DIR = FILE_DIR / "data" / "tasks" / "sqa"
EXPECTED_FILES = (
    "rubrics_v1_recomputed.json",
    "rubrics_v2_recomputed.json",
)


def main() -> None:
    RUBRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"SQA local rubric directory is ready: {RUBRICS_DIR}")
    print("Expected files:")
    missing = []
    for name in EXPECTED_FILES:
        path = RUBRICS_DIR / name
        exists = path.exists()
        print(f"  - {name}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(path)

    if missing:
        print("\nNext step: copy the missing rubric files into the directory above.")
    else:
        print("\nAll required local rubric files are present.")


if __name__ == "__main__":
    main()
