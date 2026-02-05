#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GITIGNORE = REPO_ROOT / ".gitignore"
DATA_DIR = REPO_ROOT / "DATA"

DUMMY_NAME = "dummy_file.txt"
BACKGROUND_REL = Path("DATA") / "background.png"  # exception that should be tracked

AUTO_BLOCK_HEADER = "# --- auto-added: DATA dummy files + ignore all other DATA files ---"
REQUIRED_GITIGNORE_LINES = [
    AUTO_BLOCK_HEADER,
    # If you (now or later) have broad ignore rules like DATA/**, these ensure dummies & background can still be tracked:
    "!DATA/**/",  # don't ignore directories (needed for any later un-ignores to work)
    f"!DATA/**/{DUMMY_NAME}",
    f"!{BACKGROUND_REL.as_posix()}",
]


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def append_missing_lines(path: Path, lines_to_ensure: list[str]) -> int:
    existing_lines = read_lines(path)
    existing_set = set(existing_lines)

    to_add = [ln for ln in lines_to_ensure if ln not in existing_set]
    if not to_add:
        return 0

    # Add a clean separation from previous content
    prefix = ""
    if existing_lines and existing_lines[-1].strip() != "":
        prefix = "\n"

    new_text = "\n".join(existing_lines) + prefix + "\n" + "\n".join(to_add) + "\n"
    path.write_text(new_text, encoding="utf-8")
    return len(to_add)


def ensure_dummy_files(data_dir: Path) -> int:
    """Create dummy_file.txt in DATA/ and every subdirectory under DATA/."""
    if not data_dir.exists() or not data_dir.is_dir():
        return 0

    created = 0

    # Include DATA itself
    all_dirs = [data_dir] + [p for p in data_dir.rglob("*") if p.is_dir()]
    for d in all_dirs:
        dummy = d / DUMMY_NAME
        if not dummy.exists():
            dummy.write_text("", encoding="utf-8")
            created += 1

    return created


def data_files_to_ignore(data_dir: Path) -> list[str]:
    """
    Return repo-relative POSIX paths for all files under DATA/
    that should be ignored (everything except dummy_file.txt and background.png).
    """
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    ignore_paths: list[str] = []
    for f in data_dir.rglob("*"):
        if not f.is_file():
            continue

        rel = f.relative_to(REPO_ROOT)

        # Never ignore dummy files
        if rel.name == DUMMY_NAME:
            continue

        # Never ignore background.png exception
        if rel.as_posix() == BACKGROUND_REL.as_posix():
            continue

        # Add this specific file path to .gitignore
        ignore_paths.append(rel.as_posix())

    # Stable order to reduce churn
    ignore_paths.sort()
    return ignore_paths


def main() -> None:
    dummy_created = ensure_dummy_files(DATA_DIR)

    # Add required header + unignore rules
    added_required = append_missing_lines(GITIGNORE, REQUIRED_GITIGNORE_LINES)

    # Add one ignore line per non-dummy DATA file
    per_file_ignores = data_files_to_ignore(DATA_DIR)
    added_per_file = append_missing_lines(GITIGNORE, per_file_ignores)

    print(f"Created {dummy_created} '{DUMMY_NAME}' file(s) under DATA/ directories.")
    print(f".gitignore: added {added_required} required line(s).")
    print(f".gitignore: added {added_per_file} per-file ignore line(s) for DATA/ non-dummy files.")
    print("Done.")
    print("\nNext:")
    print("  git add .gitignore DATA")
    print('  git commit -m "Track DATA folders via dummy files; ignore other DATA files"')


if __name__ == "__main__":
    main()
