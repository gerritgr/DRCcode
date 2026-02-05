#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GITIGNORE = REPO_ROOT / ".gitignore"
DATA_DIR = REPO_ROOT / "DATA"

# Exception specified as "DRCcode/DATA/background.png" where DRCcode is the repo name.
# In a repo-root .gitignore, the path should be relative to the repo root:
BACKGROUND_PNG = "DATA/background.png"

AUTO_BLOCK_HEADER = "# --- auto-added: ignore DATA contents but keep empty folders + background.png ---"
REQUIRED_GITIGNORE_LINES = [
    AUTO_BLOCK_HEADER,
    "DATA/**",              # ignore everything under DATA
    "!DATA/**/",            # but do not ignore directories themselves
    "!DATA/**/.gitkeep",    # allow placeholder files to track empty dirs
    f"!{BACKGROUND_PNG}",   # allow the one required exception file
]


def read_gitignore_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def append_missing_gitignore_lines(path: Path, required_lines: list[str]) -> int:
    existing_lines = read_gitignore_lines(path)
    existing_set = set(existing_lines)

    to_add: list[str] = []
    for line in required_lines:
        if line not in existing_set:
            to_add.append(line)

    if not to_add:
        return 0

    # Ensure file ends with a newline and we separate from previous content cleanly.
    prefix = ""
    if existing_lines and existing_lines[-1].strip() != "":
        prefix = "\n"

    path.write_text(
        "\n".join(existing_lines) + prefix + "\n" + "\n".join(to_add) + "\n",
        encoding="utf-8",
    )
    return len(to_add)


def create_gitkeep_in_empty_dirs(data_dir: Path) -> int:
    if not data_dir.exists() or not data_dir.is_dir():
        return 0

    created = 0
    # Walk bottom-up so we handle leaf dirs first
    for d in sorted([p for p in data_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        # Consider "empty" = contains no files and no subdirectories
        # (If it contains only .gitkeep already, treat as not empty)
        entries = list(d.iterdir())
        if not entries:
            gitkeep = d / ".gitkeep"
            gitkeep.write_text("", encoding="utf-8")
            created += 1
        else:
            # If it contains only subdirs, it's not empty for filesystem,
            # but git will still represent it via files in descendants (once .gitkeep exists).
            # If it contains only a .gitkeep, we do nothing.
            pass

    # Also handle DATA/ itself if it's empty
    if DATA_DIR.exists() and DATA_DIR.is_dir() and not any(DATA_DIR.iterdir()):
        gitkeep = DATA_DIR / ".gitkeep"
        gitkeep.write_text("", encoding="utf-8")
        created += 1

    return created


def main() -> None:
    added = append_missing_gitignore_lines(GITIGNORE, REQUIRED_GITIGNORE_LINES)
    gitkeeps = create_gitkeep_in_empty_dirs(DATA_DIR)

    print(f".gitignore: added {added} line(s).")
    if DATA_DIR.exists():
        print(f"DATA/: created {gitkeeps} .gitkeep file(s) in empty folders.")
    else:
        print("DATA/: directory not found (no .gitkeep files created).")
    print("Done.")


if __name__ == "__main__":
    main()
