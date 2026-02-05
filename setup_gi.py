#!/usr/bin/env python3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
GITIGNORE = REPO_ROOT / ".gitignore"
DATA_DIR = REPO_ROOT / "DATA"

DUMMY_NAME = "dummy_file.txt"
BACKGROUND_REL = Path("DATA/background.png")

AUTO_BLOCK_HEADER = "# --- auto-added: DATA dummy files + ignore all other DATA files ---"
REQUIRED_GITIGNORE_LINES = [
    AUTO_BLOCK_HEADER,
    "!DATA/**/",                     # allow directories
    f"!DATA/**/{DUMMY_NAME}",        # allow dummy files
    f"!{BACKGROUND_REL.as_posix()}", # allow background.png
]


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def append_missing_lines(path: Path, lines: list[str]) -> int:
    existing = read_lines(path)
    existing_set = set(existing)

    to_add = [l for l in lines if l not in existing_set]
    if not to_add:
        return 0

    sep = "\n" if existing and existing[-1].strip() else ""
    path.write_text(
        "\n".join(existing) + sep + "\n" + "\n".join(to_add) + "\n",
        encoding="utf-8",
    )
    return len(to_add)


def ensure_dummy_files_leaf_dirs(data_dir: Path) -> int:
    """
    Create dummy_file.txt ONLY in leaf directories under DATA/
    (directories that contain no subdirectories).
    """
    if not data_dir.exists():
        return 0

    created = 0
    for d in data_dir.rglob("*"):
        if not d.is_dir():
            continue

        has_subdirs = any(p.is_dir() for p in d.iterdir())
        if has_subdirs:
            continue  # skip non-leaf dirs

        dummy = d / DUMMY_NAME
        if not dummy.exists():
            dummy.write_text("", encoding="utf-8")
            created += 1

    return created


def data_files_to_ignore(data_dir: Path) -> list[str]:
    """
    Return repo-relative paths for all DATA files that are NOT:
    - dummy_file.txt
    - DATA/background.png
    """
    if not data_dir.exists():
        return []

    ignores = []
    for f in data_dir.rglob("*"):
        if not f.is_file():
            continue

        rel = f.relative_to(REPO_ROOT).as_posix()

        if f.name == DUMMY_NAME:
            continue
        if rel == BACKGROUND_REL.as_posix():
            continue

        ignores.append(rel)

    return sorted(ignores)


def main():
    dummy_created = ensure_dummy_files_leaf_dirs(DATA_DIR)

    added_required = append_missing_lines(GITIGNORE, REQUIRED_GITIGNORE_LINES)
    added_files = append_missing_lines(GITIGNORE, data_files_to_ignore(DATA_DIR))

    print(f"Created {dummy_created} dummy file(s) in leaf DATA directories.")
    print(f".gitignore: added {added_required} structural rule(s).")
    print(f".gitignore: added {added_files} per-file ignore rule(s).")
    print("Next:")
    print("  git add .gitignore DATA")
    print('  git commit -m "Track DATA leaf folders via dummy files; ignore real DATA files"')


if __name__ == "__main__":
    main()
