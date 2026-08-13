from __future__ import annotations

import argparse
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Iterable, List

from src.utils.paths import (
    CHUNKS_DIR,
    CONVERTED_DIR,
    EXTRACTED_DIR,
    LEAF_CHUNKS_DIR,
    LOGS_DIR,
    PARENT_CHUNKS_DIR,
    PROJECT_ROOT,
    REGISTRY_DIR,
)

RESET_DIRS = [
    CHUNKS_DIR,
    CONVERTED_DIR,
    EXTRACTED_DIR,
    REGISTRY_DIR,
    LOGS_DIR,
]

RECREATE_DIRS = [
    LEAF_CHUNKS_DIR,
    PARENT_CHUNKS_DIR,
    CONVERTED_DIR,
    EXTRACTED_DIR,
    REGISTRY_DIR,
    LOGS_DIR,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset local data, registry, and logs.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion of local state.",
    )
    return parser.parse_args()


def _on_rm_error(func, path: str, _exc_info: object) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, onerror=_on_rm_error)
    else:
        path.unlink(missing_ok=True)


def ensure_under_project(paths: Iterable[Path]) -> None:
    root = PROJECT_ROOT.resolve()
    for path in paths:
        if root not in path.resolve().parents and path.resolve() != root:
            raise RuntimeError(f"Refusing to delete outside project: {path}")


def main() -> None:
    args = parse_args()
    ensure_under_project(RESET_DIRS)
    ensure_under_project(RECREATE_DIRS)

    print("Project root:", PROJECT_ROOT)
    print("Will remove:")
    for path in RESET_DIRS:
        print(" -", path)
    print("Will recreate:")
    for path in RECREATE_DIRS:
        print(" -", path)

    if not args.yes:
        print("Re-run with --yes to confirm.")
        return

    for path in RESET_DIRS:
        remove_path(path)
        time.sleep(0.02)

    for path in RECREATE_DIRS:
        path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":  # pragma: no cover
    main()
