from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable

from src.utils.common_utils import utc_now_iso
from src.utils.logging_utils import build_script_logger
from src.utils.paths import BACKUPS_DIR, INDEX_DIR, LOGS_DIR, PROJECT_ROOT, REGISTRY_DIR


INDEX_DIR = INDEX_DIR.resolve()
REGISTRY_PATH = (REGISTRY_DIR / "sources.csv").resolve()
BACKUP_ROOT = (BACKUPS_DIR / "index").resolve()
LOG_PATH = (LOGS_DIR / "backup_index.log").resolve()

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOGGER = build_script_logger("backup_index", LOG_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/restore local Chroma index backups.")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Create a backup from data/index/chroma.")
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKUP_ROOT,
        help=f"Backup root directory (default: {BACKUP_ROOT}).",
    )
    export_parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Backup folder name (default: UTC timestamp).",
    )
    export_parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a .zip archive next to the backup folder.",
    )
    export_parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Do not copy registry/sources.csv into the backup.",
    )

    restore_parser = sub.add_parser("restore", help="Restore a backup into data/index/chroma.")
    restore_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Backup directory or .zip file created by this script.",
    )
    restore_parser.add_argument(
        "--skip-registry",
        action="store_true",
        help="Do not restore registry/sources.csv even if present in backup.",
    )
    restore_parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to overwrite current local index.",
    )
    return parser.parse_args()


def _clear_readonly(func: object, target: str, exc: BaseException) -> None:
    _ = func
    _ = exc
    Path(target).chmod(stat.S_IWRITE)


def remove_dir_with_retry(path: Path, retries: int = 8, base_delay_sec: float = 0.15) -> None:
    if not path.exists():
        return
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            shutil.rmtree(path, onexc=_clear_readonly)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(base_delay_sec * (2**attempt))
    if last_error is not None:
        raise last_error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_file_hashes(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not root.exists():
        return hashes
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = file_path.relative_to(root).as_posix()
        hashes[relative] = sha256_file(file_path)
    return hashes


def build_manifest(backup_dir: Path, include_registry: bool) -> Dict[str, object]:
    chroma_backup_dir = backup_dir / "chroma"
    registry_backup_path = backup_dir / "registry" / "sources.csv"

    manifest: Dict[str, object] = {
        "created_at": utc_now_iso(),
        "project_root": str(PROJECT_ROOT),
        "index_source_dir": str(INDEX_DIR),
        "backup_dir": str(backup_dir),
        "index_files_sha256": collect_file_hashes(chroma_backup_dir),
        "registry_included": include_registry and registry_backup_path.exists(),
    }
    if registry_backup_path.exists():
        manifest["registry_sha256"] = sha256_file(registry_backup_path)
    return manifest


def write_manifest(backup_dir: Path, include_registry: bool) -> Path:
    manifest = build_manifest(backup_dir=backup_dir, include_registry=include_registry)
    manifest_path = backup_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_backup_name(name: str) -> str:
    if name.strip():
        return name.strip()
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def ensure_under_project(paths: Iterable[Path]) -> None:
    root = PROJECT_ROOT.resolve()
    for path in paths:
        resolved = path.resolve()
        if root not in resolved.parents and resolved != root:
            raise ValueError(f"Refusing to use path outside project: {resolved}")


def export_backup(output_dir: Path, name: str, zip_archive: bool, include_registry: bool) -> None:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index directory not found: {INDEX_DIR}")

    output_dir = output_dir.resolve()
    backup_dir = output_dir / build_backup_name(name)
    ensure_under_project([output_dir, backup_dir])

    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")

    (backup_dir / "chroma").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(INDEX_DIR, backup_dir / "chroma")

    if include_registry and REGISTRY_PATH.exists():
        registry_target = backup_dir / "registry" / "sources.csv"
        registry_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REGISTRY_PATH, registry_target)

    manifest_path = write_manifest(backup_dir=backup_dir, include_registry=include_registry)
    LOGGER.info("Backup exported: %s", backup_dir)
    print("Backup folder:", backup_dir)
    print("Manifest:", manifest_path)

    if zip_archive:
        archive_path = Path(shutil.make_archive(str(backup_dir), "zip", root_dir=backup_dir))
        LOGGER.info("Backup archive created: %s", archive_path)
        print("Archive:", archive_path)


def resolve_backup_root(input_path: Path, temp_dir: Path | None) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix.lower() != ".zip":
        raise ValueError("--input must be a backup directory or .zip archive.")
    if temp_dir is None:
        raise RuntimeError("Internal error: temp_dir missing for zip restore.")
    with zipfile.ZipFile(input_path, "r") as archive:
        archive.extractall(temp_dir)
    return temp_dir


def restore_backup(input_path: Path, restore_registry: bool, confirmed: bool) -> None:
    if not confirmed:
        raise ValueError("Restore is destructive. Re-run with --yes.")

    input_path = input_path.resolve()
    temp_path: Path | None = None
    try:
        if input_path.suffix.lower() == ".zip":
            temp_path = Path(tempfile.mkdtemp(prefix="backup_index_restore_"))
        backup_root = resolve_backup_root(input_path=input_path, temp_dir=temp_path)

        manifest_path = backup_root / "manifest.json"
        chroma_backup_dir = backup_root / "chroma"
        registry_backup_path = backup_root / "registry" / "sources.csv"

        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in backup: {backup_root}")
        if not chroma_backup_dir.exists():
            raise FileNotFoundError(f"chroma/ not found in backup: {backup_root}")

        remove_dir_with_retry(INDEX_DIR)
        INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(chroma_backup_dir, INDEX_DIR)

        if restore_registry and registry_backup_path.exists():
            REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(registry_backup_path, REGISTRY_PATH)

        LOGGER.info("Backup restored from: %s", input_path)
        print("Restored index to:", INDEX_DIR)
        if restore_registry and registry_backup_path.exists():
            print("Restored registry to:", REGISTRY_PATH)
        else:
            print("Registry restore: skipped")
    finally:
        if temp_path is not None:
            remove_dir_with_retry(temp_path)


def main() -> None:
    args = parse_args()
    if args.command == "export":
        export_backup(
            output_dir=args.output_dir,
            name=args.name,
            zip_archive=args.zip,
            include_registry=not args.no_registry,
        )
        return
    restore_backup(
        input_path=args.input,
        restore_registry=not args.skip_registry,
        confirmed=args.yes,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
