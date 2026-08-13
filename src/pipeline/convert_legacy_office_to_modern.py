from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from dotenv import load_dotenv

from src.utils.fs_utils import path_is_under
from src.utils.logging_utils import build_script_logger
from src.utils.paths import CONVERTED_DIR, LOGS_DIR

load_dotenv()

DEFAULT_CONVERTED_DIR = CONVERTED_DIR.resolve()
LOG_PATH = (LOGS_DIR / "convert_legacy_office_to_modern.log").resolve()
DEFAULT_WINDOWS_SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")

LEGACY_TO_MODERN_FORMAT: Dict[str, str] = {
    ".doc": "docx",
    ".ppt": "pptx",
}
MODERN_CONVERTED_SUFFIXES = {".docx", ".pptx"}


@dataclass(frozen=True)
class ConversionPlan:
    source: Path
    destination: Path
    output_format: str


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    destination: Path
    status: str  # OK | SKIP | FAIL | DRYRUN
    detail: str


LOGGER = build_script_logger("convert_legacy_office", LOG_PATH)


def parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Convert .doc/.ppt files to .docx/.pptx with LibreOffice.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Root folder scanned for legacy files (default: RAW_DIR from .env).",
    )
    parser.add_argument(
        "--converted-dir",
        type=Path,
        default=Path(os.getenv("CONVERTED_DIR", str(DEFAULT_CONVERTED_DIR))).expanduser(),
        help="Output root for converted files (default: data/converted).",
    )
    parser.add_argument(
        "--soffice",
        type=str,
        default=os.getenv("SOFFICE_PATH", "").strip(),
        help="Path to soffice executable (default: SOFFICE_PATH env, PATH, then Windows default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reconversion even if target appears up to date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned conversions without executing LibreOffice.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=int(os.getenv("SOFFICE_TIMEOUT_SEC", "180")),
        help="Timeout per file conversion in seconds (default: 180).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files processed (0 means no limit).",
    )
    parser.add_argument(
        "--keep-stale-converted",
        action="store_true",
        help="Do not delete orphan converted .docx/.pptx files in converted-dir.",
    )
    return parser.parse_args()


def resolve_raw_dir(raw_dir: Optional[Path]) -> Path:
    if raw_dir is None:
        raw_value = os.getenv("RAW_DIR", "").strip()
        if not raw_value:
            raise FileNotFoundError("RAW_DIR not set. Define it in .env or pass --raw-dir.")
        raw_dir = Path(raw_value).expanduser()
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"RAW_DIR is not a directory: {raw_dir}")
    return raw_dir


def resolve_converted_dir(path: Path) -> Path:
    converted = path.resolve()
    converted.mkdir(parents=True, exist_ok=True)
    return converted


def find_in_path(candidates: List[str]) -> Optional[Path]:
    path_entries = os.getenv("PATH", "").split(os.pathsep)
    for entry in path_entries:
        if not entry:
            continue
        root = Path(entry)
        for name in candidates:
            candidate = (root / name)
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
    return None


def resolve_soffice_path(cli_path: str) -> Path:
    raw = (cli_path or "").strip()
    if raw:
        candidate = Path(raw).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"soffice executable not found: {candidate}")

    in_path = find_in_path(["soffice.exe", "soffice"])
    if in_path:
        return in_path

    if DEFAULT_WINDOWS_SOFFICE.exists():
        return DEFAULT_WINDOWS_SOFFICE.resolve()

    raise FileNotFoundError(
        "LibreOffice not found. Set SOFFICE_PATH in .env or pass --soffice."
    )


def iter_legacy_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("~$"):
            continue
        if path.suffix.lower() not in LEGACY_TO_MODERN_FORMAT:
            continue
        yield path.resolve()


def build_conversion_plan(source: Path, raw_dir: Path, converted_dir: Path) -> ConversionPlan:
    relative = source.relative_to(raw_dir)
    output_format = LEGACY_TO_MODERN_FORMAT[source.suffix.lower()]
    destination = (converted_dir / relative).with_suffix(f".{output_format}")
    return ConversionPlan(source=source, destination=destination, output_format=output_format)


def is_up_to_date(source: Path, destination: Path) -> bool:
    if not destination.exists():
        return False
    if destination.stat().st_size <= 0:
        return False
    return destination.stat().st_mtime >= source.stat().st_mtime


def find_output_candidate(destination: Path) -> Optional[Path]:
    if destination.exists():
        return destination

    wanted_suffix = destination.suffix.lower()
    stem = destination.stem.lower()
    for candidate in destination.parent.iterdir():
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() != wanted_suffix:
            continue
        if candidate.stem.lower() != stem:
            continue
        return candidate
    return None


def run_soffice(
    soffice: Path,
    source: Path,
    output_dir: Path,
    output_format: str,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(soffice),
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--norestore",
        "--convert-to",
        output_format,
        "--outdir",
        str(output_dir),
        str(source),
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_sec,
    )


def convert_one(
    plan: ConversionPlan,
    soffice: Path,
    timeout_sec: int,
    force: bool,
    dry_run: bool,
) -> ConversionResult:
    source = plan.source
    destination = plan.destination

    if not force and is_up_to_date(source, destination):
        return ConversionResult(source, destination, "SKIP", "up-to-date")

    if dry_run:
        return ConversionResult(source, destination, "DRYRUN", "conversion planned")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = run_soffice(
            soffice=soffice,
            source=source,
            output_dir=destination.parent,
            output_format=plan.output_format,
            timeout_sec=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return ConversionResult(source, destination, "FAIL", f"timeout after {timeout_sec}s")
    except Exception as exc:  # noqa: BLE001
        return ConversionResult(source, destination, "FAIL", f"execution error: {exc}")

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if result.returncode != 0:
        detail = stderr or stdout or f"soffice return code {result.returncode}"
        return ConversionResult(source, destination, "FAIL", detail)

    candidate = find_output_candidate(destination)
    if candidate is None:
        return ConversionResult(source, destination, "FAIL", "converted file not found")

    if candidate.resolve() != destination.resolve():
        candidate.replace(destination)

    if destination.stat().st_size <= 0:
        return ConversionResult(source, destination, "FAIL", "converted file is empty")

    return ConversionResult(source, destination, "OK", "converted")


def summarize(results: List[ConversionResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {"OK": 0, "SKIP": 0, "FAIL": 0, "DRYRUN": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def expected_converted_paths(legacy_files: List[Path], raw_dir: Path, converted_dir: Path) -> Set[str]:
    expected: Set[str] = set()
    for source in legacy_files:
        plan = build_conversion_plan(source=source, raw_dir=raw_dir, converted_dir=converted_dir)
        expected.add(str(plan.destination.resolve()))
    return expected


def purge_stale_converted_files(
    converted_dir: Path,
    expected_paths: Set[str],
    dry_run: bool,
) -> int:
    removed = 0
    for path in converted_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in MODERN_CONVERTED_SUFFIXES:
            continue

        resolved = path.resolve()
        if not path_is_under(converted_dir, resolved):
            LOGGER.warning("Skipped stale purge outside converted dir: %s", resolved)
            continue
        if str(resolved) in expected_paths:
            continue

        if dry_run:
            LOGGER.info("[DRYRUN][STALE] %s", resolved)
            removed += 1
            continue

        resolved.unlink()
        LOGGER.info("[REMOVED][STALE] %s", resolved)
        removed += 1
    return removed


def run_conversion(
    *,
    raw_dir: Path,
    converted_dir: Path,
    soffice_path: str,
    force: bool,
    dry_run: bool,
    timeout_sec: int,
    limit: int,
    keep_stale_converted: bool,
) -> Dict[str, object]:
    resolved_raw = resolve_raw_dir(raw_dir)
    resolved_converted = resolve_converted_dir(converted_dir)
    soffice = resolve_soffice_path(soffice_path)

    LOGGER.info("Run started | raw_dir=%s | converted_dir=%s", resolved_raw, resolved_converted)
    LOGGER.info("LibreOffice executable: %s", soffice)

    all_legacy_files = sorted(iter_legacy_files(resolved_raw))
    expected_paths = expected_converted_paths(
        legacy_files=all_legacy_files,
        raw_dir=resolved_raw,
        converted_dir=resolved_converted,
    )
    stale_count = 0
    if not keep_stale_converted:
        stale_count = purge_stale_converted_files(
            converted_dir=resolved_converted,
            expected_paths=expected_paths,
            dry_run=dry_run,
        )
        if stale_count:
            if dry_run:
                LOGGER.info("Stale converted files detected=%d (dry-run)", stale_count)
            else:
                LOGGER.info("Stale converted files removed=%d", stale_count)

    legacy_files = all_legacy_files
    if limit > 0:
        legacy_files = legacy_files[:limit]

    if not all_legacy_files:
        LOGGER.info("No legacy files found.")
    else:
        LOGGER.info("Found %d legacy files", len(all_legacy_files))

    results: List[ConversionResult] = []
    for source in legacy_files:
        plan = build_conversion_plan(source=source, raw_dir=resolved_raw, converted_dir=resolved_converted)
        result = convert_one(
            plan=plan,
            soffice=soffice,
            timeout_sec=timeout_sec,
            force=force,
            dry_run=dry_run,
        )
        results.append(result)

        if result.status == "FAIL":
            LOGGER.error("[%s] %s -> %s | %s", result.status, source, plan.destination, result.detail)
        else:
            LOGGER.info("[%s] %s -> %s | %s", result.status, source, plan.destination, result.detail)

    counts = summarize(results)
    if stale_count:
        if dry_run:
            counts["STALE_DRYRUN"] = stale_count
        else:
            counts["STALE_REMOVED"] = stale_count

    LOGGER.info("Run completed | summary=%s", counts)

    return {
        "summary": counts,
        "stale_removed": stale_count,
        "raw_dir": str(resolved_raw),
        "converted_dir": str(resolved_converted),
        "total_legacy_files": len(all_legacy_files),
        "processed_files": len(results),
        "dry_run": dry_run,
    }


def main() -> None:  # pragma: no cover
    args = parse_args()
    result = run_conversion(
        raw_dir=args.raw_dir,
        converted_dir=args.converted_dir,
        soffice_path=args.soffice,
        force=args.force,
        dry_run=args.dry_run,
        timeout_sec=args.timeout_sec,
        limit=args.limit,
        keep_stale_converted=args.keep_stale_converted,
    )

    if result.get("total_legacy_files", 0) == 0:
        print("No legacy .doc/.ppt files found.")

    print(f"Conversion summary: {result['summary']}")
    print(f"Output dir: {result['converted_dir']}")


if __name__ == "__main__":  # pragma: no cover
    main()
