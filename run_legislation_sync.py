from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


JURISDICTIONS = [
    "Australian Capital Territory",
    "Commonwealth",
    "New South Wales",
    "Northern Territory",
    "Queensland",
    "South Australia",
    "Tasmania",
    "Victoria",
    "Western Australia",
]


def _normalize_header_cell(cell: str) -> str:
    c = (cell or "").strip().replace("\ufeff", "").lower()
    return "_".join(c.split())


def _is_header_row(row: list[str]) -> bool:
    lowered = {_normalize_header_cell(c) for c in row}
    return "library_id" in lowered and "url" in lowered


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        raw_rows = list(reader)

    if not raw_rows:
        return [], []

    start_idx = 0
    if not _is_header_row(raw_rows[0]):
        start_idx = 1

    if len(raw_rows) <= start_idx:
        return [], []

    header = [_normalize_header_cell(h) for h in raw_rows[start_idx]]
    out_rows: list[dict[str, str]] = []
    for r in raw_rows[start_idx + 1 :]:
        d: dict[str, str] = {}
        for i, name in enumerate(header):
            if not name:
                continue
            d[name] = (r[i] if i < len(r) else "").strip()
        out_rows.append(d)
    return header, out_rows


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        for r in rows:
            writer.writerow([r.get(col, "") for col in header])


def _iter_source_catalogues(legislation_dir: Path) -> Iterable[tuple[str, Path]]:
    for jurisdiction in JURISDICTIONS:
        data_dir = legislation_dir / jurisdiction / "data"
        if not data_dir.exists():
            continue
        matches = sorted(data_dir.glob("*_legislation_catalogue.csv"))
        if not matches:
            continue
        # Prefer a stable non-tmp file.
        non_tmp = [p for p in matches if not p.name.endswith(".tmp")]
        yield jurisdiction, (non_tmp[0] if non_tmp else matches[0])


def build_combined_catalogue(*, legislation_dir: Path, combined_path: Path) -> None:
    headers: set[str] = set()
    by_jur: dict[str, list[dict[str, str]]] = {}

    for jurisdiction, csv_path in _iter_source_catalogues(legislation_dir):
        header, rows = _read_csv_rows(csv_path)
        if not header:
            continue
        for r in rows:
            r.setdefault("jurisdiction", jurisdiction)
        headers.update(header)
        headers.add("jurisdiction")
        by_jur[jurisdiction] = rows

    if not by_jur:
        raise SystemExit(
            f"No source catalogues found under {legislation_dir}. Did you populate Legislation/*/data/*_legislation_catalogue.csv?"
        )

    # Put key fields first, keep the rest stable.
    preferred = [
        "jurisdiction",
        "library_id",
        "title",
        "type",
        "frl_id",
        "version_id",
        "series",
        "source",
        "url",
        "when_scraped",
        "last_successful_scrape",
        "status",
        "error",
        "active",
        "citation",
        "citation_status",
        "content_format",
        "content_url",
        "download_url",
    ]
    ordered = []
    for c in preferred:
        if c in headers:
            ordered.append(c)
    for c in sorted(headers):
        if c not in ordered:
            ordered.append(c)

    combined_rows: list[dict[str, str]] = []
    for jurisdiction in JURISDICTIONS:
        combined_rows.extend(by_jur.get(jurisdiction, []))

    _write_csv(combined_path, ordered, combined_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all legislation sync bots from a single combined catalogue",
    )
    p.add_argument(
        "--combined-csv",
        default=str(
            Path(__file__).resolve().parent
            / "raw"
            / "data"
            / "Legislative_materials"
            / "legislation_catalogue.csv"
        ),
        help="Combined catalogue CSV (source of truth)",
    )
    p.add_argument(
        "--data-dir",
        default=str(
            Path(__file__).resolve().parent
            / "raw"
            / "data"
            / "Legislative_materials"
        ),
        help="Directory where the bots should write raw .txt outputs",
    )
    p.add_argument(
        "--legislation-dir",
        default=str(Path(__file__).resolve().parent / "Legislation"),
        help="Path to the legacy jurisdiction bot folders (until migrated)",
    )
    p.add_argument(
        "--build-combined-csv-if-missing",
        action="store_true",
        help="Build the combined CSV from existing per-jurisdiction catalogues if it doesn't exist",
    )
    p.add_argument(
        "--jurisdiction",
        choices=JURISDICTIONS,
        default=None,
        help="Run only one jurisdiction",
    )

    # Pass-through common bot options.
    p.add_argument("--user-agent", default=None)
    p.add_argument("--sleep-seconds", type=float, default=None)
    p.add_argument("--timeout-seconds", type=float, default=None)
    p.add_argument("--max-retries", type=int, default=None)
    p.add_argument("--backoff-base-seconds", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-already-scraped", action="store_true")
    p.add_argument("--force", action="store_true")

    return p.parse_args()


def _passthrough_args(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    if args.user_agent is not None:
        out += ["--user-agent", args.user_agent]
    if args.sleep_seconds is not None:
        out += ["--sleep-seconds", str(args.sleep_seconds)]
    if args.timeout_seconds is not None:
        out += ["--timeout-seconds", str(args.timeout_seconds)]
    if args.max_retries is not None:
        out += ["--max-retries", str(args.max_retries)]
    if args.backoff_base_seconds is not None:
        out += ["--backoff-base-seconds", str(args.backoff_base_seconds)]
    if args.limit is not None:
        out += ["--limit", str(args.limit)]
    if args.skip_already_scraped:
        out += ["--skip-already-scraped"]
    if args.force:
        out += ["--force"]
    return out


def main() -> int:
    args = parse_args()

    combined_csv = Path(args.combined_csv).resolve()
    data_dir = Path(args.data_dir).resolve()
    legislation_dir = Path(args.legislation_dir).resolve()

    data_dir.mkdir(parents=True, exist_ok=True)

    if not combined_csv.exists():
        if args.build_combined_csv_if_missing:
            print(f"Building combined CSV -> {combined_csv}")
            build_combined_catalogue(legislation_dir=legislation_dir, combined_path=combined_csv)
        else:
            raise SystemExit(
                f"Combined CSV not found: {combined_csv}. Use --build-combined-csv-if-missing to generate it."
            )

    header, master_rows = _read_csv_rows(combined_csv)
    if not header:
        raise SystemExit(f"Combined CSV appears empty/invalid: {combined_csv}")

    # Ensure required fields
    if "jurisdiction" not in header:
        header = ["jurisdiction", *header]
        for r in master_rows:
            r.setdefault("jurisdiction", "")

    # Index by (jurisdiction, library_id) for stable merging.
    idx: dict[tuple[str, str], dict[str, str]] = {}
    for r in master_rows:
        key = (r.get("jurisdiction", ""), r.get("library_id", ""))
        idx[key] = r

    jurisdictions = [args.jurisdiction] if args.jurisdiction else list(JURISDICTIONS)

    passthrough = _passthrough_args(args)
    failures: list[tuple[str, int]] = []

    with tempfile.TemporaryDirectory(prefix="atl_combined_") as tmpdir:
        tmpdir_path = Path(tmpdir)

        for jurisdiction in jurisdictions:
            # Subset rows for this jurisdiction.
            subset = [r for r in master_rows if r.get("jurisdiction") == jurisdiction]
            if not subset:
                print(f"[SKIP] {jurisdiction}: no rows in combined CSV")
                continue

            temp_csv = tmpdir_path / f"{jurisdiction.replace(' ', '_').lower()}_legislation_catalogue.csv"
            _write_csv(temp_csv, header, subset)

            jurisdiction_dir = legislation_dir / jurisdiction
            bot_entry = jurisdiction_dir / "bot" / "sync.py"
            if not bot_entry.exists():
                print(f"[SKIP] {jurisdiction}: bot not found at {bot_entry}")
                continue

            cmd = [sys.executable, "-m", "bot.sync", "--data-dir", str(data_dir), "--catalogue-path", str(temp_csv), *passthrough]
            print(f"\n=== {jurisdiction} ===")
            completed = subprocess.run(cmd, cwd=str(jurisdiction_dir))
            if completed.returncode != 0:
                print(f"[FAIL] {jurisdiction} returned {completed.returncode}")
                failures.append((jurisdiction, completed.returncode))
                continue

            # Merge results back into master.
            _, updated_subset = _read_csv_rows(temp_csv)
            for ur in updated_subset:
                key = (ur.get("jurisdiction", jurisdiction), ur.get("library_id", ""))
                mr = idx.get(key)
                if mr is None:
                    continue
                mr.update(ur)

    _write_csv(combined_csv, header, master_rows)

    print("\nAll jurisdictions completed.")
    if failures:
        print("\nFailures:")
        for name, code in failures:
            print(f"- {name}: {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
