from __future__ import annotations

import csv
from pathlib import Path


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


def find_catalogue_csv(jurisdiction_dir: Path) -> Path:
    data_dir = jurisdiction_dir / "data"
    matches = sorted(data_dir.glob("*_legislation_catalogue.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_legislation_catalogue.csv found in {data_dir}")
    return matches[0]


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.reader(f))
    if not raw_rows:
        return [], []

    header_idx = 0
    first = {c.strip().lower() for c in raw_rows[0]}
    if "library_id" not in first or "url" not in first:
        header_idx = 1

    header = [c.strip().lower() for c in raw_rows[header_idx]]
    return header, raw_rows[header_idx + 1 :]


def main() -> int:
    root = Path(__file__).resolve().parent

    any_error = False

    print("SUMMARY")
    for j in JURISDICTIONS:
        jdir = root / j
        csv_path = find_catalogue_csv(jdir)
        header, rows = read_csv_rows(csv_path)

        if not header:
            print(f"- {j}: empty_csv")
            any_error = True
            continue

        idx = {name: i for i, name in enumerate(header)}

        def cell(row: list[str], key: str) -> str:
            i = idx.get(key, -1)
            return (row[i].strip() if 0 <= i < len(row) else "")

        counts: dict[str, int] = {}
        examples: list[tuple[str, str, str]] = []

        for r in rows:
            status = cell(r, "status") or "(blank)"
            counts[status] = counts.get(status, 0) + 1
            if status == "error" and len(examples) < 3:
                examples.append(
                    (
                        cell(r, "library_id"),
                        cell(r, "url"),
                        (cell(r, "error") or "")[:160],
                    )
                )

        if counts.get("error", 0) > 0:
            any_error = True

        ordered = ["ok", "skipped_no_change", "skipped_inactive", "error", "(blank)"]
        parts: list[str] = []
        for k in ordered:
            if k in counts:
                parts.append(f"{k}={counts[k]}")
        for k in sorted(counts):
            if k not in set(ordered):
                parts.append(f"{k}={counts[k]}")

        print(f"- {j}: total={len(rows)} " + " ".join(parts))
        for lib, url, msg in examples:
            print(f"    error example: {lib} {url} :: {msg}")

    print("\nALL_SUCCESSFUL=" + ("yes" if not any_error else "no"))
    return 0 if not any_error else 2


if __name__ == "__main__":
    raise SystemExit(main())
