from __future__ import annotations

import argparse
import shutil
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect jurisdiction bot outputs (Legislation/*/data/*.txt) into a central raw folder."
        )
    )
    p.add_argument(
        "--legislation-root",
        default=str(Path(__file__).resolve().parents[1] / "Legislation"),
        help="Path to Australian-Taxation-Law-Library/Legislation",
    )
    p.add_argument(
        "--raw-dir",
        default=str(Path(__file__).resolve().parents[1] / "raw" / "legislation"),
        help="Destination raw folder",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    return p.parse_args()


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def main() -> int:
    args = parse_args()

    legislation_root = Path(args.legislation_root).resolve()
    raw_dir = Path(args.raw_dir).resolve()

    if not legislation_root.exists():
        raise SystemExit(f"Legislation root not found: {legislation_root}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0

    for jurisdiction in JURISDICTIONS:
        src_dir = legislation_root / jurisdiction / "data"
        if not src_dir.exists():
            continue

        for src_file in src_dir.glob("*.txt"):
            # Keep a jurisdiction subfolder to avoid filename collisions.
            dest_file = raw_dir / jurisdiction / src_file.name
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            if dest_file.exists() and not args.overwrite:
                skipped += 1
                continue

            shutil.copy2(src_file, dest_file)
            copied += 1

    print(f"Collected legislation outputs -> {raw_dir}")
    print(f"Copied: {copied} | Skipped (already existed): {skipped}")

    if copied == 0:
        print(
            "No .txt outputs found yet. Run the bots first, e.g. from Australian-Taxation-Law-Library/Legislation:"
        )
        print("  python run_all.py --skip-already-scraped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
