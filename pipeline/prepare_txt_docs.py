from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    out = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = out.strip()
    return out + "\n" if out else ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare cleaned .txt documents for RAG ingestion (raw -> prepared/docs)."
    )
    p.add_argument(
        "--raw-dir",
        default=str(Path(__file__).resolve().parents[1] / "raw" / "legislation"),
        help="Input raw folder (expects *.txt under jurisdiction subfolders)",
    )
    p.add_argument(
        "--prepared-docs-dir",
        default=str(Path(__file__).resolve().parents[1] / "prepared" / "docs"),
        help="Output folder of cleaned .txt files",
    )
    p.add_argument(
        "--manifest",
        default=str(Path(__file__).resolve().parents[1] / "prepared" / "manifest.jsonl"),
        help="Write a JSONL manifest with provenance + hashes",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite prepared outputs if they already exist",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    prepared_dir = Path(args.prepared_docs_dir).resolve()
    manifest_path = Path(args.manifest).resolve()

    if not raw_dir.exists():
        print(f"Raw dir not found: {raw_dir}")
        return 1

    prepared_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    written = 0
    skipped = 0

    for src in sorted(raw_dir.rglob("*.txt")):
        rel = src.relative_to(raw_dir).as_posix()
        # Flatten into a single folder while keeping jurisdiction in the filename.
        # Example: 'Victoria/Income Tax Assessment Act 1997 (Cth) (vic_123).txt'
        safe_name = rel.replace("/", "__")
        dest = prepared_dir / safe_name

        if dest.exists() and not args.overwrite:
            skipped += 1
            continue

        raw_bytes = src.read_bytes()
        raw_hash = sha256_bytes(raw_bytes)

        text = raw_bytes.decode("utf-8", errors="replace")
        cleaned = normalize_text(text)
        cleaned_bytes = cleaned.encode("utf-8")
        cleaned_hash = sha256_bytes(cleaned_bytes)

        dest.write_bytes(cleaned_bytes)
        written += 1

        records.append(
            {
                "when_prepared": utc_now_iso(),
                "source_relpath": rel,
                "raw_sha256": raw_hash,
                "prepared_relpath": dest.name,
                "prepared_sha256": cleaned_hash,
                "bytes": len(cleaned_bytes),
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Prepared docs -> {prepared_dir}")
    print(f"Manifest -> {manifest_path}")
    print(f"Written: {written} | Skipped (already existed): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
