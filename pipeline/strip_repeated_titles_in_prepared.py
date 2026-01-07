from __future__ import annotations

import argparse
import re
from pathlib import Path


_SECTION_HEADING_RE = re.compile(r"^\s*Section\s+\S+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rewrite prepared .txt documents in-place to remove repeated act title lines "
            "that appear immediately before each 'Section ...' heading."
        )
    )
    p.add_argument(
        "--prepared-dir",
        default=str(Path(__file__).resolve().parents[1] / "prepared" / "data" / "Legislative_materials"),
        help="Prepared folder to rewrite in-place",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report how many files would change, but do not write",
    )
    return p.parse_args()


def normalize_newlines(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    out = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n" if out.strip() else ""


def first_nonempty_line(text: str) -> str | None:
    for line in normalize_newlines(text).split("\n"):
        s = (line or "").strip()
        if s:
            return s
    return None


def strip_repeated_title_before_sections(body: str, *, title_hint: str | None) -> str:
    normalized = normalize_newlines(body)
    if not normalized:
        return ""

    citation = (title_hint or "").strip() or ""
    if not citation:
        return normalized

    citation_cf = citation.casefold()
    lines = normalized.split("\n")

    # In prepared files, the citation is already present in the header block,
    # so we allow removing a citation line even if it's the first line of the body.
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = (line or "").strip()
        if s and s.casefold() == citation_cf:
            j = i + 1
            while j < len(lines) and not (lines[j] or "").strip():
                j += 1
            if j < len(lines) and _SECTION_HEADING_RE.match(lines[j] or ""):
                i += 1
                continue

        out.append(line)
        i += 1

    return normalize_newlines("\n".join(out))


def compute_rewritten_text(path: Path) -> tuple[str, bool]:
    original = path.read_text(encoding="utf-8", errors="replace")
    normalized = normalize_newlines(original)
    if not normalized:
        return original, False

    # Prepared outputs from bot.sync generally look like:
    # <header lines>\n\n<body...>\n
    header = ""
    body = normalized
    parts = normalized.split("\n\n", 1)
    if len(parts) == 2:
        header, body = parts[0].rstrip("\n"), parts[1]

    title_hint = first_nonempty_line(header) if header else None
    new_body = strip_repeated_title_before_sections(body, title_hint=title_hint)

    if header:
        rewritten = (header.strip() + "\n\n" + new_body.strip() + "\n").replace("\r\n", "\n")
    else:
        rewritten = (new_body.strip() + "\n").replace("\r\n", "\n")

    changed = rewritten != original.replace("\r\n", "\n")
    return rewritten, changed


def main() -> int:
    args = parse_args()

    prepared_dir = Path(args.prepared_dir).resolve()
    if not prepared_dir.exists():
        print(f"Prepared dir not found: {prepared_dir}")
        return 1

    files = sorted(p for p in prepared_dir.rglob("*.txt") if p.is_file())
    changed = 0
    unchanged = 0

    for p in files:
        # Skip any catalogue/template CSV accidentally placed here.
        if p.name.lower().endswith(".csv"):
            continue

        rewritten, is_changed = compute_rewritten_text(p)
        if args.dry_run:
            if is_changed:
                changed += 1
            else:
                unchanged += 1
            continue

        if is_changed:
            p.write_text(rewritten, encoding="utf-8", newline="\n")
            changed += 1
        else:
            unchanged += 1

    print(f"Prepared dir: {prepared_dir}")
    print(f"Files scanned: {len(files)}")
    if args.dry_run:
        print(f"Would change: {changed} | Would remain unchanged: {unchanged}")
    else:
        print(f"Changed: {changed} | Unchanged: {unchanged}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
