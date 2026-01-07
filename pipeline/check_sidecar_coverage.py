"""Validate JSON sidecar coverage against prepared .txt inputs.

Primary check (requested):
- For each prepared .txt, every Section/Regulation listed in its TOS/TOC
    (e.g. 'TABLE OF PROVISIONS' / 'Contents') must have at least one sidecar
    chunk with key prefix 'section:{number}' or 'regulation:{number}'.

Notes:
- We intentionally treat a unit as "present" if any sidecar chunk key begins
    with '{prefix}:{number}' (to allow split chunks like 'section:995-1.p1').
- Schedules often contain additional internal material not listed in the TOS;
    we do NOT require schedule-internal provisions unless they appear as TOS
    sections/regulations.

Exit codes:
- 0: all checks passed
- 1: failures found
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path

from generate_sidecar_json import normalize_text, normalize_unit_number, strip_prepared_header


def chunk_exists_for_unit(sidecar: dict, *, key_prefix: str, unit_number: str) -> bool:
    prefix = f"{key_prefix}:{unit_number}"
    for c in sidecar.get("chunks") or []:
        if not isinstance(c, dict):
            continue
        key = c.get("key")
        if isinstance(key, str) and (key == prefix or key.startswith(prefix + ".")):
            return True
    return False


def infer_unit_kind_from_filename(name: str) -> tuple[str, str]:
    """Return (kind_word, key_prefix) based on filename heuristics."""

    s = (name or "").casefold()
    if " regulation" in s or " regulations" in s or s.endswith("regulation.txt") or s.endswith("regulations.txt"):
        return "Regulation", "regulation"
    return "Section", "section"


def extract_tos_unit_numbers(prepared_text: str, *, default_kind: str | None = None) -> dict[str, list[str]]:
    """Return {'section': [...], 'regulation': [...]} from the TOS/TOC region.

    We scan between the first TOC marker ('contents' or 'table of provisions')
    and the body start marker (e.g. '- LONG TITLE', 'An Act ...', or common
    consolidation headers like '<TITLE> - Regulation 1').
    """

    body = strip_prepared_header(prepared_text)
    body = normalize_text(body)
    if not body:
        return {"section": [], "regulation": []}

    lines = body.split("\n")

    toc_start = None
    for i in range(min(len(lines), 2500)):
        s = (lines[i] or "").strip().casefold()
        if s == "contents" or "table of provisions" in s:
            toc_start = i
            break
    if toc_start is None:
        return {"section": [], "regulation": []}

    # Determine TOC end reliably.
    # Many Regulations include a Contents block where TOC entries end with page numbers.
    # Those files often lack '- LONG TITLE' / 'An Act ...' markers, so we stop at the
    # first structural/provision heading that *doesn't* end with a trailing page number
    # after we've already seen a run of page-number TOC rows.
    scan_limit = min(len(lines), toc_start + 30000)

    toc_end: int | None = None

    def _has_trailing_page_number(line: str) -> bool:
        return re.search(r"\s\d+\s*$", (line or "").rstrip()) is not None

    def _looks_like_structural_or_unit_heading(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        return bool(
            re.match(r"^(?:Chapter|Part|Division|Subdivision)\b", s, re.IGNORECASE)
            or re.match(r"^(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE)
            or re.match(r"^\d[0-9A-Za-z]{0,6}\s*--\s*\S", s)
        )

    toc_page_rows = 0
    for i in range(toc_start + 1, scan_limit):
        s = (lines[i] or "").strip()
        if not s:
            continue
        # Many QLD-style consolidations start the body with asterisk-wrapped headings like:
        #   *** 1 Short title ***
        if re.match(r"^\*{3}\s*\d", s):
            toc_end = i
            break
        # Treat explicit schedule markers as end of TOC/contents.
        if re.match(r"^\s*-\s*SCHEDULE\s+\S+\b", s, re.IGNORECASE):
            toc_end = i
            break
        if re.match(r"^\s*Legislative\s+history\s*$", s, re.IGNORECASE):
            toc_end = i
            break
        if re.match(r"^\s*\d[0-9A-Za-z]{0,6}\s*--\s*\S", s):
            toc_end = i
            break
        # Some consolidations (notably WA) repeat a header like:
        #   <DOC TITLE> - NOTES
        #   <DOC TITLE> - Regulation 1
        # Treat these as body-start markers.
        if re.search(r"\s-\s*NOTES\s*$", s, re.IGNORECASE):
            toc_end = i
            break
        if re.search(r"\s-\s*(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE):
            toc_end = i
            break
        if s.casefold() == "endnotes":
            toc_end = i
            break
        if re.match(r"^\s*-\s*LONG\s+TITLE\s*$", s, re.IGNORECASE) or re.match(r"^\s*An\s+Act\b", s, re.IGNORECASE):
            toc_end = i
            break

        if _has_trailing_page_number(s) and _looks_like_structural_or_unit_heading(s):
            toc_page_rows += 1
            continue

        # After a decent number of TOC rows, the first repeated heading line without a
        # page number is a strong body-start indicator.
        if toc_page_rows >= 10 and _looks_like_structural_or_unit_heading(s) and not _has_trailing_page_number(s):
            toc_end = i
            break

    if toc_end is None:
        toc_end = scan_limit

    # Unit-number extraction should be strict enough to avoid false positives like
    # 'Section in ...' or 'Regulation Act ...' from noisy TOC text.
    # Match the same general shape as the generator's unit-number patterns.
    unit_re = re.compile(
        r"^\s*(?P<kind>Section|Sections|Regulation|Regulations)\s+"
        r"(?P<number>[0-9]{1,4}[A-Za-z]{0,6}(?:[\-\.][0-9A-Za-z]+)*)\.?\s+"
        r"(?P<heading>.+\S)\s*$",
        re.IGNORECASE,
    )

    def _looks_like_year(num: str) -> bool:
        if not (num or "").isdigit():
            return False
        if len(num) != 4:
            return False
        try:
            y = int(num)
        except ValueError:
            return False
        return 1800 <= y <= 2100

    def _canonicalize_number(num_raw: str) -> str:
        num = normalize_unit_number(num_raw)
        if not num:
            return num
        # Normalize leading zeros in purely-numeric unit numbers (e.g. '079' -> '79').
        if num.isdigit():
            try:
                num = str(int(num))
            except ValueError:
                pass
        return num

    out: dict[str, list[str]] = {"section": [], "regulation": []}
    seen: dict[str, set[str]] = {"section": set(), "regulation": set()}

    # First pass: collect candidates, then decide if we should expand simple ranges.
    candidates: list[tuple[str, str]] = []
    for line in lines[toc_start:toc_end]:
        m = unit_re.match(line or "")
        if not m:
            continue

        kind_raw = (m.group("kind") or "").strip().casefold()
        kind = "regulation" if kind_raw.startswith("regulation") else "section"
        num_raw = (m.group("number") or "").strip()
        num = _canonicalize_number(num_raw)
        if not num:
            continue

        # Never treat 4-digit year-like numbers as provision numbers.
        if _looks_like_year(num):
            continue

        heading = (m.group("heading") or "").strip()
        h_cf = heading.casefold()
        # Filter obvious false positives where a citation/year leaks into a TOC region.
        # Example patterns seen in the corpus: expected regulation:1999, section:2001.
        if _looks_like_year(num) and (" act" in h_cf or h_cf.endswith("act") or "regulation" in h_cf or "regulations" in h_cf):
            continue
        candidates.append((kind, num))

    # Additional TOC format: numeric rows split across lines, common in some
    # State/Territory regulations (e.g. WA AustLII exports):
    #   1
    #   .      Citation
    #   2
    #   .      Commencement
    # These blocks usually omit the literal word "Regulation".
    # We classify them using the caller-provided default kind.
    kind_word = (default_kind or "Section").strip().casefold()
    numeric_kind = "regulation" if kind_word.startswith("reg") else "section"

    num_line_re = re.compile(r"^\s*(?P<number>\d[0-9A-Za-z]{0,6})\s*$")
    dot_heading_re = re.compile(r"^\s*\.\s*(?P<heading>.+\S)\s*$")

    numeric_candidates: list[tuple[str, str]] = []
    region = lines[toc_start:toc_end]
    for i in range(len(region) - 1):
        m_num = num_line_re.match(region[i] or "")
        if not m_num:
            continue
        num = normalize_unit_number((m_num.group("number") or "").strip())
        if not num:
            continue

        j = i + 1
        while j < len(region) and not (region[j] or "").strip():
            j += 1
        if j >= len(region):
            continue

        m_head = dot_heading_re.match(region[j] or "")
        if not m_head:
            continue

        if _looks_like_year(num):
            continue
        numeric_candidates.append((numeric_kind, num))

    # Additional TOC format: simple numbered lines without 'Section/Regulation' keywords.
    # Common for many Regulations:
    #   1 Short title
    #   3 Interpretation
    simple_row_re = re.compile(
        r"^\s*(?P<number>\d[0-9A-Za-z]{0,6}(?:[\-\.][0-9A-Za-z]+)*)\.?\s+(?P<heading>.+\S)\s*$"
    )
    simple_candidates: list[tuple[str, str]] = []
    for line in region:
        s = (line or "").strip()
        if not s:
            continue
        # Avoid double-parsing lines already covered by the explicit Section/Regulation form.
        if re.match(r"^(?:Section|Regulation)\b", s, re.IGNORECASE):
            continue
        m = simple_row_re.match(s)
        if not m:
            continue
        num = _canonicalize_number((m.group("number") or "").strip())
        if not num:
            continue
        if _looks_like_year(num):
            continue
        simple_candidates.append((numeric_kind, num))

    if len(simple_candidates) >= 3:
        candidates.extend(simple_candidates)

    # Only apply numeric-dot parsing if it looks like a real TOC table.
    if len(numeric_candidates) >= 3:
        candidates.extend(numeric_candidates)

    for kind, num in candidates:
        if num in seen[kind]:
            continue
        seen[kind].add(num)
        out[kind].append(num)

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Check JSON sidecar coverage for all TOS sections/regulations")
    ap.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path(r"D:\Australian-Taxation-Law-Library\prepared\data\Legislative_materials"),
    )
    ap.add_argument(
        "--include-glob",
        type=str,
        default="",
        help="Optional glob (matched against filename) to limit which prepared .txt files are checked",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prepared_dir: Path = args.prepared_dir

    if not prepared_dir.exists():
        raise SystemExit(f"prepared-dir not found: {prepared_dir}")

    failures: list[str] = []
    notes: list[str] = []
    checked = 0

    include_glob = (args.include_glob or "").strip()

    for prepared_path in sorted(prepared_dir.glob("*.txt")):
        if include_glob and not fnmatch.fnmatch(prepared_path.name, include_glob):
            continue
        sidecar_path = prepared_path.with_suffix(".json")
        if not sidecar_path.exists():
            failures.append(f"MISSING SIDECAR: {sidecar_path.name}")
            continue

        prepared_text = prepared_path.read_text(encoding="utf-8", errors="replace")
        kind_word, _key_prefix = infer_unit_kind_from_filename(prepared_path.name)
        expected = extract_tos_unit_numbers(prepared_text, default_kind=kind_word)
        expected_sections = expected.get("section") or []
        expected_regs = expected.get("regulation") or []
        if not expected_sections and not expected_regs:
            notes.append(f"NOTE: no TOS units detected for {prepared_path.name}")
            continue

        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8", errors="replace"))

        for num in expected_sections:
            if not chunk_exists_for_unit(sidecar, key_prefix="section", unit_number=num):
                failures.append(f"NO CHUNK FOR TOS SECTION: {prepared_path.name} expected section:{num}")
        for num in expected_regs:
            if not chunk_exists_for_unit(sidecar, key_prefix="regulation", unit_number=num):
                failures.append(f"NO CHUNK FOR TOS REGULATION: {prepared_path.name} expected regulation:{num}")

        checked += 1
        if args.limit and checked >= int(args.limit):
            break

    print(f"Checked {checked} docs (with detectable TOS units)")
    if notes:
        print(f"Notes: {len(notes)}")
        for n in notes[:50]:
            print(f"- {n}")
        if len(notes) > 50:
            print(f"... plus {len(notes) - 50} more")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures[:200]:
            print(f"- {f}")
        if len(failures) > 200:
            print(f"... plus {len(failures) - 200} more")
        return 1

    print("OK: All checked docs have chunks for every TOS-listed unit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
