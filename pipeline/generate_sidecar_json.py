from __future__ import annotations

import argparse
import bisect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def normalize_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = "\n".join(line.rstrip() for line in out.split("\n"))
    out = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f]", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_DOC_HEADER_KV_RE = re.compile(
    r"^(?P<key>Jurisdiction|Source|Register|Downloaded|When\s+scraped)\s*:\s*(?P<value>.*\S)\s*$",
    re.IGNORECASE,
)


def parse_doc_header(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not (text or "").strip():
        return out

    lines = (text or "").split("\n")
    for line in lines[:80]:
        m = _DOC_HEADER_KV_RE.match((line or "").strip())
        if not m:
            continue
        key = (m.group("key") or "").strip().casefold()
        value = (m.group("value") or "").strip()
        if not value:
            continue

        if key == "jurisdiction":
            out["jurisdiction"] = value
        elif key == "source":
            out["source_url"] = value
        elif key == "register":
            out["register_url"] = value
        # Intentionally do not propagate per-run scrape/download metadata into sidecars.
        # It is noisy and ends up copied onto every chunk payload.

    return out


_CITATION_FROM_STEM_RE = re.compile(r"\s*\(\d+\)\s*$")


def citation_from_path(path: Path) -> str:
    stem = path.stem
    stem = _CITATION_FROM_STEM_RE.sub("", stem).strip()
    return stem or path.name


# Raw TOC parsing
_CHAPTER_RE = re.compile(r"^Chapter\s+(?P<number>\S+)\s*(?P<title>.*)$", re.IGNORECASE)
_PART_RE = re.compile(r"^PART\s+(?P<number>\S+)\s*(?P<title>.*)$", re.IGNORECASE)
_DIVISION_RE = re.compile(r"^Division\s+(?P<number>\S+)\s*(?P<title>.*)$", re.IGNORECASE)
_SUBDIVISION_RE = re.compile(r"^Subdivision\s+(?P<number>\S+)\s*(?P<title>.*)$", re.IGNORECASE)

_TOC_SECTION_RE = re.compile(r"^(?P<number>[0-9A-Za-z][0-9A-Za-z\-\.]*)\.\s*(?P<heading>.+\S)\s*$")
# Common modern Commonwealth consolidated style:
#   Section 1.1. Short title
#   Regulation 3.5. Something
_TOC_SECTION_WORD_RE = re.compile(
    r"^(?:Section|Regulation)\s+(?P<number>[0-9A-Za-z][0-9A-Za-z\-\.]*)\.?\s*(?P<heading>.+\S)\s*$",
    re.IGNORECASE,
)
# Some AustLII/plain-text consolidations use a columnar table-of-provisions format:
#   1     Short title      1
_TOC_SECTION_COL_RE = re.compile(
    # Note: lines are pre-cleaned by _clean_heading_line() which collapses runs
    # of whitespace; match the post-cleaned shape:
    #   1 Short title 1
    # Important: require digit-leading numbers; otherwise lines like
    #   Section 5. Relationship with ... 1996
    # can be mis-parsed with number='Section' and page='1996'.
    r"^(?P<number>\d[0-9A-Za-z\-]*)\s+(?P<heading>.+?)\s+(?P<page>\d+)\s*$"
)


def _clean_heading_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _normalize_structure_heading(line: str) -> str:
    # Preserve text mostly as-is but normalize whitespace and capitalization of the label.
    s = _clean_heading_line(line)
    if not s:
        return s

    for label in ("chapter", "part", "division", "subdivision"):
        if s.casefold().startswith(label):
            return label.capitalize() + s[len(label) :]
    if s.casefold().startswith("part"):
        return "Part" + s[4:]
    return s


def iter_txt_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.txt")):
        if p.is_file():
            yield p


def extract_toc_structure_map(raw_text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (structure_by_section_number, section_heading_by_number) from raw TOC.

    We scan the 'TABLE OF PROVISIONS' region and track current Part/Division/Subdivision.
    """

    text = normalize_text(raw_text)
    lines = text.split("\n")

    start_idx = None
    for i, line in enumerate(lines[:2000]):
        if "table of provisions" in (line or "").casefold() or "contents" == (line or "").strip().casefold():
            start_idx = i
            break

    if start_idx is None:
        return {}, {}

    # End at long title marker (body start) or endnotes marker.
    # For many Regulations, there is no long title marker; in those cases, the
    # TOC can otherwise run into the body and overwrite headings (e.g. regulation
    # numbers repeated inside schedules). We therefore also stop once we see the
    # first *real* body provision heading that is followed by prose.
    end_idx = len(lines)
    scan_limit = min(len(lines), start_idx + 12000)

    # If a long-title marker exists shortly after the TOC begins (common in Acts),
    # rely on it to terminate the TOC. The fallback heuristic that tries to detect
    # the body by looking for a provision heading followed by prose can incorrectly
    # cut off the final TOC entry (e.g., right before '- LONG TITLE').
    has_long_title_marker = False
    for k in range(start_idx + 1, min(start_idx + 8000, len(lines))):
        s_k = (lines[k] or "").strip()
        if re.match(r"^\s*-\s*LONG\s+TITLE\s*$", s_k, re.IGNORECASE) or re.match(
            r"^\s*An\s+Act\s+(?:to|for)\b", s_k, re.IGNORECASE
        ):
            has_long_title_marker = True
            break

    # Unit numbers can be plain numeric labels (common for many Regulations):
    #   1 Short title
    #   10A Exemption ...
    # or double-hyphen headings:
    #   1--Short title
    unit_num_src = r"[0-9]{1,4}[A-Za-z]{0,6}(?:[\-\.][0-9A-Za-z]+)*"
    toc_numbered_re = re.compile(rf"^(?P<number>{unit_num_src})\s+(?P<heading>.+\S)\s*$")
    toc_numbered_dash_re = re.compile(rf"^(?P<number>{unit_num_src})\s*--\s*(?P<heading>.+\S)\s*$")

    def _next_nonempty_idx(j: int) -> int | None:
        while j < len(lines):
            s = (lines[j] or "")
            if s.strip() and not _looks_like_doc_title(s.strip()):
                return j
            j += 1
        return None

    def _next_nonempty_noncontinuation_idx(j: int) -> int | None:
        # Skip indented continuation fragments (common in wrapped TOC rows).
        while j < len(lines):
            raw = lines[j] or ""
            s = raw.strip()
            if not s or _looks_like_doc_title(s):
                j += 1
                continue
            if raw.lstrip() != raw:
                # Keep indented lines that still look like TOC rows (rare but possible).
                if not _looks_like_toc_list_line(s):
                    j += 1
                    continue
            return j
        return None

    def _looks_like_toc_list_line(line: str) -> bool:
        s = _clean_heading_line(line)
        if not s:
            return False
        if _CHAPTER_RE.match(s) or _PART_RE.match(s) or _DIVISION_RE.match(s) or _SUBDIVISION_RE.match(s):
            return True
        if re.match(r"^(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE):
            return True
        if toc_numbered_dash_re.match(s) or toc_numbered_re.match(s):
            return True
        if re.match(r"^(?:Schedule)\b", s, re.IGNORECASE):
            return True
        if re.match(r"^\s*SCHEDULE\s+\S+\b", s, re.IGNORECASE):
            return True
        return False

    def _next_nonempty_lines(start: int, limit: int = 10) -> list[str]:
        out: list[str] = []
        j = start
        while j < len(lines) and len(out) < limit:
            s = (lines[j] or "").strip()
            if s and not _looks_like_doc_title(s):
                out.append(s)
            j += 1
        return out

    def _looks_like_prose_line(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        if len(s) >= 45:
            return True
        if re.search(r"[\.;:]\s*$", s):
            return True
        if re.match(r"^(?:These|This|In\s+these|In\s+this|For\s+the\s+purposes\b)", s, re.IGNORECASE):
            return True
        return False

    def _has_nearby_prose(i: int) -> bool:
        lookahead = _next_nonempty_lines(i + 1, limit=18)
        if not lookahead:
            return False

        # Ignore short wrapped fragments that are neither list lines nor real prose.
        filtered: list[str] = []
        for x in lookahead:
            if _looks_like_toc_list_line(x) or _looks_like_prose_line(x):
                filtered.append(x)

        if not filtered:
            return False

        if all(_looks_like_toc_list_line(x) for x in filtered[:5]):
            return False
        return any((not _looks_like_toc_list_line(x)) and _looks_like_prose_line(x) for x in filtered[:12])

    # Prefer explicit TOC terminators when present (common in some Regulation exports).
    explicit_end_idx: int | None = None
    for i in range(start_idx + 1, scan_limit):
        s = (lines[i] or "").strip()
        s_cf = s.casefold()
        if s_cf == "endnotes" or s_cf == "legislative history":
            explicit_end_idx = i
            break
        if _ASTERISK_UNIT_START_RE.match(s):
            explicit_end_idx = i
            break
        if re.match(r"^\s*-\s*LONG\s+TITLE\s*$", s, re.IGNORECASE) or re.match(
            r"^\s*An\s+Act\s+(?:to|for)\b", s, re.IGNORECASE
        ):
            explicit_end_idx = i
            break

    if explicit_end_idx is not None:
        end_idx = explicit_end_idx
    else:
        seen_unit_tokens: set[str] = set()
        for i in range(start_idx + 1, scan_limit):
            s = (lines[i] or "").strip()

            # Track repeated provision headings: many consolidations list a TOC and then
            # restart the same numbering in the body. The first repeated heading with
            # nearby prose is a strong body-start signal.
            token = ""
            s_clean = _clean_heading_line(s)
            m_word = _TOC_SECTION_WORD_RE.match(s_clean)
            if m_word:
                num = normalize_unit_number((m_word.group("number") or "").strip())
                kind = "regulation" if "reg" in (s_clean.casefold().split()[:1] or [""])[0] else "section"
                if num:
                    token = f"{kind}:{num}"
            else:
                m_num = toc_numbered_dash_re.match(s_clean) or toc_numbered_re.match(s_clean)
                if m_num:
                    num = normalize_unit_number((m_num.group("number") or "").strip())
                    if num:
                        token = f"num:{num}"

            if token:
                if token not in seen_unit_tokens:
                    seen_unit_tokens.add(token)
                    continue
                if _has_nearby_prose(i):
                    end_idx = i
                    break

            # If we can't find a repeat (rare), keep scanning; we prefer false-negatives
            # (shorter TOC map) over truncating late TOC entries.

    current: dict[str, str] = {}
    structure_by_section: dict[str, str] = {}
    heading_by_section: dict[str, str] = {}

    # Numeric-dot TOC rows (e.g. WA exports):
    #   1
    #   .      Citation
    #   2
    #   .      Commencement
    # These blocks usually omit the literal word "Regulation".
    num_line_re = re.compile(r"^\s*(?P<number>\d[0-9A-Za-z]{0,6})\s*$")
    dot_heading_re = re.compile(r"^\s*\.\s*(?P<heading>.+\S)\s*$")
    region = lines[start_idx:end_idx]
    for i, raw_line in enumerate(region[:-1]):
        m_num = num_line_re.match(raw_line or "")
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
        m_dot = dot_heading_re.match(region[j] or "")
        if not m_dot:
            continue
        heading = (m_dot.group("heading") or "").strip()
        if heading:
            heading_by_section.setdefault(num, heading)

    for line in lines[start_idx:end_idx]:
        s = _clean_heading_line(line)
        if not s:
            continue

        m = _CHAPTER_RE.match(s)
        if m:
            current["chapter"] = _normalize_structure_heading(s)
            # Reset lower levels
            current.pop("part", None)
            current.pop("division", None)
            current.pop("subdivision", None)
            continue

        m = _PART_RE.match(s)
        if m:
            current["part"] = _normalize_structure_heading(s)
            # Reset lower levels
            current.pop("division", None)
            current.pop("subdivision", None)
            continue

        m = _DIVISION_RE.match(s)
        if m:
            current["division"] = _normalize_structure_heading(s)
            current.pop("subdivision", None)
            continue

        m = _SUBDIVISION_RE.match(s)
        if m:
            current["subdivision"] = _normalize_structure_heading(s)
            continue

        m = _TOC_SECTION_RE.match(s) or _TOC_SECTION_COL_RE.match(s) or _TOC_SECTION_WORD_RE.match(s)
        if m:
            number = (m.group("number") or "").strip()
            number = normalize_unit_number(number)
            if number.isdigit() and len(number) == 4:
                try:
                    y = int(number)
                except ValueError:
                    y = 0
                if 1800 <= y <= 2100:
                    continue
            heading = (m.group("heading") or "").strip()
            heading_by_section[number] = heading
            parts = [current[k] for k in ("chapter", "part", "division", "subdivision") if k in current]
            if parts:
                structure_by_section[number] = " > ".join(parts)
            continue

        # Simple numbered rows without keyword and without a trailing page column.
        m_num = toc_numbered_dash_re.match(s) or toc_numbered_re.match(s)
        if m_num:
            number = normalize_unit_number((m_num.group("number") or "").strip())
            if number.isdigit() and len(number) == 4:
                try:
                    y = int(number)
                except ValueError:
                    y = 0
                if 1800 <= y <= 2100:
                    continue
            heading = (m_num.group("heading") or "").strip()
            if number and heading:
                heading_by_section[number] = heading
                parts = [current[k] for k in ("chapter", "part", "division", "subdivision") if k in current]
                if parts:
                    structure_by_section[number] = " > ".join(parts)

    return structure_by_section, heading_by_section


def extract_body_structure_map(body_text: str) -> dict[str, str]:
    """Infer section structure from in-body Chapter/Part/Division/Subdivision headings."""

    text = normalize_text(body_text)
    if not text:
        return {}

    current: dict[str, str] = {}
    out: dict[str, str] = {}

    def _is_confident_structure_heading(line: str) -> bool:
        # Be conservative: avoid treating narrative sentences like "Chapter 2 has ..." as headings.
        s = _clean_heading_line(line)
        if not s:
            return False
        if "--" in s or "—" in s:
            return True
        if s.isupper() and len(s) >= 8:
            return True
        # Allow bare headings like "Chapter 1" if they appear alone.
        if re.fullmatch(r"(?:Chapter|Part|Division|Subdivision)\s+\S+", s, re.IGNORECASE):
            return True
        return False

    prev_nonempty_cf = ""
    for raw_line in text.split("\n"):
        s = _clean_heading_line(raw_line)
        if not s:
            continue

        m = _CHAPTER_RE.match(s)
        if m:
            if not _is_confident_structure_heading(s):
                continue
            # Avoid treating in-text references like 'Chapter 3 of the ...' as structure headings.
            s_cf = s.casefold()
            if " of " in s_cf and "--" not in s and "—" not in s and "-" not in s:
                continue
            current["chapter"] = _normalize_structure_heading(s)
            current.pop("part", None)
            current.pop("division", None)
            current.pop("subdivision", None)
            continue

        m = _PART_RE.match(s)
        if m:
            if not _is_confident_structure_heading(s):
                continue
            # Avoid treating in-text references like 'Part 3 of the ...' as structure headings.
            s_cf = s.casefold()
            if " of " in s_cf and "--" not in s and "—" not in s and "-" not in s:
                continue
            current["part"] = _normalize_structure_heading(s)
            current.pop("division", None)
            current.pop("subdivision", None)
            continue

        m = _DIVISION_RE.match(s)
        if m:
            if not _is_confident_structure_heading(s):
                continue
            s_cf = s.casefold()
            if " of " in s_cf and "--" not in s and "—" not in s and "-" not in s:
                continue
            current["division"] = _normalize_structure_heading(s)
            current.pop("subdivision", None)
            continue

        m = _SUBDIVISION_RE.match(s)
        if m:
            if not _is_confident_structure_heading(s):
                continue
            s_cf = s.casefold()
            if " of " in s_cf and "--" not in s and "—" not in s and "-" not in s:
                continue
            current["subdivision"] = _normalize_structure_heading(s)
            continue

        m = _SECTION_WORD_HEADING_RE.match(s)
        if m:
            number = normalize_unit_number((m.group("number") or "").strip())
            if not number:
                continue
            # Ignore in-body examples like:
            #   Example for this section:
            #   Section 1. "Perth local government district" refers ...
            heading = (m.group("heading") or "").strip()
            if '"' in heading or "\u201c" in heading or "\u201d" in heading:
                continue
            if prev_nonempty_cf.startswith("example"):
                continue
            parts = [current[k] for k in ("chapter", "part", "division", "subdivision") if k in current]
            if parts:
                out.setdefault(number, " > ".join(parts))
            continue

        m = _SECTION_MARK_RE.match(s)
        if m:
            number = normalize_unit_number((m.group("number") or "").strip())
            if not number:
                continue
            parts = [current[k] for k in ("chapter", "part", "division", "subdivision") if k in current]
            if parts:
                out.setdefault(number, " > ".join(parts))

        prev_nonempty_cf = s.casefold()

    return out


def ensure_min_two_level_structure_path(structure_path: str | None, *, citation: str) -> str | None:
    """Ensure structure_path contains at least two levels.

    If the path is single-level (no ' > '), prefix it with the document citation.
    """

    s = (structure_path or "").strip()
    if not s:
        return (citation or "").strip() or None
    if " > " not in s:
        prefix = (citation or "").strip()
        if prefix:
            return f"{prefix} > {s}".strip()
    return s


# Raw section parsing
_SECTION_MARK_RE = re.compile(
    # Examples:
    #   - SECT 1.1
    #   - SECT 995.1
    #   - REG  12
    r"^\s*-\s*(?P<kind>SECT|REG)\s+(?P<number>[0-9]{1,4}[A-Za-z]{0,6}(?:[\-\.][0-9A-Za-z]+)*)\s*$",
    re.IGNORECASE,
)

# Some prepared/plain-text sources use marker lines like:
#   - Section 128A.1
#   - Regulation 12
# where the heading is on the following line.
_SECTION_WORD_MARK_RE = re.compile(
    r"^\s*-\s*(?P<kind>Section|Regulation)\s+(?P<number>[0-9]{1,4}[A-Za-z]{0,6}(?:[\-\.][0-9A-Za-z]+)*)\s*$",
    re.IGNORECASE,
)


_SCHEDULE_MARK_RE = re.compile(
    # Examples:
    #   - SCHEDULE 2
    #   - SCHEDULE 2F
    #   - SCHEDULE 1A  (sometimes with an inline title)
    r"^\s*-\s*SCHEDULE\s+(?P<id>[0-9A-Za-z]+)\s*(?P<title>.*\S)?\s*$",
    re.IGNORECASE,
)

# Some prepared/plain-text consolidations use a bare heading line like:
#   Schedule    Ordinances repealed
# or numbered schedules like:
#   Schedule 1  Administration
# Treat only digit-starting ids as schedule ids; otherwise the token is part of the title.
_SCHEDULE_WORD_MARK_RE = re.compile(
    r"^\s*Schedule(?:\s+(?P<id>\d[0-9A-Za-z]*))?\s+(?P<title>.+\S)\s*$",
    re.IGNORECASE,
)


_NOTES_MARK_RE = re.compile(r"^\s*-\s*NOTES\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class RawUnit:
    kind: str  # 'Section' or 'Regulation'
    number: str
    heading: str
    body: str


@dataclass(frozen=True)
class RawSchedule:
    schedule_id: str
    heading: str
    body: str


@dataclass(frozen=True)
class RawScheduleUnit:
    schedule_id: str
    schedule_heading: str
    number: str
    heading: str
    body: str
    structure_parts: tuple[str, ...]


_UNIT_NUMBER_RE_SRC = r"[0-9]{1,4}[A-Za-z]{0,6}(?:[\-\.][0-9A-Za-z]+)*"

# Some AustLII exports (notably QLD consolidations) render provision headings as:
#   *** 1 Short title ***
#   *** 10B Variation of exemptions ... ***
_ASTERISK_UNIT_START_RE = re.compile(
    rf"^\*{{3}}\s*(?P<number>{_UNIT_NUMBER_RE_SRC})\s+(?P<heading>.+\S)\s*(?:\*{{3}}\s*)?$",
    re.IGNORECASE,
)


def normalize_unit_number(num: str) -> str:
    """Normalize raw unit numbers to stable canonical forms.

    Some sources render hyphenated section numbers with dots, e.g. 995.1 for 995-1
    or 768.110 for 768-110. We normalize these to use hyphens so chunk keys remain
    consistent and match common statutory citation style.
    """

    s = (num or "").strip()
    if not s:
        return s
    # Trim trailing punctuation that commonly appears in TOCs like "1.1.".
    s = s.rstrip(".:")
    # If dots appear inside the number token, treat them as hyphen separators.
    # (This only applies to unit-number tokens at the start of a heading line.)
    if "." in s:
        s = s.replace(".", "-")
    s = s.rstrip("-")
    return s


def _normalize_heading_for_match(s: str) -> str:
    s = (s or "").casefold()
    s = re.sub(r"\(\s*(repealed|omitted)\s*\)", r"\1", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def headings_compatible(expected: str, found: str) -> bool:
    """Heuristic match between TOC/TOS heading and in-body heading."""

    e = _normalize_heading_for_match(expected)
    f = _normalize_heading_for_match(found)
    if not e or not f:
        return True
    if e == f:
        return True
    # Allow prefix containment both ways (TOCs often abbreviate or wrap headings).
    if e in f or f in e:
        return True
    # Allow matching on a few leading tokens.
    e_tokens = e.split()
    f_tokens = f.split()
    if e_tokens and f_tokens:
        e_head = " ".join(e_tokens[:4])
        f_head = " ".join(f_tokens[:4])
        if e_head and f_head and (e_head in f or f_head in e):
            return True
    return False


def _looks_like_doc_title(line: str) -> bool:
    s = (line or "").strip()
    if not s or not s.isupper():
        return False
    # Common in AustLII exports (e.g. 'INCOME TAX ASSESSMENT ACT 1936')
    return "ACT" in s and len(s) >= 8


_SECTIONISH_HEADING_LINE_RE = re.compile(r"^(?:Section|Regulation|SECT|REG)\s+\S+", re.IGNORECASE)

# Subsection/paragraph starts like "(1)" or " (12)".
_SUBSECTION_PARA_START_RE = re.compile(r"^\s*\(\s*(?P<num>[0-9]{1,4})\s*\)")

_SECTION_WORD_HEADING_RE = re.compile(
    rf"^\s*(?P<kind>Section|Regulation)\s+(?P<number>{_UNIT_NUMBER_RE_SRC})\.?\s+(?P<heading>.+\S)\s*$",
    re.IGNORECASE,
)

# Some consolidations repeat the document title and embed the unit label at the end:
#   PAY-ROLL TAX ASSESSMENT REGULATIONS 2003 - Regulation 1
_DOC_TITLE_UNIT_MARK_RE = re.compile(
    rf"^.+\s-\s*(?P<kind>Section|Regulation)\s+(?P<number>{_UNIT_NUMBER_RE_SRC})\s*$",
    re.IGNORECASE,
)


def structure_key_segments(structure_parts: tuple[str, ...]) -> list[str]:
    """Return short, stable key segments from Part/Division/Subdivision headings."""

    segs: list[str] = []
    seg_re = re.compile(
        r"^(?P<label>Chapter|Part|Division|Subdivision)\s+(?P<num>[0-9A-Za-z]+(?:-[0-9A-Za-z]+)?)\b",
        re.IGNORECASE,
    )
    for part in structure_parts:
        s = _clean_heading_line(part)
        m = seg_re.match(s)
        if not m:
            continue
        label = (m.group("label") or "").strip().casefold()
        num = normalize_unit_number((m.group("num") or "").strip()).casefold()
        if label and num:
            segs.append(f"{label}:{num}")
    return segs


def parse_raw_schedules(raw_text: str) -> list[RawSchedule]:
    """Parse schedule blocks from prepared/raw text.

    Supports both AustLII-style '- SCHEDULE <id>' blocks and prepared-style
    'Schedule ...' headings (e.g. 'Schedule    Ordinances repealed').
    """

    text = normalize_text(raw_text)
    lines = text.split("\n")

    marks: list[int] = []
    for i, line in enumerate(lines):
        raw = line or ""
        s = raw.strip()
        if _SCHEDULE_MARK_RE.match(s):
            marks.append(i)
            continue

        # Be conservative with 'Schedule ...' headings to avoid false positives from
        # in-body references like 'Schedule 1 to the Taxation Administration Act 1953'.
        m2 = _SCHEDULE_WORD_MARK_RE.match(raw)
        if not m2:
            continue

        # Must be a standalone heading line (not indented).
        if raw.lstrip() != raw:
            continue

        s_cf = s.casefold()
        if " to the " in s_cf:
            continue

        prev_blank = i == 0 or not (lines[i - 1] or "").strip()
        next_blank = i + 1 >= len(lines) or not (lines[i + 1] or "").strip()
        if not (prev_blank or next_blank or s.isupper()):
            continue

        marks.append(i)

    if not marks:
        return []

    schedules: list[RawSchedule] = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        start_line = (lines[start] or "").strip()
        m = _SCHEDULE_MARK_RE.match(start_line)
        m2 = _SCHEDULE_WORD_MARK_RE.match(start_line) if not m else None
        if not m and not m2:
            continue

        schedule_id = ""
        inline_title = ""
        if m:
            schedule_id = (m.group("id") or "").strip()
            inline_title = (m.group("title") or "").strip() if m.group("title") else ""
        else:
            schedule_id = (m2.group("id") or "").strip() if m2 else ""
            inline_title = (m2.group("title") or "").strip() if m2 else ""

        if not schedule_id:
            # Some consolidations have a single unnumbered schedule.
            schedule_id = "1"

        # Heading: prefer inline title, else next non-empty non-doc-title line.
        heading = inline_title
        j = start + 1
        if not heading:
            while j < end:
                candidate = (lines[j] or "").strip()
                if candidate and not _looks_like_doc_title(candidate):
                    heading = candidate
                    j += 1
                    break
                j += 1

        # Append wrapped heading lines until a blank line or a structural/provision line.
        while j < end:
            nxt = (lines[j] or "")
            if not nxt.strip():
                break
            nxt_clean = _clean_heading_line(nxt)
            if _CHAPTER_RE.match(nxt_clean):
                break
            if _PART_RE.match(nxt_clean) or _DIVISION_RE.match(nxt_clean) or _SUBDIVISION_RE.match(nxt_clean):
                break
            if _SECTION_HEADING_COL_RE.match(nxt) or _SECTION_HEADING_DOT_RE.match(nxt):
                break
            if _SCHEDULE_MARK_RE.match(nxt.strip()) or _SCHEDULE_WORD_MARK_RE.match(nxt.strip()):
                break
            heading = (heading + " " + nxt.strip()).strip()
            j += 1

        while j < end and not (lines[j] or "").strip():
            j += 1

        raw_body_lines = [(lines[k] or "").rstrip("\r") for k in range(j, end)]
        # Strip obvious repeated doc-title lines near schedule boundaries.
        trimmed: list[str] = []
        for idx, ln in enumerate(raw_body_lines):
            if (idx < 3 or idx >= len(raw_body_lines) - 3) and _looks_like_doc_title(ln):
                continue
            trimmed.append(ln)
        body = normalize_text("\n".join(trimmed))
        schedules.append(RawSchedule(schedule_id=schedule_id, heading=heading, body=body))

    return schedules


def parse_schedule_units(schedule: RawSchedule) -> list[RawScheduleUnit]:
    """Parse internal numbered provisions within a schedule block.

    Many schedules (e.g. ITAA 1936 Schedule 2F) use provisions like:
      265-5  What this Schedule is about

    This returns per-provision units with any in-schedule Division/Subdivision context.
    """

    if not (schedule.body or "").strip():
        return []

    # Some schedules arrive with their internal headings collapsed into a single long line
    # (e.g. "... Guide to Division 57 Section 57-1 ..."). To keep schedule parsing robust,
    # insert newlines before capitalized 'Section <number>' tokens so we can treat them as
    # discrete heading lines without disturbing normal in-text references (usually 'section').
    sched_text = schedule.body
    if sched_text.count("\n") < 3 and re.search(r"\bSection\s+" + _UNIT_NUMBER_RE_SRC + r"\b", sched_text):
        sched_text = re.sub(r"(?<!\n)(?=\bSection\s+" + _UNIT_NUMBER_RE_SRC + r"\b)", "\n", sched_text)

    lines = sched_text.split("\n")

    marks: list[tuple[int, str, str, tuple[str, ...]]] = []
    current: dict[str, str] = {}

    for i, line in enumerate(lines):
        s_clean = _clean_heading_line(line)
        if not s_clean:
            continue

        if _CHAPTER_RE.match(s_clean):
            current["chapter"] = _normalize_structure_heading(s_clean)
            current.pop("part", None)
            current.pop("division", None)
            current.pop("subdivision", None)
            continue

        if _PART_RE.match(s_clean):
            current["part"] = _normalize_structure_heading(s_clean)
            current.pop("division", None)
            current.pop("subdivision", None)
            continue

        if _DIVISION_RE.match(s_clean):
            current["division"] = _normalize_structure_heading(s_clean)
            current.pop("subdivision", None)
            continue

        if _SUBDIVISION_RE.match(s_clean):
            current["subdivision"] = _normalize_structure_heading(s_clean)
            continue

        # Schedule provisions can be rendered as:
        #   265-5  Heading
        #   265-5. Heading
        #   Section 57-1 Heading
        is_word_heading = False
        m = _SECTION_HEADING_COL_RE.match(line) or _SECTION_HEADING_DOT_RE.match(line)
        if not m:
            m = _SECTION_WORD_HEADING_RE.match(s_clean)
            is_word_heading = m is not None
        if not m:
            continue

        heading = (m.group("heading") or "").strip()
        if re.search(r"\s\d+$", heading):
            continue

        number = normalize_unit_number((m.group("number") or "").strip())
        # Heuristic:
        # - When the line explicitly says 'Section <n> ...', treat it as a statutory section,
        #   even if <n> is a plain number (e.g. 'Section 1 KEY').
        # - For other heading formats (e.g. '1  Zone name'), require hyphenated numbers to
        #   avoid interpreting numeric lists as sections.
        if not is_word_heading and "-" not in number:
            continue
        parts = tuple(current[k] for k in ("chapter", "part", "division", "subdivision") if k in current)
        marks.append((i, number, heading, parts))

    if not marks:
        return []

    out: list[RawScheduleUnit] = []
    for n, (start, number, heading, parts) in enumerate(marks):
        end = marks[n + 1][0] if n + 1 < len(marks) else len(lines)

        # If the heading wraps onto subsequent indented lines, append them.
        j = start + 1
        while j < end:
            sj = (lines[j] or "")
            if not sj.strip():
                break
            if sj.lstrip() == sj:  # not indented
                break
            if sj.strip().startswith("("):
                break
            if _SECTION_HEADING_COL_RE.match(sj.strip()) or _SECTION_HEADING_DOT_RE.match(sj.strip()):
                break
            if _PART_RE.match(sj.strip()) or _DIVISION_RE.match(sj.strip()) or _SUBDIVISION_RE.match(sj.strip()):
                break
            heading = (heading + " " + sj.strip()).strip()
            j += 1

        while j < end and not (lines[j] or "").strip():
            j += 1

        body = normalize_text("\n".join((lines[k] or "") for k in range(j, end)))
        out.append(
            RawScheduleUnit(
                schedule_id=schedule.schedule_id,
                schedule_heading=schedule.heading,
                number=number,
                heading=heading,
                body=body,
                structure_parts=parts,
            )
        )

    return out


def parse_schedule_parts(schedule: RawSchedule) -> list[tuple[str, str, str]]:
    """Split a schedule into Part blocks.

    Used for schedules that don't contain hyphenated provision numbers (e.g. Schedule 2).
    Returns list of (part_number, part_heading_line, part_body).
    """

    text = normalize_text(schedule.body)
    if not text:
        return []

    lines = text.split("\n")
    part_idxs: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        m = _PART_RE.match(_clean_heading_line(line))
        if not m:
            continue
        num = (m.group("number") or "").strip()
        if not num:
            continue
        part_idxs.append((i, num))

    if not part_idxs:
        return []

    out: list[tuple[str, str, str]] = []
    for n, (start, part_num) in enumerate(part_idxs):
        end = part_idxs[n + 1][0] if n + 1 < len(part_idxs) else len(lines)
        part_heading = _clean_heading_line(lines[start])

        j = start + 1
        while j < end and not (lines[j] or "").strip():
            j += 1

        body = normalize_text("\n".join((lines[k] or "") for k in range(j, end)))
        out.append((part_num, part_heading, body))

    return out


def strip_schedule_blocks(text: str) -> str:
    """Remove '- SCHEDULE ...' blocks from an AustLII-style prepared body.

    This prevents schedule-internal headings like 'Section 6-1 ...' (TAA 1953 Schedule 1)
    from being double-parsed as top-level Act sections.
    """

    lines = normalize_text(text).split("\n")

    def is_real_schedule_heading(i: int) -> bool:
        raw = lines[i] or ""
        s = raw.strip()
        if not s:
            return False

        # AustLII explicit schedule markers are safe to treat as true boundaries.
        if _SCHEDULE_MARK_RE.match(s):
            return True

        # Be conservative with 'Schedule ...' to avoid in-body cross-references.
        if not _SCHEDULE_WORD_MARK_RE.match(raw):
            return False
        if raw.lstrip() != raw:
            return False
        if " to the " in s.casefold():
            return False
        prev_blank = i == 0 or not (lines[i - 1] or "").strip()
        next_blank = i + 1 >= len(lines) or not (lines[i + 1] or "").strip()
        if not (prev_blank or next_blank or s.isupper()):
            return False
        return True

    sched_starts: list[int] = [i for i in range(len(lines)) if is_real_schedule_heading(i)]

    if not sched_starts:
        return normalize_text(text)

    keep: list[str] = []
    skip_ranges: list[tuple[int, int]] = []
    for n, start in enumerate(sched_starts):
        end = sched_starts[n + 1] if n + 1 < len(sched_starts) else len(lines)
        skip_ranges.append((start, end))

    skip_i = 0
    cur = skip_ranges[skip_i]
    for i, line in enumerate(lines):
        if cur and cur[0] <= i < cur[1]:
            if i == cur[1] - 1 and skip_i + 1 < len(skip_ranges):
                skip_i += 1
                cur = skip_ranges[skip_i]
            continue
        keep.append(line)
        if cur and i >= cur[1] - 1 and skip_i + 1 < len(skip_ranges):
            skip_i += 1
            cur = skip_ranges[skip_i]

    return normalize_text("\n".join(keep))


_SECTION_HEADING_COL_RE = re.compile(
    rf"^\s*(?P<number>{_UNIT_NUMBER_RE_SRC})\s{{2,}}(?P<heading>.+\S)\s*$"
)

_SECTION_HEADING_DOT_RE = re.compile(
    rf"^\s*(?P<number>{_UNIT_NUMBER_RE_SRC})\.\s*(?P<heading>.+\S)\s*$"
)

# Some consolidations (notably several State regulation exports) use headings like:
#   1--Short title
#   10A--Exemption for certain heavy vehicles
_SECTION_HEADING_DASH_RE = re.compile(
    rf"^\s*(?P<number>{_UNIT_NUMBER_RE_SRC})\s*--\s*(?P<heading>.+\S)\s*$"
)

# Other sources use single-space headings like:
#   1 Short title
#   3 Interpretation
_SECTION_HEADING_SPACE_RE = re.compile(
    rf"^\s*(?P<number>{_UNIT_NUMBER_RE_SRC})\s+(?P<heading>.+\S)\s*$"
)


def _parse_units_from_heading_lines(
    lines: list[str], *, start_idx: int, default_kind: str, tos_heading_by_number: dict[str, str] | None = None
) -> list[RawUnit]:
    marks: list[int] = []

    def _is_year_number(num: str) -> bool:
        if not (num or "").isdigit() or len(num) != 4:
            return False
        try:
            y = int(num)
        except ValueError:
            return False
        return 1800 <= y <= 2100

    def _match_heading_line(line: str):
        return (
            _SECTION_HEADING_DASH_RE.match(line)
            or _SECTION_HEADING_COL_RE.match(line)
            or _SECTION_HEADING_DOT_RE.match(line)
            or _SECTION_HEADING_SPACE_RE.match(line)
        )

    def _is_likely_date_heading(m: re.Match) -> bool:
        # Avoid interpreting legislative-history date lines like:
        #   28.6.2018
        # which often get tokenized as number='28.6' heading='2018'.
        try:
            number_raw = (m.group("number") or "").strip()
            heading_raw = (m.group("heading") or "").strip()
        except Exception:
            return False
        if not heading_raw.isdigit() or len(heading_raw) != 4:
            return False
        if not re.fullmatch(r"\d{1,2}\.\d{1,2}", number_raw):
            return False
        y = int(heading_raw)
        return 1800 <= y <= 2100

    for i in range(start_idx, len(lines)):
        s = (lines[i] or "").rstrip("\r")
        m = _match_heading_line(s)
        if not m:
            continue

        number_raw = (m.group("number") or "").strip()
        if _is_year_number(number_raw):
            continue

        if _is_likely_date_heading(m):
            continue

        number = normalize_unit_number(number_raw)
        heading = (m.group("heading") or "").strip()
        if tos_heading_by_number is not None and tos_heading_by_number:
            expected = tos_heading_by_number.get(number)
            if expected is None:
                continue
            if expected and heading and not headings_compatible(expected, heading):
                continue

        # Exclude TOC-style rows that end with a page number column.
        # (Common in 'Contents'/'Table of provisions' sections.)
        heading = (m.group("heading") or "").strip()
        if re.search(r"\s{2,}\d+$", heading) or re.search(r"\s\d+$", heading):
            # Heuristic: headings rarely end with a bare number.
            continue

        marks.append(i)

    if not marks:
        return []

    kind_word = default_kind
    units: list[RawUnit] = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)
        s0 = (lines[start] or "").rstrip("\r")
        m0 = _match_heading_line(s0)
        if not m0:
            continue

        number = normalize_unit_number((m0.group("number") or "").strip())
        heading = (m0.group("heading") or "").strip()

        # If the heading wraps onto subsequent indented lines, append them.
        j = start + 1
        while j < end:
            sj = (lines[j] or "")
            if not sj.strip():
                break
            if sj.lstrip() == sj:  # not indented
                break
            if sj.strip().startswith("("):
                break
            if _SECTION_HEADING_COL_RE.match(sj.strip()) or _SECTION_HEADING_DOT_RE.match(sj.strip()):
                break
            if _PART_RE.match(sj.strip()) or _DIVISION_RE.match(sj.strip()) or _SUBDIVISION_RE.match(sj.strip()):
                break
            heading = (heading + " " + sj.strip()).strip()
            j += 1

        # Body starts after any immediate blank line(s) following the heading.
        while j < end and not (lines[j] or "").strip():
            j += 1

        body_lines = [(lines[k] or "").rstrip("\r") for k in range(j, end)]
        body = normalize_text("\n".join(body_lines))
        units.append(RawUnit(kind=kind_word, number=number, heading=heading, body=body))

    return units


def parse_raw_units(
    raw_text: str,
    *,
    default_kind: str = "Section",
    tos_heading_by_number: dict[str, str] | None = None,
) -> list[RawUnit]:
    text = normalize_text(raw_text)
    lines = text.split("\n")

    def _maybe_add_tos_stubs(units: list[RawUnit]) -> list[RawUnit]:
        # If the TOC/TOS explicitly lists repealed/omitted sections/regulations that are
        # absent from the body text, generate stub units so downstream coverage checks
        # (and exact retrieval) still have an anchor.
        if not tos_heading_by_number:
            return units
        existing = {u.number for u in units if u.number}
        for num, expected_heading in tos_heading_by_number.items():
            if not num or num in existing:
                continue
            if not (num[:1].isdigit()):
                continue
            # If we parsed at least some real units, fill any TOC gaps with stubs.
            # This is important for consolidations that omit numbered provisions from
            # the body (common in some State exports).
            if units:
                units.append(RawUnit(kind=default_kind, number=num, heading=expected_heading, body=""))
                continue

            # If we parsed nothing, keep a stricter rule to avoid masking parser failures.
            h_cf = (expected_heading or "").casefold()
            if "repealed" in h_cf or "omitted" in h_cf:
                units.append(RawUnit(kind=default_kind, number=num, heading=expected_heading, body=""))
        return units

    schedule_marks: list[int] = []
    notes_marks: list[int] = []
    endnotes_marks: list[int] = []
    for i, line in enumerate(lines):
        s = (line or "").strip()
        if _SCHEDULE_MARK_RE.match(s):
            schedule_marks.append(i)
        if _NOTES_MARK_RE.match(s):
            notes_marks.append(i)
        if s.casefold() == "endnotes":
            endnotes_marks.append(i)

    marks: list[int] = []
    for i, line in enumerate(lines):
        s = (line or "").strip()
        if (
            _SECTION_MARK_RE.match(s)
            or _SECTION_WORD_MARK_RE.match(s)
            or _DOC_TITLE_UNIT_MARK_RE.match(s)
            or _ASTERISK_UNIT_START_RE.match(s)
        ):
            marks.append(i)

    # Some AustLII exports mostly use '- SECT n' markers, but occasionally omit the
    # marker and instead render the heading as a plain line like:
    #   Section 75-16 Margins for supplies ...
    # If we don't treat these as boundaries, they get absorbed into the previous unit.
    if marks:
        seen_numbers: set[str] = set()
        for i in marks:
            m0 = _SECTION_MARK_RE.match((lines[i] or "").strip())
            if not m0:
                continue
            seen_numbers.add(normalize_unit_number((m0.group("number") or "").strip()))

        extra_mark_idxs: list[int] = []

        def _prev_nonempty_idx(j: int) -> int | None:
            while j >= 0:
                s = (lines[j] or "").strip()
                if s and not _looks_like_doc_title(s):
                    return j
                j -= 1
            return None

        for i, line in enumerate(lines):
            raw = line or ""
            s = raw.strip()
            if not s:
                continue
            if raw.lstrip() != raw:
                continue

            m = _SECTION_WORD_HEADING_RE.match(s)
            if not m:
                continue

            number = normalize_unit_number((m.group("number") or "").strip())
            heading = (m.group("heading") or "").strip()
            if not number or number in seen_numbers:
                continue

            # When we have a TOC/TOS section list, use it as a strong signal of
            # which "Section/Regulation N ..." lines are real boundaries. This prevents
            # schedule/endnotes internal headings from becoming new units.
            # Important: only apply this allow-list when the TOC map is non-empty.
            if tos_heading_by_number is not None and tos_heading_by_number and number not in tos_heading_by_number:
                continue

            if tos_heading_by_number is not None:
                expected = tos_heading_by_number.get(number)
                if expected and not headings_compatible(expected, heading):
                    continue

            # Avoid example-style narrative headings.
            prev_i = _prev_nonempty_idx(i - 1)
            if prev_i is not None:
                prev_cf = (lines[prev_i] or "").strip().casefold()
                if prev_cf.startswith("example for this section"):
                    continue

            extra_mark_idxs.append(i)
            seen_numbers.add(number)

        if extra_mark_idxs:
            marks = sorted(set(marks + extra_mark_idxs))

    # If the file has been normalized to "Section <n> <heading>" style (prepared outputs),
    # parse those headings too. Critical: avoid treating "Table of sections" lists (and other
    # TOC/TOS blocks) as real section starts.
    if not marks:
        heading_marks: list[tuple[int, str, str, str]] = []  # (idx, kind_word, number, heading)
        table_mode = False
        saw_hyphenated_number = False
        seen_numbers: set[str] = set()
        table_triggers = {
            "table of provisions",
            "table of contents",
            "table of sections",
            "contents",
        }

        def _next_nonempty(j: int) -> tuple[int, str] | None:
            while j < len(lines):
                s = (lines[j] or "").strip()
                if s and not _looks_like_doc_title(s):
                    return j, s
                j += 1
            return None

        def _prev_nonempty(j: int) -> tuple[int, str] | None:
            while j >= 0:
                s = (lines[j] or "").strip()
                if s and not _looks_like_doc_title(s):
                    return j, s
                j -= 1
            return None

        for i, line in enumerate(lines):
            s = (line or "").strip()
            if not s:
                continue

            s_cf = s.casefold()
            if s_cf in table_triggers:
                table_mode = True
                continue
            if "- long title" in s_cf:
                table_mode = False

            m = _SECTION_WORD_HEADING_RE.match(s)
            if not m:
                continue

            kind_word = (m.group("kind") or default_kind).strip().capitalize()
            number = normalize_unit_number((m.group("number") or "").strip())
            heading = (m.group("heading") or "").strip()
            if not number:
                continue

            # If we have a TOC/TOS-derived section list, use it as a strong signal
            # of what constitutes a real unit boundary. This avoids false positives
            # from in-body references like 'Section 30 January 2001' or 'Section 19'.
            if tos_heading_by_number is not None and tos_heading_by_number and number not in tos_heading_by_number:
                continue

            if tos_heading_by_number is not None:
                expected = tos_heading_by_number.get(number)
                if expected and not headings_compatible(expected, heading):
                    continue

            # Ignore in-body examples that look like:
            #   Example for this section:
            #   Section 1. "Perth local government district" refers ...
            prev = _prev_nonempty(i - 1)
            if prev is not None:
                _, prev_line = prev
                prev_cf = (prev_line or "").strip().casefold()
                if prev_cf.startswith("example for this section"):
                    continue

            # Section numbers should be unique within a document; avoid later false positives.
            if number in seen_numbers:
                continue

            # If we are in a hyphen-numbered Act (e.g. ITAA 1997 sections like 110-25),
            # then stray in-body labels such as "Section 2 interposed entities" can
            # appear inside method statements. These should not become new chunk
            # boundaries.
            if "-" in number:
                saw_hyphenated_number = True
            if saw_hyphenated_number and number.isdigit():
                nxt = _next_nonempty(i + 1)
                if nxt is not None:
                    _, nxt_line = nxt
                    m_sub = _SUBSECTION_PARA_START_RE.match(nxt_line)
                    if m_sub and (m_sub.group("num") or "") != "1":
                        continue

            if table_mode:
                nxt = _next_nonempty(i + 1)
                if nxt is None:
                    continue
                _, nxt_line = nxt
                # Still within a table/list if the next line is another heading-like line.
                if _SECTION_WORD_HEADING_RE.match(nxt_line) or _PART_RE.match(nxt_line) or _DIVISION_RE.match(nxt_line) or _SUBDIVISION_RE.match(nxt_line):
                    continue
                # Otherwise this looks like a real section heading with body text following.
                table_mode = False

            heading_marks.append((i, kind_word, number, heading))
            seen_numbers.add(number)

        if heading_marks:
            units: list[RawUnit] = []
            for n, (start, kind_word, number, heading) in enumerate(heading_marks):
                end = heading_marks[n + 1][0] if n + 1 < len(heading_marks) else len(lines)

                # Prevent a section/regulation from absorbing schedules and compilation notes.
                for stop_marks in (schedule_marks, notes_marks, endnotes_marks):
                    if not stop_marks:
                        continue
                    j_stop = bisect.bisect_right(stop_marks, start)
                    if j_stop < len(stop_marks):
                        end = min(end, stop_marks[j_stop])

                j = start + 1
                while j < end and not (lines[j] or "").strip():
                    j += 1

                body_lines: list[str] = []
                for k in range(j, end):
                    ln = (lines[k] or "").rstrip()
                    if not ln:
                        body_lines.append("")
                        continue
                    if _looks_like_doc_title(ln):
                        continue
                    body_lines.append(ln)

                body = normalize_text("\n".join(body_lines))
                units.append(RawUnit(kind=kind_word, number=number, heading=heading, body=body))

            return _maybe_add_tos_stubs(units)

    if not marks:
        # Many State/Territory consolidations contain the substantive text only
        # after an ENDNOTES marker, and do not use '- SECT n' markers. Attempt
        # a conservative parse from section-heading lines like:
        #   1     Short title
        start_idx = 0

        # Prefer starting after ENDNOTES (common in NT-style consolidations).
        for i, line in enumerate(lines[:3000]):
            if (line or "").strip().casefold() == "endnotes":
                start_idx = i + 1
                break

        # Otherwise, prefer starting after a LONG TITLE marker (common in AustLII exports),
        # which usually comes after the Table of Provisions.
        if start_idx == 0:
            for i, line in enumerate(lines[:8000]):
                s = (line or "").strip().casefold()
                if "long title" in s:
                    start_idx = i + 1
                    break

        units = _parse_units_from_heading_lines(
            lines,
            start_idx=start_idx,
            default_kind=default_kind,
            tos_heading_by_number=tos_heading_by_number,
        )
        if units:
            return _maybe_add_tos_stubs(units)

        # Last resort: try from the top (still excluding TOC page-number rows).
        return _maybe_add_tos_stubs(
            _parse_units_from_heading_lines(
                lines,
                start_idx=0,
                default_kind=default_kind,
                tos_heading_by_number=tos_heading_by_number,
            )
        )

    units: list[RawUnit] = []
    for n, start in enumerate(marks):
        end = marks[n + 1] if n + 1 < len(marks) else len(lines)

        # Prevent a section/regulation from absorbing schedules and compilation notes.
        for stop_marks in (schedule_marks, notes_marks, endnotes_marks):
            if not stop_marks:
                continue
            j_stop = bisect.bisect_right(stop_marks, start)
            if j_stop < len(stop_marks):
                end = min(end, stop_marks[j_stop])
        start_line = (lines[start] or "").strip()
        m = _SECTION_MARK_RE.match(start_line)
        m_mark_word = None if m else _SECTION_WORD_MARK_RE.match(start_line)
        m_doc_mark = None if (m or m_mark_word) else _DOC_TITLE_UNIT_MARK_RE.match(start_line)
        m_word = None if (m or m_mark_word or m_doc_mark) else _SECTION_WORD_HEADING_RE.match(start_line)
        m_star = None if (m or m_mark_word or m_doc_mark or m_word) else _ASTERISK_UNIT_START_RE.match(start_line)
        if not m and not m_mark_word and not m_doc_mark and not m_word and not m_star:
            continue

        kind_word = "Section"
        number = ""
        heading = ""
        j = start + 1

        if m:
            kind = (m.group("kind") or "SECT").strip().upper()
            number = normalize_unit_number((m.group("number") or "").strip())
            kind_word = "Regulation" if kind == "REG" else "Section"

            # Heading is next non-empty line.
            while j < end:
                candidate = (lines[j] or "").strip()
                if candidate:
                    heading = candidate
                    j += 1
                    break
                j += 1
        elif m_mark_word:
            kind_word = (m_mark_word.group("kind") or default_kind).strip().capitalize()
            number = normalize_unit_number((m_mark_word.group("number") or "").strip())

            # Heading is next non-empty line.
            while j < end:
                candidate = (lines[j] or "").strip()
                if candidate:
                    heading = candidate
                    j += 1
                    break
                j += 1

        elif m_doc_mark:
            kind_word = (m_doc_mark.group("kind") or default_kind).strip().capitalize()
            number = normalize_unit_number((m_doc_mark.group("number") or "").strip())

            # Heading is typically on the next non-empty line, often as a dot-prefixed
            # TOC/table row: '.      Citation'. Skip known repeated headers.
            while j < end:
                candidate_raw = (lines[j] or "")
                candidate = candidate_raw.strip()
                if not candidate:
                    j += 1
                    continue
                c_cf = candidate.casefold()
                if c_cf in {"western australian current regulations", "search this regulation"}:
                    j += 1
                    continue
                if re.fullmatch(r"\d+[A-Za-z]{0,6}\.?", candidate):
                    j += 1
                    continue
                if candidate.startswith("."):
                    candidate = candidate.lstrip(". ")
                heading = candidate
                j += 1
                break

        elif m_star:
            kind_word = default_kind
            number = normalize_unit_number((m_star.group("number") or "").strip())
            heading = (m_star.group("heading") or "").strip().rstrip("*").strip()

            # Some sources split the marker over multiple lines, e.g.
            #   *** 5 Notice of amended assessments ...
            #   ***
            # Continue consuming lines until the closing '***'.
            if not start_line.rstrip().endswith("***"):
                while j < end:
                    cand_raw = lines[j] or ""
                    cand = cand_raw.strip()
                    if not cand:
                        j += 1
                        continue
                    if cand == "***":
                        j += 1
                        break
                    if cand.endswith("***"):
                        frag = cand.rstrip("*").strip()
                        if frag:
                            heading = (heading + " " + frag).strip()
                        j += 1
                        break
                    heading = (heading + " " + cand).strip()
                    j += 1

            if tos_heading_by_number is not None:
                expected = tos_heading_by_number.get(number)
                if expected and not headings_compatible(expected, heading):
                    continue

            # Body starts after any immediate blank line(s) following the marker block.
            while j < end and not (lines[j] or "").strip():
                j += 1

        else:
            kind_word = (m_word.group("kind") or default_kind).strip().capitalize()
            number = normalize_unit_number((m_word.group("number") or "").strip())
            heading = (m_word.group("heading") or "").strip()

            if tos_heading_by_number is not None:
                expected = tos_heading_by_number.get(number)
                if expected and not headings_compatible(expected, heading):
                    continue

            # Body starts after any immediate blank line(s) following the heading.
            while j < end and not (lines[j] or "").strip():
                j += 1

        # Body: from after heading line to end, remove leading repeated document title lines.
        body_lines: list[str] = []
        for k in range(j, end):
            s = (lines[k] or "").rstrip()
            if not s:
                body_lines.append("")
                continue
            # Skip repeated all-caps doc title lines.
            if _looks_like_doc_title(s):
                continue
            body_lines.append(s)

        body = normalize_text("\n".join(body_lines))
        units.append(RawUnit(kind=kind_word, number=number, heading=heading, body=body))

    return _maybe_add_tos_stubs(units)


# Match dictionary entries like:
#   "Term" means ...
#   "ABN ..." for an entity means ...
# Word-boundary (\b) does not work after a closing quote, so use a lookahead.
_DICT_TERM_START_RE = re.compile(r"^\s*(?:\"[^\"]+\"|“[^”]+”)(?=\s|:|$)")


def slug_term(term: str) -> str:
    s = (term or "").strip()
    s = s.strip("\"'“”")
    s = s.casefold()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "term"


def split_dictionary_entries(body: str) -> list[tuple[str, str]] | None:
    lines = (body or "").split("\n")
    term_idxs: list[int] = []
    for i, line in enumerate(lines):
        if _DICT_TERM_START_RE.match(line or ""):
            term_idxs.append(i)

    if not term_idxs:
        return None

    out: list[tuple[str, str]] = []
    for n, start in enumerate(term_idxs):
        end = term_idxs[n + 1] if n + 1 < len(term_idxs) else len(lines)
        entry = "\n".join((x or "").rstrip() for x in lines[start:end]).strip()
        if not entry:
            continue
        first_line = (lines[start] or "").strip()
        m = _DICT_TERM_START_RE.match(first_line)
        term = (m.group(0) if m else first_line).strip()
        out.append((term, entry))

    return out or None


def looks_like_dictionary_section(*, heading: str, body: str) -> bool:
    """Heuristic: decide if a unit is a Dictionary/Definitions-style section.

    Many acts use a section titled 'Dictionary' or 'Definitions', but some use
    other headings (e.g. 'Interpretation') while still containing a list of
    quoted defined terms. We prefer a conservative heuristic to avoid false splits.
    """

    h = (heading or "").casefold()
    if "dictionary" in h or "definitions" in h:
        return True

    # Body-driven heuristic: look for an introductory line then several quoted terms.
    lines = [(x or "").strip() for x in (body or "").split("\n")]
    nonempty = [x for x in lines if x]
    head = " ".join(nonempty[:3]).casefold()
    if "in this act" not in head:
        return False

    # Count term-entry starters near the top.
    term_starts = 0
    for line in nonempty[:200]:
        if _DICT_TERM_START_RE.match(line):
            term_starts += 1
            if term_starts >= 3:
                return True
    return False


def chunk_words(words: list[str], chunk_size: int, chunk_overlap: int) -> list[list[str]]:
    if not words:
        return []
    if chunk_size <= 0:
        return [words]

    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    out: list[list[str]] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        out.append(words[start:end])
        if end >= len(words):
            break
        start = end - overlap
    return out


def build_sidecar_for_doc(
    *,
    prepared_text: str,
    citation: str,
    source: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    include_structure_line: bool,
) -> dict:
    doc_meta = parse_doc_header(prepared_text)
    # Stable doc identity fields (kept in doc-level metadata; copied to each chunk payload by the indexer).
    doc_meta.setdefault("source", source)
    doc_meta.setdefault("filename", filename)
    doc_meta.setdefault("name", citation)
    doc_meta.setdefault("citation", citation)
    jurisdiction = (doc_meta.get("jurisdiction") or "").strip().casefold() or None

    prepared_body_with_toc = strip_prepared_header(prepared_text)
    prepared_body_for_chunking = slice_to_body_start(prepared_body_with_toc)
    # Some WA consolidations append large non-law material (compilation tables,
    # uncommenced provisions, other notes, defined-term lists) that can contain
    # misleading tokens like 'Schedule 1'. Exclude that end-matter from chunking.
    prepared_body_for_chunking = strip_known_end_matter(
        prepared_body_for_chunking,
        jurisdiction=jurisdiction,
    )

    # Structure inference uses the TOC/TOS blocks retained in the prepared body.
    structure_by_section, heading_by_section = extract_toc_structure_map(prepared_body_with_toc)
    # Fallback: infer structure from in-body Part/Division headings when no explicit TOC marker exists.
    body_structure_by_section = extract_body_structure_map(prepared_body_for_chunking)
    default_kind = "Regulation" if "regulation" in citation.casefold() else "Section"
    schedules = parse_raw_schedules(prepared_body_for_chunking)
    body_without_schedules = strip_schedule_blocks(prepared_body_for_chunking) if schedules else prepared_body_for_chunking
    units = parse_raw_units(
        body_without_schedules,
        default_kind=default_kind,
        tos_heading_by_number=heading_by_section,
    )

    chunks: list[dict] = []

    # Some sources (e.g. OCR/PDF-style consolidations or amending acts) do not use
    # AustLII-style '- SECT n' / '- REG n' markers. In that case, fall back to
    # deterministic word-chunking of the raw text so the sidecar still produces
    # usable chunks and indexing remains sidecar-driven.
    if not units and not schedules:
        fallback_text = normalize_text(prepared_body_for_chunking)
        fallback_words = fallback_text.split()
        if not fallback_words:
            return {"doc": doc_meta, "chunks": []}

        parts = chunk_words(fallback_words, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, part_words in enumerate(parts, start=1):
            key = f"fallback:p{i}"
            structure_path = ensure_min_two_level_structure_path(
                f"{citation} > Fallback p{i}".strip(),
                citation=citation,
            )
            chunk_text = " ".join(part_words).strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "key": key,
                    "title": f"{citation} (fallback p{i})",
                    "main_structural_unit": "fallback",
                    "main_structural_number": None,
                    "main_structural_heading": None,
                    "defined_term": None,
                    "structure_path": structure_path,
                    "text": chunk_text,
                }
            )

        return {"doc": doc_meta, "chunks": chunks}

    for unit in units:
        number = unit.number
        kind = unit.kind  # Section/Regulation

        section_heading = unit.heading or heading_by_section.get(number) or ""
        header_line = f"{kind} {number}".strip()
        if section_heading:
            header_line = f"{header_line} - {section_heading}".strip()

        structure_path_base = structure_by_section.get(number) or body_structure_by_section.get(number)
        # Always provide a structure_path (even if we couldn't infer higher-level structure).
        structure_path: str | None
        if structure_path_base:
            structure_path = f"{structure_path_base} > {kind} {number}".strip()
        else:
            structure_path = f"{kind} {number}".strip()

        structure_path = ensure_min_two_level_structure_path(structure_path, citation=citation)

        # Dictionary / Definitions splitting.
        is_dict = looks_like_dictionary_section(heading=section_heading, body=unit.body)
        dict_entries = split_dictionary_entries(unit.body) if is_dict else None

        if dict_entries:
            label = "Dictionary" if "dictionary" in section_heading.casefold() else "Definitions"
            for term, entry in dict_entries:
                term_key = slug_term(term)
                key = f"{kind.casefold()}:{number}.term:{term_key}"
                title = f"{kind} {number} {label} {term}".strip()
                text_lines: list[str] = []
                if include_structure_line and structure_path:
                    text_lines.append(f"Structure: {structure_path}")
                if text_lines:
                    text_lines.append("")
                text_lines.append(entry)
                chunk_text = normalize_text("\n".join(text_lines))
                chunks.append(
                    {
                        "key": key,
                        "title": title,
                        "main_structural_unit": kind.casefold(),
                        "main_structural_number": number,
                        "main_structural_heading": section_heading or None,
                        "defined_term": term,
                        "structure_path": structure_path,
                        "text": chunk_text,
                    }
                )
            continue

        # Regular section/reg chunking with optional word-splitting for very large units.
        base_lines: list[str] = []
        if include_structure_line and structure_path:
            base_lines.append(f"Structure: {structure_path}")
        if base_lines:
            base_lines.append("")

        body_words = unit.body.split()
        if chunk_size > 0 and len(body_words) > chunk_size:
            parts = chunk_words(body_words, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for i, part_words in enumerate(parts, start=1):
                part_key = f"{kind.casefold()}:{number}.p{i}"
                chunk_text = normalize_text("\n".join([*base_lines, " ".join(part_words)]))
                chunks.append(
                    {
                        "key": part_key,
                        "title": header_line,
                        "main_structural_unit": kind.casefold(),
                        "main_structural_number": number,
                        "main_structural_heading": section_heading or None,
                        "defined_term": None,
                        "structure_path": structure_path,
                        "text": chunk_text,
                    }
                )
        else:
            key = f"{kind.casefold()}:{number}"
            chunk_text = normalize_text("\n".join([*base_lines, unit.body]))
            chunks.append(
                {
                    "key": key,
                    "title": header_line,
                    "main_structural_unit": kind.casefold(),
                    "main_structural_number": number,
                    "main_structural_heading": section_heading or None,
                    "defined_term": None,
                    "structure_path": structure_path,
                    "text": chunk_text,
                }
            )

    # Schedule chunking (AustLII '- SCHEDULE ...' blocks).
    for sched in schedules:
        sched_id = (sched.schedule_id or "").strip()
        if not sched_id:
            continue
        sched_key = sched_id.casefold()
        sched_heading = (sched.heading or "").strip()
        schedule_label = f"Schedule {sched_id}".strip()

        sched_units = parse_schedule_units(sched)
        if not sched_units:
            # No hyphenated provisions detected.
            # If this schedule has Parts (e.g. Schedule 2 Zones), chunk by Part.
            part_blocks = parse_schedule_parts(sched)
            if part_blocks:
                for part_num, part_heading, part_body in part_blocks:
                    part_num_cf = (part_num or "").strip().casefold()
                    key = f"schedule:{sched_key}.part:{part_num_cf}" if part_num_cf else f"schedule:{sched_key}.part"
                    structure_path = ensure_min_two_level_structure_path(
                        f"{schedule_label} > Part {part_num}".strip(),
                        citation=citation,
                    )

                    text_lines: list[str] = []
                    if include_structure_line:
                        text_lines.append(f"Structure: {structure_path}")
                    text_lines.append(f"{schedule_label} - {sched_heading}".strip(" -"))
                    text_lines.append("")
                    text_lines.append(part_heading)
                    text_lines.append("")
                    text_lines.append(part_body)
                    chunk_text = normalize_text("\n".join(text_lines))

                    chunks.append(
                        {
                            "key": key,
                            "title": f"{schedule_label} {part_heading}".strip(),
                            "main_structural_unit": "part",
                            "main_structural_unit_number": part_num,
                            "main_structural_unit_heading": None,
                            "defined_term": None,
                            "structure_path": structure_path,
                            "schedule_id": sched_id,
                            "schedule_heading": sched_heading or None,
                            "text": chunk_text,
                        }
                    )

                continue

            # Otherwise chunk the schedule body as a whole schedule.
            base_lines: list[str] = []
            structure_path = ensure_min_two_level_structure_path(schedule_label, citation=citation)
            if include_structure_line:
                base_lines.append(f"Structure: {structure_path}")
            base_lines.append(f"{schedule_label} - {sched_heading}".strip(" -"))
            base_lines.append("")

            body_words = (sched.body or "").split()
            if chunk_size > 0 and len(body_words) > chunk_size:
                parts = chunk_words(body_words, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for i, part_words in enumerate(parts, start=1):
                    key = f"schedule:{sched_key}.p{i}"
                    chunk_text = normalize_text("\n".join([*base_lines, " ".join(part_words)]))
                    chunks.append(
                        {
                            "key": key,
                            "title": f"{schedule_label} - {sched_heading}".strip(" -"),
                            "main_structural_unit": "schedule",
                            "main_structural_unit_number": sched_id,
                            "main_structural_unit_heading": sched_heading or None,
                            "defined_term": None,
                            "structure_path": structure_path,
                            "schedule_id": sched_id,
                            "schedule_heading": sched_heading or None,
                            "text": chunk_text,
                        }
                    )
            else:
                key = f"schedule:{sched_key}"
                chunk_text = normalize_text("\n".join([*base_lines, sched.body]))
                chunks.append(
                    {
                        "key": key,
                        "title": f"{schedule_label} - {sched_heading}".strip(" -"),
                        "main_structural_unit": "schedule",
                        "main_structural_unit_number": sched_id,
                        "main_structural_unit_heading": sched_heading or None,
                        "defined_term": None,
                        "structure_path": structure_path,
                        "schedule_id": sched_id,
                        "schedule_heading": sched_heading or None,
                        "text": chunk_text,
                    }
                )
            continue

        # Parse internal schedule provisions (commonly referenced as 'sections' within the schedule).
        for su in sched_units:
            number = su.number
            heading = (su.heading or "").strip()
            header_line = f"Section {number}".strip()
            if heading:
                header_line = f"{header_line} - {heading}".strip()

            segs = structure_key_segments(su.structure_parts)
            key_prefix = f"schedule:{sched_key}" + "".join(f".{seg}" for seg in segs)

            structure_parts = [schedule_label, *su.structure_parts]
            structure_path = ensure_min_two_level_structure_path(
                " > ".join([*structure_parts, f"Section {number}".strip()]),
                citation=citation,
            )

            base_lines: list[str] = []
            if include_structure_line:
                base_lines.append(f"Structure: {structure_path}")
            base_lines.append(f"{schedule_label} - {sched_heading}".strip(" -"))
            base_lines.append("")
            base_lines.append(header_line)
            base_lines.append("")

            body_words = (su.body or "").split()
            if chunk_size > 0 and len(body_words) > chunk_size:
                parts = chunk_words(body_words, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for i, part_words in enumerate(parts, start=1):
                    key = f"{key_prefix}.section:{number}.p{i}"
                    chunk_text = normalize_text("\n".join([*base_lines, " ".join(part_words)]))
                    chunks.append(
                        {
                            "key": key,
                            "title": f"{schedule_label} {header_line}".strip(),
                            "main_structural_unit": "section",
                            "main_structural_unit_number": number,
                            "main_structural_unit_heading": heading or None,
                            "defined_term": None,
                            "structure_path": structure_path,
                            "schedule_id": sched_id,
                            "schedule_heading": sched_heading or None,
                            "text": chunk_text,
                        }
                    )
            else:
                key = f"{key_prefix}.section:{number}"
                chunk_text = normalize_text("\n".join([*base_lines, su.body]))
                chunks.append(
                    {
                        "key": key,
                        "title": f"{schedule_label} {header_line}".strip(),
                        "main_structural_unit": "section",
                        "main_structural_unit_number": number,
                        "main_structural_unit_heading": heading or None,
                        "defined_term": None,
                        "structure_path": structure_path,
                        "schedule_id": sched_id,
                        "schedule_heading": sched_heading or None,
                        "text": chunk_text,
                    }
                )

    return {"doc": doc_meta, "chunks": chunks}


def strip_known_end_matter(text: str, *, jurisdiction: str | None) -> str:
    """Strip non-provision end-matter blocks that should not be chunked.

    This is currently a conservative, WA-focused guard to prevent non-law notes
    (e.g., compilation tables) from being treated as schedules/sections.
    """

    if not text:
        return ""

    if (jurisdiction or "") != "western_australia":
        return text

    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    # Only cut once we have seen substantive body content.
    seen_body = False
    body_mark_re = re.compile(r"^(?:Section|Regulation)\s+\S+\b|^\s*(?:-\s*)?SECT\s+\S+\b|^\s*(?:-\s*)?REG\s+\S+\b",
                             re.IGNORECASE)
    end_matter_re = re.compile(
        r"^(Compilation table|Uncommenced provisions table|Other notes|Defined terms)\b",
        re.IGNORECASE,
    )

    for i, raw in enumerate(lines):
        s = (raw or "").strip()
        if not s:
            continue
        if body_mark_re.match(s) or _PART_RE.match(_clean_heading_line(s)) or _CHAPTER_RE.match(_clean_heading_line(s)):
            seen_body = True
            continue
        if seen_body and end_matter_re.match(s):
            return "\n".join(lines[:i]).strip() + "\n"

    return text


def strip_prepared_header(prepared_text: str) -> str:
    """Remove the metadata header block from a prepared .txt file.

    Prepared files start with a short header like:
      <Citation>
      Jurisdiction: ...
      Source: ...
      Downloaded: ...
      When scraped: ...

    Then a blank line, then the document body.
    """

    lines = (prepared_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return ""

    for i in range(min(len(lines), 80)):
        if (lines[i] or "").strip().casefold().startswith("when scraped"):
            # Consume following blank lines.
            j = i + 1
            while j < len(lines) and not (lines[j] or "").strip():
                j += 1
            return "\n".join(lines[j:]).strip() + "\n"

    # Fallback: if no header marker, return original.
    return (prepared_text or "").strip() + "\n"


def slice_to_body_start(text: str) -> str:
    """Ignore TOC/TOS by slicing to the first real body marker.

    Keeps TOC/TOS in the prepared file, but prevents them being chunked.
    """

    lines = normalize_text(text).split("\n")
    if not lines:
        return ""

    def _looks_like_toc_page_entry(line: str) -> bool:
        # Common TOC shapes:
        #   Section 1. Short title 2
        #   Regulation 3. Something 14
        # The key signal is a trailing page number.
        s = (line or "").strip()
        if not s:
            return False
        if re.search(r"\s\d+\s*$", s) is None:
            return False
        return bool(re.match(r"^(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE))

    def _looks_like_body_marker(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False

        def _is_year_number(num: str) -> bool:
            if not (num or "").isdigit() or len(num) != 4:
                return False
            try:
                y = int(num)
            except ValueError:
                return False
            return 1800 <= y <= 2100
        if re.match(r"^\s*-\s*LONG\s+TITLE\s*$", s, re.IGNORECASE):
            return True
        if re.match(r"^\s*An\s+Act\s+(?:to|for)\b", s, re.IGNORECASE):
            return True
        if re.match(r"^\s*(?:Chapter|Part|Division|Subdivision)\b", s, re.IGNORECASE):
            return True
        if re.match(r"^(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE) and not _looks_like_toc_page_entry(s):
            return True
        # Some consolidations embed the unit label at the end of a repeated title line:
        #   PAY-ROLL TAX ... - Regulation 1
        if re.match(r"^.+\s-\s*(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE):
            return True
        if re.match(r"^\s*(?:-\s*)?SECT\s+\S+\b", s, re.IGNORECASE):
            return True
        if re.match(r"^\s*(?:-\s*)?REG\s+\S+\b", s, re.IGNORECASE):
            return True
        if re.match(r"^\s*-\s*SCHEDULE\s+\S+\b", s, re.IGNORECASE):
            return True

        # Some exports use asterisk-wrapped headings like: '*** 1 Short title ***'.
        if _ASTERISK_UNIT_START_RE.match(s):
            return True

        # Some regulations use bare numbered headings like '1--Short title' or '1 Short title'.
        m_dash = _SECTION_HEADING_DASH_RE.match(s)
        m_col = _SECTION_HEADING_COL_RE.match(s)
        m_dot = _SECTION_HEADING_DOT_RE.match(s)
        m_space = _SECTION_HEADING_SPACE_RE.match(s)
        m_num = m_dash or m_col or m_dot or m_space
        if m_num:
            num_raw = (m_num.group("number") or "").strip()
            if _is_year_number(num_raw):
                return False
            return True
        return False

    # If a TOC/TOS marker exists, be stricter about where the real body starts.
    toc_start_idx: int | None = None
    for i in range(min(len(lines), 600)):
        s = (lines[i] or "").strip().casefold()
        if s == "contents" or "table of provisions" in s:
            toc_start_idx = i
            break

    # Many consolidations include an 'ENDNOTES' entry in the TOC/TOS region.
    # When that happens, treat the first ENDNOTES after Contents/TOP as a TOC terminator.
    if toc_start_idx is not None:
        # Prefer a long-title marker when present (common in AustLII + many consolidations).
        for i in range(toc_start_idx + 1, min(toc_start_idx + 8000, len(lines))):
            if re.match(r"^\s*-\s*LONG\s+TITLE\s*$", lines[i] or "", re.IGNORECASE) or re.match(
                r"^\s*An\s+Act\s+(?:to|for)\b", lines[i] or "", re.IGNORECASE
            ):
                return "\n".join(lines[i:]).strip() + "\n"

        def _looks_like_headingish(line: str) -> bool:
            s = (line or "").strip()
            if not s:
                return False
            if re.match(r"^\s*(?:Chapter|Part|Division|Subdivision)\b", s, re.IGNORECASE):
                return True
            if re.match(r"^(?:Section|Regulation)\s+\S+\b", s, re.IGNORECASE):
                return True
            if re.match(r"^\s*(?:-\s*)?SECT\s+\S+\b", s, re.IGNORECASE):
                return True
            if re.match(r"^\s*(?:-\s*)?REG\s+\S+\b", s, re.IGNORECASE):
                return True
            if re.match(r"^\s*-\s*SCHEDULE\s+\S+\b", s, re.IGNORECASE):
                return True

            # Treat bare numbered headings as heading-ish within TOCs.
            if _SECTION_HEADING_DASH_RE.match(s) or _SECTION_HEADING_COL_RE.match(s) or _SECTION_HEADING_DOT_RE.match(s) or _SECTION_HEADING_SPACE_RE.match(s):
                return True
            return False

        def _next_nonempty_lines(start: int, limit: int = 8) -> list[str]:
            out: list[str] = []
            j = start
            while j < len(lines) and len(out) < limit:
                raw = lines[j] or ""
                s = raw.strip()
                if not s:
                    j += 1
                    continue

                # In TOCs, entries often wrap onto indented continuation lines.
                # Those continuation fragments are not prose and should not trigger
                # body slicing.
                if raw.lstrip() != raw and out:
                    prev = out[-1]
                    if _looks_like_headingish(prev) and not _looks_like_headingish(s):
                        j += 1
                        continue

                out.append(s)
                j += 1
            return out

        unit_head_re = re.compile(
            r"^\s*(?P<kind>Section|Regulation)\s+(?P<number>\S+)\b",
            re.IGNORECASE,
        )
        seen_unit_heads: set[str] = set()

        numbered_head_re = re.compile(
            rf"^\s*(?P<number>{_UNIT_NUMBER_RE_SRC})\s*(?:--|\.|\s)\s*(?P<heading>.+\S)\s*$",
            re.IGNORECASE,
        )
        seen_numbered_heads: set[str] = set()

        def _is_year_number(num: str) -> bool:
            if not (num or "").isdigit() or len(num) != 4:
                return False
            try:
                y = int(num)
            except ValueError:
                return False
            return 1800 <= y <= 2100

        def _has_nearby_prose(start: int) -> bool:
            lookahead = _next_nonempty_lines(start + 1, limit=10)
            if not lookahead:
                return False
            # Still a list of headings? Then we're likely still in the TOC.
            if all(_looks_like_headingish(x) for x in lookahead[:5]):
                return False
            return any((not _looks_like_headingish(x)) for x in lookahead[:8])

        # Preferred heuristic when we have a TOC:
        # - Many modern consolidations list provisions in a TOC (e.g. 'Regulation 1. Name')
        #   and then restart the same numbering in the body ('Regulation 1 Name').
        # - Start at the first *repeated* provision heading that is followed soon by prose.
        # - As a secondary signal, allow a "1..." heading with nearby prose (covers docs
        #   where we don't see a repeat due to formatting differences).
        for i in range(toc_start_idx + 1, min(toc_start_idx + 20000, len(lines))):
            line = lines[i] or ""
            # In many TOCs (especially WA columnar exports), Part/Division headings
            # are pervasive and should not be treated as a body-start marker.
            # Prefer actual provision headings (Section/Regulation/SECT/REG) instead.
            if re.match(r"^\s*(?:Chapter|Part|Division|Subdivision)\b", (line or "").strip(), re.IGNORECASE):
                continue
            if not _looks_like_body_marker(line):
                continue

            if (line or "").strip().casefold() == "endnotes":
                continue

            m = unit_head_re.match(line)
            if m:
                kind = (m.group("kind") or "").strip().casefold()
                num = normalize_unit_number((m.group("number") or "").strip())
                token = f"{kind}:{num}" if num else ""

                # Record the first pass through TOC headings.
                if token and token not in seen_unit_heads:
                    seen_unit_heads.add(token)
                    continue

                if token and token in seen_unit_heads and _has_nearby_prose(i):
                    return "\n".join(lines[i:]).strip() + "\n"

                if num.startswith("1") and _has_nearby_prose(i):
                    return "\n".join(lines[i:]).strip() + "\n"
                continue

            m_num = numbered_head_re.match((line or "").strip())
            if m_num:
                num_raw = (m_num.group("number") or "").strip()
                if _is_year_number(num_raw):
                    continue
                num = normalize_unit_number(num_raw)
                if not num:
                    continue

                if num not in seen_numbered_heads:
                    seen_numbered_heads.add(num)
                    continue

                if num in seen_numbered_heads and _has_nearby_prose(i):
                    return "\n".join(lines[i:]).strip() + "\n"

                if num.startswith("1") and _has_nearby_prose(i):
                    return "\n".join(lines[i:]).strip() + "\n"
                continue

            if _has_nearby_prose(i):
                return "\n".join(lines[i:]).strip() + "\n"

        # Final fallback: if the heuristic fails, start at the first plausible marker.
        for i in range(toc_start_idx + 1, min(toc_start_idx + 20000, len(lines))):
            if _looks_like_body_marker(lines[i] or "") and (lines[i] or "").strip().casefold() != "endnotes":
                return "\n".join(lines[i:]).strip() + "\n"

    # Prefer these markers when present.
    long_title_hdr = re.compile(r"^\s*-\s*LONG\s+TITLE\s*$", re.IGNORECASE)
    section_hdr = re.compile(r"^(?:Section|Regulation)\s+\S+", re.IGNORECASE)
    sect_hdr = re.compile(r"^\s*(?:-\s*)?SECT\s+\S+", re.IGNORECASE)
    reg_hdr = re.compile(r"^\s*(?:-\s*)?REG\s+\S+", re.IGNORECASE)
    # Only treat explicit AustLII-style schedule markers as body-start triggers.
    schedule_hdr = re.compile(r"^\s*-\s*SCHEDULE\s+\S+", re.IGNORECASE)

    for i in range(min(len(lines), 8000)):
        line = lines[i] or ""
        if section_hdr.match(line) and _looks_like_toc_page_entry(line):
            continue
        if long_title_hdr.match(line) or section_hdr.match(line) or sect_hdr.match(line) or reg_hdr.match(line) or schedule_hdr.match(line):
            return "\n".join(lines[i:]).strip() + "\n"

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-document JSON sidecars with stable chunk keys and TOC-derived structure.")
    ap.add_argument("--prepared-dir", type=Path, default=Path(r"D:\Australian-Taxation-Law-Library\prepared\data\Legislative_materials"))
    ap.add_argument("--include-glob", type=str, default="")
    ap.add_argument("--chunk-size", type=int, default=768)
    ap.add_argument("--chunk-overlap", type=int, default=96)
    # Default is now to omit the redundant 'Structure: ...' line in chunk text.
    # Keep an opt-in flag for debugging / retrieval experiments.
    ap.add_argument(
        "--include-structure-line",
        action="store_true",
        help="Prepend 'Structure: ...' to chunk text (generally not needed if structure_path is present)",
    )
    # Back-compat: older scripts may pass this; it now has no effect beyond ensuring the default.
    ap.add_argument(
        "--no-structure-line",
        action="store_true",
        help="(Deprecated) Do not prepend 'Structure: ...' to chunk text",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    prepared_dir: Path = args.prepared_dir

    if not prepared_dir.exists():
        raise SystemExit(f"prepared-dir not found: {prepared_dir}")

    include_glob = (args.include_glob or "").strip()
    include_structure_line = bool(args.include_structure_line)

    count = 0
    for prepared_path in iter_txt_files(prepared_dir):
        if include_glob and not prepared_path.match(include_glob):
            continue

        prepared_text = prepared_path.read_text(encoding="utf-8", errors="replace")

        sidecar = build_sidecar_for_doc(
            prepared_text=prepared_text,
            citation=citation_from_path(prepared_path),
            source=prepared_path.relative_to(prepared_dir).as_posix(),
            filename=prepared_path.with_suffix(".json").name,
            chunk_size=int(args.chunk_size),
            chunk_overlap=int(args.chunk_overlap),
            include_structure_line=include_structure_line,
        )

        out_path = prepared_path.with_suffix(".json")
        out_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
        print(f"Wrote {out_path.name} ({len(sidecar.get('chunks') or [])} chunks)")

        if args.limit and count >= int(args.limit):
            break

    print(f"Done. Wrote {count} JSON sidecars.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
