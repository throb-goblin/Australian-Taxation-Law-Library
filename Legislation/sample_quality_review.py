from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass
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


@dataclass
class FileFindings:
    path: Path
    chars: int
    lines: int
    max_blank_run: int
    page_number_hits: int
    header_footer_hits: int
    website_boilerplate_hits: int
    examples_page: list[str]
    examples_header_footer: list[str]
    examples_web: list[str]


_PAGE_PATTERNS = [
    # Common PDF footer/header page markers
    re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),
]

_HEADER_FOOTER_PATTERNS = [
    # Phrases that are very likely compilation/footer artefacts rather than substantive provisions
    re.compile(r"\bauthori[sz]ed\s+version\b", re.IGNORECASE),
    re.compile(r"\bprepared\s+by\s+the\s+office\s+of\s+parliamentary\s+counsel\b", re.IGNORECASE),
    re.compile(r"\bprepared\s+by\s+parliamentary\s+counsel\b", re.IGNORECASE),
    re.compile(r"\bprinted\s+by\s+authority\b", re.IGNORECASE),
    re.compile(r"\bcompilation\s+no\.?\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bregistered\s+on\s+the\s+federal\s+register\s+of\s+legislation\b", re.IGNORECASE),
    re.compile(r"\bfederal\s+register\s+of\s+legislation\b", re.IGNORECASE),
    re.compile(r"\btable\s+of\s+provisions\b", re.IGNORECASE),
]

_WEBSITE_BOILERPLATE_PATTERNS = [
    # Unmistakable website navigation / chrome
    re.compile(r"\b(skip\s+to\s+main\s+content|skip\s+to\s+content)\b", re.IGNORECASE),
    re.compile(r"\b(toggle\s+navigation|breadcrumb|breadcrumbs)\b", re.IGNORECASE),
    re.compile(r"\b(cookie\s+policy|cookies?)\b", re.IGNORECASE),
    re.compile(r"\bprivacy\s+policy\b", re.IGNORECASE),
    re.compile(r"\bterms\s+of\s+use\b", re.IGNORECASE),
    re.compile(r"\bdisclaimer\b", re.IGNORECASE),
    re.compile(r"\baccessibility\b", re.IGNORECASE),
    re.compile(r"\bjavascript\b", re.IGNORECASE),
    re.compile(r"\bpowered\s+by\b", re.IGNORECASE),
    re.compile(r"\bsite\s+map\b", re.IGNORECASE),
    # URLs/domains in headers/footers are very common in PDF text extraction
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\bwww\.[A-Za-z0-9\-\.]+\b", re.IGNORECASE),
]


def _extract_example_lines(text: str, patterns: list[re.Pattern[str]], limit: int = 5) -> list[str]:
    examples: list[str] = []
    # Prefer line-based examples for readability
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if len(examples) >= limit:
            break
        if not line.strip():
            continue
        for p in patterns:
            if p.search(line):
                # Include a little context if available
                prev_line = lines[i - 1].strip() if i - 1 >= 0 else ""
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                snippet_parts = [s for s in (prev_line, line.strip(), next_line) if s]
                examples.append(" / ".join(snippet_parts)[:300])
                break
    return examples


def _max_consecutive_blank_lines(text: str) -> int:
    max_run = 0
    cur = 0
    for line in text.splitlines():
        if line.strip() == "":
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def _count_hits(patterns: Iterable[re.Pattern[str]], text: str) -> int:
    total = 0
    for p in patterns:
        # Count non-overlapping occurrences. For MULTILINE patterns this is good enough.
        total += len(p.findall(text))
    return total


def analyze_text_file(path: Path) -> FileFindings:
    text = path.read_text(encoding="utf-8", errors="replace")
    max_blank_run = _max_consecutive_blank_lines(text)

    examples_page = _extract_example_lines(text, _PAGE_PATTERNS)
    examples_header_footer = _extract_example_lines(text, _HEADER_FOOTER_PATTERNS)
    examples_web = _extract_example_lines(text, _WEBSITE_BOILERPLATE_PATTERNS)

    return FileFindings(
        path=path,
        chars=len(text),
        lines=text.count("\n") + 1,
        max_blank_run=max_blank_run,
        page_number_hits=_count_hits(_PAGE_PATTERNS, text),
        header_footer_hits=_count_hits(_HEADER_FOOTER_PATTERNS, text),
        website_boilerplate_hits=_count_hits(_WEBSITE_BOILERPLATE_PATTERNS, text),
        examples_page=examples_page,
        examples_header_footer=examples_header_footer,
        examples_web=examples_web,
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    rng = random.Random()
    rng.seed()  # system time

    any_missing = False
    report_path = root / "sample_quality_review_report.txt"
    report_lines: list[str] = []
    header = "SAMPLE QUALITY REVIEW (5 random .txt per jurisdiction)"
    print(header + "\n")
    report_lines.append(header)
    report_lines.append("")

    for j in JURISDICTIONS:
        data_dir = root / j / "data"
        txt_files = sorted(data_dir.glob("*.txt"))
        if not txt_files:
            any_missing = True
            print(f"=== {j} ===")
            print(f"No .txt files found in: {data_dir}")
            print()
            report_lines.append(f"=== {j} ===")
            report_lines.append(f"No .txt files found in: {data_dir}")
            report_lines.append("")
            continue

        sample_n = min(5, len(txt_files))
        sample = rng.sample(txt_files, sample_n)
        findings = [analyze_text_file(p) for p in sample]

        flagged = [
            f
            for f in findings
            if f.max_blank_run > 2
            or f.page_number_hits > 0
            or f.website_boilerplate_hits > 0
            # header/footer patterns can legitimately occur in endnotes; still flag if frequent
            or f.header_footer_hits >= 3
            or f.chars < 2000
        ]

        print(f"=== {j} ===")
        print(f"Sampled {sample_n} of {len(txt_files)} files")
        report_lines.append(f"=== {j} ===")
        report_lines.append(f"Sampled {sample_n} of {len(txt_files)} files")
        for f in findings:
            rel = f.path.relative_to(root)
            flags = []
            if f.chars < 2000:
                flags.append("SHORT(<2000 chars)")
            if f.max_blank_run > 2:
                flags.append(f"BLANK_RUN({f.max_blank_run})")
            if f.page_number_hits > 0:
                flags.append(f"PAGE_HITS({f.page_number_hits})")
            if f.header_footer_hits >= 3:
                flags.append(f"HDR_FTR_HITS({f.header_footer_hits})")
            if f.website_boilerplate_hits > 0:
                flags.append(f"WEB_HITS({f.website_boilerplate_hits})")
            flag_str = (" | " + ", ".join(flags)) if flags else ""
            line = (
                f"- {rel} :: chars={f.chars} lines={f.lines} max_blank_run={f.max_blank_run} "
                f"page_hits={f.page_number_hits} hdrftr_hits={f.header_footer_hits} web_hits={f.website_boilerplate_hits}{flag_str}"
            )
            print(line)
            report_lines.append(line)

            if flags:
                if f.examples_page:
                    report_lines.append("    page examples: " + " | ".join(f.examples_page))
                if f.examples_header_footer:
                    report_lines.append("    header/footer examples: " + " | ".join(f.examples_header_footer))
                if f.examples_web:
                    report_lines.append("    web examples: " + " | ".join(f.examples_web))

        if flagged:
            print(f"Flagged {len(flagged)}/{len(findings)} for closer review.")
            report_lines.append(f"Flagged {len(flagged)}/{len(findings)} for closer review.")
        else:
            print("No red flags found in sample.")
            report_lines.append("No red flags found in sample.")
        print()
        report_lines.append("")

    notes = [
        "NOTES",
        "- This check uses heuristics; it cannot guarantee the text is 100% complete without comparing to the source.",
        "- WEB_HITS focuses on unmistakable website chrome (privacy policy/cookies/JS/URLs).",
        "- HDR_FTR_HITS targets common register/compilation phrases; these may appear in endnotes.",
        f"- Full report written to: {report_path}",
    ]
    print("\n".join(notes))
    report_lines.extend(notes)

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return 2 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
