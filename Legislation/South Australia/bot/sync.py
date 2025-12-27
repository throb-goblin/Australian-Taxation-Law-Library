from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from pdfminer.high_level import extract_text as pdf_extract_text

try:
    from inscriptis import get_text as inscriptis_get_text
except Exception:
    inscriptis_get_text = None

try:
    from striprtf.striprtf import rtf_to_text as striprtf_to_text
except Exception:
    striprtf_to_text = None


BLOCKED_URL_SUBSTRINGS = [
    "/view/html/compare/",
]


@dataclass(frozen=True)
class Config:
    user_agent: str
    sleep_seconds: float
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def is_truthy_active(value: str) -> bool:
    v = (value or "").strip().lower()
    if v in {"", "1", "true", "t", "yes", "y", "in force", "active"}:
        return True
    if v in {"0", "false", "f", "no", "n", "repealed", "not in force"}:
        return False
    return True


def truncate_error(message: str, limit: int = 500) -> str:
    msg = (message or "").strip()
    if len(msg) <= limit:
        return msg
    return msg[: limit - 3] + "..."


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding=encoding,
        delete=False,
        dir=str(path.parent),
        prefix=path.name,
        suffix=".tmp",
        newline="\n",
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def atomic_write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=path.name,
        suffix=".tmp",
        newline="",
    ) as tmp:
        writer = csv.writer(tmp, lineterminator="\n")
        writer.writerows(rows)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def normalize_header_cell(cell: str) -> str:
    c = (cell or "").strip()
    c = c.replace("\ufeff", "")
    c = c.lower()
    c = re.sub(r"\s+", "_", c)
    return c


def is_valid_header_row(row: List[str]) -> bool:
    lowered = {normalize_header_cell(c) for c in row}
    return "library_id" in lowered and "url" in lowered


def dedupe_header(header: List[str]) -> Tuple[List[str], Dict[str, int]]:
    cleaned: List[str] = []
    index_by_name: Dict[str, int] = {}
    for i, raw in enumerate(header):
        name = normalize_header_cell(raw)
        if not name:
            continue
        if name in index_by_name:
            continue
        index_by_name[name] = i
        cleaned.append(name)
    return cleaned, index_by_name


class Catalogue:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.metadata_row: Optional[List[str]] = None
        self.header: List[str] = []
        self._index_by_name: Dict[str, int] = {}
        self.rows: List[Dict[str, str]] = []

    def load(self) -> None:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            raw_rows = list(reader)

        if not raw_rows:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        first = raw_rows[0]
        start_idx = 0
        if not is_valid_header_row(first):
            self.metadata_row = first
            start_idx = 1
        if len(raw_rows) <= start_idx:
            raise ValueError(f"CSV missing header row: {self.csv_path}")

        raw_header = raw_rows[start_idx]
        self.header, self._index_by_name = dedupe_header(raw_header)

        self.rows = []
        for raw in raw_rows[start_idx + 1 :]:
            row: Dict[str, str] = {}
            for name, idx in self._index_by_name.items():
                row[name] = raw[idx].strip() if idx < len(raw) else ""
            self.rows.append(row)

        self.ensure_columns(
            [
                "when_scraped",
                "last_successful_scrape",
                "status",
                "error",
                "version_id",
                "content_format",
                "content_url",
            ]
        )

    def ensure_columns(self, needed: List[str]) -> None:
        for col in needed:
            if col not in self.header:
                self.header.append(col)
                self._index_by_name[col] = -1
                for r in self.rows:
                    r[col] = ""

    def save(self) -> None:
        out_rows: List[List[str]] = []
        if self.metadata_row is not None:
            out_rows.append(self.metadata_row)
        out_rows.append(self.header)
        for r in self.rows:
            out_rows.append([r.get(col, "") for col in self.header])
        atomic_write_csv(self.csv_path, out_rows)


class HttpClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self._last_request_at: Optional[float] = None

    def _sleep_if_needed(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.time() - self._last_request_at
        if elapsed < self.config.sleep_seconds:
            time.sleep(self.config.sleep_seconds - elapsed)

    def get(self, url: str, *, allow_redirects: bool = True, headers: Optional[Dict[str, str]] = None) -> requests.Response:
        for blocked in BLOCKED_URL_SUBSTRINGS:
            if blocked in url:
                raise ValueError(f"Blocked URL by policy: contains {blocked}")

        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.config.max_retries + 2):
            self._sleep_if_needed()
            self._last_request_at = time.time()

            try:
                resp = self.session.get(
                    url,
                    timeout=self.config.timeout_seconds,
                    allow_redirects=allow_redirects,
                    headers=headers,
                )
                if resp.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt >= self.config.max_retries + 1:
                    break
                backoff = self.config.backoff_base_seconds * (2 ** (attempt - 1))
                time.sleep(backoff)
        raise RuntimeError(f"Request failed after retries: {url}: {last_exc}")


def html_to_text_preserve_blocks(html: str) -> str:
    if inscriptis_get_text is not None:
        try:
            out = inscriptis_get_text(html or "")
            out = (out or "").replace("\r\n", "\n").replace("\r", "\n")
            out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
            return out
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")
    body = soup.body or soup

    block_tags = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "pre",
        "blockquote",
        "table",
        "tr",
        "div",
    }

    nested_block_ancestors = {
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "p",
        "li",
        "pre",
        "blockquote",
        "tr",
    }

    pieces: List[str] = []
    for el in body.find_all(list(block_tags)):
        if el.name == "div" and not el.get_text(strip=True):
            continue

        if any(getattr(p, "name", None) in nested_block_ancestors for p in el.parents if p is not None):
            if not (el.name or "").startswith("h"):
                continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if (el.name or "").startswith("h"):
            pieces.append(text)
            pieces.append("")
        else:
            pieces.append(text)

    out = "\n".join(pieces)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    text = pdf_extract_text(io.BytesIO(pdf_bytes))
    return (text or "").strip() + "\n"


def rtf_bytes_to_text(rtf_bytes: bytes) -> str:
    if striprtf_to_text is None:
        raise RuntimeError(
            "striprtf is not installed but is required to parse RTF. "
            "Install with: python -m pip install striprtf"
        )

    last_exc: Optional[BaseException] = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            rtf_str = rtf_bytes.decode(enc)
            text = striprtf_to_text(rtf_str)
            out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
            out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
            return out
        except Exception as exc:
            last_exc = exc
            continue

    raise RuntimeError(f"Failed to decode/parse RTF: {last_exc}")


def stable_hash_version_id(content_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(content_bytes)
    return "sha256:" + h.hexdigest()


def safe_txt_filename(library_id: str) -> str:
    name = (library_id or "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    return f"{name}.txt"


JURIS_ABBREV = "SA"


_SECTION_HEADING_RE = re.compile(r"^(\d+[A-Za-z]*)\.?\s+(.+)$")


def _infer_source(version_id: str) -> Optional[str]:
    v = (version_id or "").strip()
    if not v:
        return None
    return v.split(":", 1)[0].strip() if ":" in v else None


def _header_block_for_row(row: Dict[str, str]) -> str:
    def get(k: str) -> str:
        return (row.get(k) or "").strip()

    citation = get("citation") or get("title")
    version_id = get("version_id")
    source = _infer_source(version_id)

    lines: List[str] = [
        f"library_id: {get('library_id')}",
        f"citation: {citation}",
        f"type: {get('type')}",
        f"version_id: {version_id}",
        f"jurisdiction: {get('jurisdiction')}",
        f"url: {get('url')}",
        f"content_format: {get('content_format')}",
    ]
    if source:
        lines.append(f"source: {source}")
    return "\n".join(lines).strip()


def _strip_leading_title(text: str, title: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    first = lines[0].strip()
    if title and first.lower() == title.strip().lower():
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    return "\n".join(lines)


def _strip_front_matter_and_contents(text: str) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in raw.split("\n")]
    if len(lines) < 30:
        return raw

    scan_limit = min(len(lines), 800)

    # SA consolidated RTF often has a plain "Contents" list with no page numbers,
    # followed later by the enacting formula. Prefer cutting to the enacting
    # formula when both are found near the start.
    contents_idx_simple: Optional[int] = None
    enacts_idx: Optional[int] = None
    for i in range(min(len(lines), 2000)):
        s = (lines[i] or "").strip().lower()
        if contents_idx_simple is None and s == "contents":
            contents_idx_simple = i
        if enacts_idx is None and s == "the parliament of south australia enacts as follows:":
            enacts_idx = i
        if contents_idx_simple is not None and enacts_idx is not None:
            break

    if (
        contents_idx_simple is not None
        and enacts_idx is not None
        and 0 <= contents_idx_simple < enacts_idx
        and contents_idx_simple <= 500
        and (enacts_idx - contents_idx_simple) >= 5
    ):
        out_lines = lines[:contents_idx_simple] + lines[enacts_idx:]
        return "\n".join(out_lines).strip() + "\n"

    # Secondary legislation often has an unlabeled table-of-provisions block made up
    # of numbered headings (no em dash), then "Legislative history", then the real
    # body which repeats the headings with an em dash (e.g. "1—Short title").
    # Example: Land Tax Regulations 2025.
    leg_hist_idx: Optional[int] = None
    for i in range(min(len(lines), 2000)):
        if (lines[i] or "").strip().lower() == "legislative history":
            leg_hist_idx = i
            break

    if leg_hist_idx is not None and leg_hist_idx <= 1500:
        # If the enacting formula exists, this is likely primary legislation; don't
        # apply this secondary-legislation rule.
        has_enacts = any(
            (ln or "").strip().lower().startswith("the parliament of south australia enacts as follows")
            for ln in lines[: min(len(lines), 2000)]
        )
        if not has_enacts:
            post_hist_start: Optional[int] = None
            for j in range(leg_hist_idx + 1, len(lines)):
                t = (lines[j] or "").strip()
                if re.match(r"^\d+\s*[\u2014\u2013\-]\s*\S", t):
                    post_hist_start = j
                    break

            def is_toc_heading(line: str) -> bool:
                t = (line or "").rstrip("\t ")
                # TOC headings are typically "1       Short title" (spaces/tabs), not "1—Short title".
                if re.match(r"^\d+\s*[\u2014\u2013\-]", t):
                    return False
                return re.match(r"^\d+\s{2,}\S", t) is not None or re.match(r"^\d+\t+\S", t) is not None

            toc_start: Optional[int] = None
            if post_hist_start is not None:
                # Find the start of a run of TOC headings before Legislative history.
                for i in range(0, leg_hist_idx):
                    if not is_toc_heading(lines[i]):
                        continue
                    # Require at least 3 TOC-looking headings within the next 40 lines.
                    hits = 0
                    for k in range(i, min(leg_hist_idx, i + 40)):
                        if is_toc_heading(lines[k]):
                            hits += 1
                    if hits >= 3:
                        toc_start = i
                        break

            if toc_start is not None and post_hist_start is not None and toc_start < post_hist_start:
                out_lines = lines[:toc_start] + lines[post_hist_start:]
                # Drop an occasional stray "Contents" heading that precedes the real body.
                for c_i in range(min(len(out_lines), 120)):
                    if (out_lines[c_i] or "").strip().lower() != "contents":
                        continue
                    c_j = c_i + 1
                    while c_j < len(out_lines) and not (out_lines[c_j] or "").strip():
                        c_j += 1
                    if c_j < len(out_lines) and re.match(r"^\d+\s*[\u2014\u2013\-]\s*\S", (out_lines[c_j] or "").strip()):
                        out_lines = out_lines[:c_i] + out_lines[c_i + 1 :]
                    break
                return "\n".join(out_lines).strip() + "\n"

    def looks_like_contents_heading(i: int) -> bool:
        if i < 0 or i >= len(lines):
            return False
        s = (lines[i] or "").strip().lower()
        if s not in {"contents", "table of provisions", "table of contents"}:
            return False
        window = [(lines[j] or "").strip().lower() for j in range(i, min(i + 12, len(lines)))]
        blob = " ".join(window)
        if "page" in blob:
            return True
        if any(re.search(r"\s\d{1,4}$", w) for w in window if w):
            return True
        return False

    def looks_like_contents_entry(line: str) -> bool:
        t = (line or "").strip()
        if not t:
            return False
        if not re.search(r"\s\d{1,4}$", t):
            return False
        if re.match(r"^(chapter|part|division|subdivision|schedule)\s+", t, flags=re.IGNORECASE):
            return True
        if re.match(r"^\d{1,4}[A-Za-z]{0,4}\.?\s+", t):
            return True
        return False

    def is_long_title_line(line: str) -> bool:
        t = (line or "").strip()
        if not t:
            return False
        if re.match(r"^(An\s+Act|A\s+(?:Regulation|Rule|Rules|By-law|Bylaws|Bylaw))\b", t, flags=re.IGNORECASE):
            return True
        if re.match(r"^(Be\s+it\s+enacted|The\s+Parliament\s+.*\s+enacts)\b", t, flags=re.IGNORECASE):
            return True
        return False

    def is_body_start_heading(line: str) -> bool:
        t = (line or "").strip()
        if not t or looks_like_contents_entry(t):
            return False
        if re.match(r"^Chapter\s+\d+\b", t, flags=re.IGNORECASE):
            return True
        if re.match(r"^Part\s+\d+(?:\.|\b)", t, flags=re.IGNORECASE):
            return True
        if re.match(r"^(\d{1,4}[A-Za-z]{0,4})\.?\s+\S", t):
            return True
        return False

    contents_idx: Optional[int] = None
    for i in range(scan_limit):
        if looks_like_contents_heading(i):
            contents_idx = i
            break
    if contents_idx is None or contents_idx > 500:
        return raw

    end_idx: Optional[int] = None
    for j in range(contents_idx + 1, len(lines)):
        if is_long_title_line(lines[j]):
            end_idx = j
            break
    if end_idx is None:
        for j in range(contents_idx + 1, len(lines)):
            if is_body_start_heading(lines[j]):
                end_idx = j
                break

    if end_idx is None:
        return raw

    out_lines = lines[:contents_idx] + lines[end_idx:]
    return "\n".join(out_lines).strip() + "\n"


def _fix_leading_label_space(line: str) -> str:
    s = (line or "").rstrip()
    m = re.match(r"^(\d+[A-Za-z]{0,6})([A-Z])", s)
    if not m:
        return s
    label = m.group(1)
    if label.isdigit() and len(label) == 4:
        return s
    return f"{label} {s[len(label):]}"


def _hard_wrap_repair(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\u00ad", "")
    out = re.sub(r"([A-Za-z])\-\n([A-Za-z])", r"\1\2", out)
    # Fix common extraction artifact where a word is broken across lines with indentation.
    # Example: "a\n dministration" -> "administration".
    out = re.sub(r"\b([A-Za-z])\n[ \t]+([a-z]{6,})", r"\1\2", out)
    # If a line wraps between words, preserve a space (do NOT merge words).
    out = re.sub(r"([A-Za-z])\n[ \t]+([a-z]{2,})", r"\1 \2", out)
    return out


def _normalize_body_text(body: str) -> str:
    raw = _hard_wrap_repair((body or "").replace("\r\n", "\n").replace("\r", "\n"))
    in_lines = [ln.rstrip() for ln in raw.split("\n")]

    def is_major_heading(line: str) -> bool:
        l = (line or "").strip().lower()
        return l.startswith(("chapter ", "part ", "division ", "subdivision ", "schedule "))

    def is_enacting_formula(line: str) -> bool:
        l = (line or "").strip().lower()
        return l == "the parliament of south australia enacts as follows:" or l.startswith(
            "the parliament of south australia enacts as follows"
        )

    def is_section_heading(line: str) -> bool:
        s = (line or "").strip()
        m = re.match(
            r"^(\d{1,4})(?:\s*[A-Za-z]{1,4})?(?:\s*[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\-]|\.|\s)\s*\S",
            s,
        )
        if not m:
            return False
        digits = m.group(1)
        if len(digits) == 4:
            try:
                year = int(digits)
                if 1900 <= year <= 2099:
                    return False
            except Exception:
                return False
        return True

    def is_subsection_line(line: str) -> bool:
        return re.match(r"^\(\d+\)\s+\S", (line or "").strip()) is not None

    def is_marker(line: str) -> bool:
        s = (line or "").strip()
        if re.fullmatch(r"\(\d+\)", s):
            return True
        if re.fullmatch(r"\([a-z]\)", s):
            return True
        if re.fullmatch(r"\([ivxlcdm]+\)", s, flags=re.IGNORECASE):
            return True
        return False

    out_lines: List[str] = []
    i = 0
    while i < len(in_lines):
        s = (in_lines[i] or "").strip()
        if not s:
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            nxt = (in_lines[j] or "").strip() if j < len(in_lines) else ""

            # Avoid blank lines directly before subsection/marker lines.
            if nxt and (is_subsection_line(nxt) or is_marker(nxt)):
                i += 1
                continue

            # Avoid blank lines between a section heading and its first subsection.
            if out_lines and out_lines[-1] and is_section_heading(out_lines[-1]):
                i += 1
                continue

            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            i += 1
            continue

        if re.fullmatch(r"\d+[A-Za-z]{0,4}", s):
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            if j < len(in_lines):
                nxt = (in_lines[j] or "").strip()
                if nxt and not is_major_heading(nxt) and not is_marker(nxt):
                    out_lines.append(_fix_leading_label_space(f"{s} {nxt}"))
                    i = j + 1
                    continue

        if is_marker(s):
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            if j < len(in_lines):
                nxt = (in_lines[j] or "").strip()
                if is_marker(nxt):
                    out_lines.append(s)
                    i += 1
                    continue
                if nxt and not is_major_heading(nxt):
                    out_lines.append(f"{s} {nxt}")
                    i = j + 1
                    continue

        s = re.sub(r"^\((\d+)\)(\S)", r"(\1) \2", s)
        s = re.sub(r"^\(([a-z])\)(\S)", r"(\1) \2", s)
        s = re.sub(r"^\(([ivxlcdm]+)\)(\S)", r"(\1) \2", s, flags=re.IGNORECASE)
        s = re.sub(r"^(\(\d+\)|\([a-z]\)|\([ivxlcdm]+\))\s{2,}", r"\1 ", s, flags=re.IGNORECASE)
        s = _fix_leading_label_space(s)

        if is_enacting_formula(s):
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append(s)
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            i += 1
            continue

        if is_major_heading(s):
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append(s)
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            i += 1
            continue

        if is_section_heading(s) and out_lines and out_lines[-1] != "":
            out_lines.append("")

        # Keep subsections contiguous (no extra blank lines inserted).

        out_lines.append(s)
        i += 1

    while out_lines and out_lines[0] == "":
        out_lines.pop(0)
    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    return "\n".join(out_lines).strip() + "\n"


def finalize_output_text(row: Dict[str, str], text: str) -> str:
    title = (row.get("citation") or row.get("title") or "").strip()
    body = _strip_leading_title(text, title)
    body = _strip_front_matter_and_contents(body)
    body = _normalize_body_text(body)
    header = _header_block_for_row(row)
    return f"{header}\n\n{body.strip()}\n"


def find_catalogue_csv(data_dir: Path) -> Path:
    matches = sorted(data_dir.glob("*_legislation_catalogue.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_legislation_catalogue.csv found in {data_dir}")
    return matches[0]


def extract_date_like_version_from_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    patterns = [
        r"Last\s+updated\s*[:\-]\s*([^\.|\n]+)",
        r"Updated\s+on\s*[:\-]?\s*([^\.|\n]+)",
        r"Current\s+as\s+at\s*[:\-]?\s*([^\.|\n]+)",
        r"As\s+at\s*[:\-]\s*([^\.|\n]+)",
        r"Version\s+as\s+at\s*[:\-]\s*([^\.|\n]+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                dt = dateparser.parse(m.group(1), dayfirst=True)
                if dt:
                    return dt.date().isoformat()
            except Exception:
                pass
    return None


def _extract_sa_latest_point_in_time_from_html(html: str) -> Optional[str]:
    """South Australia exposes point-in-time versions via URLs containing YYYY.MM.DD.

    Example observed:
      /__legislation/lz/c/a/<title>/2015.06.30/1986.78.auth.pdf
    """

    dates: List[str] = []
    for m in re.finditer(r"/__legislation/lz/[^\"']+/(20\d{2}\.\d{2}\.\d{2})(?:_[^/\"']+)?/", html, flags=re.IGNORECASE):
        dates.append(m.group(1))

    if not dates:
        return None

    def key(d: str) -> str:
        return d.replace(".", "")

    latest = sorted(set(dates), key=key)[-1]
    return latest.replace(".", "-")


def _is_sa_act_url(url: str) -> bool:
    u = (url or "").lower()
    return (
        "/c/a/" in u
        or "path=/c/a/" in u
        or "path=%2fc%2fa%2f" in u
    )


def _conditional_headers_for_version_id(version_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not version_id:
        return None
    v = version_id.strip()
    if not v:
        return None

    # Allow format prefixes like fmt:v2|etag:... or fmt:v2|last_modified:...
    if v.startswith("fmt:") and "|" in v:
        v = v.split("|", 1)[1].strip()

    if v.startswith("etag:"):
        etag = v[len("etag:") :].strip()
        return {"If-None-Match": etag} if etag else None

    if v.startswith("last_modified:"):
        ims = v[len("last_modified:") :].strip()
        return {"If-Modified-Since": ims} if ims else None

    return None


def decorate_version_id(version_id: str) -> str:
    v = (version_id or "").strip()
    if not v:
        return v
    if v.startswith("fmt:v2|"):
        return v
    return f"fmt:v2|{v}"


def looks_like_xml(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ctype:
        return True
    content = resp.content.lstrip()[:50]
    return content.startswith(b"<?xml")


def generic_xml_to_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)

    def local(tag: str) -> str:
        return tag.split("}", 1)[1] if "}" in tag else tag

    heading_tags = {"title", "heading", "head", "part", "chapter", "section"}
    out: List[str] = []
    for el in root.iter():
        name = local(el.tag).lower()
        txt = " ".join(" ".join(el.itertext()).split())
        if not txt:
            continue
        if name in heading_tags:
            out.append(txt)
            out.append("")
        else:
            out.append(txt)
    text_out = "\n".join(out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip() + "\n"
    return text_out


def try_find_download_link(html: str, base_url: str, kind: str) -> Optional[str]:
    kind = kind.lower()
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        if kind == "rtf":
            if abs_url.lower().endswith(".rtf") or "rtf" in href.lower():
                return abs_url
        if kind == "xml":
            if abs_url.lower().endswith(".xml") or "xml" in href.lower():
                return abs_url
        if kind == "pdf":
            if abs_url.lower().endswith(".pdf") or "pdf" in href.lower():
                return abs_url
    return None


def guess_version_id_and_fetch(
    client: HttpClient,
    url: str,
    *,
    current_version_id: Optional[str] = None,
) -> Tuple[str, requests.Response]:
    # If the stored version_id isn't format-decorated yet, force a full fetch so we
    # can regenerate the output with the new extraction pipeline.
    use_conditional = bool((current_version_id or "").strip()) and (current_version_id or "").strip().startswith("fmt:")
    resp = client.get(url, headers=_conditional_headers_for_version_id(current_version_id) if use_conditional else None)
    if resp.status_code == 304 and current_version_id and use_conditional:
        return current_version_id, resp
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "html" in ctype:
        if _is_sa_act_url(url):
            pit = _extract_sa_latest_point_in_time_from_html(resp.text)
            if pit:
                return decorate_version_id(pit), resp

        page_version = extract_date_like_version_from_html(resp.text)
        if page_version:
            return decorate_version_id(page_version), resp

    etag = (resp.headers.get("ETag") or "").strip()
    if etag:
        return decorate_version_id(f"etag:{etag}"), resp

    last_mod = resp.headers.get("Last-Modified")
    if last_mod:
        return decorate_version_id(f"last_modified:{last_mod}"), resp

    if "html" in ctype:
        normalized = " ".join(html_to_text_preserve_blocks(resp.text).split())
        return decorate_version_id(stable_hash_version_id(normalized.encode("utf-8"))), resp
    return decorate_version_id(stable_hash_version_id(resp.content)), resp


def download_text_best_effort(client: HttpClient, resp: requests.Response) -> Tuple[str, str, str]:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "rtf" in ctype or resp.url.lower().endswith(".rtf"):
        return rtf_bytes_to_text(resp.content), resp.url, "rtf"
    if looks_like_xml(resp):
        return generic_xml_to_text(resp.content), resp.url, "xml"
    if "pdf" in ctype or resp.url.lower().endswith(".pdf"):
        return pdf_bytes_to_text(resp.content), resp.url, "pdf"

    if "html" in ctype:
        rtf_link = try_find_download_link(resp.text, resp.url, "rtf")
        if rtf_link:
            try:
                rtf_resp = client.get(rtf_link)
                rtf_ct = (rtf_resp.headers.get("Content-Type") or "").lower()
                if "rtf" in rtf_ct or rtf_resp.url.lower().endswith(".rtf"):
                    return rtf_bytes_to_text(rtf_resp.content), rtf_resp.url, "rtf"
            except Exception:
                pass

        xml_link = try_find_download_link(resp.text, resp.url, "xml")
        if xml_link:
            try:
                xml_resp = client.get(xml_link)
                if looks_like_xml(xml_resp):
                    return generic_xml_to_text(xml_resp.content), xml_resp.url, "xml"
            except Exception:
                pass

        pdf_link = try_find_download_link(resp.text, resp.url, "pdf")
        if pdf_link:
            try:
                pdf_resp = client.get(pdf_link)
                pdf_ct = (pdf_resp.headers.get("Content-Type") or "").lower()
                if "pdf" in pdf_ct or pdf_resp.url.lower().endswith(".pdf"):
                    return pdf_bytes_to_text(pdf_resp.content), pdf_resp.url, "pdf"
            except Exception:
                pass

        return html_to_text_preserve_blocks(resp.text), resp.url, "html"

    return (resp.text or "").strip() + "\n", resp.url, "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="South Australia legislation sync bot")
    parser.add_argument("--user-agent", default="AustralianTaxLawLibrarySyncBot/SA")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--backoff-base-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0, help="Process only N rows (0 = all)")
    parser.add_argument("--force", action="store_true", help="Force re-download/rewrite even when version_id unchanged")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        user_agent=args.user_agent,
        sleep_seconds=max(0.0, float(args.sleep_seconds)),
        timeout_seconds=max(1.0, float(args.timeout_seconds)),
        max_retries=max(0, int(args.max_retries)),
        backoff_base_seconds=max(0.0, float(args.backoff_base_seconds)),
    )

    data_dir = Path.cwd() / "data"
    csv_path = find_catalogue_csv(data_dir)

    cat = Catalogue(csv_path)
    cat.load()
    client = HttpClient(cfg)

    processed = 0
    for row in cat.rows:
        if args.limit and processed >= args.limit:
            break

        row["when_scraped"] = utc_now_iso()

        if "active" in row and not is_truthy_active(row.get("active", "")):
            row["status"] = "skipped_inactive"
            row["error"] = ""
            cat.save()
            processed += 1
            continue

        library_id = (row.get("library_id") or "").strip()
        url = (row.get("url") or "").strip()
        if not library_id or not url:
            row["status"] = "error"
            row["error"] = "Missing library_id or url"
            cat.save()
            processed += 1
            continue

        try:
            current_version_id = (row.get("version_id") or "").strip()
            fetch_version_id = None if args.force else current_version_id
            latest_version_id, resp = guess_version_id_and_fetch(client, url, current_version_id=fetch_version_id)

            if (not args.force) and current_version_id == latest_version_id:
                row["status"] = "skipped_no_change"
                row["error"] = ""
                cat.save()
                processed += 1
                continue

            text, content_url, content_format = download_text_best_effort(client, resp)
            row["content_url"] = (content_url or url).strip()
            row["content_format"] = (content_format or "html").strip()
            row_for_meta = dict(row)
            row_for_meta["version_id"] = latest_version_id
            text = finalize_output_text(row_for_meta, text)
            out_path = data_dir / safe_txt_filename(library_id)
            atomic_write_text(out_path, text)

            row["version_id"] = latest_version_id
            row["last_successful_scrape"] = utc_now_iso()
            row["status"] = "ok"
            row["error"] = ""
            cat.save()
        except Exception as exc:
            row["status"] = "error"
            row["error"] = truncate_error(str(exc))
            cat.save()

        processed += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
