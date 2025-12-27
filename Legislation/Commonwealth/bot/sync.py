from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
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
            except requests.RequestException as exc:
                last_exc = exc

                # Do not retry on ordinary 4xx (except 429), since these are typically
                # permanent request errors and backing off just slows the run.
                if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
                    status = exc.response.status_code
                    if 400 <= status < 500 and status != 429:
                        break

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
    """Extract text from a PDF with a hard timeout.

    pdfminer can take an extremely long time (or appear hung) on some PDFs.
    To keep runs restartable and avoid wedging a full-jurisdiction scrape,
    extraction is run in a subprocess with a timeout.
    """

    timeout_seconds = 45.0
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        code = (
            "from pdfminer.high_level import extract_text; "
            "import sys; "
            "sys.stdout.reconfigure(encoding='utf-8', errors='replace'); "
            "print((extract_text(sys.argv[1]) or '').strip())"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code, tmp_path],
            capture_output=True,
            text=False,
            timeout=timeout_seconds,
        )
        if proc.returncode != 0:
            stderr_bytes = proc.stderr or b""
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"PDF text extraction failed: {stderr or 'non-zero exit'}")

        stdout_bytes = proc.stdout or b""
        out = stdout_bytes.decode("utf-8", errors="replace")
        out = cleanup_commonwealth_pdf_text(out)
        return out
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"PDF text extraction timed out after {timeout_seconds:.0f}s")
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def looks_like_rtf_bytes(content: bytes) -> bool:
    head = (content or b"").lstrip()[:20]
    return head.startswith(b"{\\rtf") or head.startswith(b"{\\urtf")


def rtf_bytes_to_text(rtf_bytes: bytes) -> str:
    last_exc: Optional[Exception] = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            rtf_str = (rtf_bytes or b"").decode(enc, errors="replace")
            if striprtf_to_text is not None:
                text = striprtf_to_text(rtf_str)
            else:
                text = rtf_str
            out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
            out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
            return out
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"Failed to decode RTF: {last_exc}")


def normalize_plain_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def cleanup_commonwealth_pdf_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    out = out.replace("\u00ad", "")
    # De-hyphenate common line wraps: "exam-\nple" -> "example".
    out = re.sub(r"([A-Za-z])\-\n([a-z])", r"\1\2", out)

    noise_patterns = [
        re.compile(r"^federal\s+register\s+of\s+legislation\s*$", re.IGNORECASE),
        re.compile(r"^authorised\s+version\b.*$", re.IGNORECASE),
        re.compile(r"^compilation\s+no\.?\s*\d+\b.*$", re.IGNORECASE),
        re.compile(r"^register\s+id\b.*$", re.IGNORECASE),
        re.compile(r"^as\s+at\b.*federal\s+register\s+of\s+legislation.*$", re.IGNORECASE),
        re.compile(r"^prepared\s+by\s+the\s+office\s+of\s+parliamentary\s+counsel\b.*$", re.IGNORECASE),
        re.compile(r"^https?://\S*legislation\.gov\.au\S*$", re.IGNORECASE),
        re.compile(r"^page\s*\d+(\s*of\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"\bprivacy\b|\bcookies?\b|\bterms\s+of\s+use\b|\bdisclaimer\b|\bcopyright\b", re.IGNORECASE),
    ]

    lines: List[str] = []
    for raw in out.split("\n"):
        line = (raw or "").strip()
        if not line:
            lines.append("")
            continue

        if re.search(r"https?://", line, flags=re.IGNORECASE) and len(line) < 250:
            continue

        if any(p.search(line) for p in noise_patterns):
            continue
        lines.append(line)

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def stable_hash_version_id(content_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(content_bytes)
    return "sha256:" + h.hexdigest()


def safe_txt_filename(library_id: str) -> str:
    name = (library_id or "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    return f"{name}.txt"


JURIS_ABBREV = "Cth"


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


def _fix_leading_label_space(line: str) -> str:
    s = (line or "").rstrip()
    # Repair missing whitespace between a section label and the start of a title.
    # Example: "3AALevy" -> "3AA Levy".
    # Avoid splitting labels like "3AB" into "3A B".
    m = re.match(r"^(\d+[A-Za-z]{0,6})([A-Z][a-z])", s)
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
    out = re.sub(r"([A-Za-z])\n([a-z])", r"\1\2", out)
    return out


def _normalize_body_text(body: str) -> str:
    raw = _hard_wrap_repair((body or "").replace("\r\n", "\n").replace("\r", "\n"))
    in_lines = [ln.rstrip() for ln in raw.split("\n")]

    def is_major_heading(line: str) -> bool:
        l = (line or "").strip().lower()
        return l.startswith(("chapter ", "part ", "division ", "subdivision ", "schedule "))

    def is_section_heading(line: str) -> bool:
        s = (line or "").strip()
        m = re.match(r"^(\d{1,4}[A-Za-z]{0,4})\.?\s+\S", s)
        if not m:
            return False
        label = m.group(1)
        digits_m = re.match(r"^(\d+)", label)
        digits = digits_m.group(1) if digits_m else ""
        if label == digits and len(digits) == 4:
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
            # Drop blank lines immediately before subsection/marker lines, and drop
            # blank lines immediately after a section heading.
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            nxt = (in_lines[j] or "").strip() if j < len(in_lines) else ""
            if nxt and (is_subsection_line(nxt) or is_marker(nxt)):
                i += 1
                continue
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
    body = _normalize_body_text(body)
    header = _header_block_for_row(row)
    return f"{header}\n\n{body.strip()}\n"


def find_catalogue_csv(data_dir: Path) -> Path:
    matches = sorted(data_dir.glob("*_legislation_catalogue.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_legislation_catalogue.csv found in {data_dir}")
    return matches[0]


def try_frl_versions_api(client: HttpClient, title_id: str) -> Optional[str]:
    """Best-effort API check.

    This bot prefers an API endpoint when available, but will gracefully fall back
    to URL-based scraping if the API is unreachable or the response shape differs.
    """
    title_id = (title_id or "").strip()
    if not title_id:
        return None

    # Federal Register of Legislation public API base.
    # This is an OData endpoint; entity set names and property names are case-sensitive.
    url = "https://api.prod.legislation.gov.au/v1/versions"
    params = {
        "$filter": f"titleId eq '{title_id}'",
        "$top": "1",
        "$orderby": "start desc",
        "$select": "titleId,start,end",
    }

    # Keep this extremely defensive: any network/DNS/API shape issue should degrade
    # to HTML scraping rather than crashing the whole bot.
    try:
        # Use a manual querystring to avoid requests percent-encoding surprises in logs.
        qs = "&".join([f"{k}={requests.utils.quote(v)}" for k, v in params.items()])
        resp = client.get(f"{url}?{qs}")
    except Exception:
        return None


def try_frl_latest_primary_pdf_document(client: HttpClient, title_id: str) -> Optional[Tuple[str, bytes, str]]:
    """Download the latest authorised Primary PDF for a titleId.

    The FRL website is SPA-rendered, so HTML scraping often returns only site chrome.
    The public OData API exposes document entities which can be requested directly and
    returns raw document bytes for PDF/Word formats.
    """

    title_id = (title_id or "").strip()
    if not title_id:
        return None

    docs_url = "https://api.prod.legislation.gov.au/v1/Documents"

    # Prefer authorised Primary PDFs when available, but fall back to non-authorised
    # Primary PDFs (some titles only expose isAuthorised=false via the API).
    filter_candidates = [
        f"titleId eq '{title_id}' and type eq 'Primary' and format eq 'Pdf' and isAuthorised eq true",
        f"titleId eq '{title_id}' and type eq 'Primary' and format eq 'Pdf'",
    ]

    try:
        doc: Optional[dict] = None
        for flt in filter_candidates:
            params = {
                "$top": "1",
                "$orderby": "start desc",
                "$filter": flt,
                "$select": "titleId,start,retrospectiveStart,rectificationVersionNumber,type,uniqueTypeNumber,volumeNumber,format,registerId,compilationNumber,rectified",
            }
            qs = "&".join([f"{k}={requests.utils.quote(v)}" for k, v in params.items()])
            resp = client.get(f"{docs_url}?{qs}")
            data = resp.json() if resp is not None else None
            items = data.get("value") if isinstance(data, dict) else None
            if not items or not isinstance(items, list):
                continue
            first = items[0] if items else None
            if isinstance(first, dict):
                doc = first
                break
        if doc is None:
            return None

        required = [
            "titleId",
            "start",
            "retrospectiveStart",
            "rectificationVersionNumber",
            "type",
            "uniqueTypeNumber",
            "volumeNumber",
            "format",
        ]
        if any(doc.get(k) is None for k in required):
            return None

        # Important: The Document entity is addressable via a composite key, and requesting
        # the entity URL returns the raw document bytes (PDF).
        key_url = (
            "https://api.prod.legislation.gov.au/v1/Documents("
            f"titleId='{doc['titleId']}',"
            f"start={doc['start']},"
            f"retrospectiveStart={doc['retrospectiveStart']},"
            f"rectificationVersionNumber={int(doc['rectificationVersionNumber'])},"
            f"type='{doc['type']}',"
            f"uniqueTypeNumber={int(doc['uniqueTypeNumber'])},"
            f"volumeNumber={int(doc['volumeNumber'])},"
            f"format='{doc['format']}'"
            ")"
        )

        pdf_resp = client.get(key_url)
        if not pdf_resp.content:
            return None

        register_id = (doc.get("registerId") or "").strip()
        compilation_no = (doc.get("compilationNumber") or "").strip()
        rectified = (doc.get("rectified") or "").strip()

        version_parts = [
            f"titleId:{title_id}",
            f"start:{doc['start']}",
        ]
        if register_id:
            version_parts.append(f"registerId:{register_id}")
        if compilation_no:
            version_parts.append(f"compilation:{compilation_no}")
        if rectified:
            version_parts.append(f"rectified:{rectified}")
        version_id = "frl_pdf|" + "|".join(version_parts)
        return version_id, pdf_resp.content, key_url
    except Exception:
        return None

    try:
        data = resp.json()
        items = data.get("value") if isinstance(data, dict) else None
        if not items or not isinstance(items, list):
            return None

        item = items[0] if items else None
        if not isinstance(item, dict):
            return None

        start = item.get("start")
        end = item.get("end")
        if start is None:
            return None

        parts: List[str] = [title_id, f"start={start}"]
        if end is not None:
            parts.append(f"end={end}")
        return "|".join(parts)
    except Exception:
        return None


def extract_last_updated_date_from_commonwealth_html(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # Try a few common labels
    patterns = [
        r"Registered\s*[:\-]\s*([^\.|\n]+)",
        r"Compilation\s+date\s*[:\-]\s*([^\.|\n]+)",
        r"Date\s+of\s+assent\s*[:\-]\s*([^\.|\n]+)",
        r"Last\s+updated\s*[:\-]\s*([^\.|\n]+)",
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


def guess_commonwealth_version_id(client: HttpClient, url: str, title_id: str) -> Tuple[str, requests.Response]:
    api_version = try_frl_versions_api(client, title_id)
    if api_version:
        # Still fetch the HTML content for text extraction when required.
        resp = client.get(url)
        return api_version, resp

    resp = client.get(url)

    last_updated = extract_last_updated_date_from_commonwealth_html(resp.text)
    if last_updated:
        return last_updated, resp

    last_mod = resp.headers.get("Last-Modified")
    if last_mod:
        try:
            dt = dateparser.parse(last_mod)
            if dt:
                return dt.date().isoformat(), resp
        except Exception:
            pass

    return stable_hash_version_id(resp.content), resp


def try_find_download_link(html: str, base_url: str, kind: str) -> Optional[str]:
    # kind: "xml", "rtf" or "pdf"
    soup = BeautifulSoup(html, "html.parser")
    kind = kind.lower()
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        if kind == "xml":
            if ".xml" in abs_url.lower() or "xml" in href.lower():
                return abs_url
        if kind == "rtf":
            if abs_url.lower().endswith(".rtf") or "rtf" in href.lower():
                return abs_url
        if kind == "pdf":
            if abs_url.lower().endswith(".pdf") or "pdf" in href.lower():
                return abs_url
    return None


def _looks_like_frl_site_chrome(text: str) -> bool:
    t = (text or "").lower()
    needles = [
        "sign in to my account",
        "register for my account",
        "help and resources",
        "site navigation",
        "scheduled maintenance",
        "order print copy",
        "save this title to my account",
        "set up an alert",
        "enter text to search the table of contents",
        "we acknowledge the traditional owners and custodians of country",
    ]
    hits = sum(1 for n in needles if n in t)
    return hits >= 2


def _guess_title_id_from_url_or_html(url: str, html: str) -> Optional[str]:
    # FRL identifiers often look like:
    # - C1971A00104 (Acts)
    # - F2025L01234 (Legislative instruments)
    # Keep this permissive and validate by attempting the API call.
    combined = f"{url}\n{html or ''}"
    m = re.search(r"\b[A-Z]\d{4}[A-Z]\d{5}\b", combined)
    if m:
        return m.group(0)
    return None


def _best_effort_decode_html_bytes(content: bytes) -> str:
    data = content or b""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def try_find_frl_epub_document_html_link(html: str, base_url: str) -> Optional[str]:
    """Find the FRL 'epub' HTML document that contains the full act text.

    On /latest/text pages, the visible HTML is largely navigation + TOC. The actual
    legislation body is served as an EPUB-derived HTML file linked from the TOC.
    """

    soup = BeautifulSoup(html or "", "html.parser")
    candidates: List[str] = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "/epub/" not in href.lower():
            continue
        if ".html" not in href.lower():
            continue
        abs_url = urljoin(base_url, href.split("#", 1)[0])
        if abs_url.lower().endswith(".html"):
            candidates.append(abs_url)

    if not candidates:
        return None

    # Prefer the canonical document_1/document_1.html when present.
    preferred = [c for c in candidates if "document_1/document_1.html" in c.lower()]
    if preferred:
        return sorted(set(preferred), key=len)[0]
    return sorted(set(candidates), key=len)[0]


def commonwealth_epub_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    title = ""
    title_p = soup.find("p", class_="ShortT")
    if title_p:
        title = " ".join((title_p.get_text(" ", strip=True) or "").split())
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = " ".join((h1.get_text(" ", strip=True) or "").split())

    paragraphs = soup.find_all("p")
    start_idx: Optional[int] = None
    for i, p in enumerate(paragraphs):
        classes = p.get("class") or []
        if any((c or "").startswith("ActHead") for c in classes):
            start_idx = i
            break
    if start_idx is None:
        return normalize_plain_text(html_to_text_preserve_blocks(html))

    pieces: List[str] = []

    def push(line: str, *, heading: bool) -> None:
        line = " ".join((line or "").split()).strip()
        if not line:
            return
        if heading and pieces and pieces[-1] != "":
            pieces.append("")
        pieces.append(line)

    if title:
        pieces.append(title)
        pieces.append("")

    for p in paragraphs[start_idx:]:
        classes = p.get("class") or []
        cls_strs = [c for c in classes if isinstance(c, str)]
        if any(c.startswith("TOC") for c in cls_strs):
            continue
        if any(c.startswith("ENotesHeading") for c in cls_strs):
            break
        if "Header" in cls_strs:
            continue

        txt = " ".join((p.get_text(" ", strip=True) or "").split())
        if not txt:
            continue

        is_heading = any(c.startswith("ActHead") for c in cls_strs)
        push(txt, heading=is_heading)

    out = "\n".join(pieces)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def looks_like_xml_response(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ctype:
        return True
    content = resp.content.lstrip()[:50]
    return content.startswith(b"<?xml") or content.startswith(b"<")


def generic_xml_to_text(xml_bytes: bytes) -> str:
    # Best-effort: preserve obvious heading-ish tags, otherwise join text in order.
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


def download_commonwealth_text(client: HttpClient, resp: requests.Response) -> Tuple[str, str, str]:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    # Prefer the full body HTML for /latest/text pages (avoids PDF page header repeats).
    if "legislation.gov.au" in (resp.url or "").lower() and "/text" in (resp.url or "").lower():
        epub_url = try_find_frl_epub_document_html_link(resp.text, resp.url)
        if epub_url:
            epub_resp = client.get(epub_url)
            epub_html = _best_effort_decode_html_bytes(epub_resp.content)
            text = commonwealth_epub_html_to_text(epub_html)
            return normalize_plain_text(text), epub_resp.url, "html"

    if "rtf" in ctype or resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(resp.content):
        return rtf_bytes_to_text(resp.content), resp.url, "rtf"
    if "xml" in ctype:
        return generic_xml_to_text(resp.content), resp.url, "xml"
    if "pdf" in ctype or resp.url.lower().endswith(".pdf"):
        return pdf_bytes_to_text(resp.content), resp.url, "pdf"

    # If we're on an HTML page, prefer RTF/XML/PDF if a download link is present.
    rtf_link = try_find_download_link(resp.text, resp.url, "rtf")
    if rtf_link:
        try:
            rtf_resp = client.get(rtf_link)
            rtf_ct = (rtf_resp.headers.get("Content-Type") or "").lower()
            if "rtf" in rtf_ct or rtf_resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(rtf_resp.content):
                return rtf_bytes_to_text(rtf_resp.content), rtf_resp.url, "rtf"
        except Exception:
            pass

    xml_link = try_find_download_link(resp.text, resp.url, "xml")
    if xml_link:
        try:
            xml_resp = client.get(xml_link)
            if looks_like_xml_response(xml_resp):
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

    text = normalize_plain_text(html_to_text_preserve_blocks(resp.text))
    if _looks_like_frl_site_chrome(text):
        raise RuntimeError("Commonwealth HTML appears to be FRL site chrome; no usable download link found")
    return text, resp.url, "html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commonwealth legislation sync bot")
    parser.add_argument("--user-agent", default="AustralianTaxLawLibrarySyncBot/Commonwealth")
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

    jurisdiction_dir = Path.cwd()
    data_dir = jurisdiction_dir / "data"
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
        title_id = (row.get("title_id") or "").strip()
        if not library_id or not url:
            row["status"] = "error"
            row["error"] = "Missing library_id or url"
            cat.save()
            processed += 1
            continue

        try:
            current_version_id = (row.get("version_id") or "").strip()

            # Prefer HTML-based full-text extraction for /text pages; fall back to API PDF only if needed.
            latest_version_id, resp = guess_commonwealth_version_id(client, url, title_id)
            if (not args.force) and current_version_id == latest_version_id:
                row["status"] = "skipped_no_change"
                row["error"] = ""
                cat.save()
                processed += 1
                continue

            guessed_title_id = _guess_title_id_from_url_or_html(resp.url, resp.text)

            try:
                text, content_url, content_format = download_commonwealth_text(client, resp)
            except Exception:
                # If HTML extraction fails (e.g., unexpected structure), fall back to authorised Primary PDF.
                official = try_frl_latest_primary_pdf_document(client, title_id or (guessed_title_id or ""))
                if official is None:
                    raise
                latest_version_id, pdf_bytes, pdf_url = official
                text = pdf_bytes_to_text(pdf_bytes)
                content_url, content_format = pdf_url, "pdf"

            row["content_url"] = (content_url or url).strip()
            row["content_format"] = (content_format or "html").strip()

            text = normalize_plain_text(text)
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
