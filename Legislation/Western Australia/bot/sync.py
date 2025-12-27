from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import tempfile
import time
import zipfile
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

try:
    import win32com.client as win32_client  # type: ignore
except Exception:
    win32_client = None


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
    out = (text or "")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


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


def looks_like_docx_bytes(content: bytes) -> bool:
    head = (content or b"")[:4]
    if head != b"PK\x03\x04":
        return False
    b = content or b""
    # Quick signature check for OOXML Word packages.
    return (b"word/document.xml" in b) or (b"[Content_Types].xml" in b and b"word/" in b)


def docx_bytes_to_text(docx_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
            xml = z.read("word/document.xml")
    except Exception as exc:
        raise RuntimeError(f"Failed to read DOCX: {exc}")

    # Drop textbox content blocks (common source of amendment notes/callouts).
    xml_str = (xml or b"").decode("utf-8", errors="replace")
    xml_str = re.sub(r"<w:txbxContent\b.*?</w:txbxContent>", "", xml_str, flags=re.DOTALL)

    try:
        root = ET.fromstring(xml_str)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DOCX XML: {exc}")

    def local(tag: str) -> str:
        return tag.split("}", 1)[1] if "}" in tag else tag

    paras: List[str] = []
    for p in root.iter():
        if local(p.tag) != "p":
            continue
        parts: List[str] = []
        for node in p.iter():
            name = local(node.tag)
            if name == "t":
                if node.text:
                    parts.append(node.text)
            elif name == "tab":
                parts.append("\t")
            elif name == "br":
                parts.append("\n")

        txt = "".join(parts)
        txt = txt.replace("\r\n", "\n").replace("\r", "\n")
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = "\n".join([ln.rstrip() for ln in txt.split("\n")]).strip()
        if txt:
            paras.append(txt)

    out = "\n".join(paras)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def looks_like_doc_ole_bytes(content: bytes) -> bool:
    # OLE Compound File signature (legacy .doc).
    head = (content or b"")[:8]
    return head == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"


def doc_bytes_to_text_via_word_com(doc_bytes: bytes) -> str:
    if win32_client is None:
        raise RuntimeError("pywin32 is not available to extract legacy .doc")

    tmp_path: Optional[str] = None
    word = None
    doc = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp:
            tmp.write(doc_bytes)
            tmp.flush()
            tmp_path = tmp.name

        word = win32_client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(tmp_path, ReadOnly=True, AddToRecentFiles=False)
        text = (doc.Content.Text or "")
        out = text.replace("\r\n", "\n").replace("\r", "\n")
        out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
        return out
    finally:
        try:
            if doc is not None:
                doc.Close(False)
        except Exception:
            pass
        try:
            if word is not None:
                word.Quit()
        except Exception:
            pass
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def cleanup_wa_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")

    lines: List[str] = []
    for raw in out.split("\n"):
        line = (raw or "").strip()
        if not line:
            lines.append("")
            continue

        # Drop common Gazette/PDF header/footer noise.
        if re.search(r"\bgovernment\s+gazette\b", line, flags=re.IGNORECASE):
            continue
        if re.search(r"\bissn\b", line, flags=re.IGNORECASE):
            continue
        if re.search(r"\bwestern\s+australia\b", line, flags=re.IGNORECASE) and re.search(
            r"\bgazette\b", line, flags=re.IGNORECASE
        ):
            continue
        if re.search(r"\bpublished\s+on\b", line, flags=re.IGNORECASE):
            continue

        # Drop page numbers.
        if re.fullmatch(r"\d+", line):
            continue
        if re.fullmatch(r"page\s*\d+(\s*of\s*\d+)?", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"page\s*[ivxlcdm]+", line, flags=re.IGNORECASE):
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


JURIS_ABBREV = "WA"


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
    out = re.sub(r"\b([A-Za-z])\n[ \t]+([a-z]{6,})", r"\1\2", out)
    out = re.sub(r"([A-Za-z])\n[ \t]+([a-z]{2,})", r"\1 \2", out)
    return out


def _normalize_body_text(body: str) -> str:
    raw = _hard_wrap_repair((body or "").replace("\r\n", "\n").replace("\r", "\n"))
    in_lines = [ln.rstrip() for ln in raw.split("\n")]

    def is_bracket_note(line: str) -> bool:
        s = (line or "").strip()
        if not (s.startswith("[") and s.endswith("]")):
            return False
        # Keep this conservative: WA Word exports contain editor notes like
        # "[Heading inserted: ...]" and "[Section X inserted: ...]".
        return True

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
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            i += 1
            continue

        # Treat bracketed editor notes as their own lines, but ensure they end a block.
        if is_bracket_note(s):
            out_lines.append(s)
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            # Only insert a blank line after the last note in a run.
            if j < len(in_lines):
                nxt = (in_lines[j] or "").strip()
                if not is_bracket_note(nxt):
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
            # Do not force a blank line before a bracket note like "[Heading inserted: ...]".
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            if j < len(in_lines):
                nxt = (in_lines[j] or "").strip()
                if not is_bracket_note(nxt):
                    if out_lines and out_lines[-1] != "":
                        out_lines.append("")
            else:
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


def _extract_wa_currency_version_from_html(html: str) -> Optional[str]:
    """Western Australia consolidated pages list 'Currency start' for the current version.

    We treat the current row's currency start date as the official consolidation marker.
    """

    soup = BeautifulSoup(html, "html.parser")

    # Prefer the row explicitly marked as current.
    current_td = soup.select_one("td.current")
    row = current_td.parent if current_td else None
    if row is None:
        for tr in soup.select("table tbody tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"]) if c.get_text(strip=True)]
            if any(c.lower() == "current" for c in cells):
                row = tr
                break

    if row is None:
        return None

    cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
    if len(cells) < 2:
        return None

    currency_start_raw = (cells[1] or "").strip()
    if not currency_start_raw:
        return None

    try:
        dt = dateparser.parse(currency_start_raw, dayfirst=True)
        currency_start = dt.date().isoformat() if dt else None
    except Exception:
        currency_start = None

    if not currency_start:
        return None

    suffix = (cells[3] or "").strip() if len(cells) >= 4 else ""
    if suffix:
        return f"currency_start:{currency_start}|suffix:{suffix}"
    return f"currency_start:{currency_start}"


def _conditional_headers_for_version_id(version_id: Optional[str]) -> Optional[Dict[str, str]]:
    if not version_id:
        return None
    v = version_id.strip()
    if not v:
        return None

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
    # Preserve validator semantics when explicitly stored.
    if v.startswith("etag:") or v.startswith("last_modified:"):
        return v
    # Force one-time regeneration when output strategy changes.
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
        if kind == "xml":
            if abs_url.lower().endswith(".xml") or "xml" in href.lower():
                return abs_url
        if kind == "docx":
            if abs_url.lower().endswith(".docx") or "docx" in href.lower() or "word" in href.lower():
                return abs_url
        if kind == "doc":
            # Match .doc but avoid treating .docx as .doc
            low = abs_url.lower()
            if low.endswith(".doc") or (".doc" in low and not low.endswith(".docx")):
                return abs_url
        if kind == "rtf":
            if abs_url.lower().endswith(".rtf") or "rtf" in href.lower():
                return abs_url
        if kind == "pdf":
            if abs_url.lower().endswith(".pdf") or "pdf" in href.lower():
                return abs_url
    return None


def try_find_wa_official_source_link(html: str, base_url: str) -> Optional[str]:
    """Prefer WA 'official version' filestore (mrdoc) HTML/PDF links over Gazette PDFs."""
    soup = BeautifulSoup(html, "html.parser")
    candidates: List[Tuple[int, str]] = []

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        label = (a.get_text(" ", strip=True) or "").strip().lower()
        abs_url = urljoin(base_url, href)
        low = abs_url.lower()

        # Skip obvious Gazette targets.
        if "gg" in low and ".pdf" in low:
            candidates.append((200, abs_url))
            continue
        if "gazette" in low:
            candidates.append((200, abs_url))
            continue

        score = 100

        # Strongly prefer filestore mrdoc HTML (cleanest for text).
        if "filestore.nsf/fileurl/" in low and ("mrdoc_" in low):
            score = 0
            # Prefer Word downloads over everything else.
            if low.endswith(".docx"):
                score = -20
            elif low.endswith(".doc"):
                score = -18
            elif low.endswith(".htm") or low.endswith(".html"):
                score = -10
            if "openelement" in low:
                score -= 1
        # Also accept RedirectURL that resolves to mrdoc.
        elif "redirecturl" in low and "mrdoc_" in low:
            score = 10
            # Prefer HTML over PDF when the query includes an extension.
            if ".docx" in low:
                score = -20
            elif ".doc" in low:
                score = -18
            elif ".htm" in low or ".html" in low:
                score = -10
            elif ".pdf" in low:
                score = 20
        # Generic PDF link (lower priority)
        elif low.endswith(".pdf") or "pdf" in low or "pdf" in label:
            score = 50

        # Prefer links explicitly labelled as official.
        if "official" in label:
            score -= 10
        if "download" in label:
            score -= 2

        candidates.append((score, abs_url))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def wa_filestore_html_to_text(html: str) -> str:
    """Extract text from WA filestore HTML downloads.

    These pages are much cleaner than PDF output for page markers.
    """

    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # Start from the act title when possible.
    h1 = None
    for cand in soup.find_all("h1"):
        txt = (cand.get_text(" ", strip=True) or "").strip()
        if txt and "western australia" not in txt.lower() and "contents" not in txt.lower():
            h1 = cand
            break

    root = h1.find_parent("body") if h1 else (soup.body or soup)

    block_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"}

    pieces: List[str] = []
    started = h1 is None
    for el in root.find_all(list(block_tags)):
        if el is h1:
            started = True
        if not started:
            continue

        sep = "\n" if (el.name or "").lower() in {"p", "li", "pre"} else " "
        raw_txt = (el.get_text(sep, strip=True) or "").replace("\xa0", " ")
        raw_lines = [" ".join((ln or "").split()) for ln in (raw_txt.splitlines() or [])]
        raw_lines = [ln for ln in raw_lines if ln]
        if not raw_lines:
            continue

        is_heading = (el.name or "").lower().startswith("h")
        if is_heading and pieces:
            pieces.append("")
        pieces.extend(raw_lines)

    out = "\n".join(pieces)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def decode_html_bytes_best_effort(content: bytes, content_type: str) -> str:
    ct = (content_type or "").lower()
    m = re.search(r"charset=([^;]+)", ct)
    if m:
        enc = m.group(1).strip().strip('"').strip("'")
        try:
            return (content or b"").decode(enc, errors="replace")
        except Exception:
            pass

    b = content or b""
    # WA filestore HTML frequently omits charset but is UTF-8.
    s_utf8 = b.decode("utf-8", errors="replace")
    if s_utf8.count("\ufffd") <= max(2, len(s_utf8) // 2000):
        return s_utf8
    return b.decode("cp1252", errors="replace")


def guess_version_id_and_fetch(
    client: HttpClient,
    url: str,
    *,
    current_version_id: Optional[str] = None,
) -> Tuple[str, requests.Response]:
    resp = client.get(url, headers=_conditional_headers_for_version_id(current_version_id))
    if resp.status_code == 304 and current_version_id:
        return current_version_id, resp
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "html" in ctype:
        wa_version = _extract_wa_currency_version_from_html(resp.text)
        if wa_version:
            return decorate_version_id(wa_version), resp

        page_version = extract_date_like_version_from_html(resp.text)
        if page_version:
            return decorate_version_id(page_version), resp

    etag = (resp.headers.get("ETag") or "").strip()
    if etag:
        return f"etag:{etag}", resp

    last_mod = resp.headers.get("Last-Modified")
    if last_mod:
        return f"last_modified:{last_mod}", resp

    if "html" in ctype:
        normalized = " ".join(BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True).split())
        return decorate_version_id(stable_hash_version_id(normalized.encode("utf-8"))), resp
    return decorate_version_id(stable_hash_version_id(resp.content)), resp


def download_text_best_effort(client: HttpClient, resp: requests.Response) -> Tuple[str, str, str]:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "docx" in ctype or resp.url.lower().endswith(".docx") or looks_like_docx_bytes(resp.content):
        return cleanup_wa_text(docx_bytes_to_text(resp.content)), resp.url, "docx"
    if "msword" in ctype or resp.url.lower().endswith(".doc") or looks_like_doc_ole_bytes(resp.content):
        return cleanup_wa_text(doc_bytes_to_text_via_word_com(resp.content)), resp.url, "doc"
    if "rtf" in ctype or resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(resp.content):
        return cleanup_wa_text(rtf_bytes_to_text(resp.content)), resp.url, "rtf"
    if looks_like_xml(resp):
        return cleanup_wa_text(generic_xml_to_text(resp.content)), resp.url, "xml"
    if "pdf" in ctype or resp.url.lower().endswith(".pdf"):
        return cleanup_wa_text(pdf_bytes_to_text(resp.content)), resp.url, "pdf"

    if "html" in ctype:
        # WA consolidated pages are often SPA shells; prefer official filestore sources.
        preferred = try_find_wa_official_source_link(resp.text, resp.url)
        if preferred:
            try:
                pref_resp = client.get(preferred)
                pref_ct = (pref_resp.headers.get("Content-Type") or "").lower()
                if "docx" in pref_ct or pref_resp.url.lower().endswith(".docx") or looks_like_docx_bytes(pref_resp.content):
                    return cleanup_wa_text(docx_bytes_to_text(pref_resp.content)), pref_resp.url, "docx"
                if "msword" in pref_ct or pref_resp.url.lower().endswith(".doc") or looks_like_doc_ole_bytes(pref_resp.content):
                    return cleanup_wa_text(doc_bytes_to_text_via_word_com(pref_resp.content)), pref_resp.url, "doc"
                if "rtf" in pref_ct or pref_resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(pref_resp.content):
                    return cleanup_wa_text(rtf_bytes_to_text(pref_resp.content)), pref_resp.url, "rtf"
                if looks_like_xml(pref_resp):
                    return cleanup_wa_text(generic_xml_to_text(pref_resp.content)), pref_resp.url, "xml"
                if "html" in pref_ct or pref_resp.url.lower().endswith((".htm", ".html")):
                    html_text = decode_html_bytes_best_effort(pref_resp.content, pref_ct)
                    return cleanup_wa_text(wa_filestore_html_to_text(html_text)), pref_resp.url, "html"
                if "pdf" in pref_ct or pref_resp.url.lower().endswith(".pdf"):
                    return cleanup_wa_text(pdf_bytes_to_text(pref_resp.content)), pref_resp.url, "pdf"
                return cleanup_wa_text(html_to_text_preserve_blocks(pref_resp.text)), pref_resp.url, "html"
            except Exception:
                pass

        docx_link = try_find_download_link(resp.text, resp.url, "docx")
        if docx_link:
            try:
                docx_resp = client.get(docx_link)
                docx_ct = (docx_resp.headers.get("Content-Type") or "").lower()
                if "docx" in docx_ct or docx_resp.url.lower().endswith(".docx") or looks_like_docx_bytes(docx_resp.content):
                    return cleanup_wa_text(docx_bytes_to_text(docx_resp.content)), docx_resp.url, "docx"
            except Exception:
                pass

        doc_link = try_find_download_link(resp.text, resp.url, "doc")
        if doc_link:
            try:
                doc_resp = client.get(doc_link)
                doc_ct = (doc_resp.headers.get("Content-Type") or "").lower()
                if "msword" in doc_ct or doc_resp.url.lower().endswith(".doc") or looks_like_doc_ole_bytes(doc_resp.content):
                    return cleanup_wa_text(doc_bytes_to_text_via_word_com(doc_resp.content)), doc_resp.url, "doc"
            except Exception:
                pass

        rtf_link = try_find_download_link(resp.text, resp.url, "rtf")
        if rtf_link:
            try:
                rtf_resp = client.get(rtf_link)
                rtf_ct = (rtf_resp.headers.get("Content-Type") or "").lower()
                if "rtf" in rtf_ct or rtf_resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(rtf_resp.content):
                    return cleanup_wa_text(rtf_bytes_to_text(rtf_resp.content)), rtf_resp.url, "rtf"
            except Exception:
                pass

        xml_link = try_find_download_link(resp.text, resp.url, "xml")
        if xml_link:
            try:
                xml_resp = client.get(xml_link)
                if looks_like_xml(xml_resp):
                    return cleanup_wa_text(generic_xml_to_text(xml_resp.content)), xml_resp.url, "xml"
            except Exception:
                pass

        pdf_link = try_find_download_link(resp.text, resp.url, "pdf")
        if pdf_link:
            try:
                pdf_resp = client.get(pdf_link)
                pdf_ct = (pdf_resp.headers.get("Content-Type") or "").lower()
                if "pdf" in pdf_ct or pdf_resp.url.lower().endswith(".pdf"):
                    return cleanup_wa_text(pdf_bytes_to_text(pdf_resp.content)), pdf_resp.url, "pdf"
            except Exception:
                pass

        return cleanup_wa_text(html_to_text_preserve_blocks(resp.text)), resp.url, "html"

    return cleanup_wa_text((resp.text or "").strip() + "\n"), resp.url, "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Western Australia legislation sync bot")
    parser.add_argument("--user-agent", default="AustralianTaxLawLibrarySyncBot/WA")
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
