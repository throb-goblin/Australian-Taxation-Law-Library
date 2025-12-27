from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import zipfile
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

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
    return cleanup_vic_pdf_text(text or "")


def cleanup_vic_pdf_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    out = out.replace("\u00ad", "")
    out = re.sub(r"([A-Za-z])\-\n([a-z])", r"\1\2", out)

    noise_patterns = [
        re.compile(r"^page\s*\d+(\s*of\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^https?://\S+$", re.IGNORECASE),
        re.compile(r"^authorised\s+version\b.*$", re.IGNORECASE),
        re.compile(r"\bchief\s+parliamentary\s+counsel\b", re.IGNORECASE),
        re.compile(r"\bcontent\.legislation\.vic\.gov\.au\b", re.IGNORECASE),
        re.compile(r"\bprivacy\b|\bcookies?\b|\bterms\b|\bdisclaimer\b|\bcopyright\b", re.IGNORECASE),
    ]

    lines: List[str] = []
    for raw in out.split("\n"):
        line = (raw or "").strip()
        if not line:
            lines.append("")
            continue
        if re.search(r"https?://", line, flags=re.IGNORECASE) and len(line) < 250:
            continue
        if any(p.search(line) for p in noise_patterns) and len(line) < 160:
            continue
        lines.append(line)

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def docx_bytes_to_text(docx_bytes: bytes) -> str:
    # Minimal DOCX text extraction (word/document.xml) without external deps.
    # DOCX is a zip containing OOXML.
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception as exc:
        raise RuntimeError(f"Failed to read DOCX: {exc}")

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DOCX XML: {exc}")

    # Extract runs of text while preserving paragraph breaks.
    # Skip textbox/drawing content (commonly amendment-note callouts on VIC).
    ns_w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    textbox_paragraph_ids: set[int] = set()
    for txbx in root.iter(ns_w + "txbxContent"):
        for p in txbx.iter(ns_w + "p"):
            textbox_paragraph_ids.add(id(p))

    out: List[str] = []
    for p in root.iter(ns_w + "p"):
        if id(p) in textbox_paragraph_ids:
            continue

        parts: List[str] = []
        for t in p.iter(ns_w + "t"):
            if t.text:
                parts.append(t.text)
        line = "".join(parts).strip()
        if line:
            out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text


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


def stable_hash_version_id(content_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(content_bytes)
    return "sha256:" + h.hexdigest()


def safe_txt_filename(library_id: str) -> str:
    name = (library_id or "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    return f"{name}.txt"


JURIS_ABBREV = "VIC"


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
            j = i + 1
            while j < len(in_lines) and not (in_lines[j] or "").strip():
                j += 1
            nxt = (in_lines[j] or "").strip() if j < len(in_lines) else ""

            # Avoid blank lines directly before subsection/marker lines.
            if nxt and (is_subsection_line(nxt) or is_marker(nxt)):
                i += 1
                continue

            # Avoid blank lines between a section heading and its first content.
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
    body = _normalize_body_text(body)
    header = _header_block_for_row(row)
    return f"{header}\n\n{body.strip()}\n"


def find_catalogue_csv(data_dir: Path) -> Path:
    matches = sorted(data_dir.glob("*_legislation_catalogue.csv"))
    if not matches:
        raise FileNotFoundError(f"No *_legislation_catalogue.csv found in {data_dir}")
    return matches[0]


def extract_version_segment_from_url(url: str) -> Optional[str]:
    # Prefer a trailing /### segment
    m = re.search(r"/(\d{3})/?$", url)
    if m:
        return m.group(1)
    return None


def extract_latest_version_id_vic(client: HttpClient, url: str) -> Tuple[str, requests.Response]:
    resp = client.get(url, allow_redirects=True)

    # 1) Final URL after redirects
    seg = extract_version_segment_from_url(resp.url)
    if seg:
        return seg, resp

    # 2) Canonical link
    soup = BeautifulSoup(resp.text, "html.parser")
    canonical = soup.find("link", attrs={"rel": re.compile(r"canonical", re.IGNORECASE)})
    if canonical and canonical.get("href"):
        canon_url = urljoin(resp.url, canonical.get("href"))
        seg = extract_version_segment_from_url(canon_url)
        if seg:
            return seg, resp

    # 3) Newest /### link on the page
    candidates: List[int] = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(resp.url, href)
        seg = extract_version_segment_from_url(abs_url)
        if seg:
            try:
                candidates.append(int(seg))
            except ValueError:
                pass
    if candidates:
        return f"{max(candidates):03d}", resp

    # 4) Date if explicitly present
    text = soup.get_text(" ", strip=True)
    for m in re.finditer(r"Last\s+updated\s*[:\-]\s*([^\.|\n]+)", text, flags=re.IGNORECASE):
        try:
            dt = dateparser.parse(m.group(1), dayfirst=True)
            if dt:
                return dt.date().isoformat(), resp
        except Exception:
            pass

    # 5) HTTP last-modified
    last_mod = resp.headers.get("Last-Modified")
    if last_mod:
        try:
            dt = dateparser.parse(last_mod)
            if dt:
                return dt.date().isoformat(), resp
        except Exception:
            pass

    # 6) Content hash
    # 6a) Nuxt SPA pages often don't expose versioning/downloads in HTML.
    # Try the Drupal content backend and infer a stable version id from the top download link.
    try:
        content_url = vic_content_url_from_main_site(resp.url)
        if content_url:
            content_resp = client.get(content_url, allow_redirects=True)
            best = find_first_vic_content_download_link(content_resp.text, content_resp.url)
            if best:
                inferred = infer_vic_version_id_from_download_url(best)
                if inferred:
                    return inferred, resp
    except Exception:
        pass

    return stable_hash_version_id(resp.content), resp


def vic_content_url_from_main_site(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
    except Exception:
        return None

    host = (p.netloc or "").lower()
    if not host.endswith("legislation.vic.gov.au"):
        return None

    path = (p.path or "").strip()
    if not path.startswith("/"):
        path = "/" + path
    path = path.rstrip("/")
    if not path:
        return None

    # Empirically, the Vic Drupal backend mirrors these paths under site-6.
    return "https://content.legislation.vic.gov.au/site-6" + path


def find_first_vic_content_download_link(html: str, base_url: str) -> Optional[str]:
    """Find the first 'current version' download link from the Vic content (Drupal) page.

    The content page lists current consolidated downloads first in DOM order.
    Prefer Word (DOCX/DOC) over PDF for cleaner text extraction.
    """

    soup = BeautifulSoup(html or "", "html.parser")

    def iter_links():
        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            abs_url = urljoin(base_url, href)
            low = abs_url.lower()
            if "sites/default/files/" not in low:
                continue
            yield abs_url

    # Prefer DOCX/DOC, then PDF.
    for u in iter_links():
        low = u.lower()
        if low.endswith(".docx") or low.endswith(".doc"):
            return u
    for u in iter_links():
        if u.lower().endswith(".pdf"):
            return u
    return None


def infer_vic_version_id_from_download_url(download_url: str) -> Optional[str]:
    """Infer VIC version id from filenames like 00-79a140.docx or 00-79aa140-authorised.pdf."""
    u = (download_url or "").lower()
    m = re.search(r"00-79a[a-z]?(\d{3,4})", u)
    if m:
        return m.group(1)
    return None


def _looks_like_vic_site_chrome(text: str) -> bool:
    t = (text or "").lower()
    # Heuristics tuned to Victorian legislation site chrome captured in samples.
    needles = [
        "skip to main content",
        "version history",
        "acts in force",
        "statutory rules in force",
        "bills",
        "privacy",
        "disclaimer",
        "copyright",
        "contact us",
        "accessibility",
    ]
    hits = sum(1 for n in needles if n in t)
    return hits >= 4 and len(t) < 200_000


def _try_find_download_link_vic(html: str, base_url: str, kind: str) -> Optional[str]:
    kind = kind.lower()
    soup = BeautifulSoup(html, "html.parser")

    candidates: List[Tuple[int, str]] = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        label = (a.get_text(" ", strip=True) or "").strip().lower()
        abs_url = urljoin(base_url, href)
        low = abs_url.lower()
        href_low = href.lower()

        if kind == "rtf":
            if low.endswith(".rtf") or "rtf" in href_low or "rtf" in label:
                score = 0
                if "download" in label:
                    score -= 2
                candidates.append((score, abs_url))
            continue

        if kind == "docx":
            if low.endswith(".docx") or "docx" in href_low or "word" in label:
                score = 5
                if low.endswith(".docx"):
                    score -= 2
                if "download" in label:
                    score -= 2
                candidates.append((score, abs_url))
            continue

        if kind == "pdf":
            if low.endswith(".pdf") or "pdf" in href_low or "pdf" in label:
                score = 10
                if "download" in label:
                    score -= 2
                candidates.append((score, abs_url))
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def download_vic_text(client: HttpClient, resp: requests.Response) -> Tuple[str, str, str]:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "rtf" in ctype or resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(resp.content):
        return rtf_bytes_to_text(resp.content), resp.url, "rtf"
    if "word" in ctype or resp.url.lower().endswith(".docx"):
        return docx_bytes_to_text(resp.content), resp.url, "docx"
    if "pdf" in ctype or resp.url.lower().endswith(".pdf"):
        return pdf_bytes_to_text(resp.content), resp.url, "pdf"
    if "html" in ctype:
        # Prefer downloads to avoid saving site chrome.
        rtf_link = _try_find_download_link_vic(resp.text, resp.url, "rtf")
        if rtf_link:
            try:
                rtf_resp = client.get(rtf_link)
                rtf_ct = (rtf_resp.headers.get("Content-Type") or "").lower()
                if "rtf" in rtf_ct or rtf_resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(rtf_resp.content):
                    return rtf_bytes_to_text(rtf_resp.content), rtf_resp.url, "rtf"
            except Exception:
                pass

        docx_link = _try_find_download_link_vic(resp.text, resp.url, "docx")
        if docx_link:
            try:
                docx_resp = client.get(docx_link)
                docx_ct = (docx_resp.headers.get("Content-Type") or "").lower()
                if "word" in docx_ct or docx_resp.url.lower().endswith(".docx"):
                    return docx_bytes_to_text(docx_resp.content), docx_resp.url, "docx"
            except Exception:
                pass

        # PDF last resort.
        pdf_link = _try_find_download_link_vic(resp.text, resp.url, "pdf")
        if pdf_link:
            try:
                pdf_resp = client.get(pdf_link)
                pdf_ct = (pdf_resp.headers.get("Content-Type") or "").lower()
                if "pdf" in pdf_ct or pdf_resp.url.lower().endswith(".pdf"):
                    return pdf_bytes_to_text(pdf_resp.content), pdf_resp.url, "pdf"
            except Exception:
                pass

        # Nuxt SPA pages: fall back to the Drupal content backend, which exposes
        # direct file downloads (Word/PDF) in DOM order.
        try:
            content_url = vic_content_url_from_main_site(resp.url)
            if content_url:
                content_resp = client.get(content_url, allow_redirects=True)
                best = find_first_vic_content_download_link(content_resp.text, content_resp.url)
                if best:
                    best_resp = client.get(best)
                    best_ct = (best_resp.headers.get("Content-Type") or "").lower()
                    if "rtf" in best_ct or best_resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(best_resp.content):
                        return rtf_bytes_to_text(best_resp.content), best_resp.url, "rtf"
                    if "word" in best_ct or best_resp.url.lower().endswith(".docx") or best_resp.url.lower().endswith(".doc"):
                        return docx_bytes_to_text(best_resp.content), best_resp.url, "docx"
                    if "pdf" in best_ct or best_resp.url.lower().endswith(".pdf"):
                        return pdf_bytes_to_text(best_resp.content), best_resp.url, "pdf"
        except Exception:
            pass

        text = html_to_text_preserve_blocks(resp.text)
        if _looks_like_vic_site_chrome(text):
            raise RuntimeError("Victoria HTML appears to be site chrome; no usable download link found")
        return text, resp.url, "html"

    return html_to_text_preserve_blocks(resp.text), resp.url, "html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Victoria legislation sync bot")
    parser.add_argument("--user-agent", default="AustralianTaxLawLibrarySyncBot/VIC")
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
        if not library_id or not url:
            row["status"] = "error"
            row["error"] = "Missing library_id or url"
            cat.save()
            processed += 1
            continue

        try:
            latest_version_id, resp = extract_latest_version_id_vic(client, url)
            current_version_id = (row.get("version_id") or "").strip()

            out_path = data_dir / safe_txt_filename(library_id)
            has_provenance = bool((row.get("content_url") or "").strip()) and bool((row.get("content_format") or "").strip())

            if (not args.force) and current_version_id == latest_version_id and has_provenance and out_path.exists():
                row["status"] = "skipped_no_change"
                row["error"] = ""
                cat.save()
                processed += 1
                continue

            text, content_url, content_format = download_vic_text(client, resp)
            row["content_url"] = (content_url or url).strip()
            row["content_format"] = (content_format or "html").strip()
            row_for_meta = dict(row)
            row_for_meta["version_id"] = latest_version_id
            text = finalize_output_text(row_for_meta, text)
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
