from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import tempfile
import textwrap
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
        if text and len(text) > 20000:
            # Some register HTML pages place large amounts of content inside a single container
            # with <br> and inline spans. Re-extract with a newline separator so the output
            # doesn't become a single megabyte-scale line.
            text = el.get_text("\n", strip=True)
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


def normalize_qld_url_to_whole_html(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if "legislation.qld.gov.au" not in u.lower():
        return u
    if "/view/whole/html/" in u.lower():
        return u
    if "/view/html/" in u.lower():
        return re.sub(r"/view/html/", "/view/whole/html/", u, flags=re.IGNORECASE)
    return u


def normalize_qld_url_to_whole_xml(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    if "legislation.qld.gov.au" not in u.lower():
        return u
    if "/view/whole/xml/" in u.lower():
        return u
    if "/view/whole/html/" in u.lower():
        return re.sub(r"/view/whole/html/", "/view/whole/xml/", u, flags=re.IGNORECASE)
    if "/view/xml/" in u.lower():
        return re.sub(r"/view/xml/", "/view/whole/xml/", u, flags=re.IGNORECASE)
    if "/view/html/" in u.lower():
        return re.sub(r"/view/html/", "/view/whole/xml/", u, flags=re.IGNORECASE)
    return u


def qld_whole_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    h1 = None
    for cand in soup.find_all("h1"):
        txt = (cand.get_text(" ", strip=True) or "").strip()
        if txt and "site" not in txt.lower():
            h1 = cand
            break
    if not h1:
        return html_to_text_preserve_blocks(html)

    main = h1.find_parent(id="main") or h1.find_parent("body") or soup

    block_tags = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"}
    headingish_p_classes = {
        "ChapterHeadingParagraph",
        "PartHeadingName",
        "PartHeadingParagraph",
    }
    toc_ids = {"toc", "tocnav", "toc-col"}

    def is_in_toc(el) -> bool:
        for p in [el] + list(getattr(el, "parents", []) or []):
            pid = (getattr(p, "get", lambda *_: None)("id") or "")
            if pid and pid.lower() in toc_ids:
                return True
            classes = getattr(p, "get", lambda *_: None)("class") or []
            if any("toc" in (c or "").lower() for c in classes):
                return True
        return False

    pieces: List[str] = []
    started = False
    for el in main.find_all(list(block_tags)):
        if el is h1:
            started = True
        if not started:
            continue
        if is_in_toc(el):
            continue

        sep = "\n" if (el.name or "").lower() in {"p", "li", "pre"} else " "
        raw_txt = (el.get_text(sep, strip=True) or "").replace("\xa0", " ")
        raw_lines = [" ".join((ln or "").split()) for ln in (raw_txt.splitlines() or [])]
        raw_lines = [ln for ln in raw_lines if ln]
        if not raw_lines:
            continue

        is_heading = (el.name or "").lower().startswith("h")
        if (el.name or "").lower() == "p":
            classes = set((el.get("class") or []))
            if classes.intersection(headingish_p_classes):
                is_heading = True

        if is_heading and pieces:
            pieces.append("")
        pieces.extend(raw_lines)

    out = "\n".join(pieces)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def qld_whole_xml_to_text(xml_bytes: bytes) -> str:
    xml_text = ""
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            xml_text = (xml_bytes or b"").decode(enc)
            break
        except Exception:
            continue
    if not xml_text:
        xml_text = (xml_bytes or b"").decode("utf-8", errors="replace")

    # ElementTree doesn't handle DTD/DOCTYPE; strip it.
    xml_text = re.sub(r"<!DOCTYPE[^>]*>", "", xml_text, flags=re.IGNORECASE | re.DOTALL)
    xml_text = xml_text.replace("\r\n", "\n").replace("\r", "\n")

    root = ET.fromstring(xml_text)

    def local(tag: str) -> str:
        return (tag or "").split("}")[-1]

    def child_text(parent: ET.Element, name: str) -> str:
        for c in list(parent):
            if local(getattr(c, "tag", "")) == name:
                return " ".join("".join(c.itertext()).split())
        return ""

    def txt_from_blocks(parent: ET.Element) -> List[str]:
        paras: List[str] = []
        for block in list(parent):
            if local(getattr(block, "tag", "")) != "block":
                continue
            for txt in list(block):
                if local(getattr(txt, "tag", "")) != "txt":
                    continue
                s = " ".join("".join(txt.itertext()).split())
                if s:
                    paras.append(s)
        return paras

    lines: List[str] = []

    short_title = ""
    for el in root.iter():
        if local(getattr(el, "tag", "")) == "shorttitle":
            short_title = " ".join("".join(el.itertext()).split())
            break

    if short_title:
        lines.append(short_title)
        lines.append("")

    def add_para(s: str) -> None:
        s = " ".join((s or "").split()).strip()
        if not s:
            return
        if lines and lines[-1] != "":
            lines.append("")
        lines.append(s)

    def add_body_para(s: str) -> None:
        s = " ".join((s or "").split()).strip()
        if not s:
            return
        lines.append(s)

    def process_container(el: ET.Element) -> None:
        tag = local(getattr(el, "tag", ""))
        if tag == "historynote":
            return

        if tag in {"chapter", "part", "division", "subdivision", "schedule"}:
            no = child_text(el, "no")
            he = child_text(el, "heading")
            heading = " ".join([p for p in [no, he] if p]).strip()
            if heading:
                add_para(heading)
            for child in list(el):
                if local(getattr(child, "tag", "")) in {"chapter", "part", "division", "subdivision", "schedule", "clause"}:
                    process_container(child)
            return

        if tag == "clause":
            no = child_text(el, "no")
            he = child_text(el, "heading")
            if he:
                head = f"{no}. {he}" if no and not no.endswith(".") else f"{no} {he}".strip()
            else:
                head = no
            if head:
                add_para(head)

            for p in txt_from_blocks(el):
                add_body_para(p)

            for child in list(el):
                if local(getattr(child, "tag", "")) == "subclause":
                    sub_no = child_text(child, "no")
                    for p in txt_from_blocks(child):
                        if sub_no:
                            add_body_para(f"{sub_no} {p}")
                        else:
                            add_body_para(p)
            return

        for child in list(el):
            process_container(child)

    process_container(root)
    out = "\n".join(lines).strip() + "\n"
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    text = pdf_extract_text(io.BytesIO(pdf_bytes))
    return cleanup_qld_pdf_text(text or "")


def cleanup_qld_pdf_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    out = out.replace("\u00ad", "")
    out = re.sub(r"([A-Za-z])\-\n([a-z])", r"\1\2", out)

    # pdfminer can sometimes emit extremely long lines (hundreds of KB) with minimal
    # newline structure. When that happens, hard-wrap those lines at word boundaries.
    raw_lines = out.split("\n")
    if any(len(l) > 20000 for l in raw_lines):
        repaired: List[str] = []
        for l in raw_lines:
            if len(l) <= 20000:
                repaired.append(l)
                continue
            compact = re.sub(r"\s+", " ", l).strip()
            repaired.extend(
                textwrap.wrap(
                    compact,
                    width=120,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
            repaired.append("")
        out = "\n".join(repaired)

    noise_patterns = [
        re.compile(r"^page\s*\d+(\s*of\s*\d+)?\s*$", re.IGNORECASE),
        re.compile(r"^https?://\S+$", re.IGNORECASE),
        re.compile(r"\bqueensland\s+legislation\b", re.IGNORECASE),
        re.compile(r"\bqueensland\s+government\b", re.IGNORECASE),
        re.compile(r"\boffice\s+of\s+the\s+queensland\s+parliamentary\s+counsel\b", re.IGNORECASE),
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


def wrap_extremely_long_lines(text: str, *, width: int = 120, trigger_len: int = 20000) -> str:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines_in = raw.split("\n")
    if not any(len(l) > trigger_len for l in lines_in):
        return raw

    wrapped: List[str] = []
    for l in lines_in:
        if len(l) <= trigger_len:
            wrapped.append(l)
            continue
        compact = re.sub(r"\s+", " ", l).strip()
        parts = textwrap.wrap(
            compact,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        wrapped.extend(parts if parts else [""])
    return "\n".join(wrapped)


def looks_like_rtf_bytes(content: bytes) -> bool:
    head = (content or b"").lstrip()[:20]
    return head.startswith(b"{\\rtf") or head.startswith(b"{\\urtf")


def rtf_bytes_to_text(rtf_bytes: bytes) -> str:
    # Best-effort: many sites serve RTF with inconsistent encodings.
    last_exc: Optional[Exception] = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            rtf_str = (rtf_bytes or b"").decode(enc, errors="replace")
            if striprtf_to_text is not None:
                text = striprtf_to_text(rtf_str)
            else:
                # Fallback: return decoded RTF (may include markup)
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


JURIS_ABBREV = "QLD"


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
    title = (row.get("citation") or "").strip()
    body = _strip_leading_title(text, title)
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


def _extract_qld_as_at_date_from_html(html: str) -> Optional[str]:
    """Queensland exposes a dated inforce link for the current consolidated view.

    Example observed in page source:
      /view/whole/html/inforce/2024-02-01/act-1954-003
    """

    m = re.search(r"/view/(?:whole/)?html/inforce/(20\d{2}-\d{2}-\d{2})/", html)
    if m:
        return m.group(1)

    # Secondary signal: PublicationDate embedded in related/search links.
    m = re.search(r"PublicationDate%3D(20\d{2})(\d{2})(\d{2})\d{6}", html)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    return None


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
        if kind == "rtf":
            if abs_url.lower().endswith(".rtf") or "rtf" in href.lower():
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
    resp = client.get(url, headers=_conditional_headers_for_version_id(current_version_id))
    if resp.status_code == 304 and current_version_id:
        return current_version_id, resp
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if "html" in ctype:
        as_at = _extract_qld_as_at_date_from_html(resp.text)
        if as_at:
            return as_at, resp

        page_version = extract_date_like_version_from_html(resp.text)
        if page_version:
            return page_version, resp

    etag = (resp.headers.get("ETag") or "").strip()
    if etag:
        return f"etag:{etag}", resp

    last_mod = resp.headers.get("Last-Modified")
    if last_mod:
        return f"last_modified:{last_mod}", resp

    if "html" in ctype:
        normalized = " ".join(BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True).split())
        return stable_hash_version_id(normalized.encode("utf-8")), resp
    return stable_hash_version_id(resp.content), resp


def download_text_best_effort(client: HttpClient, resp: requests.Response) -> Tuple[str, str, str]:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "rtf" in ctype or resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(resp.content):
        return rtf_bytes_to_text(resp.content), resp.url, "rtf"
    if looks_like_xml(resp):
        if "legislation.qld.gov.au" in resp.url.lower() and "/view/whole/xml/" in resp.url.lower():
            return qld_whole_xml_to_text(resp.content), resp.url, "xml"
        return generic_xml_to_text(resp.content), resp.url, "xml"
    if "pdf" in ctype or resp.url.lower().endswith(".pdf"):
        return pdf_bytes_to_text(resp.content), resp.url, "pdf"

    if "html" in ctype:
        if "legislation.qld.gov.au" in resp.url.lower() and "/view/whole/html/" in resp.url.lower():
            try:
                derived_xml = resp.url.replace("/view/whole/html/", "/view/whole/xml/")
                xml_resp = client.get(derived_xml)
                if looks_like_xml(xml_resp):
                    return qld_whole_xml_to_text(xml_resp.content), xml_resp.url, "xml"
            except Exception:
                pass
            return qld_whole_html_to_text(resp.text), resp.url, "html"

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
    parser = argparse.ArgumentParser(description="Queensland legislation sync bot")
    parser.add_argument("--user-agent", default="AustralianTaxLawLibrarySyncBot/QLD")
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
        url = normalize_qld_url_to_whole_xml((row.get("url") or "").strip())
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
            text = wrap_extremely_long_lines(text)
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
