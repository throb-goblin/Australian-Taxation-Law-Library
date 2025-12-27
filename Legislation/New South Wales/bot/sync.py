from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
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
    "/view/html/compare/",  # explicitly disallowed by policy
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
    c = c.replace("\ufeff", "")  # BOM safety
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
            # Duplicate: ignore later occurrences (prefer the first occurrence)
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
            out = [r.get(col, "") for col in self.header]
            out_rows.append(out)

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

    # Only treat certain ancestors as meaningfully “nested” for de-duplication.
    # Container tags like div/table are common and should not cause us to drop content.
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
        if el.name in {"div"} and not el.get_text(strip=True):
            continue

        # Skip nested “real” blocks (e.g. <p> inside <li>) to reduce duplication,
        # but do not skip content just because it's inside a container <div>.
        if any(getattr(p, "name", None) in nested_block_ancestors for p in el.parents if p is not None):
            # Headings nested in other blocks are still useful; keep them.
            if not (el.name or "").startswith("h"):
                continue

        text = el.get_text(" ", strip=True)
        if not text:
            continue

        if el.name and el.name.startswith("h"):
            pieces.append(text)
            pieces.append("")
        else:
            pieces.append(text)
    # Normalize spacing
    out = "\n".join(pieces)
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    # pdfminer expects a file-like object
    text = pdf_extract_text(io.BytesIO(pdf_bytes))
    text = (text or "").strip() + "\n"
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


def nswh_exchange_xml_to_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)

    def local(tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    out: List[str] = []
    for el in root.iter():
        name = local(el.tag)
        if name not in {"head", "txt"}:
            continue
        text = " ".join(" ".join(el.itertext()).split())
        if not text:
            continue
        if name == "head":
            out.append(text)
            out.append("")
        else:
            out.append(text)

    text_out = "\n".join(out)
    text_out = re.sub(r"\n{3,}", "\n\n", text_out).strip() + "\n"
    return text_out


def extract_candidate_dates_from_nsw_html(html: str) -> List[datetime]:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    dates: List[datetime] = []

    # Pattern: "Current version for <DATE> to date"
    for m in re.finditer(r"Current version for\s+(.+?)\s+to\s+date", text, flags=re.IGNORECASE):
        try:
            dt = dateparser.parse(m.group(1), dayfirst=True)
            if dt:
                dates.append(dt)
        except Exception:
            pass

    # pointInTime=YYYY-MM-DD
    for m in re.finditer(r"pointInTime=(\d{4}-\d{2}-\d{2})", html, flags=re.IGNORECASE):
        try:
            dt = dateparser.parse(m.group(1))
            if dt:
                dates.append(dt)
        except Exception:
            pass

    # PublicationDate=YYYYMMDD
    for m in re.finditer(r"PublicationDate=(\d{8})", html, flags=re.IGNORECASE):
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
            dates.append(dt)
        except Exception:
            pass

    return dates


def stable_hash_version_id(content_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(content_bytes)
    return "sha256:" + h.hexdigest()


def guess_version_id_for_nsw(client: HttpClient, url: str) -> Tuple[str, requests.Response]:
    resp = client.get(url)
    dates = extract_candidate_dates_from_nsw_html(resp.text)
    if dates:
        latest = max(dates)
        return f"fmt:v2|{latest.date().isoformat()}", resp

    last_mod = resp.headers.get("Last-Modified")
    if last_mod:
        try:
            dt = dateparser.parse(last_mod)
            if dt:
                return f"fmt:v2|{dt.date().isoformat()}", resp
        except Exception:
            pass

    return f"fmt:v2|{stable_hash_version_id(resp.content)}", resp


def nsw_build_xml_candidates(url: str) -> List[str]:
    candidates: List[str] = []
    if "/view/html/" in url:
        candidates.append(url.replace("/view/html/", "/view/whole/xml/"))
        candidates.append(url.replace("/view/html/", "/view/xml/"))
    if "/view/whole/html/" in url:
        candidates.append(url.replace("/view/whole/html/", "/view/whole/xml/"))
        candidates.append(url.replace("/view/whole/html/", "/view/xml/"))
    if "/view/whole/xml/" in url or "/view/xml/" in url:
        candidates.append(url)

    # De-dupe preserving order
    seen = set()
    out: List[str] = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def looks_like_xml(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ctype:
        return True
    content = resp.content.lstrip()[:50]
    return content.startswith(b"<?xml") or content.startswith(b"<exdoc")


def nsw_find_export_xml_link(export_page_html: str, base_url: str) -> Optional[str]:
    soup = BeautifulSoup(export_page_html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "/export/xml/" not in href:
            continue
        links.append(urljoin(base_url, href))

    if not links:
        # Fallback: raw regex
        for m in re.finditer(r"(/export/xml/[^\"'\s>]+)", export_page_html):
            links.append(urljoin(base_url, m.group(1)))

    # Prefer not /lh
    non_lh = [u for u in links if "/lh" not in u]
    return (non_lh or links)[0] if links else None


def download_nsw_consolidated_text(
    client: HttpClient,
    url: str,
    prefetched_html_resp: Optional[requests.Response],
) -> Tuple[str, str, str]:
    # Prefer NSW exchange XML
    for candidate in nsw_build_xml_candidates(url):
        resp = client.get(candidate)
        if looks_like_xml(resp):
            return nswh_exchange_xml_to_text(resp.content), resp.url, "xml"

        # Export page: follow /export/xml/... link
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "html" in ctype:
            export_link = nsw_find_export_xml_link(resp.text, resp.url)
            if export_link:
                export_resp = client.get(export_link)
                if looks_like_xml(export_resp):
                    return nswh_exchange_xml_to_text(export_resp.content), export_resp.url, "xml"

    # Fallback to HTML
    html_resp = prefetched_html_resp or client.get(url)
    ctype = (html_resp.headers.get("Content-Type") or "").lower()
    if "rtf" in ctype or html_resp.url.lower().endswith(".rtf") or looks_like_rtf_bytes(html_resp.content):
        return rtf_bytes_to_text(html_resp.content), html_resp.url, "rtf"
    if "pdf" in ctype or html_resp.url.lower().endswith(".pdf"):
        return pdf_bytes_to_text(html_resp.content), html_resp.url, "pdf"
    return html_to_text_preserve_blocks(html_resp.text), html_resp.url, "html"


def safe_txt_filename(library_id: str) -> str:
    # Preserve library_id as much as possible but prevent path traversal and Windows-invalid separators.
    name = (library_id or "").strip()
    name = name.replace("/", "_").replace("\\", "_")
    return f"{name}.txt"


JURIS_ABBREV = "NSW"


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
    # Example: "3AALevy of land tax" -> "3AA Levy of land tax".
    # Avoid splitting labels like "3AA" into "3A A".
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="New South Wales legislation sync bot")
    parser.add_argument("--user-agent", default="AustralianTaxLawLibrarySyncBot/NSW")
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
    for i, row in enumerate(cat.rows):
        if args.limit and processed >= args.limit:
            break

        library_id = row.get("library_id", "").strip()
        url = row.get("url", "").strip()

        row["when_scraped"] = utc_now_iso()

        if "active" in row and not is_truthy_active(row.get("active", "")):
            row["status"] = "skipped_inactive"
            row["error"] = ""
            cat.save()
            processed += 1
            continue

        if not library_id or not url:
            row["status"] = "error"
            row["error"] = "Missing library_id or url"
            cat.save()
            processed += 1
            continue

        try:
            latest_version_id, prefetched_html = guess_version_id_for_nsw(client, url)
            current_version_id = (row.get("version_id") or "").strip()

            if (not args.force) and current_version_id == latest_version_id:
                row["status"] = "skipped_no_change"
                row["error"] = ""
                cat.save()
                processed += 1
                continue

            text, content_url, content_format = download_nsw_consolidated_text(client, url, prefetched_html)
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
