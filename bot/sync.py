from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# -----------------
# CLI + config
# -----------------


@dataclass(frozen=True)
class Config:
    user_agent: str
    sleep_seconds: float
    timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Single legislation sync bot (combined catalogue) -> raw .txt + prepared cleaned .txt. "
            "Designed to primarily follow the AustLII (classic) Downloads→Plain text flow."
        )
    )

    p.add_argument(
        "--catalogue-path",
        default=str(
            Path(__file__).resolve().parents[1]
            / "raw"
            / "data"
            / "Legislative_materials"
            / "legislation_catalogue.csv"
        ),
        help="Combined catalogue CSV",
    )
    p.add_argument(
        "--raw-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "raw"
            / "data"
            / "Legislative_materials"
        ),
        help="Where to write raw .txt files (no formatting changes)",
    )
    p.add_argument(
        "--prepared-dir",
        default=str(
            Path(__file__).resolve().parents[1]
            / "prepared"
            / "data"
            / "Legislative_materials"
        ),
        help="Where to copy and write cleaned .txt files",
    )

    p.add_argument("--user-agent", default="Australian-Tax-Law-Library/1.0")
    p.add_argument("--sleep-seconds", type=float, default=1.5)
    p.add_argument("--timeout-seconds", type=float, default=60.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--backoff-base-seconds", type=float, default=1.0)

    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--only-library-id",
        default=None,
        help="Only process the row with this library_id (useful for targeted re-scrapes)",
    )
    p.add_argument(
        "--repair-bad-raw",
        action="store_true",
        help=(
            "Scan existing raw .txt outputs; re-scrape only rows whose raw output is missing or looks like HTML/website boilerplate. "
            "This mode always re-downloads and rewrites the affected files."
        ),
    )
    p.add_argument(
        "--skip-already-scraped",
        action="store_true",
        help="Skip rows that already have a raw output file",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force rewrite even if version_id appears unchanged",
    )

    p.add_argument(
        "--rebuild-prepared-only",
        action="store_true",
        help=(
            "Rewrite prepared outputs from existing raw files only (no network). "
            "Useful after changing prepared-cleanup rules."
        ),
    )

    return p.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------
# CSV catalogue
# -----------------


def normalize_header_cell(cell: str) -> str:
    c = (cell or "").strip().replace("\ufeff", "")
    c = c.lower()
    c = re.sub(r"\s+", "_", c)
    return c


def is_valid_header_row(row: List[str]) -> bool:
    lowered = {normalize_header_cell(c) for c in row}
    return "library_id" in lowered and "url" in lowered


def dedupe_header(header: List[str]) -> tuple[List[str], Dict[str, int]]:
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
        self.header: List[str] = []
        self._index_by_name: Dict[str, int] = {}
        self.rows: List[Dict[str, str]] = []

    def load(self) -> None:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            raw_rows = list(reader)

        if not raw_rows:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        start_idx = 0
        if not is_valid_header_row(raw_rows[0]):
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
        rows: List[List[str]] = []
        rows.append(self.header)
        for r in self.rows:
            rows.append([r.get(col, "") for col in self.header])
        try:
            atomic_write_csv(self.csv_path, rows)
        except PermissionError as exc:
            # Common on Windows when the CSV is open in Excel.
            pending_path = self.csv_path.with_suffix(self.csv_path.suffix + ".pending")
            try:
                atomic_write_csv(pending_path, rows)
                print(
                    f"WARNING: Could not write catalogue (locked): {self.csv_path}. "
                    f"Wrote pending copy to: {pending_path}"
                )
                return
            except Exception:
                raise exc


def atomic_write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_attempts = 12
    for attempt in range(1, max_attempts + 1):
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

        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            winerror = getattr(exc, "winerror", None)
            if attempt >= max_attempts:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise
            if winerror not in (5, 32, None):
                raise
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            time.sleep(0.5 * attempt)


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


# -----------------
# HTTP + crawling
# -----------------


class HttpClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.user_agent})
        self._last_request_at_by_host: Dict[str, float] = {}

    def _sleep_if_needed(self, url: str) -> None:
        host = (urlparse(url).hostname or "").lower()
        min_delay = float(self.config.sleep_seconds)
        # Be polite to AustLII. Default is ON; disable explicitly with RESPECT_AUSTLII_CRAWL_DELAY=false.
        val = os.getenv("RESPECT_AUSTLII_CRAWL_DELAY")
        respect = True
        if val is not None:
            v = val.strip().lower()
            respect = v in {"1", "true", "t", "yes", "y"}

        if host.endswith("austlii.edu.au") and respect:
            crawl_delay = 30.0
            raw = os.getenv("AUSTLII_CRAWL_DELAY_SECONDS")
            if raw is not None and raw.strip() != "":
                try:
                    crawl_delay = float(raw.strip())
                except ValueError:
                    crawl_delay = 30.0
            min_delay = max(min_delay, crawl_delay)

        last_at = self._last_request_at_by_host.get(host)
        if last_at is None:
            return
        elapsed = time.time() - last_at
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)

    def get(
        self,
        url: str,
        *,
        allow_redirects: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, self.config.max_retries + 2):
            self._sleep_if_needed(url)
            host = (urlparse(url).hostname or "").lower()
            self._last_request_at_by_host[host] = time.time()
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


# -----------------
# AustLII download
# -----------------


def is_austlii_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host.endswith("austlii.edu.au")


def download_austlii_plain_text(client: HttpClient, txt_url: str, *, referer: Optional[str] = None) -> tuple[str, str, str]:
    headers = {"Referer": referer} if referer else None
    resp = client.get(txt_url, headers=headers)
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if "charset=" in ctype:
        text = resp.text or ""
    else:
        raw = resp.content or b""
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
    if text and not text.endswith("\n"):
        text += "\n"
    return text, (resp.url or txt_url), "text"


def download_austlii_from_base_page(client: HttpClient, base_url: str) -> tuple[str, str, str]:
    """Follow the AustLII UI path: base page → Download menu → Plain text (ASCII)."""

    base_resp = client.get(base_url)
    base_html = base_resp.text or ""
    base_canonical = base_resp.url or base_url

    soup = BeautifulSoup(base_html, "html.parser")
    download_href: Optional[str] = None
    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        label = " ".join((a.get_text(" ", strip=True) or "").split())
        if not href:
            continue
        low_href = href.lower()
        low_label = label.lower()
        if low_label == "download" or "download.cgi" in low_href:
            download_href = href
            break

    if not download_href:
        raise RuntimeError("AustLII base page did not contain a Download link")

    downloads_url = urljoin(base_canonical, download_href)
    downloads_resp = client.get(downloads_url, headers={"Referer": base_canonical})
    downloads_html = downloads_resp.text or ""
    downloads_canonical = downloads_resp.url or downloads_url

    dsoup = BeautifulSoup(downloads_html, "html.parser")
    plain_href: Optional[str] = None
    txt_candidates: list[str] = []
    for a in dsoup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        label = " ".join((a.get_text(" ", strip=True) or "").split())
        low_label = label.lower()
        low_href = href.lower()
        if "plain text" in low_label and "ascii" in low_label:
            plain_href = href
            break
        if low_href.endswith(".txt"):
            txt_candidates.append(href)

    if not plain_href and txt_candidates:
        plain_href = txt_candidates[0]
    if not plain_href:
        raise RuntimeError("AustLII download menu did not contain a Plain text (ASCII) link")

    plain_url = urljoin(downloads_canonical, plain_href)

    # Some AustLII pages include odd doubled paths; just follow what they provide.
    if plain_url.lower().endswith(".txt"):
        text, final_url, fmt = download_austlii_plain_text(client, plain_url, referer=downloads_canonical)
        return text, final_url, fmt

    # Otherwise, treat it as an intermediate page and find the .txt.
    plain_resp = client.get(plain_url, headers={"Referer": downloads_canonical})
    psoup = BeautifulSoup(plain_resp.text or "", "html.parser")
    txt_href2: Optional[str] = None
    for a in psoup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if href.lower().endswith(".txt"):
            txt_href2 = href
            break
    if not txt_href2:
        raise RuntimeError("AustLII plain-text page did not contain a .txt link")
    txt_url = urljoin((plain_resp.url or plain_url), txt_href2)
    text, final_url, fmt = download_austlii_plain_text(client, txt_url, referer=(plain_resp.url or plain_url))
    return text, final_url, fmt


def find_austlii_downloads_url(client: HttpClient, base_url: str) -> str:
    resp = client.get(base_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # Common pattern on classic pages: /cgi-bin/download.cgi/... (may be gone, but keep as a hint).
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "download.cgi" in href:
            return urljoin(resp.url, href)

    # Prefer explicit "Downloads" link.
    for a in soup.find_all("a"):
        text = (a.get_text(" ", strip=True) or "").strip().lower()
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "downloads" in text:
            return urljoin(resp.url, href)

    # Fallback: heuristic common pattern.
    return resp.url.rstrip("/") + "/download"


def find_austlii_plain_text_url(client: HttpClient, downloads_url: str) -> str:
    resp = client.get(downloads_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    candidates: list[str] = []
    for a in soup.find_all("a"):
        label = (a.get_text(" ", strip=True) or "").strip().lower()
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if "plain text" in label and "ascii" in label:
            return urljoin(resp.url, href)
        if "plain text" in label:
            candidates.append(urljoin(resp.url, href))

    if candidates:
        return candidates[0]

    raise RuntimeError(f"Could not find plain text link on AustLII downloads page: {downloads_url}")


def looks_like_legislation_gov_au_shell_html(html: str) -> bool:
    s = (html or "")
    if not s:
        return False
    # The current site is an Angular SPA; the initial HTML is mostly a shell.
    return (
        "<base href=\"/\">" in s
        and "<app-root" in s
        and "document" not in s.lower()  # avoid false positives on real text pages
    )


def extract_visible_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # Prefer content-like containers when available.
    candidates = [
        soup.select_one("pre"),
        soup.select_one("main"),
        soup.select_one("article"),
        soup.select_one("div#content"),
        soup.select_one("div#page"),
        soup.select_one("body"),
    ]
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return soup.get_text("\n", strip=False)

    best = max(candidates, key=lambda el: len(el.get_text("\n", strip=False)))
    return best.get_text("\n", strip=False)


def is_probably_austlii_toc(text: str) -> bool:
    s = (text or "")
    if not s:
        return False
    # Many AustLII consolidated act landing pages are TOC-only.
    if "TABLE OF PROVISIONS" in s.upper() or "TABLE OF CONTENTS" in s.upper():
        # If there is almost no section body text, treat as TOC.
        # (Heuristic: not many long lines, and few occurrences of 'section' content words.)
        return True
    return False


def collect_austlii_section_urls(*, base_url: str, html: str, max_urls: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    seen: set[str] = set()

    parsed = urlparse(base_url)
    base_dir = base_url
    if not base_dir.endswith("/"):
        base_dir = base_dir.rsplit("/", 1)[0] + "/"
    base_prefix_path = urlparse(base_dir).path
    if not base_prefix_path.endswith("/"):
        base_prefix_path += "/"

    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if href.startswith("#"):
            continue
        if "sinosrch.pl" in href:
            continue
        if href.lower().startswith("mailto:"):
            continue

        abs_url = urljoin(base_dir, href)
        u = urlparse(abs_url)
        if (u.hostname or "").lower() != (parsed.hostname or "").lower():
            continue

        # Keep URLs within the same act directory.
        if not u.path.startswith(base_prefix_path):
            continue

        name = Path(u.path).name.lower()
        if not name:
            continue

        if not (
            name.endswith(".html")
            or re.match(r"^s\d+[a-z]*$", name)
            or re.match(r"^s\d+[a-z]*\.html$", name)
            or name in {"longtitle", "longtitle.html", "notes", "notes.html"}
        ):
            continue

        norm = u._replace(fragment="").geturl()
        if norm in seen:
            continue
        seen.add(norm)
        urls.append(norm)
        if len(urls) >= max_urls:
            break

    return urls


def download_austlii_text(client: HttpClient, url: str) -> tuple[str, str]:
    """Attempt to get full text from an AustLII consolidated-act landing page."""
    resp = client.get(url)
    base_text = extract_visible_text_from_html(resp.text)

    # If it looks like a TOC page, follow section links and concatenate.
    if is_probably_austlii_toc(base_text):
        max_pages = int(os.getenv("AUSTLII_MAX_PAGES", "250"))
        section_urls = collect_austlii_section_urls(base_url=resp.url, html=resp.text, max_urls=max_pages)
        if os.getenv("DEBUG_AUSTLII", "").strip() in {"1", "true", "yes", "y"}:
            print(f"[DEBUG_AUSTLII] toc_url={resp.url}")
            print(f"[DEBUG_AUSTLII] found_section_urls={len(section_urls)} max={max_pages}")
            if section_urls:
                print(f"[DEBUG_AUSTLII] first_section_url={section_urls[0]}")
        if section_urls:
            parts: list[str] = []
            for section_url in section_urls:
                _, sec_text = download_text(client, section_url)
                parts.append(normalize_newlines(sec_text).strip())
            combined = "\n\n".join([p for p in parts if p])
            return resp.url, combined

    return resp.url, base_text


def download_text(client: HttpClient, url: str) -> tuple[str, str]:
    """Return (download_url, text)."""
    resp = client.get(url)
    ct = (resp.headers.get("content-type") or "").lower()

    # Some sites return text/plain with a charset, others HTML with preformatted body.
    if "text/plain" in ct or resp.url.lower().endswith(".txt"):
        return resp.url, resp.text

    return resp.url, extract_visible_text_from_html(resp.text)


def download_text_with_raw_html_check(client: HttpClient, url: str) -> tuple[str, str, str]:
    """Return (final_url, extracted_text, raw_html_or_text)."""
    resp = client.get(url)
    ct = (resp.headers.get("content-type") or "").lower()
    if "text/plain" in ct or resp.url.lower().endswith(".txt"):
        return resp.url, resp.text, resp.text
    return resp.url, extract_visible_text_from_html(resp.text), resp.text


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def normalize_newlines(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


# -----------------
# Prepared cleanup (Commonwealth-style)
# -----------------


# Some Commonwealth acts have long letter suffixes (e.g. 8AAZAA), so allow more.
_AUSTLII_SECT_DOT_RE = re.compile(r"^\s*-\s*SECT\s+(\d{1,5})\.(\d{1,5})([A-Za-z]{0,12})?\s*$")
_AUSTLII_SECT_SIMPLE_RE = re.compile(r"^\s*-\s*SECT\s+(\d{1,5}[A-Za-z]{0,12})\s*$")
_AUSTLII_INLINE_SECT_DOT_RE = re.compile(r"\bSECT\s+(\d{1,5})\.(\d{1,5})([A-Za-z]{0,12})?\b")
_AUSTLII_INLINE_SECT_SIMPLE_RE = re.compile(r"\bSECT\s+(\d{1,5}[A-Za-z]{0,12})\b")

# Regulations commonly use REG markers in AustLII dumps.
_AUSTLII_REG_DOT_RE = re.compile(r"^\s*-\s*REG\s+(\d{1,5})\.(\d{1,5})([A-Za-z]{0,12})?\s*$", re.IGNORECASE)
_AUSTLII_REG_SIMPLE_RE = re.compile(r"^\s*-\s*REG\s+(\d{1,5}[A-Za-z]{0,12})\s*$", re.IGNORECASE)
_AUSTLII_INLINE_REG_DOT_RE = re.compile(r"\bREG\s+(\d{1,5})\.(\d{1,5})([A-Za-z]{0,12})?\b", re.IGNORECASE)
_AUSTLII_INLINE_REG_SIMPLE_RE = re.compile(r"\bREG\s+(\d{1,5}[A-Za-z]{0,12})\b", re.IGNORECASE)


def _rewrite_austlii_sect_lines(text: str) -> str:
    lines = normalize_newlines(text).split("\n")
    out: list[str] = []
    for line in lines:
        s = (line or "").rstrip("\n")
        m = _AUSTLII_SECT_DOT_RE.match(s)
        if m:
            a, b, suffix = m.group(1), m.group(2), m.group(3) or ""
            out.append(f"Section {a}-{b}{suffix}")
            continue
        m2 = _AUSTLII_SECT_SIMPLE_RE.match(s)
        if m2:
            out.append(f"Section {m2.group(1)}")
            continue
        m3 = _AUSTLII_REG_DOT_RE.match(s)
        if m3:
            a, b, suffix = m3.group(1), m3.group(2), m3.group(3) or ""
            out.append(f"Regulation {a}-{b}{suffix}")
            continue
        m4 = _AUSTLII_REG_SIMPLE_RE.match(s)
        if m4:
            out.append(f"Regulation {m4.group(1)}")
            continue
        out.append(s)
    return "\n".join(out).strip() + "\n"


def _rewrite_austlii_sect_inline(text: str) -> str:
    out = _AUSTLII_INLINE_SECT_DOT_RE.sub(lambda m: f"Section {m.group(1)}-{m.group(2)}{m.group(3) or ''}", text)
    out = _AUSTLII_INLINE_SECT_SIMPLE_RE.sub(lambda m: f"Section {m.group(1)}", out)
    out = _AUSTLII_INLINE_REG_DOT_RE.sub(lambda m: f"Regulation {m.group(1)}-{m.group(2)}{m.group(3) or ''}", out)
    out = _AUSTLII_INLINE_REG_SIMPLE_RE.sub(lambda m: f"Regulation {m.group(1)}", out)
    return out


def _strip_leading_toc(text: str) -> str:
    """Best-effort: if the document begins with a TOC-like block, drop it."""
    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    # If we see many TOC markers early on, skip until first real section.
    sect_like = 0
    for i in range(min(len(lines), 250)):
        if (
            _AUSTLII_SECT_DOT_RE.match(lines[i])
            or _AUSTLII_SECT_SIMPLE_RE.match(lines[i])
            or _AUSTLII_REG_DOT_RE.match(lines[i] or "")
            or _AUSTLII_REG_SIMPLE_RE.match(lines[i] or "")
        ):
            sect_like += 1
    if sect_like < 5:
        return text

    heading_re = re.compile(r"^(?:Section|Regulation)\s+\d", re.IGNORECASE)
    for i in range(min(len(lines), 800)):
        if heading_re.match((lines[i] or "").lstrip()):
            return "\n".join(lines[i:]).strip() + "\n"
    return text


def _strip_internal_tables_of_sections(text: str) -> str:
    """Remove internal 'Table of sections/subdivisions' blocks (best-effort)."""
    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    triggers = {
        "table of sections",
        "table of provisions",
        "table of contents",
        "contents",
    }

    def is_trigger(line: str) -> bool:
        s = (line or "").strip().lower()
        return any(t in s for t in triggers)

    def looks_like_section_heading(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        if s.startswith("Section ") or s.startswith("Regulation "):
            return True
        # Common non-hyphenated headings: "1 Short title" or "1A ..."
        if re.match(r"^\d+[A-Za-z]*\s+\S+", s):
            return True
        return False

    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if is_trigger(cur):
            # Skip until it looks like we are back in body.
            j = i + 1
            while j < len(lines) and not looks_like_section_heading(lines[j]):
                j += 1
            if j < len(lines):
                i = j
                continue
        out.append(cur)
        i += 1

    return "\n".join(out).strip() + "\n"


# Many AustLII text dumps render TOCs as lines ending with a page number, often
# separated by tabs or wide spacing, e.g.:
#   Regulation 1 Name\t1
#   1     Short title      1
_PAGINATED_TOC_LINE_RE = re.compile(r"^\s*\S.*(?:\t|\s{2,})\d+\s*$")
_PAGINATED_TOC_PART_RE = re.compile(r"^\s*Part\s+\d+\b", re.IGNORECASE)
_PAGINATED_TOC_DIV_RE = re.compile(r"^\s*Division\s+\d+\b", re.IGNORECASE)


def _strip_leading_paginated_toc(text: str) -> str:
    """Strip a leading TOC that uses trailing page numbers (common in some states/territories).

    Example (NT):
      1     Short title      1
      2     Commencement     1

    We keep from the first plausible body heading onward, where the section headings
    no longer end with page numbers.
    """

    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    scan = lines[:300]
    paginated = 0
    structure = 0
    for line in scan:
        if _PAGINATED_TOC_LINE_RE.match(line or ""):
            paginated += 1
        if _PAGINATED_TOC_PART_RE.match(line or "") or _PAGINATED_TOC_DIV_RE.match(line or ""):
            structure += 1

    # Require a meaningful number of TOC-like lines to avoid false positives.
    if paginated < 20:
        return text

    # If the TOC includes an ENDNOTES marker, it's a strong signal the real body follows.
    for i in range(min(len(lines), 2500)):
        if (lines[i] or "").strip().casefold() == "endnotes":
            return "\n".join(lines[i:]).strip() + "\n"

    # Regulations often repeat structural headings after the TOC without page numbers.
    body_structural = re.compile(r"^\s*(?:Chapter\s+\d+\b|Part\s+\d+\b)", re.IGNORECASE)
    for i in range(min(len(lines), 5000)):
        s = (lines[i] or "").rstrip()
        if not s:
            continue
        if _PAGINATED_TOC_LINE_RE.match(s):
            continue
        if body_structural.match(s):
            return "\n".join(lines[i:]).strip() + "\n"

    # Otherwise, find the first heading-like line that does NOT end with a page number,
    # and require nearby body-like text so we don't cut on wrapped TOC lines.
    heading_like = re.compile(r"^\s*\d+[A-Za-z]*\s+\S+.*\S\s*$")
    body_line_re = re.compile(r"^\s{2,}\S+.*\S\s*$")
    for i in range(min(len(lines), 5000)):
        s = (lines[i] or "").rstrip()
        if not s:
            continue
        if _PAGINATED_TOC_LINE_RE.match(s):
            continue
        if heading_like.match(s) and not s.lower().startswith("part ") and not s.lower().startswith("division "):
            # Look ahead for indented body-like lines (common in AustLII text).
            lookahead = lines[i + 1 : i + 12]
            if any(body_line_re.match((x or "").rstrip()) for x in lookahead):
                return "\n".join(lines[i:]).strip() + "\n"

    return text


def _strip_early_paginated_toc_block(text: str) -> str:
    """Remove an early TOC block that uses trailing page numbers.

    Unlike _strip_leading_paginated_toc(), this attempts to preserve any preamble
    before the TOC by removing only the detected TOC block.
    """

    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    # Prefer explicit TOC triggers when present.
    toc_start: int | None = None
    toc_triggers = {"contents", "table of contents", "table of provisions"}
    for i in range(min(len(lines), 700)):
        if (lines[i] or "").strip().casefold() in toc_triggers:
            toc_start = i + 1
            break

    # Otherwise find a plausible TOC start within the first ~500 lines.
    if toc_start is None:
        for i in range(min(len(lines), 500)):
            if _PAGINATED_TOC_LINE_RE.match(lines[i] or ""):
                # Require a dense run of paginated lines after this point.
                window = lines[i : i + 250]
                if sum(1 for x in window if _PAGINATED_TOC_LINE_RE.match(x or "")) >= 20:
                    toc_start = i
                    break

    if toc_start is None:
        return text

    # Find a plausible end marker.
    body_structural = re.compile(r"^\s*(?:Chapter\s+\d+\b|Part\s+\d+\b)", re.IGNORECASE)
    heading_like = re.compile(r"^\s*\d[0-9A-Za-z\-\.]*\s+\S+.*\S\s*$")
    body_line_re = re.compile(r"^\s{2,}\S+.*\S\s*$")

    toc_end: int | None = None
    for j in range(toc_start + 1, min(len(lines), 8000)):
        s = (lines[j] or "").rstrip()
        if not s:
            continue
        if (s or "").strip().casefold() == "endnotes":
            toc_end = j
            break
        if _PAGINATED_TOC_LINE_RE.match(s):
            continue
        if body_structural.match(s):
            toc_end = j
            break
        if heading_like.match(s):
            lookahead = lines[j + 1 : j + 12]
            if any(body_line_re.match((x or "").rstrip()) for x in lookahead):
                toc_end = j
                break

    if toc_end is None:
        return text

    kept = [*lines[:toc_start], *lines[toc_end:]]
    return "\n".join(kept).strip() + "\n"


def _strip_paginated_lines_right_before_body(text: str) -> str:
    """Drop a small paginated-TOC remnant immediately before body start.

    Some documents have a paginated TOC block and then repeat a structural header
    (e.g. 'Chapter 1-...') for the actual body. After block removal, a few TOC lines
    can remain right before that repeated header.
    """

    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    body_structural = re.compile(r"^\s*(?:Chapter\s+\d+\b|Part\s+\d+\b)", re.IGNORECASE)

    body_start: int | None = None
    for i in range(min(len(lines), 1200)):
        s = (lines[i] or "").rstrip()
        if not s:
            continue
        if _PAGINATED_TOC_LINE_RE.match(s):
            continue
        if body_structural.match(s):
            body_start = i
            break

    if body_start is None or body_start == 0:
        return text

    window_start = max(0, body_start - 120)
    to_remove: set[int] = set()

    for idx in range(window_start, body_start):
        s = (lines[idx] or "").rstrip()
        if not s:
            continue
        if _PAGINATED_TOC_LINE_RE.match(s):
            to_remove.add(idx)
            # Often the descriptive portion wraps onto the prior line.
            if idx - 1 >= window_start:
                prev = (lines[idx - 1] or "").rstrip()
                if prev and (not _PAGINATED_TOC_LINE_RE.match(prev)) and (not body_structural.match(prev)):
                    to_remove.add(idx - 1)

    if not to_remove:
        return text

    kept = [line for i, line in enumerate(lines) if i not in to_remove]
    return "\n".join(kept).strip() + "\n"


def _join_wrapped_bare_heading_refs(text: str, *, heading_word: str) -> str:
    """Join lines like 'Section 60.' back into the previous line when they appear
    to be a wrapped reference mid-paragraph rather than a real heading.

    This prevents false section boundaries in definitions/paragraphs.
    """

    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    bare_ref = re.compile(rf"^\s*{re.escape(heading_word)}\s+\d+[A-Za-z]*(?:\.)?\s*$")
    out: list[str] = []
    for line in lines:
        s = (line or "").rstrip()
        if out and bare_ref.match(s):
            prev = (out[-1] or "").rstrip()
            if prev and not prev.strip().lower().startswith(heading_word.lower()):
                # If previous line looks like a sentence fragment, treat this as wrap.
                if not prev.rstrip().endswith((".", ":", ";")):
                    out[-1] = (prev + " " + s.strip()).rstrip()
                    continue
        out.append(s)

    return "\n".join(out).strip() + "\n"


def _collapse_whitespace(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _normalize_heading_text(heading: str, *, number: str) -> str:
    """Normalize extracted heading lines.

    - Removes duplicated leading number patterns like '5--Meaning' or '5 Meaning'
      when the number matches the section/regulation number.
    """

    h = _collapse_whitespace(heading)
    if not h:
        return ""

    # Remove patterns like '5--Meaning' / '5 Meaning' / '5. Meaning'
    m = re.match(r"^(?P<num>\d+[A-Za-z]*)\s*(?:\.|--|-)\s*(?P<rest>.*\S)$", h)
    if m and m.group("num") == number:
        return _collapse_whitespace(m.group("rest"))

    m2 = re.match(r"^(?P<num>\d+[A-Za-z]*)\s+(?P<rest>.*\S)$", h)
    if m2 and m2.group("num") == number:
        return _collapse_whitespace(m2.group("rest"))

    return h


def _merge_split_headings(text: str, *, heading_word: str) -> str:
    """Merge split headings into a single line.

    Handles patterns like:
      Section 1\nName of Act\n\n1 Name of Act\nThis Act...
      Section 5\n5--Meaning\nof non-reviewable ...\n\n...
    """

    lines = normalize_newlines(text).split("\n")
    out: list[str] = []

    only_number_hdr = re.compile(rf"^\s*{re.escape(heading_word)}\s+(?P<num>\S+)\s*$", re.IGNORECASE)
    # Only convert flush-left numeric headings (avoid indented lists like examples).
    numbered_hdr = re.compile(r"^(?P<num>\d[0-9A-Za-z\-\.]*)\s*(?:\.)?\s+(?P<head>\S.*\S)\s*$")

    i = 0
    while i < len(lines):
        line = lines[i] or ""
        m = only_number_hdr.match(line)
        if m:
            num = (m.group("num") or "").strip()

            # Collect up to 3 non-empty heading lines (stop before a blank line).
            heading_parts: list[str] = []
            j = i + 1
            while j < len(lines) and len(heading_parts) < 3:
                nxt = (lines[j] or "").strip()
                if not nxt:
                    break
                # Stop if it looks like body text starting (e.g. '(1) ...')
                if nxt.startswith("(") or re.match(r"^\(?\d+\)\s", nxt):
                    break
                heading_parts.append(nxt)
                j += 1

            heading = _normalize_heading_text(" ".join(heading_parts), number=num) if heading_parts else ""
            merged = f"{heading_word} {num}".strip()
            if heading:
                merged = f"{merged} {heading}".strip()
            out.append(merged)

            # Skip consumed heading lines (but keep the blank line, if any)
            i = j
            continue

        # Add missing heading_word for numeric headings like '1. Short title' or '1     Short title'
        m2 = numbered_hdr.match(line)
        if m2 and not re.match(rf"^\s*{re.escape(heading_word)}\b", line, re.IGNORECASE):
            num = (m2.group("num") or "").strip()
            head = (m2.group("head") or "").strip()
            out.append(f"{heading_word} {num} {head}".strip())
            i += 1
            continue

        out.append(line.rstrip())
        i += 1

    # Remove duplicated numeric heading lines that immediately follow a merged header.
    cleaned: list[str] = []
    merged_re = re.compile(rf"^\s*{re.escape(heading_word)}\s+(?P<num>\S+)\s+(?P<head>\S.*\S)\s*$", re.IGNORECASE)
    for k, line in enumerate(out):
        m3 = merged_re.match(line or "")
        if m3 and k + 1 < len(out):
            nxt = _collapse_whitespace(out[k + 1])
            num = (m3.group("num") or "").strip()
            head = _collapse_whitespace(m3.group("head") or "")
            dup1 = _collapse_whitespace(f"{num} {head}")
            dup2 = _collapse_whitespace(f"{num}--{head}")
            if nxt in {dup1, dup2}:
                cleaned.append(line)
                # Skip the duplicate line
                out[k + 1] = ""
                continue
        if (line or "") != "":
            cleaned.append(line)

    # Drop consecutive duplicate section/regulation headers.
    deduped: list[str] = []
    prev_key: str | None = None
    for line in cleaned:
        key = _collapse_whitespace(line).casefold()
        if prev_key is not None and key and key == prev_key:
            continue
        deduped.append(line)
        prev_key = key

    return "\n".join([l for l in deduped if l is not None]).strip() + "\n"


def _strip_citation_lines_before_headings(
    text: str,
    *,
    citation: str | None,
    heading_word: str,
    allow_removing_first_nonempty_line: bool,
) -> str:
    """Strip redundant citation/title lines in the body immediately before headings.

    Rule (as requested): only strip if the citation is:
    - on a line of its own (line.strip() == citation.strip())
    - not the first non-empty line (unless allow_removing_first_nonempty_line=True)
    - immediately before a '{heading_word} ...' line (skipping blank lines)
    """

    normalized = normalize_newlines(text)
    citation_s = (citation or "").strip()
    if not normalized or not citation_s:
        return normalized

    citation_cf = citation_s.casefold()
    lines = normalized.split("\n")

    heading_re = re.compile(rf"^\s*{re.escape(heading_word)}\s+\S+", re.IGNORECASE)

    first_nonempty_idx: int | None = None
    for k, line in enumerate(lines):
        if (line or "").strip():
            first_nonempty_idx = k
            break

    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = (line or "").strip()
        if s and s.casefold() == citation_cf:
            if (not allow_removing_first_nonempty_line) and (first_nonempty_idx is not None) and i == first_nonempty_idx:
                out.append(line)
                i += 1
                continue

            # If the next non-empty line is a Section heading, drop this citation line.
            j = i + 1
            while j < len(lines) and not (lines[j] or "").strip():
                j += 1
            if j < len(lines) and heading_re.match(lines[j] or ""):
                i += 1
                continue

        out.append(line)
        i += 1

    return normalize_newlines("\n".join(out))


def _strip_table_of_provisions(text: str) -> str:
    lines = normalize_newlines(text).split("\n")
    if not lines:
        return text

    toc_idx = None
    for i, line in enumerate(lines[:800]):
        if "table of provisions" in (line or "").strip().lower() or "table of contents" in (line or "").strip().lower():
            toc_idx = i
            break
    if toc_idx is None:
        return text

    # Otherwise, keep from the first real body marker onwards.
    # Important: earlier cleanup may rewrite '- SECT 1' -> 'Section 1', so we must
    # look for both 'SECT' and 'Section'. Also prefer structural headers like
    # 'SCHEDULE 1 ...' or '- LONG TITLE' when present.
    # Require uppercase SCHEDULE headings to avoid false positives like
    # 'Schedule 1, Chapter ...' references.
    schedule_hdr = re.compile(r"^\s*SCHEDULE\s+\d")
    long_title_hdr = re.compile(r"^\s*-\s*LONG\s+TITLE\s*$", re.IGNORECASE)
    # Important: require flush-left headings. Indented lines like
    # '   section 52C.' can appear mid-paragraph and must not be treated as
    # a body marker.
    section_hdr = re.compile(r"^(?:Section|Regulation)\s+\S+", re.IGNORECASE)
    sect_hdr = re.compile(r"^\s*(?:-\s*)?SECT\s+\S+", re.IGNORECASE)

    first_idx: int | None = None
    for j in range(toc_idx + 1, min(len(lines), 5000)):
        line = lines[j] or ""
        if schedule_hdr.match(line) or long_title_hdr.match(line) or section_hdr.match(line) or sect_hdr.match(line):
            first_idx = j
            break

    if first_idx is not None:
        return "\n".join(lines[first_idx:]).strip() + "\n"

    return text


def _strip_austlii_nav_blocks(text: str) -> str:
    lines = normalize_newlines(text).split("\n")
    nav_words = {
        "index",
        "table",
        "search",
        "search this act",
        "notes",
        "noteup",
        "download",
        "help",
        "previous",
        "next",
        "austlii:",
        "copyright policy",
        "disclaimers",
        "privacy policy",
        "feedback",
        "commonwealth consolidated acts",
    }
    out: list[str] = []
    for line in lines:
        s = (line or "").strip()
        if not s:
            out.append(line)
            continue
        if s in {"[", "]", "[ ]"}:
            continue
        if s == "|":
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        if s.lower() in nav_words:
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def _header_block_for_row(row: Dict[str, str]) -> str:
    parts: list[str] = []
    citation = (row.get("citation") or row.get("title") or "").strip()
    if citation:
        parts.append(citation)

    jurisdiction = (row.get("jurisdiction") or "").strip()
    if jurisdiction:
        parts.append(f"Jurisdiction: {jurisdiction}")

    if row.get("url"):
        parts.append(f"Source: {row.get('url')}")

    if row.get("government_register_url"):
        parts.append(f"Register: {row.get('government_register_url')}")

    if row.get("content_url"):
        parts.append(f"Downloaded: {row.get('content_url')}")

    parts.append(f"When scraped: {row.get('when_scraped') or ''}".strip())

    return "\n".join([p for p in parts if p])


def finalize_prepared_text(row: Dict[str, str], raw_text: str) -> str:
    body = normalize_newlines(raw_text)
    body = _rewrite_austlii_sect_lines(body)
    body = _rewrite_austlii_sect_inline(body)
    body = _strip_austlii_nav_blocks(body)
    # Intentionally keep TOC/TOS blocks in prepared outputs.
    # Sidecar generation (and therefore embeddings) will ignore TOC/TOS when chunking.

    doc_type = (row.get("type") or "").strip().casefold()
    heading_word = "Regulation" if ("secondary" in doc_type or "regulation" in doc_type or "rules" in doc_type) else "Section"
    body = _merge_split_headings(body, heading_word=heading_word)
    body = _join_wrapped_bare_heading_refs(body, heading_word=heading_word)

    citation = (row.get("citation") or row.get("title") or "").strip() or None
    # Prepared outputs already include the citation in the header block, so any
    # repeated citation line inside the body is redundant and safe to strip.
    body = _strip_citation_lines_before_headings(
        body,
        citation=citation,
        heading_word=heading_word,
        allow_removing_first_nonempty_line=True,
    )
    header = _header_block_for_row(row)
    return f"{header}\n\n{body.strip()}\n"


# -----------------
# Filenames
# -----------------


def sanitize_filename(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "document"
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(".")
    return s[:180]


def output_filename_for_row(row: Dict[str, str]) -> str:
    library_id = (row.get("library_id") or "").strip()
    citation = (row.get("citation") or row.get("title") or "").strip()
    base = citation
    if library_id:
        base = f"{citation} ({library_id})" if citation else library_id
    return sanitize_filename(base) + ".txt"


def raw_looks_like_htmlish(text: str) -> bool:
    head = (text or "")[:8000].lower()
    if not head.strip():
        return True
    markers = [
        "<!doctype html",
        "<html",
        "<head",
        "<body",
        "skip to content",
        "skip to main content",
        "site header",
        "toggle navigation",
        "site navigation",
        "south australian legislation",
        "legislation.sa.gov.au",
        "western australian legislation",
        "return to cart",
    ]
    return any(m in head for m in markers)


def read_text_head(path: Path, max_chars: int = 12000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception:
        return ""


# -----------------
# Main
# -----------------


def main() -> int:
    args = parse_args()

    cfg = Config(
        user_agent=args.user_agent,
        sleep_seconds=float(args.sleep_seconds),
        timeout_seconds=float(args.timeout_seconds),
        max_retries=int(args.max_retries),
        backoff_base_seconds=float(args.backoff_base_seconds),
    )

    catalogue_path = Path(args.catalogue_path).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    prepared_dir = Path(args.prepared_dir).resolve()

    raw_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    cat = Catalogue(catalogue_path)
    cat.load()

    client = HttpClient(cfg)

    filtered_rows = cat.rows
    if args.only_library_id is not None and str(args.only_library_id).strip() != "":
        target_id = str(args.only_library_id).strip()
        filtered_rows = [r for r in cat.rows if (r.get("library_id") or "").strip() == target_id]

    if args.repair_bad_raw:
        repair_rows: list[Dict[str, str]] = []
        for r in filtered_rows:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            raw_path = raw_dir / output_filename_for_row(r)
            if not raw_path.exists():
                repair_rows.append(r)
                continue
            head = read_text_head(raw_path)
            if raw_looks_like_htmlish(head):
                repair_rows.append(r)
        filtered_rows = repair_rows

    total_rows = len(filtered_rows)
    if args.limit is not None:
        total_rows = min(total_rows, int(args.limit))
    print(
        f"Loaded {len(cat.rows)} catalogue rows. Starting scrape (only_library_id={args.only_library_id}, limit={args.limit})."
    )

    processed = 0
    try:
        for row in filtered_rows:
            if args.limit is not None and processed >= int(args.limit):
                break

            library_id = (row.get("library_id") or "").strip()
            url = (row.get("url") or "").strip()
            if not url:
                continue

            out_name = output_filename_for_row(row)
            raw_path = raw_dir / out_name
            prepared_path = prepared_dir / out_name

            if args.skip_already_scraped and raw_path.exists():
                continue

            display_id = library_id or (row.get("citation") or row.get("title") or "").strip() or out_name
            print(f"[{processed + 1}/{total_rows}] {display_id}")

            if not args.rebuild_prepared_only:
                row["when_scraped"] = utc_now_iso()
                row["status"] = "running"
                row["error"] = ""
                cat.save()

            try:
                if args.rebuild_prepared_only:
                    if not raw_path.exists():
                        raise RuntimeError(f"Raw file missing for rebuild: {raw_path}")
                    raw_text = normalize_newlines(raw_path.read_text(encoding="utf-8", errors="replace"))
                else:
                    download_url = url
                    text = ""

                    # This bot intentionally only supports AustLII catalogue URLs.
                    # Do not use government_register_url or any HTML-to-text fallback.
                    if not is_austlii_url(url):
                        raise RuntimeError(f"Non-AustLII url not supported (refusing HTML scrape): {url}")

                    text, download_url, _fmt = download_austlii_from_base_page(client, url)
                    row["content_format"] = "text"
                    row["content_url"] = download_url
                    vid = f"austlii:sha256:{sha256_text(text)}"

                    if (
                        not args.force
                        and not args.repair_bad_raw
                        and row.get("version_id")
                        and row["version_id"].strip() == vid
                        and raw_path.exists()
                    ):
                        row["status"] = "skipped_unchanged"
                        row["last_successful_scrape"] = row["last_successful_scrape"] or utc_now_iso()
                        row["when_scraped"] = utc_now_iso()
                        cat.save()
                        processed += 1
                        continue

                    row["version_id"] = vid

                    raw_text = normalize_newlines(text)
                    atomic_write_text(raw_path, raw_text)

                # Copy raw -> prepared BEFORE any edits, then edit prepared only.
                atomic_write_text(prepared_path, raw_text)
                prepared_text = finalize_prepared_text(row, raw_text)
                atomic_write_text(prepared_path, prepared_text)

                if not args.rebuild_prepared_only:
                    row["status"] = "ok"
                    row["last_successful_scrape"] = utc_now_iso()
                    row["error"] = ""

            except Exception as exc:
                if not args.rebuild_prepared_only:
                    row["status"] = "error"
                    row["error"] = str(exc)[:500]
                else:
                    raise

            if not args.rebuild_prepared_only:
                cat.save()
            processed += 1
    except KeyboardInterrupt:
        try:
            cat.save()
        finally:
            print("Interrupted. Progress saved to catalogue.")
        return 130

    print(f"Done. Processed: {processed}")
    print(f"Raw outputs: {raw_dir}")
    print(f"Prepared outputs: {prepared_dir}")
    print(f"Catalogue: {catalogue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
