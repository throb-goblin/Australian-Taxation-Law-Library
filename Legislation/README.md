# Legislation Sync Bots (Plain-Text Offline Library)

This repo contains 9 separate jurisdiction-specific “sync bots”. Each bot:

- Reads its jurisdiction CSV catalogue in `data/*_legislation_catalogue.csv`
- Downloads the latest consolidated legislation content
- Writes plain text to `data/<library_id>.txt` (overwriting on update)
- Updates the CSV **after each row** so runs are restartable

No embeddings, databases, or RAG indexing are performed here; the output is a plain-text library intended for later use.

## Install

Python 3.10+ recommended.

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Optional (higher-quality HTML-to-text via Inscriptis):

```powershell
python -m pip install -r requirements_optional.txt
```

Note: `inscriptis` depends on `lxml`. On Windows with very new Python versions (e.g. 3.14),
`lxml` wheels may not yet be available and installation can require Microsoft C++ Build Tools.

## Run one bot

Example (New South Wales):

```powershell
cd "New South Wales"
python -m bot.sync
```

Common options (supported by every jurisdiction bot):

- `--user-agent "..."`
- `--sleep-seconds 1.0`
- `--timeout-seconds 60`
- `--max-retries 4`
- `--backoff-base-seconds 1.0`
- `--limit 10` (process only N rows)

## Run all bots

Run all 9 jurisdictions sequentially (polite delays still apply inside each bot):

```powershell
python run_all.py
```

You can pass the same common options through to all bots, for example:

```powershell
python run_all.py --sleep-seconds 1.5 --max-retries 5
```

## How change detection + extraction works (per bot)

Each bot stores a per-row `version_id` in its catalogue CSV and skips rows where the latest `version_id` matches the stored value (unless `--force` is used).

Some bots also support conditional requests when the stored `version_id` is a validator (e.g. `etag:...` or `last_modified:...`) and will treat HTTP `304 Not Modified` as unchanged.

Below is a high-level summary of (a) how each bot detects a change and (b) how it extracts the substantive legislation text, in priority order.

### Australian Capital Territory (ACT)

- Change detection (`version_id`) priority:
	1) republication/version parsed from HTML (when present)
	2) date-like version parsed from HTML
	3) HTTP `ETag` (`etag:...`)
	4) HTTP `Last-Modified` (parsed to ISO date)
	5) stable hash of normalized HTML text (or raw bytes)
- Substantive text extraction priority:
	1) if the “current” page exposes a DOCX link: download DOCX → extract text
	2) else if the “current” page exposes a PDF link: download PDF → extract text
	3) else if the fetched response is already RTF/PDF/XML: extract directly
	4) else from HTML: follow linked RTF/XML/PDF downloads when present
	5) fallback: HTML-to-text

### Commonwealth

- Change detection (`version_id`) priority:
	1) FRL OData “versions” API for the row’s `title_id` (best-effort)
	2) “Registered / Compilation date / Last updated / …” parsed from HTML
	3) HTTP `Last-Modified` (parsed to ISO date)
	4) stable hash of bytes
- Substantive text extraction priority:
	1) for `/text` pages: follow the linked EPUB-derived HTML (`/epub/.../document_1/document_1.html`) and extract the act body (skips TOC, stops before end-notes)
	2) if response is RTF: extract RTF
	3) if response is PDF: extract PDF
	4) if response is XML: extract XML
	5) from HTML: prefer linked RTF → XML → PDF downloads
	6) fallback: HTML-to-text (errors if it looks like FRL site chrome)
- Special fallback:
	- If HTML/EPUB extraction fails, the bot falls back to the FRL Documents API to download the latest “Primary PDF” and updates `version_id` to an `frl_pdf|...` composite id.

### New South Wales (NSW)

- Change detection (`version_id`) priority (stored as `fmt:v2|...`):
	1) latest of multiple candidate dates extracted from NSW HTML
	2) HTTP `Last-Modified` date
	3) stable hash
- Substantive text extraction priority:
	1) NSW Exchange XML (derived “whole” XML candidates from the row URL) → parse and format
	2) if an “export” HTML page is encountered: follow its `/export/xml/...` link and parse XML
	3) fallback: HTML (or RTF/PDF if the fetched content is actually those)

### Northern Territory (NT)

- Change detection (`version_id`) priority:
	1) “Reprint” code extracted from HTML (`reprint:...`)
	2) date-like version parsed from HTML
	3) HTTP `ETag` (`etag:...`)
	4) HTTP `Last-Modified` (`last_modified:...`)
	5) stable hash of normalized HTML text (or bytes)
- Substantive text extraction priority:
	1) if response is RTF: extract RTF
	2) if response is PDF: extract PDF
	3) if response is XML: extract XML
	4) from HTML: prefer linked RTF → XML → PDF downloads
	5) fallback: HTML-to-text
- Conditional requests:
	- Used only when stored `version_id` is `etag:...` or `last_modified:...` (sends `If-None-Match` / `If-Modified-Since`).

### Queensland (QLD)

- Change detection (`version_id`) priority:
	1) “as at / in force” date extracted from HTML (`/view/.../inforce/YYYY-MM-DD/...`) or embedded `PublicationDate`
	2) date-like version parsed from HTML
	3) HTTP `ETag` (`etag:...`)
	4) HTTP `Last-Modified` (`last_modified:...`)
	5) stable hash of normalized HTML text (or bytes)
- Substantive text extraction priority:
	1) if response is RTF: extract RTF
	2) if response is PDF: extract PDF
	3) if response is whole-act XML (`/view/whole/xml/...`): use the structured QLD formatter (adds act citation line + blank line)
	4) other XML: generic XML-to-text
	5) if response is whole-act HTML: attempt to derive whole-act XML and use it; otherwise parse whole-act HTML
	6) from HTML: prefer linked RTF → XML → PDF downloads
	7) fallback: HTML-to-text
- Conditional requests:
	- Used only when stored `version_id` is `etag:...` or `last_modified:...`.

### South Australia (SA)

- Change detection (`version_id`) priority (stored as `fmt:v2|...`):
	1) for SA “act” URLs: latest “point in time” inferred from embedded dated URLs
	2) date-like version parsed from HTML
	3) HTTP `ETag` (wrapped inside `fmt:v2|...`)
	4) HTTP `Last-Modified` (wrapped inside `fmt:v2|...`)
	5) stable hash of normalized HTML-to-text (or bytes)
- Substantive text extraction priority:
	1) if response is RTF: extract RTF
	2) if response is PDF: extract PDF
	3) if response is XML: extract XML
	4) from HTML: prefer linked RTF → XML → PDF downloads
	5) fallback: HTML-to-text
- Conditional requests:
	- Used only once the row has a `fmt:`-decorated `version_id`. Supports `fmt:...|etag:...` / `fmt:...|last_modified:...`.

### Tasmania (TAS)

- Change detection (`version_id`) priority (stored as `fmt:v4|...`, but preserves validators):
	1) `PublicationDate=YYYYMMDD...` parsed from HTML (or `VersionDescId` GUID as a secondary signal)
	2) date-like version parsed from HTML
	3) HTTP `ETag` (`etag:...`)
	4) HTTP `Last-Modified` (`last_modified:...`)
	5) stable hash of normalized HTML text (or bytes)
- Substantive text extraction priority:
	1) if response is RTF: extract RTF
	2) if response is PDF: extract PDF
	3) if response is XML: extract XML
	4) if HTML is whole-act view (`/view/whole/html/`): parse whole-act HTML for paragraph fidelity
	5) else if HTML is non-whole view: try switching to whole-act HTML and parse
	6) try derived XML endpoint and parse
	7) try derived PDF endpoint and parse
	8) from HTML: prefer linked RTF → XML → PDF downloads
	9) fallback: HTML-to-text
- Conditional requests:
	- Used only when stored `version_id` is `etag:...` or `last_modified:...`.

### Victoria (VIC)

- Change detection (`version_id`) priority:
	1) version segment inferred from the final redirected URL
	2) version segment from a canonical link
	3) highest “/###” version link found in the page
	4) “Last updated” date in page text
	5) HTTP `Last-Modified` date
	6) if the HTML is SPA/chrome: query the Drupal content backend and infer version from download filename
	7) stable hash
- Substantive text extraction priority:
	1) if response is RTF: extract RTF
	2) if response is Word/DOCX: extract DOCX
	3) if response is PDF: extract PDF
	4) from HTML: prefer linked RTF → DOCX → PDF downloads
	5) for Nuxt SPA pages: use the Drupal content backend and download the first “current version” file (prefer DOCX/DOC, else PDF)
	6) fallback: HTML-to-text (errors if it still looks like site chrome)

### Western Australia (WA)

- Change detection (`version_id`) priority (stored as `fmt:v2|...`, but preserves validators):
	1) currency metadata extracted from HTML (“currency start” date + optional suffix)
	2) date-like version parsed from HTML
	3) HTTP `ETag` (`etag:...`)
	4) HTTP `Last-Modified` (`last_modified:...`)
	5) stable hash of normalized HTML-to-text (or bytes)
- Substantive text extraction priority (always runs WA-specific cleanup on the result):
	1) if response is RTF: extract RTF
	2) if response is PDF: extract PDF
	3) if response is XML: extract XML
	4) from HTML: prefer the official “filestore/source” link; fetch it and extract based on its true type (including special handling for filestore HTML)
	5) otherwise from HTML: prefer linked RTF → XML → PDF downloads
	6) fallback: HTML-to-text
- Conditional requests:
	- Used only when stored `version_id` is `etag:...` or `last_modified:...`.
