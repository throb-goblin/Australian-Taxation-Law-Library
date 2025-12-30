# Australian Taxation Law Library (local RAG corpus)

## Single legislation bot

This repo uses one combined catalogue CSV and one bot.

**Inputs**
- Catalogue CSV: `raw/data/Legislative_materials/legislation_catalogue.csv`
  - This file is the source of truth for what gets scraped.
  - You must populate the `url` column with Classic AustLII Act URLs, for example:
    - `https://classic.austlii.edu.au/au/legis/cth/consol_act/antsbna1999470/` (A New Tax System (Australian Business Number) Act 1999)
  - A template file is provided at `raw/data/Legislative_materials/legislation_catalogue.template.csv`.
    - It includes the rows/columns but blanks the run-state columns: `when_scraped`, `last_successful_scrape`, `status`, `error`.

**Outputs**
- Raw (unmodified) text files: `raw/data/Legislative_materials/*.txt`
- Prepared (cleaned) text files: `prepared/data/Legislative_materials/*.txt`
  - Raw files are copied to prepared *before* any cleanup.
  - Cleanup/formatting only touches the prepared copy.

## Run (Windows)

## Quickstart (new Windows machine)

Prereqs
- Windows 10/11
- Python 3.x installed and available as `python` in PowerShell

GitHub install (no git required)
1) Download and run the installer script:
- `powershell -ExecutionPolicy Bypass -File .\scripts\install_from_github_windows.ps1`

One-liner (from any folder)
- `powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-RestMethod 'https://raw.githubusercontent.com/throb-goblin/Australian-Taxation-Law-Library/main/scripts/install_from_github_windows.ps1' | Invoke-Expression"`

2) The script downloads the repo ZIP from GitHub, extracts it, and runs the bootstrap.

Setup (recommended)
1) Open PowerShell and run the bootstrap script:
- `powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1`

Catalogue note
- On a fresh install, if `raw/data/Legislative_materials/legislation_catalogue.csv` does not exist, the bootstrap script will copy it from `legislation_catalogue.template.csv`.
- You then need to edit `legislation_catalogue.csv` and ensure the `url` column contains the Classic AustLII links you want scraped.

2) Run a small test scrape (from the repo root):
- `.\.venv\Scripts\python.exe .\bot\sync.py --limit 5 --skip-already-scraped`

From `Australian-Taxation-Law-Library/`:

1) Install deps
- `python -m pip install -r requirements.txt`

2) Test with 5 rows
- `python .\bot\sync.py --limit 5 --skip-already-scraped`

Useful flags
- `--force` to re-download/rewrite even if unchanged
- `--catalogue-path`, `--raw-dir`, `--prepared-dir` to override paths

Environment knobs (optional)
- `AUSTLII_MAX_PAGES` (default `250`): max linked section pages to crawl per Act.
- Polite crawling (recommended): by default the bot enforces a minimum delay between requests to `*.austlii.edu.au`.
  - `RESPECT_AUSTLII_CRAWL_DELAY=true` enables this (this is the default).
  - `AUSTLII_CRAWL_DELAY_SECONDS` (default `30`): minimum delay between requests to `*.austlii.edu.au` when enforcement is enabled.
  - `RESPECT_AUSTLII_CRAWL_DELAY=false`: disable the minimum delay (not recommended).

## Notes

- The bot primarily follows the AustLII classic “Downloads → Plain text (ASCII)” flow when the catalogue `url` is on `*.austlii.edu.au`.
- Best-effort cleanup on prepared copies includes rewriting `- SECT 31.10` → `Section 31-10` and removing TOC-like blocks.
