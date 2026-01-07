@echo off
setlocal enableextensions

REM --- Edit these two paths if you keep the repos elsewhere ---
set "ATL_REPO=D:\Australian-Taxation-Law-Library"
set "INDEXER_REPO=D:\RAGStack\indexer"

REM --- Optional: be polite to AustLII (defaults to 30s in your bot when enabled) ---
set "RESPECT_AUSTLII_CRAWL_DELAY=true"
set "AUSTLII_CRAWL_DELAY_SECONDS=30"

echo.
echo === 1) Scrape -> prepared corpus ===
echo Repo: %ATL_REPO%

if not exist "%ATL_REPO%\bot\sync.py" (
  echo ERROR: bot.sync.py not found under %ATL_REPO%
  exit /b 1
)

pushd "%ATL_REPO%" || exit /b 1
python -m bot.sync --force
if errorlevel 1 (
  popd
  echo ERROR: scrape failed.
  exit /b 1
)
popd

echo.
echo === 2) Re-index -> Qdrant (clear per-file, section chunking) ===
echo Indexer: %INDEXER_REPO%

if not exist "%INDEXER_REPO%\docker-compose.yml" (
  echo ERROR: docker-compose.yml not found under %INDEXER_REPO%
  exit /b 1
)

pushd "%INDEXER_REPO%" || exit /b 1

REM Open WebUI (Qdrant multitenancy) typically uses the Knowledge base ID (UUID) as the tenant key.
REM If the Knowledge base name is 'australian_tax_law_corpus', detect its ID from Open WebUI's SQLite DB.
set "KB_NAME=australian_tax_law_corpus"
for /f "usebackq delims=" %%i in (`docker exec open-webui sh -lc "python - << \"PY\"^\
import sqlite3^\
con=sqlite3.connect('/app/backend/data/webui.db')^\
cur=con.cursor()^\
row=cur.execute('select id from knowledge where name=? limit 1',(\"%KB_NAME%\",)).fetchone()^\
print(row[0] if row else '')^\
PY"`) do set "KB_ID=%%i"
if "%KB_ID%"=="" (
  echo WARN: Could not find Knowledge base '%KB_NAME%' in Open WebUI.
  echo       Create it in Open WebUI first, or set KB_ID manually in this script.
) else (
  echo Using Knowledge base ID: %KB_ID%
)

REM CLEAR_STRATEGY=file avoids duplicates without wiping the whole tenant.
REM CHUNK_STRATEGY=sections matches the new default but is passed explicitly here.
docker compose run --rm --build -e CLEAR_STRATEGY=file -e CHUNK_STRATEGY=sections -e OPENWEBUI_COLLECTION=%KB_ID% indexer
set "IDX_EXIT=%ERRORLEVEL%"

popd

if not "%IDX_EXIT%"=="0" (
  echo ERROR: indexing failed.
  echo NOTE: If the error mentions host.docker.internal:11434, Ollama is likely bound to 127.0.0.1 only.
  echo       Fix by setting OLLAMA_HOST=0.0.0.0 and restarting Ollama.
  echo       Quick check: docker run --rm curlimages/curl:8.5.0 http://host.docker.internal:11434/api/tags
  exit /b %IDX_EXIT%
)

echo.
echo Done.
echo Next: open Open WebUI and ask a statute question.
exit /b 0
