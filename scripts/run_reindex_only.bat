@echo off
setlocal enableextensions

REM --- Edit this path if you keep the repo elsewhere ---
set "INDEXER_REPO=D:\RAGStack\indexer"

echo.
echo === Re-index -> Qdrant (clear per-file, section chunking) ===
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

docker compose run --rm --build -e CLEAR_STRATEGY=file -e CHUNK_STRATEGY=sections -e OPENWEBUI_COLLECTION=%KB_ID% indexer
set "IDX_EXIT=%ERRORLEVEL%"

popd

if not "%IDX_EXIT%"=="0" (
  echo ERROR: indexing failed.
  echo NOTE: If the error mentions host.docker.internal:11434, Ollama is likely not reachable from Docker.
  exit /b %IDX_EXIT%
)

echo.
echo Done.
exit /b 0
