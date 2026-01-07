param(
  [switch]$ForceScrape,
  [string]$OnlyLibraryId = "",
  [switch]$RespectAustliiDelay,
  [int]$AustliiDelaySeconds = 30,
  [ValidateSet('none','file','tenant')][string]$ClearStrategy = 'file',
  [ValidateSet('words','sections')][string]$ChunkStrategy = 'sections'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot  # Australian-Taxation-Law-Library
$ragstackIndexer = Join-Path (Split-Path -Parent $repoRoot) 'RAGStack\indexer'

if (-not (Test-Path -LiteralPath $ragstackIndexer)) {
  throw "RAGStack indexer folder not found at: $ragstackIndexer. Pass a different layout or run indexing manually from RAGStack/indexer."
}

Write-Host "1) Scrape -> prepared corpus" -ForegroundColor Cyan

if ($RespectAustliiDelay) {
  $env:RESPECT_AUSTLII_CRAWL_DELAY = 'true'
  $env:AUSTLII_CRAWL_DELAY_SECONDS = "$AustliiDelaySeconds"
}

$syncArgs = @()
if ($ForceScrape) { $syncArgs += '--force' }
if ($OnlyLibraryId) { $syncArgs += @('--only-library-id', $OnlyLibraryId) }

Push-Location $repoRoot
try {
  python -m bot.sync @syncArgs
} finally {
  Pop-Location
}

Write-Host "2) Re-index -> Qdrant" -ForegroundColor Cyan

# Preflight: Ollama must be reachable from containers via host.docker.internal.
# On Windows, if Ollama is bound to 127.0.0.1 only, containers will fail.
try {
  Invoke-RestMethod -Uri 'http://localhost:11434/api/tags' -TimeoutSec 5 | Out-Null
} catch {
  throw "Ollama not reachable on host at http://localhost:11434. Start Ollama first."
}

Push-Location $ragstackIndexer
try {
  docker compose run --rm --build `
    -e CLEAR_STRATEGY=$ClearStrategy `
    -e CHUNK_STRATEGY=$ChunkStrategy `
    indexer
} finally {
  Pop-Location
}

Write-Host "Done." -ForegroundColor Green
Write-Host "If Open WebUI is running, ask a question and confirm citations reference your corpus."