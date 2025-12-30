Param(
  [switch]$CreateVenv = $true,
  [switch]$InstallDeps = $true,
  [switch]$InitCatalogueFromTemplate = $true,
  [switch]$ForceCatalogueOverwrite = $false,
  [switch]$RegenerateTemplateFromCatalogue = $false
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

Write-Host "Repo root: $repoRoot"

# Create required folder structure
$rawDir = Join-Path $repoRoot 'raw\data\Legislative_materials'
$preparedDir = Join-Path $repoRoot 'prepared\data\Legislative_materials'
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null
New-Item -ItemType Directory -Force -Path $preparedDir | Out-Null

$catalogue = Join-Path $rawDir 'legislation_catalogue.csv'
$templateCatalogue = Join-Path $rawDir 'legislation_catalogue.template.csv'

if ($RegenerateTemplateFromCatalogue) {
  if (-not (Test-Path -LiteralPath $catalogue)) {
    throw "Cannot regenerate template: catalogue not found at $catalogue"
  }
  Write-Host "Regenerating template from catalogue (clearing when_scraped/last_successful_scrape/status/error)"
  $python = Get-Command python -ErrorAction SilentlyContinue
  if (-not $python) {
    throw "Python is not on PATH. Install Python 3.x and ensure 'python' is available."
  }
  python -c "import csv; from pathlib import Path; in_path=Path(r'$catalogue'); out_path=Path(r'$templateCatalogue'); r=csv.DictReader(in_path.open('r',encoding='utf-8',newline='')); f=r.fieldnames or []; rows=list(r); clear={'when_scraped','last_successful_scrape','status','error'}; missing=clear-set(f); assert not missing, f'Missing columns: {sorted(missing)}';
for row in rows:
  for c in clear:
    row[c]='';
w=csv.DictWriter(out_path.open('w',encoding='utf-8',newline=''), fieldnames=f); w.writeheader(); w.writerows(rows); print(out_path)"
}

if ((-not (Test-Path -LiteralPath $catalogue)) -and $InitCatalogueFromTemplate) {
  if (Test-Path -LiteralPath $templateCatalogue) {
    Write-Host "Initialising catalogue from template: $templateCatalogue"
    Copy-Item -LiteralPath $templateCatalogue -Destination $catalogue -Force:$ForceCatalogueOverwrite
  } else {
    Write-Warning "Catalogue CSV not found at: $catalogue"
    Write-Warning "Template CSV not found at: $templateCatalogue"
    Write-Warning "Create/copy legislation_catalogue.csv before running the bot."
  }
} elseif (Test-Path -LiteralPath $catalogue) {
  Write-Host "Found catalogue: $catalogue"
}

if ($CreateVenv) {
  $venvPath = Join-Path $repoRoot '.venv'
  if (-not (Test-Path -LiteralPath $venvPath)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
      throw "Python is not on PATH. Install Python 3.x and ensure 'python' is available."
    }
    Write-Host "Creating venv at $venvPath"
    python -m venv $venvPath
  } else {
    Write-Host "Venv already exists at $venvPath"
  }

  $pythonExe = Join-Path $venvPath 'Scripts\python.exe'
  if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Venv python not found at $pythonExe"
  }

  if ($InstallDeps) {
    Write-Host "Installing deps from requirements.txt"
    & $pythonExe -m pip install --upgrade pip
    & $pythonExe -m pip install -r (Join-Path $repoRoot 'requirements.txt')
  }

  Write-Host "Bootstrap complete. Next:" 
  Write-Host "  & '$pythonExe' bot\\sync.py --limit 5 --skip-already-scraped"
} else {
  Write-Host "Folders created. Next: install deps and run bot\\sync.py"
}
