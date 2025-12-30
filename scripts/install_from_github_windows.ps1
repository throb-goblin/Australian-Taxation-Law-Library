Param(
  [string]$RepoOwner = "throb-goblin",
  [string]$RepoName = "Australian-Taxation-Law-Library",
  [string]$Branch = "main",
  [string]$InstallDir = "",
  [switch]$CreateVenv = $true,
  [switch]$InstallDeps = $true
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
  $InstallDir = Join-Path $env:USERPROFILE ("src\\" + $RepoName)
}

$zipUrl = "https://github.com/$RepoOwner/$RepoName/archive/refs/heads/$Branch.zip"
$tempRoot = Join-Path $env:TEMP ("${RepoName}_install_" + [Guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $tempRoot "$RepoName.zip"
$extractDir = Join-Path $tempRoot "extract"

New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

Write-Host "Downloading: $zipUrl"
Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

Write-Host "Extracting to: $extractDir"
Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force

$expandedRoot = Get-ChildItem -LiteralPath $extractDir -Directory | Select-Object -First 1
if (-not $expandedRoot) {
  throw "Unexpected ZIP structure: no folder found after extraction."
}

Write-Host "Installing to: $InstallDir"
if (Test-Path -LiteralPath $InstallDir) {
  throw "InstallDir already exists: $InstallDir (choose a different path or delete it first)"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $InstallDir) | Out-Null
Move-Item -LiteralPath $expandedRoot.FullName -Destination $InstallDir

$bootstrap = Join-Path $InstallDir "Australian-Taxation-Law-Library\\scripts\\bootstrap_windows.ps1"
if (-not (Test-Path -LiteralPath $bootstrap)) {
  throw "Bootstrap script not found at: $bootstrap"
}

Write-Host "Running bootstrap: $bootstrap"
powershell -ExecutionPolicy Bypass -File $bootstrap -CreateVenv:$CreateVenv -InstallDeps:$InstallDeps

Write-Host "Done. Repo installed at: $InstallDir"
