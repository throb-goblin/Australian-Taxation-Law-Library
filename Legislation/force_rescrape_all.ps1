param(
    [int]$SleepSeconds = 1,
  [int]$Limit = 0,
  [int]$TimeoutSeconds = 60,
  [int]$MaxRetries = 4,
  [double]$BackoffBaseSeconds = 1.0,
  [switch]$ContinueOnError
)

$ErrorActionPreference = 'Stop'

$jurisdictions = @(
  'Australian Capital Territory',
  'Commonwealth',
  'New South Wales',
  'Northern Territory',
  'Queensland',
  'South Australia',
  'Tasmania',
  'Victoria',
  'Western Australia'
)

$root = Split-Path -Parent $PSCommandPath

$failures = New-Object System.Collections.Generic.List[string]

foreach($j in $jurisdictions){
  $dir = Join-Path $root $j
  Write-Host "=== $j ==="
  Push-Location $dir
  try {
    $args = @(
      '-m','bot.sync',
      '--force',
      '--sleep-seconds', $SleepSeconds,
      '--timeout-seconds', $TimeoutSeconds,
      '--max-retries', $MaxRetries,
      '--backoff-base-seconds', $BackoffBaseSeconds
    )
    if($Limit -gt 0){
      $args += @('--limit', $Limit)
    } else {
      # no limit
    }
    $output = & python @args 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    if($exitCode -ne 0){
      $detail = ($output)
      if($null -eq $detail){ $detail = '' }
      $detail = $detail.Trim()
      if($detail){
        throw "bot.sync exited with code $exitCode for $j`n$detail"
      }
      throw "bot.sync exited with code $exitCode for $j"
    }
  } catch {
    $msg = ($_ | Out-String)
    if($null -eq $msg){ $msg = '' }
    $msg = $msg.Trim()
    if(-not $msg){
      $msg = $_.Exception.ToString()
      if($null -eq $msg){ $msg = '' }
      $msg = $msg.Trim()
    }
    if(-not $msg){
      $msg = 'Unknown error (no exception message)'
    }
    $failures.Add("$j :: $msg") | Out-Null
    Write-Host "ERROR in ${j}: $msg"
    if(-not $ContinueOnError){
      throw
    }
  } finally {
    Pop-Location
  }
}


if($failures.Count -gt 0){
  Write-Host 'FAILED'
  $failures | ForEach-Object { Write-Host "- $_" }
  exit 1
}

Write-Host 'DONE'
