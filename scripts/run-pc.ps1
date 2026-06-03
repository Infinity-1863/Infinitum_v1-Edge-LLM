param(
  [Parameter(Mandatory=$true)][string]$PackageDir,
  [string]$ServerBin = "",
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8080,
  [int]$Context = 4096,
  [int]$Threads = 4,
  [int]$ThreadsBatch = 4,
  [int]$GpuLayers = 99,
  [int]$PredictTokens = -1,
  [int]$Batch = 512,
  [int]$UBatch = 512,
  [switch]$NoWarmup,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-PackagePath {
  param(
    [string]$PackageDir,
    [string]$ManifestPath,
    [string]$ManifestKey,
    [string[]]$FallbackPatterns
  )
  if ($ManifestPath) {
    if ([System.IO.Path]::IsPathRooted($ManifestPath)) {
      if (Test-Path -LiteralPath $ManifestPath) {
        return (Resolve-Path -LiteralPath $ManifestPath).Path
      }
    } else {
      $relativePath = Join-Path $PackageDir $ManifestPath
      if (Test-Path -LiteralPath $relativePath) {
        return (Resolve-Path -LiteralPath $relativePath).Path
      }
    }
  }
  foreach ($pattern in $FallbackPatterns) {
    $hit = Get-ChildItem -LiteralPath $PackageDir -Recurse -File -Filter $pattern -ErrorAction SilentlyContinue |
      Sort-Object Length -Descending |
      Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  throw "Could not resolve package file for $ManifestKey"
}

function Quote-Arg {
  param([string]$Value)
  if ($Value -match '\s') { return '"' + $Value + '"' }
  return $Value
}

$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$manifestFile = Join-Path $PackageDir "package.manifest.json"
if (-not (Test-Path -LiteralPath $manifestFile)) {
  throw "Missing package manifest: $manifestFile"
}
$manifest = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json

if (-not $ServerBin) {
  $ServerBin = if ($env:LLAMA_SERVER_BIN) { $env:LLAMA_SERVER_BIN } else { "llama-server.exe" }
}

$core = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.core.destination -ManifestKey "core" -FallbackPatterns @("*.gguf")
$index = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_index.destination -ManifestKey "expert_index" -FallbackPatterns @("*index*.json", "*manifest*.json")
$pack = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_pack.destination -ManifestKey "expert_pack" -FallbackPatterns @("*experts*.bin", "*expert*.bin")

$envPlan = [ordered]@{
  LLAMA_INFINITUM_SELECTIVE_MOE = "1"
  LLAMA_INFINITUM_EXPERT_INDEX = $index
  LLAMA_INFINITUM_V2_GGML_EXPERT_PACK = $pack
}

$args = @(
  "-m", $core,
  "-c", "$Context",
  "-t", "$Threads",
  "-tb", "$ThreadsBatch",
  "-n", "$PredictTokens",
  "-ngl", "$GpuLayers",
  "-b", "$Batch",
  "-ub", "$UBatch",
  "--host", $BindHost,
  "--port", "$Port"
)
if ($NoWarmup) { $args += "--no-warmup" }

if ($DryRun) {
  Write-Output "dry-run: PC launch plan"
  foreach ($key in $envPlan.Keys) {
    Write-Output (' $env:' + $key + '=' + $envPlan[$key])
  }
  $command = (@(Quote-Arg $ServerBin) + ($args | ForEach-Object { Quote-Arg $_ })) -join " "
  Write-Output $command
  return
}

if (-not (Test-Path -LiteralPath $ServerBin)) {
  $serverCommand = Get-Command $ServerBin -ErrorAction SilentlyContinue
  if (-not $serverCommand) {
    throw "Missing llama-server executable. Pass -ServerBin, set LLAMA_SERVER_BIN, or put it in PATH: $ServerBin"
  }
  $ServerBin = $serverCommand.Source
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  $pids = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
  throw "Port $Port is already listening. Owning process id(s): $pids"
}

$env:LLAMA_INFINITUM_SELECTIVE_MOE = $envPlan.LLAMA_INFINITUM_SELECTIVE_MOE
$env:LLAMA_INFINITUM_EXPERT_INDEX = $envPlan.LLAMA_INFINITUM_EXPERT_INDEX
$env:LLAMA_INFINITUM_V2_GGML_EXPERT_PACK = $envPlan.LLAMA_INFINITUM_V2_GGML_EXPERT_PACK

Write-Host "Starting PC server on http://$BindHost`:$Port"
& $ServerBin @args
