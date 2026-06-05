param(
  [Parameter(Mandatory=$true)][string]$PackageDir,
  [string]$ServerBin = "",
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8080,
  [int]$Context = 4096,
  [int]$Threads = 8,
  [int]$ThreadsBatch = 8,
  [int]$GpuLayers = 99,
  [int]$PredictTokens = -1,
  [int]$Batch = 512,
  [int]$UBatch = 512,
  [int]$ExpertWorkers = 1,
  [int]$ExpertRowThreads = 6,
  [switch]$UseOneApi,
  [string]$OneApiRoot = "",
  [string]$OneApiDeviceSelector = "level_zero:gpu",
  [ValidateSet("auto", "on", "off")]
  [string]$FlashAttention = "on",
  [string]$SyclTempDir = "artifacts/sycl-temp-run",
  [switch]$NoWarmup,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

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

function Resolve-OptionalPackagePath {
  param(
    [string]$PackageDir,
    [string]$ManifestPath,
    [string]$ManifestKey,
    [string[]]$FallbackPatterns
  )
  try {
    return Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $ManifestPath -ManifestKey $ManifestKey -FallbackPatterns $FallbackPatterns
  } catch {
    return ""
  }
}

function Quote-Arg {
  param([string]$Value)
  if ($Value -match '\s') { return '"' + $Value + '"' }
  return $Value
}

function Import-InfinitumOneApiEnvironment {
  param([string]$Root)
  $candidates = @()
  if ($Root) { $candidates += $Root }
  if ($env:ONEAPI_ROOT) { $candidates += $env:ONEAPI_ROOT }
  $candidates += "C:\Program Files (x86)\Intel\oneAPI"

  foreach ($candidate in $candidates) {
    $setvars = Join-Path $candidate "setvars.bat"
    if (-not (Test-Path -LiteralPath $setvars)) {
      continue
    }
    $cmd = 'call "' + $setvars + '" intel64 --force >nul && set'
    $lines = & cmd.exe /d /c $cmd
    if ($LASTEXITCODE -ne 0) {
      throw "oneAPI setvars failed: $setvars"
    }
    foreach ($line in $lines) {
      $idx = $line.IndexOf("=")
      if ($idx -gt 0) {
        [Environment]::SetEnvironmentVariable($line.Substring(0, $idx), $line.Substring($idx + 1), "Process")
      }
    }
    return $setvars
  }
  return ""
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

$useOneApiRuntime = $UseOneApi -or ($ServerBin -match '(?i)sycl')

$core = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.core.destination -ManifestKey "core" -FallbackPatterns @("*.gguf")
$index = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_index.destination -ManifestKey "expert_index" -FallbackPatterns @("*index*.json", "*manifest*.json")
$pack = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_pack.destination -ManifestKey "expert_pack" -FallbackPatterns @("*experts*.bin", "*expert*.bin")
$splitPackManifestPath = ""
if ($manifest.files.PSObject.Properties.Name -contains "expert_split_pack") {
  $splitPackManifestPath = $manifest.files.expert_split_pack.destination
}
$splitPack = Resolve-OptionalPackagePath -PackageDir $PackageDir -ManifestPath $splitPackManifestPath -ManifestKey "expert_split_pack" -FallbackPatterns @("experts.split.ggml_mxfp4.bin")

$envPlan = [ordered]@{
  LLAMA_INFINITUM_SELECTIVE_MOE = "1"
  LLAMA_INFINITUM_EXPERT_INDEX = $index
  LLAMA_INFINITUM_V2_GGML_EXPERT_PACK = $pack
  LLAMA_INFINITUM_EXPERT_WORKERS = "$ExpertWorkers"
  LLAMA_INFINITUM_EXPERT_ROW_THREADS = "$ExpertRowThreads"
}
if ($splitPack -and -not $useOneApiRuntime) {
  $envPlan.LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK = $splitPack
}
if ($useOneApiRuntime) {
  $SyclTempDir = if ([System.IO.Path]::IsPathRooted($SyclTempDir)) { $SyclTempDir } else { Join-Path $RepoRoot $SyclTempDir }
  $envPlan.ONEAPI_DEVICE_SELECTOR = $OneApiDeviceSelector
  $envPlan.ZES_ENABLE_SYSMAN = "1"
  $envPlan.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS = "1"
  $envPlan.TEMP = $SyclTempDir
  $envPlan.TMP = $SyclTempDir
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
  "--flash-attn", "$FlashAttention",
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
if ($useOneApiRuntime) {
  $setvars = Import-InfinitumOneApiEnvironment -Root $OneApiRoot
  if (-not $setvars) {
    throw "oneAPI setvars.bat not found. Pass -OneApiRoot or install Intel oneAPI."
  }
  New-Item -ItemType Directory -Force -Path $SyclTempDir | Out-Null
}
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
  $pids = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
  throw "Port $Port is already listening. Owning process id(s): $pids"
}

$env:LLAMA_INFINITUM_SELECTIVE_MOE = $envPlan.LLAMA_INFINITUM_SELECTIVE_MOE
$env:LLAMA_INFINITUM_EXPERT_INDEX = $envPlan.LLAMA_INFINITUM_EXPERT_INDEX
$env:LLAMA_INFINITUM_V2_GGML_EXPERT_PACK = $envPlan.LLAMA_INFINITUM_V2_GGML_EXPERT_PACK
if ($envPlan.Contains("LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK")) {
  $env:LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK = $envPlan.LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK
}
$env:LLAMA_INFINITUM_EXPERT_WORKERS = $envPlan.LLAMA_INFINITUM_EXPERT_WORKERS
$env:LLAMA_INFINITUM_EXPERT_ROW_THREADS = $envPlan.LLAMA_INFINITUM_EXPERT_ROW_THREADS
if ($useOneApiRuntime) {
  $env:ONEAPI_DEVICE_SELECTOR = $envPlan.ONEAPI_DEVICE_SELECTOR
  $env:ZES_ENABLE_SYSMAN = $envPlan.ZES_ENABLE_SYSMAN
  $env:UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS = $envPlan.UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS
  $env:TEMP = $envPlan.TEMP
  $env:TMP = $envPlan.TMP
}

Write-Host "Starting PC server on http://$BindHost`:$Port"
& $ServerBin @args
