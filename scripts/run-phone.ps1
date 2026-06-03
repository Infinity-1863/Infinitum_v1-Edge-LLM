param(
  [Parameter(Mandatory=$true)][string]$PackageDir,
  [string]$DeviceDir = "/data/local/tmp/infinitum-edge-llm",
  [int]$PcPort = 18080,
  [int]$PhonePort = 8080,
  [int]$Context = 4096,
  [int]$Threads = 4,
  [int]$ThreadsBatch = 4,
  [int]$GpuLayers = 0,
  [int]$SmokeSeconds = 0,
  [string]$ServerRelativePath = "bin/llama-server",
  [string]$AndroidBinDir,
  [switch]$SkipPush,
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

function Write-Plan {
  param([string]$Line)
  Write-Output $Line
}

$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$manifestFile = Join-Path $PackageDir "package.manifest.json"
if (-not (Test-Path -LiteralPath $manifestFile)) {
  throw "Missing package manifest: $manifestFile"
}
$manifest = Get-Content -LiteralPath $manifestFile -Raw | ConvertFrom-Json

$core = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.core.destination -ManifestKey "core" -FallbackPatterns @("*.gguf")
$index = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_index.destination -ManifestKey "expert_index" -FallbackPatterns @("*index*.json", "*manifest*.json")
$pack = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_pack.destination -ManifestKey "expert_pack" -FallbackPatterns @("*experts*.bin", "*expert*.bin")

$coreName = Split-Path -Leaf $core
$indexName = Split-Path -Leaf $index
$packName = Split-Path -Leaf $pack

$resolvedAndroidBinDir = $null
$androidBinFiles = @()
if ($AndroidBinDir) {
  $resolvedAndroidBinDir = (Resolve-Path -LiteralPath $AndroidBinDir).Path
  $serverName = Split-Path -Leaf $ServerRelativePath
  $localServer = Join-Path $resolvedAndroidBinDir $serverName
  if (-not (Test-Path -LiteralPath $localServer)) {
    throw "AndroidBinDir does not contain $serverName`: $resolvedAndroidBinDir"
  }
  $androidBinFiles = @(Get-ChildItem -LiteralPath $resolvedAndroidBinDir -File)
}

$remoteModel = "$DeviceDir/model/$coreName"
$remoteIndex = "$DeviceDir/moe/$indexName"
$remotePack = "$DeviceDir/moe_ggml_pack/$packName"
$remoteServer = "$DeviceDir/$ServerRelativePath"

$serverCommand = "env LD_LIBRARY_PATH=bin LLAMA_INFINITUM_SELECTIVE_MOE=1 LLAMA_INFINITUM_EXPERT_INDEX=$remoteIndex LLAMA_INFINITUM_V2_GGML_EXPERT_PACK=$remotePack ./$ServerRelativePath -m $remoteModel -c $Context -t $Threads -tb $ThreadsBatch -ngl $GpuLayers --host 0.0.0.0 --port $PhonePort --parallel 1 --no-warmup"
if ($SmokeSeconds -gt 0) {
  $serverCommand = "timeout $SmokeSeconds $serverCommand"
}
$remoteCommand = "cd $DeviceDir && $serverCommand"

if ($DryRun) {
  Write-Plan "dry-run: phone launch plan"
  Write-Plan "adb shell `"mkdir -p $DeviceDir/bin $DeviceDir/model $DeviceDir/moe $DeviceDir/moe_ggml_pack`""
  if (-not $SkipPush) {
    Write-Plan "adb push `"$core`" `"$DeviceDir/model/`""
    Write-Plan "adb push `"$index`" `"$DeviceDir/moe/`""
    Write-Plan "adb push `"$pack`" `"$DeviceDir/moe_ggml_pack/`""
    if ($resolvedAndroidBinDir) {
      foreach ($binFile in $androidBinFiles) {
        Write-Plan "adb push `"$($binFile.FullName)`" `"$DeviceDir/bin/`""
      }
    } else {
      Write-Plan "adb push <android-llama-bin-dir> `"$DeviceDir/bin/`""
    }
  }
  Write-Plan "adb shell `"chmod 755 $remoteServer 2>/dev/null || true`""
  Write-Plan "adb forward tcp:$PcPort tcp:$PhonePort"
  Write-Plan "adb shell `"$remoteCommand`""
  return
}

$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
  throw "adb was not found in PATH"
}

adb shell "mkdir -p $DeviceDir/bin $DeviceDir/model $DeviceDir/moe $DeviceDir/moe_ggml_pack" | Out-Null
if (-not $SkipPush) {
  adb push $core "$DeviceDir/model/"
  adb push $index "$DeviceDir/moe/"
  adb push $pack "$DeviceDir/moe_ggml_pack/"
  if ($resolvedAndroidBinDir) {
    foreach ($binFile in $androidBinFiles) {
      adb push $binFile.FullName "$DeviceDir/bin/"
    }
  }
}
adb shell "chmod 755 $remoteServer 2>/dev/null || true" | Out-Null
adb forward "tcp:$PcPort" "tcp:$PhonePort" | Out-Null

Write-Host "Starting phone server, forwarded to http://127.0.0.1:$PcPort"
adb shell $remoteCommand
