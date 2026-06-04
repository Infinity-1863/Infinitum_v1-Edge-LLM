param(
  [string]$DeviceDir = "/data/local/tmp/infinitum-edge-llm",
  [ValidateSet("baseline", "page-prefetch")]
  [string]$Profile = "baseline",
  [int]$PcPort = 18110,
  [int]$PhonePort = 8080,
  [int]$Context = 512,
  [int]$Threads = 4,
  [int]$ThreadsBatch = 4,
  [int]$GpuLayers = 0,
  [int]$MaxTokens = 32,
  [string]$Message = "привет",
  [int]$PrefetchMaxExperts = 1,
  [string]$RunLabel = "",
  [string]$ServerRelativePath = "bin/llama-server",
  [string]$CoreRelativePath = "model/gpt-oss-24l-32e-core-only.gguf",
  [string]$IndexRelativePath = "moe/gpt_oss_expert_store_index.fixed.json",
  [string]$PackRelativePath = "moe_ggml_pack/experts.ggml_mxfp4.bin",
  [string[]]$ExtraEnv = @(),
  [string]$OutDir = "artifacts/phone-bench",
  [switch]$KeepServer,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Join-RemotePath {
  param([string]$Root, [string]$Child)
  $Root.TrimEnd("/") + "/" + $Child.TrimStart("/")
}

function Quote-RemoteSh {
  param([string]$Value)
  "'" + $Value.Replace("'", "'\''") + "'"
}

function New-GptOssPrompt {
  param(
    [string]$SystemPrompt,
    [string]$UserMessage
  )
  $cleanSystem = $SystemPrompt.Replace("<|start|>", "").Replace("<|end|>", "").Replace("<|message|>", "").Replace("<|channel|>", "").Replace("<|return|>", "")
  $cleanUser = $UserMessage.Replace("<|start|>", "").Replace("<|end|>", "").Replace("<|message|>", "").Replace("<|channel|>", "").Replace("<|return|>", "")
  "<|start|>system<|message|>$cleanSystem<|end|><|start|>user<|message|>$cleanUser<|end|><|start|>assistant<|channel|>final<|message|>"
}

function Invoke-Utf8Json {
  param(
    [string]$Uri,
    [object]$Payload,
    [int]$TimeoutSec = 900
  )
  $json = $Payload | ConvertTo-Json -Depth 12
  $body = [System.Text.Encoding]::UTF8.GetBytes($json)
  Invoke-WebRequest -UseBasicParsing -Uri $Uri -Method POST -ContentType "application/json; charset=utf-8" -Body $body -TimeoutSec $TimeoutSec
}

function Measure-ProfileLines {
  param([string[]]$Lines)
  $count = 0
  $loadMs = 0.0
  $computeMs = 0.0
  $gateUpMs = 0.0
  $activationMs = 0.0
  $downMs = 0.0
  $loadedBytes = [int64]0
  $cacheMisses = [int64]0
  $cacheHits = [int64]0
  $predictionHits = [int64]0
  $predictionCount = [int64]0
  $predictionWaste = [int64]0
  $prefetchSubmitMs = 0.0
  foreach ($line in $Lines) {
    $marker = "LLAMA_INFINITUM_PROFILE "
    $idx = $line.IndexOf($marker)
    if ($idx -lt 0) { continue }
    $json = $line.Substring($idx + $marker.Length)
    try {
      $profile = $json | ConvertFrom-Json
      $count += 1
      $loadMs += [double]$profile.load_ms
      $computeMs += [double]$profile.total_compute_ms
      $gateUpMs += [double]$profile.gate_up_ms
      $activationMs += [double]$profile.activation_ms
      $downMs += [double]$profile.down_ms
      $loadedBytes += [int64]$profile.loaded_bytes_delta
      $cacheMisses += [int64]$profile.cache_misses_delta
      $cacheHits += [int64]$profile.cache_hits_delta
      $predictionHits += [int64]$profile.prediction_hits
      $predictionCount += [int64]$profile.prediction_count
      $predictionWaste += [int64]$profile.prediction_waste
      $prefetchSubmitMs += [double]$profile.prefetch_submit_ms
    } catch {}
  }
  $totalCache = $cacheHits + $cacheMisses
  [ordered]@{
    profile_lines = $count
    load_ms_sum = [math]::Round($loadMs, 3)
    compute_ms_sum = [math]::Round($computeMs, 3)
    gate_up_ms_sum = [math]::Round($gateUpMs, 3)
    activation_ms_sum = [math]::Round($activationMs, 3)
    down_ms_sum = [math]::Round($downMs, 3)
    loaded_bytes_sum = $loadedBytes
    cache_hits_sum = $cacheHits
    cache_misses_sum = $cacheMisses
    cache_hit_rate = if ($totalCache -gt 0) { [math]::Round($cacheHits / $totalCache, 4) } else { $null }
    prediction_hits_sum = $predictionHits
    prediction_count_sum = $predictionCount
    prediction_waste_sum = $predictionWaste
    prediction_hit_rate = if ($predictionCount -gt 0) { [math]::Round($predictionHits / $predictionCount, 4) } else { $null }
    prefetch_submit_ms_sum = [math]::Round($prefetchSubmitMs, 3)
  }
}

$remoteServer = Join-RemotePath $DeviceDir $ServerRelativePath
$remoteCore = Join-RemotePath $DeviceDir $CoreRelativePath
$remoteIndex = Join-RemotePath $DeviceDir $IndexRelativePath
$remotePack = Join-RemotePath $DeviceDir $PackRelativePath
if ($RunLabel -and -not ($RunLabel -match '^[A-Za-z0-9_.-]+$')) {
  throw "RunLabel may only contain letters, digits, dot, underscore, and dash: $RunLabel"
}
$artifactLabel = if ($RunLabel) { "$Profile-$RunLabel" } else { $Profile }
$logName = "server-$artifactLabel.log"

$envParts = @(
  "LD_LIBRARY_PATH=$(Quote-RemoteSh "bin")",
  "LLAMA_INFINITUM_SELECTIVE_MOE=1",
  "LLAMA_INFINITUM_EXPERT_INDEX=$(Quote-RemoteSh $remoteIndex)",
  "LLAMA_INFINITUM_V2_GGML_EXPERT_PACK=$(Quote-RemoteSh $remotePack)",
  "LLAMA_INFINITUM_PACK_REPORT=1"
)
if ($Profile -eq "page-prefetch") {
  $envParts += @(
    "LLAMA_INFINITUM_EXPERT_PREFETCH=1",
    "LLAMA_INFINITUM_GGML_PACK_PREFETCH=1",
    "LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS=$PrefetchMaxExperts",
    "LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD=1"
  )
}
foreach ($entry in $ExtraEnv) {
  $name, $value = $entry.Split("=", 2)
  if (-not $name -or $null -eq $value) {
    throw "ExtraEnv entries must use NAME=VALUE format: $entry"
  }
  if (-not ($name -match '^[A-Z_][A-Z0-9_]*$')) {
    throw "ExtraEnv name is not a safe environment variable name: $name"
  }
  $envParts += "$name=$(Quote-RemoteSh $value)"
}

$serverArgs = @(
  "-m", (Quote-RemoteSh $remoteCore),
  "-c", "$Context",
  "-t", "$Threads",
  "-tb", "$ThreadsBatch",
  "-ngl", "$GpuLayers",
  "--host", (Quote-RemoteSh "0.0.0.0"),
  "--port", "$PhonePort",
  "--parallel", "1",
  "--no-warmup"
)
$remoteCommand = Quote-RemoteSh ("./" + $ServerRelativePath.TrimStart("/"))
$quotedDeviceDir = Quote-RemoteSh $DeviceDir
$quotedRemoteServer = Quote-RemoteSh $remoteServer
$quotedLogName = Quote-RemoteSh $logName

$launchCommand = @(
  "cd $quotedDeviceDir",
  'old=$(cat server.pid 2>/dev/null); if [ -n "$old" ]; then kill "$old" 2>/dev/null || true; fi',
  "rm -f $quotedLogName server.pid server.exit",
  "chmod 755 $quotedRemoteServer 2>/dev/null || true",
  "nohup env $($envParts -join " ") $remoteCommand $($serverArgs -join " ") > $quotedLogName 2>&1 < /dev/null & echo `$! > server.pid"
) -join "; "

if ($DryRun) {
  Write-Output "dry-run: phone benchmark plan"
  Write-Output "adb shell '$launchCommand'"
  Write-Output "adb forward tcp:$PcPort tcp:$PhonePort"
  Write-Output "POST http://127.0.0.1:$PcPort/completion"
  return
}

$adb = Get-Command adb -ErrorAction SilentlyContinue
if (-not $adb) {
  throw "adb was not found in PATH"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$resultPath = Join-Path $OutDir "$stamp-$artifactLabel.result.json"

try {
  & adb shell $launchCommand | Out-Null
  & adb forward "tcp:$PcPort" "tcp:$PhonePort" | Out-Null

  $healthy = $false
  for ($i = 0; $i -lt 180; $i++) {
    Start-Sleep -Seconds 1
    try {
      $health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$PcPort/health" -TimeoutSec 2
      if ($health.StatusCode -eq 200) {
        $healthy = $true
        break
      }
    } catch {}
  }
  if (-not $healthy) {
    $tail = & adb shell "cd $quotedDeviceDir; tail -80 $quotedLogName 2>/dev/null"
    throw "phone llama-server did not become healthy. Log tail:`n$($tail -join "`n")"
  }

  $prompt = New-GptOssPrompt `
    -SystemPrompt "You are a helpful assistant. Reply only to the user. Do not describe the conversation." `
    -UserMessage $Message
  $payload = @{
    prompt = $prompt
    n_predict = $MaxTokens
    temperature = 0.0
    stop = @("<|return|>", "<|end|>", "<|start|>")
  }
  $started = Get-Date
  $response = Invoke-Utf8Json -Uri "http://127.0.0.1:$PcPort/completion" -Payload $payload
  $ended = Get-Date
  $parsed = $response.Content | ConvertFrom-Json
  $serverPid = (& adb shell "cd $quotedDeviceDir; cat server.pid 2>/dev/null").Trim()
  $rss = $null
  if ($serverPid) {
    $rssLine = & adb shell "ps -p $serverPid -o RSS= 2>/dev/null"
    if ($rssLine) {
      $rss = [int64](([string]$rssLine).Trim())
    }
  }
  $logLines = & adb shell "cd $quotedDeviceDir; cat $quotedLogName 2>/dev/null"
  $profileSummary = Measure-ProfileLines -Lines $logLines

  $summary = [ordered]@{
    profile = $Profile
    run_label = $RunLabel
    device_dir = $DeviceDir
    pc_port = $PcPort
    phone_port = $PhonePort
    context = $Context
    threads = $Threads
    threads_batch = $ThreadsBatch
    gpu_layers = $GpuLayers
    max_tokens = $MaxTokens
    wall_ms = [math]::Round(($ended - $started).TotalMilliseconds, 3)
    content = ([string]$parsed.content).Trim()
    prompt_n = $parsed.timings.prompt_n
    prompt_ms = $parsed.timings.prompt_ms
    prompt_per_second = $parsed.timings.prompt_per_second
    predicted_n = $parsed.timings.predicted_n
    predicted_ms = $parsed.timings.predicted_ms
    predicted_per_second = $parsed.timings.predicted_per_second
    rss_kib = $rss
    prefetch_log_seen = (($logLines -join "`n").Contains("infinitum_pack_prefetch"))
    runtime_profile = $profileSummary
    remote_log = "$DeviceDir/$logName"
    env = $envParts
    extra_env = $ExtraEnv
  }
  $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding UTF8
  $summary | ConvertTo-Json -Depth 8
} finally {
  if (-not $KeepServer) {
    & adb shell "cd $quotedDeviceDir; old=`$(cat server.pid 2>/dev/null); if [ -n `"`$old`" ]; then kill `"`$old`" 2>/dev/null || true; fi" | Out-Null
    & adb forward --remove "tcp:$PcPort" 2>$null | Out-Null
  }
}
