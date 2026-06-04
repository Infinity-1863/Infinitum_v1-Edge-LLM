param(
  [Parameter(Mandatory=$true)][string]$PackageDir,
  [string]$ServerBin = "",
  [ValidateSet("baseline", "page-prefetch", "streaming")]
  [string]$Profile = "baseline",
  [int]$Port = 18082,
  [int]$Context = 512,
  [int]$Threads = 0,
  [int]$ThreadsBatch = 0,
  [int]$GpuLayers = 0,
  [int]$Batch = 512,
  [int]$UBatch = 512,
  [int]$MaxTokens = 32,
  [string]$Message = "привет",
  [int]$GpuExpertGlobalSlots = 64,
  [int]$GpuExpertLayerSlots = 8,
  [int]$GpuExpertCacheMiB = 4096,
  [int]$ExpertCacheMiB = 512,
  [int]$PackedExpertCacheMiB = 512,
  [int]$PrefetchMaxExperts = 1,
  [string]$OutDir = "artifacts/pc-bench"
)

$ErrorActionPreference = "Stop"

if ($Threads -le 0) {
  $Threads = [Environment]::ProcessorCount
}
if ($ThreadsBatch -le 0) {
  $ThreadsBatch = $Threads
}

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

function New-GptOssPrompt {
  param(
    [string]$SystemPrompt,
    [string]$UserMessage
  )
  $cleanSystem = $SystemPrompt.Replace("<|start|>", "").Replace("<|end|>", "").Replace("<|message|>", "").Replace("<|channel|>", "").Replace("<|return|>", "")
  $cleanUser = $UserMessage.Replace("<|start|>", "").Replace("<|end|>", "").Replace("<|message|>", "").Replace("<|channel|>", "").Replace("<|return|>", "")
  "<|start|>system<|message|>$cleanSystem<|end|><|start|>user<|message|>$cleanUser<|end|><|start|>assistant<|channel|>final<|message|>"
}

$PackageDir = (Resolve-Path -LiteralPath $PackageDir).Path
$manifestPath = Join-Path $PackageDir "package.manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "Missing package manifest: $manifestPath"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$core = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.core.destination -ManifestKey "core" -FallbackPatterns @("*.gguf")
$index = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_index.destination -ManifestKey "expert_index" -FallbackPatterns @("*index*.json", "*manifest*.json")
$pack = Resolve-PackagePath -PackageDir $PackageDir -ManifestPath $manifest.files.expert_pack.destination -ManifestKey "expert_pack" -FallbackPatterns @("*experts*.bin", "*expert*.bin")

if (-not $ServerBin) {
  $ServerBin = if ($env:LLAMA_SERVER_BIN) { $env:LLAMA_SERVER_BIN } else { "llama-server.exe" }
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

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdout = Join-Path $OutDir "$stamp-$Profile.out.log"
$stderr = Join-Path $OutDir "$stamp-$Profile.err.log"
$resultPath = Join-Path $OutDir "$stamp-$Profile.result.json"

$oldEnv = @{}
$envNames = @(
  "LLAMA_INFINITUM_SELECTIVE_MOE",
  "LLAMA_INFINITUM_EXPERT_INDEX",
  "LLAMA_INFINITUM_V2_GGML_EXPERT_PACK",
  "LLAMA_INFINITUM_EXPERT_BACKEND",
  "LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS",
  "LLAMA_INFINITUM_EXPERT_GPU_LAYER_SLOTS",
  "LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS",
  "LLAMA_INFINITUM_EXPERT_GPU_CACHE_MB",
  "LLAMA_INFINITUM_EXPERT_VULKAN_F16_INPUT",
  "LLAMA_INFINITUM_EXPERT_PREFETCH",
  "LLAMA_INFINITUM_GGML_PACK_PREFETCH",
  "LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS",
  "LLAMA_INFINITUM_GGML_PACK_PREFETCH_TOUCH_FALLBACK",
  "LLAMA_INFINITUM_PREFETCH_MAX_PENDING",
  "LLAMA_INFINITUM_EXPERT_PREDICTOR",
  "LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K",
  "LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD",
  "LLAMA_INFINITUM_EXPERT_CACHE_MB",
  "LLAMA_INFINITUM_EXPERT_PACKED_CACHE_MB",
  "LLAMA_INFINITUM_PACK_REPORT"
)
foreach ($name in $envNames) {
  $oldEnv[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
  Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$env:LLAMA_INFINITUM_SELECTIVE_MOE = "1"
$env:LLAMA_INFINITUM_EXPERT_INDEX = $index
$env:LLAMA_INFINITUM_V2_GGML_EXPERT_PACK = $pack
$env:LLAMA_INFINITUM_PACK_REPORT = "1"
if ($Profile -eq "page-prefetch") {
  $env:LLAMA_INFINITUM_EXPERT_PREFETCH = "1"
  $env:LLAMA_INFINITUM_GGML_PACK_PREFETCH = "1"
  $env:LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS = "$PrefetchMaxExperts"
  $env:LLAMA_INFINITUM_PREFETCH_MAX_PENDING = "1"
  $env:LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K = "$PrefetchMaxExperts"
  $env:LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD = "1"
  $env:LLAMA_INFINITUM_EXPERT_CACHE_MB = "$ExpertCacheMiB"
  $env:LLAMA_INFINITUM_EXPERT_PACKED_CACHE_MB = "$PackedExpertCacheMiB"
} elseif ($Profile -eq "streaming") {
  $env:LLAMA_INFINITUM_EXPERT_BACKEND = "vulkan"
  $env:LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS = "1"
  $env:LLAMA_INFINITUM_EXPERT_GPU_LAYER_SLOTS = "$GpuExpertLayerSlots"
  $env:LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS = "$GpuExpertGlobalSlots"
  $env:LLAMA_INFINITUM_EXPERT_GPU_CACHE_MB = "$GpuExpertCacheMiB"
  $env:LLAMA_INFINITUM_EXPERT_VULKAN_F16_INPUT = "1"
  $env:LLAMA_INFINITUM_EXPERT_PREFETCH = "1"
  $env:LLAMA_INFINITUM_PREFETCH_MAX_PENDING = "1"
  $env:LLAMA_INFINITUM_EXPERT_PREDICTOR = "learned"
  $env:LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K = "4"
  $env:LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD = "1"
  $env:LLAMA_INFINITUM_EXPERT_CACHE_MB = "$ExpertCacheMiB"
  $env:LLAMA_INFINITUM_EXPERT_PACKED_CACHE_MB = "$PackedExpertCacheMiB"
}

$args = @(
  "-m", $core,
  "-c", "$Context",
  "-t", "$Threads",
  "-tb", "$ThreadsBatch",
  "-n", "-1",
  "-ngl", "$GpuLayers",
  "-b", "$Batch",
  "-ub", "$UBatch",
  "--host", "127.0.0.1",
  "--port", "$Port",
  "--no-warmup"
)

$process = $null
try {
  $process = Start-Process -FilePath $ServerBin -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
  $healthy = $false
  for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) { break }
    try {
      $health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
      if ($health.StatusCode -eq 200) { $healthy = $true; break }
    } catch {}
  }
  if (-not $healthy) {
    throw "llama-server did not become healthy. stderr: $(Get-Content -Tail 60 $stderr -ErrorAction SilentlyContinue | Out-String)"
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
  $response = Invoke-Utf8Json -Uri "http://127.0.0.1:$Port/completion" -Payload $payload
  $ended = Get-Date
  $parsed = $response.Content | ConvertFrom-Json
  $timings = $parsed.timings
  $summary = [ordered]@{
    profile = $Profile
    server_bin = $ServerBin
    package_dir = $PackageDir
    port = $Port
    context = $Context
    threads = $Threads
    threads_batch = $ThreadsBatch
    gpu_layers = $GpuLayers
    batch = $Batch
    ubatch = $UBatch
    max_tokens = $MaxTokens
    wall_ms = [math]::Round(($ended - $started).TotalMilliseconds, 3)
    content = $parsed.content
    prompt_n = $timings.prompt_n
    prompt_ms = $timings.prompt_ms
    prompt_per_second = $timings.prompt_per_second
    predicted_n = $timings.predicted_n
    predicted_ms = $timings.predicted_ms
    predicted_per_second = $timings.predicted_per_second
    env = [ordered]@{
      LLAMA_INFINITUM_EXPERT_BACKEND = $env:LLAMA_INFINITUM_EXPERT_BACKEND
      LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS = $env:LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS
      LLAMA_INFINITUM_EXPERT_GPU_LAYER_SLOTS = $env:LLAMA_INFINITUM_EXPERT_GPU_LAYER_SLOTS
      LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS = $env:LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS
      LLAMA_INFINITUM_EXPERT_GPU_CACHE_MB = $env:LLAMA_INFINITUM_EXPERT_GPU_CACHE_MB
      LLAMA_INFINITUM_EXPERT_PREFETCH = $env:LLAMA_INFINITUM_EXPERT_PREFETCH
      LLAMA_INFINITUM_GGML_PACK_PREFETCH = $env:LLAMA_INFINITUM_GGML_PACK_PREFETCH
      LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS = $env:LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS
      LLAMA_INFINITUM_GGML_PACK_PREFETCH_TOUCH_FALLBACK = $env:LLAMA_INFINITUM_GGML_PACK_PREFETCH_TOUCH_FALLBACK
      LLAMA_INFINITUM_PREFETCH_MAX_PENDING = $env:LLAMA_INFINITUM_PREFETCH_MAX_PENDING
      LLAMA_INFINITUM_EXPERT_PREDICTOR = $env:LLAMA_INFINITUM_EXPERT_PREDICTOR
      LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K = $env:LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K
      LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD = $env:LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD
      LLAMA_INFINITUM_EXPERT_CACHE_MB = $env:LLAMA_INFINITUM_EXPERT_CACHE_MB
      LLAMA_INFINITUM_EXPERT_PACKED_CACHE_MB = $env:LLAMA_INFINITUM_EXPERT_PACKED_CACHE_MB
    }
    stdout = $stdout
    stderr = $stderr
  }
  $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resultPath -Encoding UTF8
  $summary | ConvertTo-Json -Depth 8
} finally {
  if ($process -and -not $process.HasExited) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
  }
  foreach ($name in $envNames) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    if ($null -ne $oldEnv[$name]) {
      [Environment]::SetEnvironmentVariable($name, $oldEnv[$name], "Process")
    }
  }
}
