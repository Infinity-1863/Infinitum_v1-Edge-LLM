param(
  [string]$DeviceDir = "/data/local/tmp/infinitum-edge-gpt-oss-20b",
  [int]$PcPort = 18110,
  [int]$PhonePort = 8080,
  [int]$MaxTokens = 32,
  [string]$Message = "Напиши 6 коротких пунктов о том, как ускорить запуск модели на телефоне.",
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Bench = Join-Path $ScriptRoot "bench-phone.ps1"

$runs = @(
  @{ Label = "full-cpu"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=cpu") },
  @{ Label = "full-prefetch"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=cpu") },
  @{ Label = "full-prefetch4"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=cpu"); PrefetchMaxExperts = 4 },
  @{ Label = "full-simd"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=simd") },
  @{ Label = "full-simd-prefetch4"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=simd"); PrefetchMaxExperts = 4 },
  @{ Label = "full-simd-profile"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=simd", "LLAMA_INFINITUM_PROFILE=1") },
  @{ Label = "full-cpu-row2"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=cpu", "LLAMA_INFINITUM_EXPERT_ROW_THREADS=2") },
  @{ Label = "full-cpu-row4"; Env = @("LLAMA_INFINITUM_EXPERT_BACKEND=cpu", "LLAMA_INFINITUM_EXPERT_ROW_THREADS=4") },
  @{ Label = "full-workers2"; Env = @("LLAMA_INFINITUM_EXPERT_WORKERS=2") },
  @{ Label = "full-workers6"; Env = @("LLAMA_INFINITUM_EXPERT_WORKERS=6") }
)

foreach ($run in $runs) {
  $profile = if ($run.Label -eq "full-cpu") { "baseline" } else { "page-prefetch" }
  $benchArgs = @{
    DeviceDir = $DeviceDir
    Profile = $profile
    RunLabel = $run.Label
    PcPort = $PcPort
    PhonePort = $PhonePort
    Context = 512
    Threads = 4
    ThreadsBatch = 4
    GpuLayers = 0
    MaxTokens = $MaxTokens
    Message = $Message
  }
  if ($run.ContainsKey("PrefetchMaxExperts")) {
    $benchArgs.PrefetchMaxExperts = $run.PrefetchMaxExperts
  }
  if ($run.Env.Count -gt 0) {
    $benchArgs.ExtraEnv = $run.Env
  }
  if ($DryRun) {
    $benchArgs.DryRun = $true
  }
  Write-Host "sweep: $($run.Label) env=$($run.Env -join ',')"
  & $Bench @benchArgs
}
