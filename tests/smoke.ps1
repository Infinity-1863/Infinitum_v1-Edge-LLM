$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Assert-True {
  param(
    [bool]$Condition,
    [string]$Message
  )
  if (-not $Condition) {
    throw "ASSERT FAILED: $Message"
  }
}

function Assert-Contains {
  param(
    [string]$Haystack,
    [string]$Needle,
    [string]$Message
  )
  if (-not $Haystack.Contains($Needle)) {
    throw "ASSERT FAILED: $Message`nExpected to find: $Needle`nActual: $Haystack"
  }
}

function Assert-Matches {
  param(
    [string]$Haystack,
    [string]$Pattern,
    [string]$Message
  )
  if ($Haystack -notmatch $Pattern) {
    throw "ASSERT FAILED: $Message`nExpected pattern: $Pattern"
  }
}

$requiredFiles = @(
  "README.md",
  "docs\INDEX.md",
  "docs\PERFORMANCE.md",
  "docs\RUNTIME.md",
  "vendor\llama.cpp\CMakeLists.txt",
  "vendor\llama.cpp\src\llama-infinitum-moe.cpp",
  "vendor\llama.cpp\src\models\openai-moe.cpp",
  "scripts\split-model.ps1",
  "scripts\build-expert-pack.ps1",
  "scripts\build-split-pack.ps1",
  "scripts\build-runtime-pc.ps1",
  "scripts\build-runtime-android.ps1",
  "scripts\bench-pc.ps1",
  "scripts\bench-phone.ps1",
  "scripts\sweep-phone-full.ps1",
  "scripts\run-pc.ps1",
  "scripts\run-phone.ps1",
  "scripts\chat.ps1",
  "tools\build_expert_pack.py",
  "tools\split_ggml_expert_pack.py"
)

foreach ($file in $requiredFiles) {
  Assert-True (Test-Path -LiteralPath (Join-Path $RepoRoot $file)) "missing required file $file"
}

$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("infinitum-edge-smoke-" + [System.Guid]::NewGuid().ToString("N"))
$SourceDir = Join-Path $TempRoot "source"
$PackageDir = Join-Path $TempRoot "package"

New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null
try {
  $core = Join-Path $SourceDir "core-model.gguf"
  $index = Join-Path $SourceDir "expert-store-index.json"
  $pack = Join-Path $SourceDir "experts.ggml_f16.bin"

  Set-Content -LiteralPath $core -Value "core" -Encoding ASCII
  Set-Content -LiteralPath $index -Value '{"experts":[]}' -Encoding ASCII
  Set-Content -LiteralPath $pack -Value "experts" -Encoding ASCII

  & (Join-Path $RepoRoot "scripts\split-model.ps1") `
    -ModelDir $SourceDir `
    -OutDir $PackageDir `
    -ModelName "smoke-model" `
    -CoreFile $core `
    -ExpertIndex $index `
    -ExpertPack $pack `
    -LinkMode Copy | Out-Null

  $manifestPath = Join-Path $PackageDir "package.manifest.json"
  Assert-True (Test-Path -LiteralPath $manifestPath) "package manifest was not created"
  Assert-True ((Get-Item -LiteralPath $manifestPath).Length -gt 0) "package manifest is empty"
  Assert-True (Test-Path -LiteralPath (Join-Path $PackageDir "model\core-model.gguf")) "core model was not copied"
  Assert-True (Test-Path -LiteralPath (Join-Path $PackageDir "moe\expert-store-index.json")) "expert index was not copied"
  Assert-True (Test-Path -LiteralPath (Join-Path $PackageDir "moe_ggml_pack\experts.ggml_f16.bin")) "expert pack was not copied"

  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  $manifestRaw = Get-Content -LiteralPath $manifestPath -Raw
  Assert-True ($manifest.model_name -eq "smoke-model") "manifest model_name mismatch"
  Assert-True ($manifest.layout -eq "core-plus-external-experts") "manifest layout mismatch"
  Assert-True (-not $manifestRaw.Contains($TempRoot)) "package manifest should not leak temp absolute paths"
  Assert-True ($manifest.files.core.destination -eq "model/core-model.gguf") "manifest core destination should be package-relative"

  $pcPlan = & (Join-Path $RepoRoot "scripts\run-pc.ps1") `
    -PackageDir $PackageDir `
    -ServerBin "C:\tools\llama-server.exe" `
    -Port 18082 `
    -DryRun 2>&1 | Out-String
  Assert-Contains $pcPlan "llama-server.exe" "PC dry run should mention llama-server"
  Assert-Contains $pcPlan "LLAMA_INFINITUM_EXPERT_INDEX" "PC dry run should show expert index env"
  Assert-Contains $pcPlan "--host 127.0.0.1 --port 18082" "PC dry run should show bind and port"

  $conflictRoot = Join-Path $TempRoot "conflict-cwd"
  New-Item -ItemType Directory -Force -Path (Join-Path $conflictRoot "model"), (Join-Path $conflictRoot "moe"), (Join-Path $conflictRoot "moe_ggml_pack") | Out-Null
  Set-Content -LiteralPath (Join-Path $conflictRoot "model\core-model.gguf") -Value "wrong core" -Encoding ASCII
  Set-Content -LiteralPath (Join-Path $conflictRoot "moe\expert-store-index.json") -Value '{"wrong":true}' -Encoding ASCII
  Set-Content -LiteralPath (Join-Path $conflictRoot "moe_ggml_pack\experts.ggml_f16.bin") -Value "wrong experts" -Encoding ASCII
  Push-Location $conflictRoot
  try {
    $pcPlanFromConflict = & (Join-Path $RepoRoot "scripts\run-pc.ps1") `
      -PackageDir $PackageDir `
      -ServerBin "C:\tools\llama-server.exe" `
      -DryRun 2>&1 | Out-String
  } finally {
    Pop-Location
  }
  Assert-Contains $pcPlanFromConflict $PackageDir "PC dry run should resolve relative manifest paths under PackageDir, not CWD"
  Assert-True (-not $pcPlanFromConflict.Contains($conflictRoot)) "PC dry run should not pick conflicting files from CWD"

  $benchSource = Get-Content -LiteralPath (Join-Path $RepoRoot "scripts\bench-pc.ps1") -Raw
  Assert-Contains $benchSource "page-prefetch" "PC bench should expose universal page-prefetch profile"
  Assert-Contains $benchSource "LLAMA_INFINITUM_GGML_PACK_PREFETCH" "PC bench should enable GGML pack page prefetch"
  Assert-Contains $benchSource "LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS" "PC bench should bound page prefetch pressure"
  Assert-Contains $benchSource "LLAMA_INFINITUM_PREFETCH_MAX_PENDING = `"1`"" "PC page-prefetch profile should keep prefetch queue short"
  Assert-Contains $benchSource 'LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K = "$PrefetchMaxExperts"' "PC page-prefetch profile should bound predictor fanout"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD = `"1`"" "PC page-prefetch profile should keep the stable L+1 default"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD = `$env:LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD" "PC bench result should record predictor lookahead"
  Assert-Contains $benchSource "LLAMA_INFINITUM_GGML_PACK_PREFETCH_TOUCH_FALLBACK" "PC bench should clear opt-in touch fallback"
  Assert-Contains $benchSource "LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK" "PC bench should auto-use split GGML expert packs when present"
  Assert-Contains $benchSource "expert_split_pack" "PC bench should read split expert packs from the package manifest"
  Assert-Contains $benchSource '-not $useOneApiRuntime' "PC bench should keep merged expert packs on SYCL when oneAPI can handle them"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_PREDICTOR" "PC bench should keep learned prediction available for streaming diagnostics"
  Assert-Contains $benchSource "router-shadow" "PC bench should expose the router-shadow predictor profile"
  Assert-Contains $benchSource "sycl-hybrid" "PC bench should expose a SYCL merged-prefill split-decode profile"
  Assert-Contains $benchSource "sycl-slots" "PC bench should expose a SYCL persistent slot-cache profile"
  Assert-Contains $benchSource "sycl-router-slots" "PC bench should expose a SYCL router-shadow slot-cache profile"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-hybrid".*LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK = \$splitPack' "SYCL hybrid profile should force split-pack decode on oneAPI runtimes"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-slots".*LLAMA_INFINITUM_EXPERT_BACKEND = "sycl_arena"' "SYCL slot-cache profile should route decode through the fused SYCL arena"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-slots".*LLAMA_INFINITUM_V2_GGML_EXPERT_PACK_SLOTS_PREFILL = "0"' "SYCL slot-cache profile should keep prompt prefill out of GPU slot admission by default"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-router-slots".*LLAMA_INFINITUM_EXPERT_BACKEND = "sycl_arena"' "SYCL router-shadow slot-cache profile should route decode through the fused SYCL arena"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-router-slots".*LLAMA_INFINITUM_EXPERT_PREDICTOR = "router-shadow"' "SYCL router-shadow slot-cache profile should use future-router prediction"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-router-slots".*LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K = "8"' "SYCL router-shadow slot-cache profile should use top-8 future-router candidates"
  Assert-Matches $benchSource '(?s)Profile -eq "sycl-router-slots".*LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD = "1"' "SYCL router-shadow slot-cache profile should keep the best measured L+1 future-router prefetch"
  Assert-Matches $benchSource '(?s)LLAMA_INFINITUM_EXPERT_PREDICTOR = "router-shadow".*LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K = "6"' "router-shadow profile should use wider top-6 future-router candidates"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS" "PC bench should support bounded global GPU expert slots"
  Assert-Matches $benchSource '(?s)Profile -eq "streaming".*LLAMA_INFINITUM_GGML_PACK_PREFETCH = "1"' "PC streaming profile should enable router-driven GGML pack prefetch"
  Assert-Matches $benchSource '(?s)Profile -eq "streaming".*LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS = "4"' "PC streaming profile should prefetch top-4 router candidates"
  $pcBuildSource = Get-Content -LiteralPath (Join-Path $RepoRoot "scripts\build-runtime-pc.ps1") -Raw
  Assert-Contains $pcBuildSource '[ValidateSet("auto", "cpu", "vulkan", "sycl")]' "PC runtime build should expose an explicit backend selector"
  Assert-Contains $pcBuildSource '-DGGML_VULKAN=ON' "PC runtime build should be able to enable Vulkan for Intel Arc"
  Assert-Contains $pcBuildSource '-DGGML_SYCL=ON' "PC runtime build should document the SYCL path for oneAPI-capable Intel GPUs"
  Assert-Contains $pcBuildSource 'setvars.bat' "PC SYCL build should activate an installed oneAPI environment"
  Assert-Contains $pcBuildSource '-DCMAKE_CXX_COMPILER=icx' "PC SYCL build should use the Intel oneAPI compiler on Windows"
  Assert-Contains $pcBuildSource '-DGGML_SYCL_DEVICE_ARCH=' "PC SYCL build should support AOT device architecture selection"
  Assert-Contains $pcBuildSource '--verbose 2>nul' "PC SYCL architecture detection should ignore noisy loader stderr"
  Assert-Contains $pcBuildSource 'artifacts\sycl-temp-build' "PC SYCL build should use a writable local temp directory"
  Assert-Contains $pcBuildSource 'llama-ls-sycl-device' "PC SYCL build should include the SYCL device inspection tool"

  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_WORKERS" "PC bench should expose expert worker scheduling"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_ROW_THREADS" "PC bench should expose row-thread scheduling"
  Assert-Contains $benchSource "ONEAPI_DEVICE_SELECTOR" "PC bench should set the Intel Level Zero SYCL device selector"
  Assert-Contains $benchSource "UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS" "PC bench should set the Level Zero relaxed allocation guard"
  Assert-Contains $benchSource "FlashAttention" "PC bench should expose flash attention mode for backend graph comparisons"
  Assert-Contains $benchSource '"--flash-attn", "$FlashAttention"' "PC bench should pass flash attention mode to llama-server"

  $runPcSource = Get-Content -LiteralPath (Join-Path $RepoRoot "scripts\run-pc.ps1") -Raw
  Assert-Contains $runPcSource "LLAMA_INFINITUM_EXPERT_WORKERS" "PC run should expose expert worker scheduling"
  Assert-Contains $runPcSource "LLAMA_INFINITUM_EXPERT_ROW_THREADS" "PC run should expose row-thread scheduling"
  Assert-Contains $runPcSource "ONEAPI_DEVICE_SELECTOR" "PC run should support oneAPI SYCL runtime environment"
  Assert-Contains $runPcSource "FlashAttention" "PC run should expose flash attention mode"
  Assert-Contains $runPcSource '"--flash-attn", "$FlashAttention"' "PC run should pass flash attention mode to llama-server"
  Assert-Contains $runPcSource "LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK" "PC run should auto-use split GGML expert packs when present"
  Assert-Contains $runPcSource "expert_split_pack" "PC run should read split expert packs from the package manifest"
  Assert-Contains $runPcSource '-not $useOneApiRuntime' "PC run should keep merged expert packs on SYCL when oneAPI can handle them"

  $runtimeSource = Get-Content -LiteralPath (Join-Path $RepoRoot "vendor\llama.cpp\src\llama-infinitum-moe.cpp") -Raw
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_SELECTIVE_MOE" "bundled runtime should contain external expert support"
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_EXPERT_ROW_THREADS" "bundled runtime should contain full-model row threading knob"
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_PREFETCH_MAX_PENDING" "bundled runtime should expose bounded prefetch queue control"
  Assert-Contains $runtimeSource "infinitum_prefetch_queue" "bundled runtime should report dropped stale prefetch work"
  Assert-Matches $runtimeSource "llama_infinitum_moe_ggml_pack_slots_enabled\(\) &&\s*!llama_infinitum_moe_gpu_global_slots_enabled\(\) &&\s*llama_infinitum_moe_backend_uses_gpu_slots\(llama_infinitum_moe_backend_kind_from_env\(\)\)(?s:.*?)llama_infinitum_moe_prefetch_selected_gpu_experts(?s:.*?)llama_infinitum_moe_ggml_pack_prefetch_enabled\(\)(?s:.*?)llama_infinitum_moe_prefetch_selected_ggml_pack_pages" "prefetch worker should prefer safe per-layer GPU slots over page prefetch"
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_PREFETCH_GPU_BLOCKING" "GPU expert prefetch should expose an opt-in blocking mode"
  Assert-Contains $runtimeSource "llama_infinitum_ggml_gpu_upload_backend" "GPU expert prefetch should use a dedicated upload backend"
  Assert-Contains $runtimeSource "upload_backend" "GPU expert prefetch should enqueue uploads on a backend separate from foreground compute"
  Assert-True (-not $runtimeSource.Contains("std::unique_lock<std::mutex> compute_guard(state.compute_mutex(), std::defer_lock)")) "GPU expert prefetch must not take the foreground compute mutex"
  Assert-True (-not $runtimeSource.Contains("compute_guard.try_lock()")) "GPU expert prefetch must not poll the foreground compute lock"
  Assert-Contains $runtimeSource "foreground_waiters" "GPU backend should track foreground compute demand"
  Assert-Contains $runtimeSource "llama_infinitum_moe_backend_uses_gpu_slots" "GPU expert prefetch should share the Vulkan/SYCL slot-cache eligibility check"
  Assert-Matches $runtimeSource "llama_infinitum_moe_backend_uses_gpu_slots\(llama_infinitum_moe_backend_kind_from_env\(\)\)(?s:.*?)llama_infinitum_moe_prefetch_selected_gpu_experts" "prefetch worker should use GPU slots for both Vulkan and fused SYCL arenas"
  Assert-Matches $runtimeSource "llama_infinitum_moe_prefetch_selected_gpu_experts(?s:.*?)ggml_backend_synchronize\(upload_backend\);" "GPU expert prefetch should synchronize uploaded slots before they can be reused as hits"
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_V2_GGML_EXPERT_SPLIT_PACK" "GGML expert pack runtime should expose split gate/up physical layout"
  Assert-Contains $runtimeSource "llama_infinitum_moe_ggml_split_pack_enabled" "GGML expert pack runtime should detect split-pack mode"
  Assert-Contains $runtimeSource 'std::strcmp(kind, "gate")' "GGML expert pack runtime should create separate gate tensors"
  Assert-Contains $runtimeSource 'std::strcmp(kind, "up")' "GGML expert pack runtime should create separate up tensors"
  $backendSource = Get-Content -LiteralPath (Join-Path $RepoRoot "vendor\llama.cpp\ggml\src\ggml-backend.cpp") -Raw
  Assert-Matches $backendSource "ggml_backend_sched_moe_copy_cache_enabled\(\)(?s:.*?)LLAMA_INFINITUM_V2_GGML_EXPERT_PACK(?s:.*?)return true;" "native GGML MoE scheduler should enable resident expert copy cache automatically for external expert packs"
  Assert-Contains $backendSource "GGML_SCHED_MOE_COPY_CACHE_DISABLE" "native GGML MoE scheduler cache should keep an explicit opt-out"
  Assert-Contains $backendSource "ggml_backend_sched_moe_copy_cache_report" "native GGML MoE scheduler should expose copy-cache telemetry"
  Assert-Contains $backendSource "ggml_moe_copy_cache_report_final" "native GGML MoE scheduler should print a final aggregate copy-cache report"
  Assert-Contains $backendSource "id_wait_us" "native GGML MoE scheduler telemetry should measure router id readback stalls"
  Assert-Contains $backendSource "resident_hits" "native GGML MoE scheduler telemetry should measure resident expert hits"
  Assert-Contains $backendSource "GGML_SCHED_MOE_COPY_TRACE" "native GGML MoE scheduler should expose opt-in split/input tracing"
  Assert-Contains $backendSource "ggml_backend_sched_moe_copy_trace_enabled" "native GGML MoE scheduler should guard trace output behind a helper"
  Assert-Contains $backendSource "ggml_moe_copy_trace:" "native GGML MoE scheduler trace should use a stable prefix"
  Assert-Contains $backendSource "buffer_not_weights" "native GGML MoE scheduler trace should explain non-weight inputs"
  Assert-Contains $backendSource "not_host_buffer" "native GGML MoE scheduler trace should explain non-host inputs"
  Assert-Contains $backendSource "consumer_missing" "native GGML MoE scheduler trace should explain missing MoE consumers"
  Assert-Contains $backendSource "ggml_backend_sched_find_moe_weight_consumer" "native GGML MoE scheduler should find MUL_MAT_ID consumers across the full split graph"
  Assert-Contains $backendSource "ggml_backend_sched_moe_copy_cache_key" "native GGML MoE scheduler cache should be keyed by stable destination slots"
  Assert-Contains $backendSource "source_data" "native GGML MoE scheduler cache should survive rebuilt graph tensors for the same mapped weights"
  Assert-Contains $backendSource "dst_data" "native GGML MoE scheduler cache should detect when a destination slot is reused"
  Assert-True (-not $backendSource.Contains("ggml_tensor * node = split->graph.nodes[0];")) "native GGML MoE scheduler must not assume the MoE consumer is the first split node"
  $openAiMoeSource = Get-Content -LiteralPath (Join-Path $RepoRoot "vendor\llama.cpp\src\models\openai-moe.cpp") -Raw
  Assert-Matches $openAiMoeSource "llama_openai_moe_infinitum_prefetch_selected_op(?s:.*?)llama_openai_moe_infinitum_prefetch_learned_next_layers(?s:.*?)llama_openai_moe_infinitum_learned_predictor_record" "OpenAI-MoE graph prefetch should feed predictor state from router output"
  Assert-Contains $openAiMoeSource "prefetch_submit_ms" "OpenAI-MoE profile should expose early prefetch submit latency"
  Assert-Contains $openAiMoeSource "prediction_hits" "OpenAI-MoE profile should expose predictor hit count"
  Assert-Contains $openAiMoeSource "router_shadow" "OpenAI-MoE predictor should implement a router-shadow mode"
  Assert-Contains $openAiMoeSource "llama_openai_moe_infinitum_shadow_prefetch_op" "OpenAI-MoE should have a dedicated future-router shadow prefetch op"
  Assert-Contains $openAiMoeSource "build_lora_mm(model.layers[il + lookahead].ffn_gate_inp, cur)" "router-shadow should use real future router weights for lookahead prefetch"
  Assert-Contains $openAiMoeSource 'llama_infinitum_moe_ggml_pack_tensor(ctx0, expert_index, static_cast<int>(il), "gate")' "OpenAI-MoE GGML path should use separate gate tensors when split-pack is present"
  Assert-Contains $openAiMoeSource 'llama_infinitum_moe_ggml_pack_tensor(ctx0, expert_index, static_cast<int>(il), "up")' "OpenAI-MoE GGML path should use separate up tensors when split-pack is present"
  Assert-Contains $openAiMoeSource "ffn_moe_gate_external_ggml" "OpenAI-MoE GGML path should name separate gate matmuls"
  Assert-Contains $openAiMoeSource "ffn_moe_up_external_ggml" "OpenAI-MoE GGML path should name separate up matmuls"
  Assert-Contains $openAiMoeSource "cur->ne[1] == 1" "OpenAI-MoE GGML path should use split gate/up only for single-token decode"
  Assert-Matches $openAiMoeSource "cur->ne\[1\] == 1 && llama_infinitum_moe_ggml_pack_enabled\(\) &&\s*llama_openai_moe_infinitum_ggml_pack_prefetch_enabled\(\)" "OpenAI-MoE graph prefetch should run for GGML pack slots too"
  Assert-Matches $openAiMoeSource "llama_openai_moe_infinitum_expert_predictor_top_k\(\)\s*\{(?s:.*?)value == nullptr(?s:.*?)return 4;(?s:.*?)parsed <= 0(?s:.*?)return 4;" "OpenAI-MoE predictor top-k fallback should stay at safe top-4 unless explicitly overridden"
  Assert-Matches $openAiMoeSource "llama_openai_moe_infinitum_expert_predictor_lookahead\(\)\s*\{(?s:.*?)value == nullptr(?s:.*?)return 1;(?s:.*?)parsed <= 0(?s:.*?)return 1;" "OpenAI-MoE predictor lookahead fallback should stay at safe L+1 unless explicitly overridden"
  $prefetchIndex = $openAiMoeSource.IndexOf("prefetch_prediction =`r`n            llama_openai_moe_infinitum_prefetch_learned_next_layers")
  if ($prefetchIndex -lt 0) {
    $prefetchIndex = $openAiMoeSource.IndexOf("prefetch_prediction =`n            llama_openai_moe_infinitum_prefetch_learned_next_layers")
  }
  $executeIndex = if ($prefetchIndex -ge 0) { $openAiMoeSource.IndexOf("llama_infinitum_moe_execute_selected_experts_into", $prefetchIndex) } else { -1 }
  Assert-True ($prefetchIndex -ge 0 -and $executeIndex -ge 0 -and $prefetchIndex -lt $executeIndex) "OpenAI-MoE should submit lookahead prefetch before selected expert compute"

  $splitPackToolSource = Get-Content -LiteralPath (Join-Path $RepoRoot "tools\split_ggml_expert_pack.py") -Raw
  Assert-Contains $splitPackToolSource "experts.split.ggml_mxfp4.bin" "split-pack tool should emit the canonical split pack name"
  Assert-Contains $splitPackToolSource "expert_split_pack" "split-pack tool should add the split pack to the package manifest"
  Assert-Contains $splitPackToolSource "gate_up_expert_bytes" "split-pack tool should know the old merged gate_up layout"

  $phoneBenchPlan = & (Join-Path $RepoRoot "scripts\bench-phone.ps1") `
    -DeviceDir "/data/local/tmp/infinitum-edge-smoke" `
    -Profile page-prefetch `
    -PcPort 18110 `
    -PhonePort 8080 `
    -DryRun 2>&1 | Out-String
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_GGML_PACK_PREFETCH=1" "phone bench should enable GGML pack page prefetch"
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS=1" "phone bench should bound page prefetch pressure"
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_PREFETCH_MAX_PENDING=1" "phone bench should keep prefetch queue short"
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_EXPERT_PREDICTOR_TOP_K=1" "phone bench should bound predictor fanout"
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_EXPERT_PREDICTOR_LOOKAHEAD=1" "phone bench should keep the stable L+1 page-prefetch default"
  Assert-Contains $phoneBenchPlan "POST http://127.0.0.1:18110/completion" "phone bench should describe the forwarded completion endpoint"
  $phoneBenchSource = Get-Content -LiteralPath (Join-Path $RepoRoot "scripts\bench-phone.ps1") -Raw
  Assert-Contains $phoneBenchSource "LLAMA_INFINITUM_PROFILE" "phone bench should preserve profile env support through ExtraEnv"
  Assert-Contains $phoneBenchSource "compute_ms_sum" "phone bench should aggregate runtime profile compute time"
  Assert-Contains $phoneBenchSource "prediction_hit_rate" "phone bench should aggregate runtime predictor hit rate"
  $phoneSweepSource = Get-Content -LiteralPath (Join-Path $RepoRoot "scripts\sweep-phone-full.ps1") -Raw
  Assert-Contains $phoneSweepSource "LLAMA_INFINITUM_EXPERT_BACKEND=simd" "phone sweep should test full-model SIMD backend"
  Assert-Contains $phoneSweepSource "PrefetchMaxExperts = 4" "phone sweep should test wider full-model prefetch"
  Assert-True (-not $phoneSweepSource.Contains("LLAMA_INFINITUM_EXPERT_TOP_K")) "phone sweep must not reduce selected experts"

  $phonePlan = & (Join-Path $RepoRoot "scripts\run-phone.ps1") `
    -PackageDir $PackageDir `
    -DeviceDir "/data/local/tmp/infinitum-edge-smoke" `
    -PcPort 18080 `
    -PhonePort 8080 `
    -SmokeSeconds 15 `
    -DryRun 2>&1 | Out-String
  Assert-Contains $phonePlan "adb push" "phone dry run should show adb push"
  Assert-Contains $phonePlan "adb forward tcp:18080 tcp:8080" "phone dry run should show adb forward"
  Assert-Contains $phonePlan "LLAMA_INFINITUM_EXPERT_INDEX" "phone dry run should show expert index env"
  Assert-Contains $phonePlan "timeout 15 env" "phone dry run should support bounded smoke launches"

  $androidBin = Join-Path $TempRoot "android-bin"
  New-Item -ItemType Directory -Force -Path $androidBin | Out-Null
  Set-Content -LiteralPath (Join-Path $androidBin "llama-server") -Value "server" -Encoding ASCII
  $phonePlanWithBin = & (Join-Path $RepoRoot "scripts\run-phone.ps1") `
    -PackageDir $PackageDir `
    -AndroidBinDir $androidBin `
    -DeviceDir "/data/local/tmp/infinitum-edge-smoke" `
    -DryRun 2>&1 | Out-String
  Assert-Contains $phonePlanWithBin $androidBin "phone dry run should show concrete Android binary directory"
  Assert-Contains $phonePlanWithBin "adb push `"$androidBin\llama-server`" `"/data/local/tmp/infinitum-edge-smoke/bin/`"" "phone dry run should push Android server binary"

  $chatPayload = & (Join-Path $RepoRoot "scripts\chat.ps1") `
    -Message "привет" `
    -MaxTokens 16 `
    -Temperature 0 `
    -DryRun 2>&1 | Out-String
  $chatPayloadJson = $chatPayload | ConvertFrom-Json
  Assert-Contains $chatPayloadJson.prompt "<|start|>system<|message|>" "chat dry run should use GPT-OSS Harmony system turn"
  Assert-Contains $chatPayloadJson.prompt "<|start|>user<|message|>привет<|end|>" "chat dry run should preserve UTF-8 user text"
  Assert-Contains $chatPayloadJson.prompt "<|start|>assistant<|channel|>final<|message|>" "chat dry run should request final-channel assistant output"
  Assert-True ($chatPayloadJson.stop -contains "<|return|>") "chat dry run should include GPT-OSS return stop token"

  $mockServer = Join-Path $TempRoot "mock_chat_server.py"
  $mockOut = Join-Path $TempRoot "mock_chat_server.out"
  $mockErr = Join-Path $TempRoot "mock_chat_server.err"
  @'
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(raw.decode("utf-8"))
        prompt = payload.get("prompt", "")
        errors = []
        if self.path != "/completion":
            errors.append("wrong endpoint")
        if "привет" not in prompt:
            errors.append("missing utf8 cyrillic")
        if "<|start|>assistant<|channel|>final<|message|>" not in prompt:
            errors.append("missing final channel")
        if "<|return|>" not in payload.get("stop", []):
            errors.append("missing return stop")
        if errors:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(json.dumps({"content": "; ".join(errors)}).encode("utf-8"))
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"content": "mock ok"}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, *_args):
        return

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).handle_request()
'@ | Set-Content -LiteralPath $mockServer -Encoding UTF8
  $mockPort = 18993
  $mockProcess = Start-Process `
    -FilePath python `
    -ArgumentList @($mockServer, "$mockPort") `
    -PassThru `
    -WindowStyle Hidden `
    -RedirectStandardOutput $mockOut `
    -RedirectStandardError $mockErr
  Start-Sleep -Seconds 1
  try {
    $chatResult = & (Join-Path $RepoRoot "scripts\chat.ps1") `
      -ServerUrl "http://127.0.0.1:$mockPort" `
      -Message "привет" `
      -MaxTokens 16 `
      -Temperature 0 2>&1 | Out-String
    Assert-Contains $chatResult "mock ok" "chat should send UTF-8 Harmony payload to /completion"
  } finally {
    if ($mockProcess -and -not $mockProcess.HasExited) {
      Stop-Process -Id $mockProcess.Id -Force -ErrorAction SilentlyContinue
    }
  }

  $hfSource = Join-Path $TempRoot "hf-source"
  $hfPackage = Join-Path $TempRoot "hf-package"
  New-Item -ItemType Directory -Force -Path $hfSource | Out-Null
  Set-Content -LiteralPath (Join-Path $hfSource "config.json") -Value '{"model_type":"gpt_oss","architectures":["GptOssForCausalLM"],"num_hidden_layers":1,"num_local_experts":32}' -Encoding ASCII
  Set-Content -LiteralPath (Join-Path $hfSource "model-00000-of-00001.safetensors") -Value "dummy shard" -Encoding ASCII
  $hfIndex = @{
    metadata = @{ total_size = 1234 }
    weight_map = @{
      "model.layers.0.input_layernorm.weight" = "model-00000-of-00001.safetensors"
      "model.layers.0.mlp.router.weight" = "model-00000-of-00001.safetensors"
      "model.layers.0.mlp.experts.gate_up_proj_blocks" = "model-00000-of-00001.safetensors"
    }
  } | ConvertTo-Json -Depth 6
  Set-Content -LiteralPath (Join-Path $hfSource "model.safetensors.index.json") -Value $hfIndex -Encoding ASCII

  & (Join-Path $RepoRoot "scripts\split-model.ps1") `
    -ModelDir $hfSource `
    -OutDir $hfPackage `
    -ModelName "hf-smoke" | Out-Null

  $hfManifest = Get-Content -LiteralPath (Join-Path $hfPackage "package.manifest.json") -Raw | ConvertFrom-Json
  $hfManifestRaw = Get-Content -LiteralPath (Join-Path $hfPackage "package.manifest.json") -Raw
  Assert-True ($hfManifest.layout -eq "hf-safetensors-source") "HF source layout should be detected"
  Assert-True ($hfManifest.tensor_summary.expert_tensor_count -eq 2) "HF expert tensor count mismatch"
  Assert-True ($hfManifest.tensor_summary.core_tensor_count -eq 1) "HF core tensor count mismatch"
  Assert-True (-not $hfManifestRaw.Contains($TempRoot)) "HF source manifest should not leak temp absolute paths"

  $packSource = Join-Path $TempRoot "pack-source"
  $packOut = Join-Path $TempRoot "pack-out"
  New-Item -ItemType Directory -Force -Path $packSource | Out-Null
  @'
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

root = Path(r"PACK_SOURCE")
tensor_path = root / "model-00000-of-00001.safetensors"
tensors = {
    "model.layers.0.input_layernorm.weight": torch.zeros(2, dtype=torch.float32),
    "model.layers.0.mlp.router.weight": torch.zeros((2, 2), dtype=torch.float32),
    "model.layers.0.mlp.experts.gate_up_proj_bias": torch.zeros((3, 2), dtype=torch.bfloat16),
    "model.layers.0.mlp.experts.gate_up_proj_blocks": torch.arange(12, dtype=torch.uint8).reshape(3, 4),
    "model.layers.0.mlp.experts.down_proj_blocks": torch.arange(12, 24, dtype=torch.uint8).reshape(3, 4),
}
save_file(tensors, str(tensor_path))
index = {"metadata": {"total_size": tensor_path.stat().st_size}, "weight_map": {name: tensor_path.name for name in tensors}}
(root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
(root / "config.json").write_text(json.dumps({"model_type": "gpt_oss", "num_hidden_layers": 1, "num_local_experts": 3}), encoding="utf-8")
'@.Replace("PACK_SOURCE", ($packSource -replace '\\', '\\')) | python -

  & (Join-Path $RepoRoot "scripts\build-expert-pack.ps1") `
    -ModelDir $packSource `
    -OutDir $packOut `
    -ModelName "pack-smoke" | Out-Null

  $packManifestPath = Join-Path $packOut "expert-pack.manifest.json"
  $packBinPath = Join-Path $packOut "experts.pack.bin"
  Assert-True (Test-Path -LiteralPath $packManifestPath) "expert pack manifest was not created"
  Assert-True (Test-Path -LiteralPath $packBinPath) "expert pack binary was not created"
  $packManifest = Get-Content -LiteralPath $packManifestPath -Raw | ConvertFrom-Json
  $packManifestRaw = Get-Content -LiteralPath $packManifestPath -Raw
  Assert-True ($packManifest.format -eq "infinitum_edge_expert_pack_v1") "expert pack format mismatch"
  Assert-True ($packManifest.summary.entry_count -eq 9) "expert pack entry count mismatch"
  Assert-True ($packManifest.summary.layer_count -eq 1) "expert pack layer count mismatch"
  Assert-True (-not $packManifestRaw.Contains($TempRoot)) "expert pack manifest should not leak temp absolute paths"
} finally {
  Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "smoke: ok"
