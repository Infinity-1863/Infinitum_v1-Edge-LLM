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
  "scripts\build-runtime-pc.ps1",
  "scripts\build-runtime-android.ps1",
  "scripts\bench-pc.ps1",
  "scripts\bench-phone.ps1",
  "scripts\sweep-phone-full.ps1",
  "scripts\run-pc.ps1",
  "scripts\run-phone.ps1",
  "scripts\chat.ps1",
  "tools\build_expert_pack.py"
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
  Assert-Contains $benchSource "LLAMA_INFINITUM_GGML_PACK_PREFETCH_TOUCH_FALLBACK" "PC bench should clear opt-in touch fallback"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_PREDICTOR" "PC bench should keep learned prediction available for streaming diagnostics"
  Assert-Contains $benchSource "LLAMA_INFINITUM_EXPERT_GPU_GLOBAL_SLOTS" "PC bench should support bounded global GPU expert slots"

  $runtimeSource = Get-Content -LiteralPath (Join-Path $RepoRoot "vendor\llama.cpp\src\llama-infinitum-moe.cpp") -Raw
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_SELECTIVE_MOE" "bundled runtime should contain external expert support"
  Assert-Contains $runtimeSource "LLAMA_INFINITUM_EXPERT_ROW_THREADS" "bundled runtime should contain full-model row threading knob"

  $phoneBenchPlan = & (Join-Path $RepoRoot "scripts\bench-phone.ps1") `
    -DeviceDir "/data/local/tmp/infinitum-edge-smoke" `
    -Profile page-prefetch `
    -PcPort 18110 `
    -PhonePort 8080 `
    -DryRun 2>&1 | Out-String
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_GGML_PACK_PREFETCH=1" "phone bench should enable GGML pack page prefetch"
  Assert-Contains $phoneBenchPlan "LLAMA_INFINITUM_GGML_PACK_PREFETCH_MAX_EXPERTS=1" "phone bench should bound page prefetch pressure"
  Assert-Contains $phoneBenchPlan "POST http://127.0.0.1:18110/completion" "phone bench should describe the forwarded completion endpoint"
  $phoneBenchSource = Get-Content -LiteralPath (Join-Path $RepoRoot "scripts\bench-phone.ps1") -Raw
  Assert-Contains $phoneBenchSource "LLAMA_INFINITUM_PROFILE" "phone bench should preserve profile env support through ExtraEnv"
  Assert-Contains $phoneBenchSource "compute_ms_sum" "phone bench should aggregate runtime profile compute time"
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
