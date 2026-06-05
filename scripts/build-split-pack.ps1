param(
  [Parameter(Mandatory=$true)][string]$PackageDir,
  [string]$Python = "python",
  [string]$SourcePack = "",
  [string]$SourceManifest = "",
  [string]$OutPack = "",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$tool = Join-Path $RepoRoot "tools\split_ggml_expert_pack.py"
if (-not (Test-Path -LiteralPath $tool)) {
  throw "Missing split-pack builder: $tool"
}

$args = @(
  $tool,
  "--package-dir", $PackageDir
)
if ($SourcePack) { $args += @("--src-pack", $SourcePack) }
if ($SourceManifest) { $args += @("--src-manifest", $SourceManifest) }
if ($OutPack) { $args += @("--out-pack", $OutPack) }
if ($Force) { $args += "--force" }

& $Python @args
if ($LASTEXITCODE -ne 0) {
  throw "split expert pack build failed with exit code $LASTEXITCODE"
}
