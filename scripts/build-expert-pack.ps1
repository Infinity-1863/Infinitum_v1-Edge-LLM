param(
  [Parameter(Mandatory=$true)][string]$ModelDir,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [string]$ModelName = "",
  [string]$Python = "python",
  [string]$PackName = "experts.pack.bin",
  [string]$ManifestName = "expert-pack.manifest.json",
  [switch]$Force
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$tool = Join-Path $RepoRoot "tools\build_expert_pack.py"
if (-not (Test-Path -LiteralPath $tool)) {
  throw "Missing pack builder: $tool"
}

$args = @(
  $tool,
  "--model-dir", $ModelDir,
  "--out-dir", $OutDir,
  "--pack-name", $PackName,
  "--manifest-name", $ManifestName
)
if ($ModelName) { $args += @("--model-name", $ModelName) }
if ($Force) { $args += "--force" }

& $Python @args
if ($LASTEXITCODE -ne 0) {
  throw "expert pack build failed with exit code $LASTEXITCODE"
}
