param(
  [string]$BuildDir = "build/runtime-pc",
  [string]$InstallDir = "artifacts/runtime-pc",
  [string]$CmakePath = "",
  [string]$Generator = "",
  [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LlamaDir = Join-Path $RepoRoot "vendor\llama.cpp"

if (-not (Test-Path -LiteralPath (Join-Path $LlamaDir "CMakeLists.txt"))) {
  throw "Missing bundled runtime source: $LlamaDir"
}
if (-not $CmakePath) {
  $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
  if ($cmd) { $CmakePath = $cmd.Source }
}
if (-not $CmakePath) {
  throw "cmake.exe not found. Pass -CmakePath or install CMake."
}

$BuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $RepoRoot $BuildDir }
$InstallDir = if ([System.IO.Path]::IsPathRooted($InstallDir)) { $InstallDir } else { Join-Path $RepoRoot $InstallDir }

$configureArgs = @(
  "-S", $LlamaDir,
  "-B", $BuildDir,
  "-DCMAKE_BUILD_TYPE=$Config",
  "-DCMAKE_INSTALL_PREFIX=$InstallDir",
  "-DLLAMA_BUILD_TESTS=OFF",
  "-DLLAMA_BUILD_EXAMPLES=ON",
  "-DLLAMA_BUILD_SERVER=ON"
)
if ($Generator) {
  $configureArgs = @("-G", $Generator) + $configureArgs
}

& $CmakePath @configureArgs
if ($LASTEXITCODE -ne 0) {
  throw "CMake configure failed with exit code $LASTEXITCODE"
}

& $CmakePath --build $BuildDir --config $Config --target llama-cli llama-server llama-bench
if ($LASTEXITCODE -ne 0) {
  throw "Runtime build failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "bin") | Out-Null
$binRoot = Join-Path $BuildDir "bin"
if (Test-Path -LiteralPath (Join-Path $binRoot $Config)) {
  $binRoot = Join-Path $binRoot $Config
}
Copy-Item -Force (Join-Path $binRoot "*") (Join-Path $InstallDir "bin")
Write-Output "runtime pc bin: $(Join-Path $InstallDir "bin")"
