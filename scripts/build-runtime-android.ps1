param(
  [string]$BuildDir = "build/runtime-android-arm64",
  [string]$InstallDir = "artifacts/runtime-android-arm64",
  [string]$AndroidNdk = "",
  [string]$CmakePath = "",
  [string]$NinjaPath = "",
  [string]$Abi = "arm64-v8a",
  [string]$Platform = "android-28",
  [string]$CpuArch = "armv8.2-a+fp16",
  [switch]$WithVulkan
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LlamaDir = Join-Path $RepoRoot "vendor\llama.cpp"

if (-not (Test-Path -LiteralPath (Join-Path $LlamaDir "CMakeLists.txt"))) {
  throw "Missing bundled runtime source: $LlamaDir"
}
if (-not $AndroidNdk) {
  $sdk = Join-Path $env:LOCALAPPDATA "Android\Sdk"
  $ndkRoot = Join-Path $sdk "ndk"
  $AndroidNdk = Get-ChildItem $ndkRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    Select-Object -First 1 -ExpandProperty FullName
}
if (-not $AndroidNdk) {
  throw "Android NDK not found. Pass -AndroidNdk or install it under %LOCALAPPDATA%\Android\Sdk\ndk."
}
if (-not $CmakePath) {
  $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
  if ($cmd) { $CmakePath = $cmd.Source }
}
if (-not $CmakePath) {
  $candidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    "C:\Program Files\Unity\Hub\Editor\6000.0.62f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\cmake\3.22.1\bin\cmake.exe"
  )
  $CmakePath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $CmakePath) {
  throw "cmake.exe not found. Pass -CmakePath."
}
if (-not $NinjaPath) {
  $cmd = Get-Command ninja.exe -ErrorAction SilentlyContinue
  if ($cmd) { $NinjaPath = $cmd.Source }
}
if (-not $NinjaPath) {
  $candidates = @(
    "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe",
    "C:\Program Files\Unity\Hub\Editor\6000.0.62f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\cmake\3.22.1\bin\ninja.exe"
  )
  $NinjaPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $NinjaPath) {
  throw "ninja.exe not found. Pass -NinjaPath."
}

$BuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $RepoRoot $BuildDir }
$InstallDir = if ([System.IO.Path]::IsPathRooted($InstallDir)) { $InstallDir } else { Join-Path $RepoRoot $InstallDir }
$env:PATH = "$(Split-Path -Parent $CmakePath);$(Split-Path -Parent $NinjaPath);$env:PATH"

$toolchain = Join-Path $AndroidNdk "build\cmake\android.toolchain.cmake"
if (-not (Test-Path -LiteralPath $toolchain)) {
  throw "Android CMake toolchain not found: $toolchain"
}

$vulkan = if ($WithVulkan) { "ON" } else { "OFF" }
$configureArgs = @(
  "-S", $LlamaDir,
  "-B", $BuildDir,
  "-G", "Ninja",
  "-DCMAKE_MAKE_PROGRAM=$NinjaPath",
  "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
  "-DANDROID_ABI=$Abi",
  "-DANDROID_PLATFORM=$Platform",
  "-DGGML_CPU_ARM_ARCH=$CpuArch",
  "-DCMAKE_C_FLAGS=-march=$CpuArch",
  "-DCMAKE_CXX_FLAGS=-march=$CpuArch",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=$InstallDir",
  "-DGGML_OPENMP=OFF",
  "-DGGML_LLAMAFILE=OFF",
  "-DGGML_VULKAN=$vulkan",
  "-DLLAMA_BUILD_TESTS=OFF",
  "-DLLAMA_BUILD_EXAMPLES=ON",
  "-DLLAMA_BUILD_SERVER=ON"
)

& $CmakePath @configureArgs
if ($LASTEXITCODE -ne 0) {
  throw "CMake configure failed with exit code $LASTEXITCODE"
}

& $CmakePath --build $BuildDir --config Release --target llama-cli llama-server llama-bench
if ($LASTEXITCODE -ne 0) {
  throw "Android runtime build failed with exit code $LASTEXITCODE"
}

New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "bin") | Out-Null
if (-not $WithVulkan) {
  Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $BuildDir "bin\libggml-vulkan.so")
  Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $InstallDir "bin\libggml-vulkan.so")
}
Copy-Item -Force (Join-Path $BuildDir "bin\*") (Join-Path $InstallDir "bin")
Write-Output "runtime android bin: $(Join-Path $InstallDir "bin")"
