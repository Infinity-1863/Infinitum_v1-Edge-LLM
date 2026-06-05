param(
  [string]$BuildDir = "build/runtime-pc",
  [string]$InstallDir = "artifacts/runtime-pc",
  [string]$CmakePath = "",
  [string]$Generator = "",
  [string]$Config = "Release",
  [ValidateSet("auto", "cpu", "vulkan", "sycl")]
  [string]$Backend = "auto",
  [string]$OneApiRoot = "",
  [string]$SyclDeviceArch = "auto",
  [string]$SyclTempDir = "artifacts\sycl-temp-build",
  [switch]$SyclF16,
  [int]$Parallel = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LlamaDir = Join-Path $RepoRoot "vendor\llama.cpp"

if (-not (Test-Path -LiteralPath (Join-Path $LlamaDir "CMakeLists.txt"))) {
  throw "Missing bundled runtime source: $LlamaDir"
}

function Test-InfinitumCommand {
  param([string]$Name)
  return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Import-InfinitumOneApiEnvironment {
  param([string]$Root)
  $candidates = @()
  if ($Root) { $candidates += $Root }
  if ($env:ONEAPI_ROOT) { $candidates += $env:ONEAPI_ROOT }
  $candidates += "C:\Program Files (x86)\Intel\oneAPI"

  foreach ($candidate in $candidates) {
    $setvars = Join-Path $candidate "setvars.bat"
    if (-not (Test-Path -LiteralPath $setvars)) {
      continue
    }
    $cmd = 'call "' + $setvars + '" intel64 --force >nul && set'
    $lines = & cmd.exe /d /c $cmd
    if ($LASTEXITCODE -ne 0) {
      throw "oneAPI setvars failed: $setvars"
    }
    foreach ($line in $lines) {
      $idx = $line.IndexOf("=")
      if ($idx -gt 0) {
        [Environment]::SetEnvironmentVariable($line.Substring(0, $idx), $line.Substring($idx + 1), "Process")
      }
    }
    return $setvars
  }
  return ""
}

function Convert-InfinitumSyclArchitecture {
  param([string]$Architecture)
  if ($Architecture -match '^intel_gpu_(.+)$') {
    return $matches[1].Replace("_", "-")
  }
  return ""
}

function Get-InfinitumSyclDeviceArch {
  $syclLs = Get-Command sycl-ls.exe -ErrorAction SilentlyContinue
  if (-not $syclLs) {
    return ""
  }
  $cmd = '"' + $syclLs.Source + '" --verbose 2>nul'
  $lines = & cmd.exe /d /c $cmd
  foreach ($line in $lines) {
    if ($line -match 'Architecture:\s*(\S+)') {
      $arch = Convert-InfinitumSyclArchitecture $matches[1]
      if ($arch) { return $arch }
    }
  }
  return ""
}

if (($Backend -eq "auto") -or ($Backend -eq "sycl")) {
  $setvars = Import-InfinitumOneApiEnvironment -Root $OneApiRoot
  if ($setvars) {
    Write-Output "oneAPI env: $setvars"
  }
}

if (-not $CmakePath) {
  $cmd = Get-Command cmake.exe -ErrorAction SilentlyContinue
  if ($cmd) { $CmakePath = $cmd.Source }
}
if (-not $CmakePath) {
  throw "cmake.exe not found. Pass -CmakePath or install CMake."
}

if ($Backend -eq "auto") {
  $hasSycl = (Test-InfinitumCommand "icx.exe") -or (Test-InfinitumCommand "icpx.exe") -or (Test-InfinitumCommand "dpcpp.exe")
  $hasMkl = [bool]$env:MKLROOT
  $hasVulkan = [bool]$env:VULKAN_SDK -or (Test-InfinitumCommand "glslc.exe") -or (Test-InfinitumCommand "vulkaninfo.exe")
  if ($hasSycl -and $hasMkl) {
    $Backend = "sycl"
  } elseif ($hasVulkan) {
    $Backend = "vulkan"
  } else {
    $Backend = "cpu"
  }
}

if ($Backend -eq "sycl") {
  if (-not ((Test-InfinitumCommand "icx.exe") -or (Test-InfinitumCommand "dpcpp.exe"))) {
    throw "oneAPI DPC++ compiler not found. Install Intel oneAPI or pass -OneApiRoot."
  }
  if (-not $env:MKLROOT) {
    throw "oneAPI MKLROOT is not set. Install Intel oneAPI Base Toolkit or pass -OneApiRoot."
  }
  if (-not $Generator) {
    if (Test-InfinitumCommand "ninja.exe") {
      $Generator = "Ninja"
    }
  }
}

$BuildDir = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $RepoRoot $BuildDir }
$InstallDir = if ([System.IO.Path]::IsPathRooted($InstallDir)) { $InstallDir } else { Join-Path $RepoRoot $InstallDir }
$SyclTempDir = if ([System.IO.Path]::IsPathRooted($SyclTempDir)) { $SyclTempDir } else { Join-Path $RepoRoot $SyclTempDir }

$oldTemp = $env:TEMP
$oldTmp = $env:TMP
try {
  $configureArgs = @(
    "-S", $LlamaDir,
    "-B", $BuildDir,
    "-DCMAKE_BUILD_TYPE=$Config",
    "-DCMAKE_INSTALL_PREFIX=$InstallDir",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DGGML_OPENCL=OFF",
    "-DGGML_CUDA=OFF"
  )
  if ($Backend -eq "vulkan") {
    $configureArgs += @(
      "-DGGML_VULKAN=ON",
      "-DGGML_SYCL=OFF"
    )
  } elseif ($Backend -eq "sycl") {
    New-Item -ItemType Directory -Force -Path $SyclTempDir | Out-Null
    $env:TEMP = $SyclTempDir
    $env:TMP = $SyclTempDir

    $configureArgs += @(
      "-DGGML_VULKAN=OFF",
      "-DGGML_SYCL=ON",
      "-DGGML_SYCL_TARGET=INTEL",
      "-DCMAKE_C_COMPILER=cl",
      "-DCMAKE_CXX_COMPILER=icx"
    )
    $arch = $SyclDeviceArch
    if ($arch -eq "auto") {
      $arch = Get-InfinitumSyclDeviceArch
    }
    if ($arch) {
      $configureArgs += "-DGGML_SYCL_DEVICE_ARCH=$arch"
      Write-Output "SYCL device arch: $arch"
    }
    if ($SyclF16) {
      $configureArgs += "-DGGML_SYCL_F16=ON"
    }
  } else {
    $configureArgs += @(
      "-DGGML_VULKAN=OFF",
      "-DGGML_SYCL=OFF"
    )
  }
  if ($Generator) {
    $configureArgs = @("-G", $Generator) + $configureArgs
  }

  Write-Output "runtime pc backend: $Backend"
  & $CmakePath @configureArgs
  if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
  }

  $targets = @("llama-cli", "llama-server", "llama-bench")
  if ($Backend -eq "sycl") {
    $targets += "llama-ls-sycl-device"
  }
  $buildArgs = @("--build", $BuildDir, "--config", $Config, "--target") + $targets
  if ($Parallel -gt 0) {
    $buildArgs += @("--parallel", "$Parallel")
  }
  & $CmakePath @buildArgs
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
} finally {
  $env:TEMP = $oldTemp
  $env:TMP = $oldTmp
}
