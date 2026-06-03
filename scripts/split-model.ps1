param(
  [Parameter(Mandatory=$true)][string]$ModelDir,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [string]$ModelName = "",
  [string]$CoreFile = "",
  [string]$ExpertIndex = "",
  [string]$ExpertPack = "",
  [ValidateSet("ManifestOnly", "Copy", "Hardlink", "Symlink")]
  [string]$LinkMode = "ManifestOnly",
  [switch]$DryRun,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
  param(
    [string]$Path,
    [string]$Name
  )
  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "$Name does not exist: $Path"
  }
  return (Resolve-Path -LiteralPath $Path).Path
}

function Find-FirstFile {
  param(
    [string]$Root,
    [string[]]$Include,
    [string[]]$PreferNameRegex,
    [string[]]$RejectNameRegex = @()
  )
  $files = Get-ChildItem -LiteralPath $Root -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object {
      $name = $_.Name
      $extOk = $false
      foreach ($pattern in $Include) {
        if ($name -like $pattern) { $extOk = $true; break }
      }
      if (-not $extOk) { return $false }
      foreach ($reject in $RejectNameRegex) {
        if ($name -match $reject) { return $false }
      }
      return $true
    }

  foreach ($prefer in $PreferNameRegex) {
    $hit = $files | Where-Object { $_.Name -match $prefer } | Sort-Object Length -Descending | Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }

  $fallback = $files | Sort-Object Length -Descending | Select-Object -First 1
  if ($fallback) { return $fallback.FullName }
  return ""
}

function Format-Size {
  param([string]$Path)
  if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return 0 }
  return (Get-Item -LiteralPath $Path).Length
}

function Get-RelativePackagePath {
  param([string]$Path)
  $base = [System.IO.Path]::GetFullPath($OutDirFull)
  if (-not $base.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
    $base += [System.IO.Path]::DirectorySeparatorChar
  }
  $target = [System.IO.Path]::GetFullPath($Path)
  $relative = (New-Object System.Uri($base)).MakeRelativeUri((New-Object System.Uri($target))).ToString()
  return [System.Uri]::UnescapeDataString($relative)
}

function Copy-Or-Link {
  param(
    [string]$Source,
    [string]$Destination,
    [string]$Mode
  )
  if ($DryRun -or $Mode -eq "ManifestOnly") {
    Write-Host "plan: $Mode $Source -> $Destination"
    return
  }

  $parent = Split-Path -Parent $Destination
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  if ((Test-Path -LiteralPath $Destination) -and -not $Force) {
    throw "Destination exists. Pass -Force to overwrite: $Destination"
  }
  if (Test-Path -LiteralPath $Destination) {
    Remove-Item -LiteralPath $Destination -Force
  }

  switch ($Mode) {
    "Copy" {
      Copy-Item -LiteralPath $Source -Destination $Destination
    }
    "Hardlink" {
      New-Item -ItemType HardLink -Path $Destination -Target $Source | Out-Null
    }
    "Symlink" {
      New-Item -ItemType SymbolicLink -Path $Destination -Target $Source | Out-Null
    }
  }
}

$ModelDir = Resolve-ExistingPath -Path $ModelDir -Name "ModelDir"
if (-not $ModelName) {
  $ModelName = Split-Path -Leaf $ModelDir
}
$OutDirFull = [System.IO.Path]::GetFullPath($OutDir)

$CoreFile = Resolve-ExistingPath -Path $CoreFile -Name "CoreFile"
$ExpertIndex = Resolve-ExistingPath -Path $ExpertIndex -Name "ExpertIndex"
$ExpertPack = Resolve-ExistingPath -Path $ExpertPack -Name "ExpertPack"

if (-not $CoreFile) {
  $CoreFile = Find-FirstFile `
    -Root $ModelDir `
    -Include @("*.gguf") `
    -PreferNameRegex @("core", "model") `
    -RejectNameRegex @("expert", "moe", "pack")
}
if (-not $ExpertIndex) {
  $ExpertIndex = Find-FirstFile `
    -Root $ModelDir `
    -Include @("*.json") `
    -PreferNameRegex @("expert.*index", "moe.*index", "manifest")
}
if (-not $ExpertPack) {
  $ExpertPack = Find-FirstFile `
    -Root $ModelDir `
    -Include @("*.bin", "*.gguf") `
    -PreferNameRegex @("experts", "expert", "moe.*pack")
}

if (-not $CoreFile) {
  $hfIndexPath = Join-Path $ModelDir "model.safetensors.index.json"
  if (Test-Path -LiteralPath $hfIndexPath) {
    $hfIndex = Get-Content -LiteralPath $hfIndexPath -Raw | ConvertFrom-Json
    $tensorNames = @($hfIndex.weight_map.PSObject.Properties.Name)
    $expertTensorNames = @($tensorNames | Where-Object { $_ -match '(^|\.)(mlp\.)?(router|experts)(\.|$)|expert|moe' })
    $expertSet = @{}
    foreach ($name in $expertTensorNames) { $expertSet[$name] = $true }
    $coreTensorNames = @($tensorNames | Where-Object { -not $expertSet.ContainsKey($_) })
    $shardNames = @($hfIndex.weight_map.PSObject.Properties.Value | Sort-Object -Unique)
    $shards = @(
      foreach ($shard in $shardNames) {
        $path = Join-Path $ModelDir $shard
        [ordered]@{
          name = $shard
          bytes = Format-Size $path
        }
      }
    )
    $configPath = Join-Path $ModelDir "config.json"
    $config = $null
    if (Test-Path -LiteralPath $configPath) {
      $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    }

    $manifestDest = Join-Path $OutDirFull "package.manifest.json"
    $manifest = [ordered]@{
      schema_version = 1
      model_name = $ModelName
      layout = "hf-safetensors-source"
      created_utc = [DateTime]::UtcNow.ToString("o")
      source_model = Split-Path -Leaf $ModelDir
      model_config = [ordered]@{
        model_type = if ($config) { $config.model_type } else { $null }
        architectures = if ($config) { $config.architectures } else { @() }
        num_hidden_layers = if ($config) { $config.num_hidden_layers } else { $null }
        num_local_experts = if ($config) { $config.num_local_experts } else { $null }
      }
      tensor_summary = [ordered]@{
        tensor_count = $tensorNames.Count
        core_tensor_count = $coreTensorNames.Count
        expert_tensor_count = $expertTensorNames.Count
        shard_count = $shardNames.Count
      }
      shards = $shards
      tensor_groups = [ordered]@{
        core = $coreTensorNames
        experts = $expertTensorNames
      }
      notes = @(
        "HF safetensors source detected. This manifest classifies tensors but does not rewrite shard bytes.",
        "Use a converter/packer to build core GGUF and external expert pack before run-pc.ps1 or run-phone.ps1.",
        "Public repos should not commit safetensors shards unless model licensing allows redistribution."
      )
    }

    if ($DryRun) {
      Write-Host "dry-run: would write HF source manifest $manifestDest"
      $manifest | ConvertTo-Json -Depth 10
    } else {
      New-Item -ItemType Directory -Force -Path $OutDirFull | Out-Null
      $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestDest -Encoding ASCII
      Write-Host "created HF source manifest: $manifestDest"
    }
    return
  }
}

if (-not $CoreFile) { throw "Could not find a core GGUF. Pass -CoreFile explicitly." }
if (-not $ExpertIndex) { throw "Could not find an expert index or MoE manifest. Pass -ExpertIndex explicitly." }
if (-not $ExpertPack) { throw "Could not find an expert pack. Pass -ExpertPack explicitly." }

$modelOut = Join-Path $OutDirFull "model"
$moeOut = Join-Path $OutDirFull "moe"
$packOut = Join-Path $OutDirFull "moe_ggml_pack"

$coreDest = Join-Path $modelOut (Split-Path -Leaf $CoreFile)
$indexDest = Join-Path $moeOut (Split-Path -Leaf $ExpertIndex)
$packDest = Join-Path $packOut (Split-Path -Leaf $ExpertPack)
$manifestDest = Join-Path $OutDirFull "package.manifest.json"

if ($DryRun) {
  Write-Host "dry-run: would create package at $OutDirFull"
} else {
  New-Item -ItemType Directory -Force -Path $modelOut, $moeOut, $packOut | Out-Null
}

Copy-Or-Link -Source $CoreFile -Destination $coreDest -Mode $LinkMode
Copy-Or-Link -Source $ExpertIndex -Destination $indexDest -Mode $LinkMode
Copy-Or-Link -Source $ExpertPack -Destination $packDest -Mode $LinkMode

$manifest = [ordered]@{
  schema_version = 1
  model_name = $ModelName
  layout = "core-plus-external-experts"
  created_utc = [DateTime]::UtcNow.ToString("o")
  link_mode = $LinkMode
  source_model = Split-Path -Leaf $ModelDir
  files = [ordered]@{
    core = [ordered]@{
      source = Split-Path -Leaf $CoreFile
      destination = Get-RelativePackagePath $coreDest
      bytes = Format-Size $CoreFile
    }
    expert_index = [ordered]@{
      source = Split-Path -Leaf $ExpertIndex
      destination = Get-RelativePackagePath $indexDest
      bytes = Format-Size $ExpertIndex
    }
    expert_pack = [ordered]@{
      source = Split-Path -Leaf $ExpertPack
      destination = Get-RelativePackagePath $packDest
      bytes = Format-Size $ExpertPack
    }
  }
  runtime_env = [ordered]@{
    LLAMA_INFINITUM_SELECTIVE_MOE = "1"
    LLAMA_INFINITUM_EXPERT_INDEX = Get-RelativePackagePath $indexDest
    LLAMA_INFINITUM_V2_GGML_EXPERT_PACK = Get-RelativePackagePath $packDest
  }
  notes = @(
    "This script organizes already-built core and expert artifacts.",
    "It does not download model weights and does not convert Hugging Face tensors into GGUF.",
    "Keep model licenses, private paths, and secrets out of public commits."
  )
}

if ($DryRun) {
  Write-Host "dry-run: would write $manifestDest"
  $manifest | ConvertTo-Json -Depth 8
} else {
  $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestDest -Encoding ASCII
  Write-Host "created package manifest: $manifestDest"
}
