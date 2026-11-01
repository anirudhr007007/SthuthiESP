#requires -Version 5.1
<#
.SYNOPSIS
Plan or remove the project-local runtime and generated caches.

.DESCRIPTION
By default this removes only generated runtime artefacts: .venv, .runtime,
.cache, tool caches, and Python bytecode. Source files, .env, and biometric
enrollment images are retained unless explicitly requested.

.PARAMETER Force
Actually delete. Without this switch the script prints the deletion plan.

.PARAMETER DeleteBiometrics
Also delete enrollment images under data/training_images. The README is kept.
Use only after confirming consent and retention requirements.

.PARAMETER DeleteEnv
Also delete .env and .env.* files (except .env.example). This may contain API keys.

.EXAMPLE
.\cleanup_runtime.ps1
Show exactly what would be removed.

.EXAMPLE
.\cleanup_runtime.ps1 -Force
Remove the venv, model/embedding caches, and Python caches.

.EXAMPLE
.\cleanup_runtime.ps1 -Force -DeleteBiometrics -DeleteEnv
Remove the runtime plus consented biometric data and local secrets.
#>
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$DeleteBiometrics,
    [switch]$DeleteEnv,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$targets = [System.Collections.Generic.List[object]]::new()
$seen = @{}

function Add-Target {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the project: $full"
    }
    foreach ($existing in $targets) {
        $prefix = $existing.Path.TrimEnd("\") + [System.IO.Path]::DirectorySeparatorChar
        if ($full.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            return
        }
    }
    if ($seen.ContainsKey($full)) { return }
    $seen[$full] = $true
    $targets.Add([pscustomobject]@{ Path = $full; Label = $Label })
}

function Get-TargetBytes {
    param([string]$Path)
    try {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            return (Get-Item -LiteralPath $Path -Force).Length
        }
        $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
        if ($null -eq $sum) { return [int64]0 }
        return [int64]$sum
    } catch {
        return [int64]0
    }
}

foreach ($entry in @(
    @{ Name = ".venv"; Label = "project virtual environment" },
    @{ Name = ".runtime"; Label = "model, pip, and temporary caches" },
    @{ Name = ".cache"; Label = "embedding cache" },
    @{ Name = ".ruff_cache"; Label = "Ruff cache" },
    @{ Name = ".pytest_cache"; Label = "pytest cache" }
)) {
    Add-Target -Path (Join-Path $root $entry.Name) -Label $entry.Label
}

Get-ChildItem -LiteralPath $root -Directory -Recurse -Filter "__pycache__" -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notlike "$(Join-Path $root '.venv')*" -and
        $_.FullName -notlike "$(Join-Path $root '.runtime')*"
    } |
    ForEach-Object { Add-Target -Path $_.FullName -Label "Python bytecode cache" }

Get-ChildItem -LiteralPath $root -File -Recurse -Filter "*.pyc" -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -notlike "$(Join-Path $root '.venv')*" -and
        $_.FullName -notlike "$(Join-Path $root '.runtime')*"
    } |
    ForEach-Object { Add-Target -Path $_.FullName -Label "Python bytecode" }

if ($DeleteBiometrics) {
    $training = Join-Path $root "data\training_images"
    if (Test-Path -LiteralPath $training) {
        Get-ChildItem -LiteralPath $training -Force |
            Where-Object { $_.Name -ne "README.md" } |
            ForEach-Object { Add-Target -Path $_.FullName -Label "biometric enrollment data" }
    }
}

if ($DeleteEnv) {
    Get-ChildItem -LiteralPath $root -Force -File -Filter ".env*" |
        Where-Object { $_.Name -ne ".env.example" } |
        ForEach-Object { Add-Target -Path $_.FullName -Label "local secret/configuration" }
}

if ($targets.Count -eq 0) {
    Write-Host "Nothing to remove; the project-local runtime is already clean."
    exit 0
}

$totalBytes = [int64]0
Write-Host "Project-local cleanup plan for $root"
Write-Host ""
foreach ($target in $targets) {
    $bytes = Get-TargetBytes -Path $target.Path
    $totalBytes += $bytes
    $relative = $target.Path.Substring($root.Length).TrimStart("\")
    Write-Host ("  {0,-48} {1,10:N1} MB  {2}" -f $relative, ($bytes / 1MB), $target.Label)
}
Write-Host ""
Write-Host ("Total: {0:N2} GB" -f ($totalBytes / 1GB))

if (-not $Force -or $DryRun) {
    Write-Host ""
    Write-Host "Dry run only. Re-run with -Force to delete these items."
    if (-not $DeleteBiometrics) {
        Write-Host "Enrollment images are retained. Add -DeleteBiometrics only after checking consent."
    }
    if (-not $DeleteEnv) {
        Write-Host ".env is retained. Add -DeleteEnv to remove local secrets/configuration."
    }
    exit 0
}

Write-Host ""
foreach ($target in $targets) {
    try {
        if (Test-Path -LiteralPath $target.Path -PathType Leaf) {
            Remove-Item -LiteralPath $target.Path -Force
        } else {
            Remove-Item -LiteralPath $target.Path -Recurse -Force
        }
    } catch {
        Write-Warning "Could not remove $($target.Path): $($_.Exception.Message)"
    }
}

$remaining = @($targets | Where-Object { Test-Path -LiteralPath $_.Path })
if ($remaining.Count -gt 0) {
    Write-Warning "Some files could not be removed. Close running app/venv processes and retry."
    exit 1
}

Write-Host "Cleanup complete."
if (-not $DeleteBiometrics) {
    Write-Host "Enrollment images were retained under data/training_images."
}
if (-not $DeleteEnv) {
    Write-Host ".env was retained. Revoke any exposed keys separately."
}
