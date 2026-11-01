param ([string]$Destination = "")

# Create a shareable source archive of this project.
#
# The archive is safe to hand out: it never contains secrets (`.env`),
# biometric images or embedding caches, a virtual environment, or model
# caches. Run it whenever a copy of the project has to be shared:
#
#   .\make_archive.ps1
#   .\make_archive.ps1 -Destination D:\somewhere\espsthuthi2-src.zip
#
# Review the archive content before sharing it anyway; `Consent` for the
# enclosed training README still applies to the recipient.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $Destination) {
    $Destination = Join-Path (Split-Path -Parent $root) "espsthuthi2-src.zip"
}

$staging = Join-Path ([System.IO.Path]::GetTempPath()) (
    "espsthuthi2-src-" + [guid]::NewGuid().ToString("N")
)
New-Item -ItemType Directory -Path $staging | Out-Null

try {
    # Copy the source tree without heavy/local-only artefacts.
    robocopy $root $staging /E /NFL /NDL /NJH /NJS /NP `
        /XD .venv venv .runtime .cache .pytest_cache .ruff_cache `
            __pycache__ .idea .vscode build dist node_modules `
        /XF .env *.bak *.log *.tmp *.pyc *.wav *.mp3 Thumbs.db desktop.ini .DS_Store
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed with exit code $LASTEXITCODE."
    }
    $global:LASTEXITCODE = 0

    # Biometric folder: keep the consent/retention README only.
    $trainingStaging = Join-Path $staging "data\training_images"
    New-Item -ItemType Directory -Force -Path $trainingStaging | Out-Null
    Copy-Item -LiteralPath (Join-Path $root "data\training_images\README.md") `
        -Destination $trainingStaging -Force

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Force
    }
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $Destination

    $sizeMb = [math]::Round((Get-Item -LiteralPath $Destination).Length / 1MB, 2)
    Write-Host "Created $Destination ($sizeMb MB)."
    Write-Host "Confirmed exclusions: .env, .venv, .runtime, .cache, biometric images."
}
finally {
    Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
}
