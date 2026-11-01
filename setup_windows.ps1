$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtime = Join-Path $root ".runtime"
$python = Join-Path $root ".venv\Scripts\python.exe"

foreach ($name in @(
    "pip-cache", "tmp", "deepface", "huggingface", "keras", "ollama",
    "logs", "matplotlib", "numba", "cache", "torch"
)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $runtime $name) | Out-Null
}

# Keep pip, temporary files, and model caches inside the checkout.
$env:PIP_CACHE_DIR = Join-Path $runtime "pip-cache"
$env:TMP = Join-Path $runtime "tmp"
$env:TEMP = Join-Path $runtime "tmp"
$env:TMPDIR = Join-Path $runtime "tmp"
$env:DEEPFACE_HOME = Join-Path $runtime "deepface"
$env:HF_HOME = Join-Path $runtime "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $runtime "huggingface\hub"
$env:TRANSFORMERS_CACHE = Join-Path $runtime "huggingface\transformers"
$env:KERAS_HOME = Join-Path $runtime "keras"
$env:XDG_CACHE_HOME = Join-Path $runtime "cache"
$env:MPLCONFIGDIR = Join-Path $runtime "matplotlib"
$env:NUMBA_CACHE_DIR = Join-Path $runtime "numba"
$env:TORCH_HOME = Join-Path $runtime "torch"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PYTHONUTF8 = "1"

Set-Location -LiteralPath $root
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating project-local virtual environment at .venv ..."
    py -3.12 -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create .venv. Install Python 3.12 and retry."
    }
}

$log = Join-Path $runtime "logs\install.log"
# Native tools write warnings to stderr; do not let that abort the installer.
$ErrorActionPreference = "Continue"
Write-Host "Installing pinned dependencies into .venv (local cache: .runtime\pip-cache) ..."
& $python -m pip install --upgrade pip setuptools wheel *>> $log
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed. See $log" }

& $python -m pip install `
    -r (Join-Path $root "requirements-windows.txt") `
    -r (Join-Path $root "requirements-offline-voice.txt") `
    -r (Join-Path $root "requirements-cloud.txt") `
    -r (Join-Path $root "requirements-dev.txt") *>> $log
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See $log" }

& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "pip reported broken requirements." }

# retina-face (a DeepFace dependency) installs opencv-contrib-python-headless,
# which can overwrite cv2 with a build that cannot open a window. Restore the
# GUI-enabled wheel pinned in requirements.txt and verify it before finishing.
Write-Host "Ensuring the GUI-enabled OpenCV build is active ..."
$opencvPin = (Select-String -Path (Join-Path $root "requirements.txt") -Pattern "^opencv-python==").Line
if (-not $opencvPin) { throw "opencv-python pin missing from requirements.txt." }
& $python -m pip install --force-reinstall --no-deps $opencvPin *>> $log
if ($LASTEXITCODE -ne 0) { throw "Could not restore GUI-enabled OpenCV. See $log" }
& $python -c "import cv2, sys; ok = 'WIN32UI' in cv2.getBuildInformation(); print('OpenCV GUI: ' + ('WIN32UI' if ok else 'missing')); sys.exit(0 if ok else 1)" *>> $log
if ($LASTEXITCODE -ne 0) { throw "OpenCV has no GUI backend; the camera window would fail. See $log" }

Write-Host ""
Write-Host "Project-local runtime is ready."
Write-Host "  Python:  $python"
Write-Host "  Caches:  $runtime"
Write-Host "  Log:     $log"
Write-Host ""
Write-Host "Next:"
Write-Host "  .\run_windows.ps1 --doctor"
Write-Host '  .\run_windows.ps1 --ask "Who is the principal of MCE?"'
Write-Host ""
Write-Host "Ollama is optional and manages its own model store; see docs/operations.md."
