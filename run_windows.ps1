$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project runtime missing. Run .\setup_windows.ps1 first."
}
$runtime = Join-Path $PSScriptRoot ".runtime"
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
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -B (Join-Path $PSScriptRoot "app.py") @args
$code = $LASTEXITCODE
$ErrorActionPreference = $previousErrorAction
exit $code
