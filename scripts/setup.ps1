$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = (Get-Command py -ErrorAction SilentlyContinue)
if ($Python) {
    $PythonCommand = 'py'
    $PythonArgs = @('-3.11')
} else {
    $PythonCommand = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (!$PythonCommand) { throw 'Python 3.11 was not found. Install Python 3.11 and ensure py.exe or python.exe is on PATH.' }
    $PythonArgs = @()
}
& $PythonCommand @PythonArgs --version
if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }

if (!(Test-Path -LiteralPath (Join-Path $Root '.venv\Scripts\python.exe'))) {
    & $PythonCommand @PythonArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
}
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }
& $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'Could not install CPU PyTorch.' }
& $VenvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Could not install application dependencies.' }
Write-Host 'Environment is ready. Run .\scripts\run.ps1 to start the local app.'
