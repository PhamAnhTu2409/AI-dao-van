$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $Python)) { throw 'Environment missing. Run .\scripts\setup.ps1 first.' }
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& $Python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --server.showEmailPrompt false --browser.gatherUsageStats false
if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE" }
