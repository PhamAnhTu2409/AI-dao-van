$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Models = Join-Path $Root 'models'
$Qwen = Join-Path $Models 'Qwen3-1.7B'
$DeepX = Join-Path $Models 'deepx-embedding-v1'
$E5 = Join-Path $Models 'multilingual-e5-small'
New-Item -ItemType Directory -Force -Path $Qwen, $DeepX, $E5 | Out-Null

function Download-HFFile([string]$Repo, [string]$Revision, [string]$File, [string]$Destination) {
    $Uri = "https://huggingface.co/$Repo/resolve/$Revision/$File"
    Write-Host "Downloading $Repo/$File ..."
    & curl.exe --fail --location --retry 5 --retry-delay 2 --output $Destination $Uri
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Repo/$File (curl exit $LASTEXITCODE)" }
    if (!(Test-Path -LiteralPath $Destination) -or (Get-Item -LiteralPath $Destination).Length -eq 0) {
        throw "Downloaded file is missing or empty: $Destination"
    }
}

# Pin immutable revisions so the files are reproducible.
$QwenRevision = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
$DeepXRevision = '91d7a8427ed08eece9241c2be65d4be776e067ab'
$E5Revision = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
$QwenFiles = @(
    '.gitattributes', 'LICENSE', 'README.md', 'config.json', 'generation_config.json',
    'merges.txt', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
    'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'
)
$DeepXFiles = @('README.md', 'config.json', 'deepx_v1.0.pt', 'id_remap.pt', 'token_embedding.pt', 'tokenizer.json', 'tokenizer_config.json')
$E5Files = @('README.md', 'config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model')

foreach ($File in $QwenFiles) {
    $Target = Join-Path $Qwen $File
    if (!(Test-Path -LiteralPath $Target) -or (Get-Item -LiteralPath $Target).Length -eq 0) {
        Download-HFFile 'Qwen/Qwen3-1.7B' $QwenRevision $File $Target
    } else {
        Write-Host "Already present: $Target"
    }
}
foreach ($File in $DeepXFiles) {
    $Target = Join-Path $DeepX $File
    if (!(Test-Path -LiteralPath $Target) -or (Get-Item -LiteralPath $Target).Length -eq 0) {
        Download-HFFile 'dxtech-asia/deepx-embedding-v1' $DeepXRevision $File $Target
    } else {
        Write-Host "Already present: $Target"
    }
}

foreach ($File in $E5Files) {
    $Target = Join-Path $E5 $File
    if (!(Test-Path -LiteralPath $Target) -or (Get-Item -LiteralPath $Target).Length -eq 0) {
        Download-HFFile 'intfloat/multilingual-e5-small' $E5Revision $File $Target
    } else {
        Write-Host "Already present: $Target"
    }
}

Write-Host "Models downloaded. Qwen: $Qwen; DeepX: $DeepX; E5: $E5"
