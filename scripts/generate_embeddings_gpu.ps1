[CmdletBinding()]
param(
    [ValidateRange(1, 2048)]
    [int]$BatchSize = 256,

    [ValidateSet("float32", "float16")]
    [string]$Dtype = "float32",

    [string]$PythonVersion = "3.11",

    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",

    [switch]$SkipInstall,
    [switch]$Force,
    [switch]$CreateArchive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentPath = Join-Path $repositoryRoot ".venv-embed"
$pythonPath = Join-Path $environmentPath "Scripts\python.exe"
$catalogPath = Join-Path $repositoryRoot "data\catalog.jsonl"
$catalogArchivePath = Join-Path $repositoryRoot "data\catalog.jsonl.gz"
$indexPath = Join-Path $repositoryRoot "data\cartographer_index"
$embeddingPath = Join-Path $indexPath "embeddings.npy"
$manifestPath = Join-Path $indexPath "embeddings_manifest.json"
$modelPath = Join-Path $indexPath "bge-small-en-v1.5"
$transferArchive = Join-Path $repositoryRoot "cartographer-bge-artifacts.zip"

$expectedCatalogHash = "DA979B05A68AF864CB0DCF9EE6A81C010C7E66A57978AD286C7A2E005FC69A67"
$expectedArchiveHash = "07FD142631FD6B03E2B4D09988C3EB7D53720E9D57010C79DB48EEAADA50A8F8"
$catalogUrl = "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz"

Set-Location $repositoryRoot

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Creating Python $PythonVersion environment at $environmentPath"
    & py "-$PythonVersion" -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python environment. Install Python $PythonVersion and retry."
    }
}

if (-not $SkipInstall) {
    Write-Host "Installing CUDA PyTorch from $TorchIndexUrl"
    & $pythonPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $pythonPath -m pip install torch --index-url $TorchIndexUrl
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed. Pass the index URL selected at https://pytorch.org/get-started/locally/ with -TorchIndexUrl."
    }
    & $pythonPath -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Embedding dependency installation failed." }
}

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install or update the NVIDIA driver before generating embeddings."
}
nvidia-smi

& $pythonPath -c "import sys, torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); sys.exit(0 if torch.cuda.is_available() else 3)"
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch cannot access CUDA. Re-run with the correct -TorchIndexUrl from the official PyTorch installer."
}

if (-not (Test-Path -LiteralPath $catalogPath)) {
    if (-not (Test-Path -LiteralPath $catalogArchivePath)) {
        $temporaryArchive = "$catalogArchivePath.download"
        Write-Host "Downloading the official frozen catalog"
        Invoke-WebRequest -Uri $catalogUrl -OutFile $temporaryArchive
        $downloadHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporaryArchive).Hash
        if ($downloadHash -ne $expectedArchiveHash) {
            throw "Downloaded catalog archive checksum mismatch: $downloadHash"
        }
        Move-Item -LiteralPath $temporaryArchive -Destination $catalogArchivePath -Force
    }

    $archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $catalogArchivePath).Hash
    if ($archiveHash -ne $expectedArchiveHash) {
        throw "Catalog archive checksum mismatch: $archiveHash"
    }
    Write-Host "Decompressing the official catalog"
    & $pythonPath -c "import gzip, shutil; source=gzip.open(r'$catalogArchivePath','rb'); target=open(r'$catalogPath','wb'); shutil.copyfileobj(source,target); source.close(); target.close()"
    if ($LASTEXITCODE -ne 0) { throw "Catalog decompression failed." }
}

$catalogHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $catalogPath).Hash
if ($catalogHash -ne $expectedCatalogHash) {
    throw "Catalog checksum mismatch: $catalogHash"
}
$catalogRows = (Get-Content -LiteralPath $catalogPath | Measure-Object -Line).Lines
if ($catalogRows -ne 50000) {
    throw "Expected 50,000 catalog rows, found $catalogRows."
}
Write-Host "Catalog verified: 50,000 rows, SHA256 $catalogHash"

$completeArtifact = (
    (Test-Path -LiteralPath $embeddingPath) -and
    (Test-Path -LiteralPath $manifestPath) -and
    (Test-Path -LiteralPath $modelPath)
)
if ($completeArtifact -and -not $Force) {
    Write-Host "A complete embedding artifact already exists; verifying it instead of rebuilding. Use -Force to regenerate."
    & $pythonPath -m cartographer.build_embeddings --verify-only
    if ($LASTEXITCODE -ne 0) { throw "Existing embedding verification failed." }
} else {
    $candidateBatches = @($BatchSize, 128, 64, 32) |
        Where-Object { $_ -le $BatchSize } |
        Select-Object -Unique
    $generated = $false
    foreach ($candidateBatch in $candidateBatches) {
        Write-Host "Generating BGE embeddings on CUDA with batch size $candidateBatch and dtype $Dtype"
        & $pythonPath -m cartographer.build_embeddings --device cuda --batch-size $candidateBatch --dtype $Dtype
        if ($LASTEXITCODE -eq 0) {
            $generated = $true
            break
        }
        Write-Warning "Embedding generation failed at batch size $candidateBatch; trying a smaller batch."
    }
    if (-not $generated) {
        throw "Embedding generation failed at every attempted batch size: $($candidateBatches -join ', ')."
    }
}

Write-Host "Running final portable-artifact verification"
& $pythonPath -m cartographer.build_embeddings --verify-only
if ($LASTEXITCODE -ne 0) { throw "Final embedding verification failed." }

if ($CreateArchive) {
    Write-Host "Creating $transferArchive"
    Compress-Archive -Path $embeddingPath, $manifestPath, $modelPath -DestinationPath $transferArchive -Force
    $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $transferArchive).Hash
    Write-Host "Transfer ZIP SHA256: $zipHash"
}

Write-Host "Embedding generation is complete. Generated artifacts remain ignored by Git under data\cartographer_index\."
