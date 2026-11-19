<#
Run the three Week 5 P ablations for exactly 50 optimizer updates each.

Run from the repository root after activating the project's Python environment:
  .\scripts\run_ablations_smoke.ps1

Use -AllowCpu only when deliberately running without an NVIDIA CUDA GPU.
#>
[CmdletBinding()]
param(
    [int]$Steps = 50,
    [switch]$AllowCpu
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repoRoot

function Stop-WithGuidance([string]$Message) {
    Write-Error $Message
    exit 1
}

if ($Steps -lt 1) {
    Stop-WithGuidance 'Steps must be a positive integer.'
}

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Stop-WithGuidance 'Python was not found. Install Python 3.11, activate the project virtual environment, then rerun this script.'
}

$probe = & python -c "import sys; import torch; print(f'PY={sys.version_info.major}.{sys.version_info.minor}'); print(f'TORCH={torch.__version__}'); print(f'CUDA={torch.cuda.is_available()}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Stop-WithGuidance "PyTorch could not be imported. Activate the configured environment and install the dependencies first.`n$probe"
}
$probeText = ($probe | Out-String)
Write-Host $probeText.Trim()
if ($probeText -notmatch 'PY=3\.11') {
    Stop-WithGuidance 'This project specifies Python 3.11. Activate the Python 3.11 virtual environment and rerun.'
}
if (-not $AllowCpu -and $probeText -notmatch 'CUDA=True') {
    Stop-WithGuidance 'CUDA is not available to PyTorch. Install the CUDA-enabled PyTorch build for this Alienware GPU, or explicitly use -AllowCpu.'
}

$requiredFiles = @(
    'data/manifest.csv',
    'data/splits.csv',
    'data/stats/feature_norm.npz',
    'checkpoints/b1_audio_only_best.pt',
    'checkpoints/b2_visual_only_v2_best.pt',
    'configs/x1_no_modality_dropout.yaml',
    'configs/x2_concat_fusion.yaml',
    'configs/x3_2d_frontend.yaml'
)
$missing = @($requiredFiles | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
if ($missing.Count -gt 0) {
    $details = @("Required files are missing:") + @($missing | ForEach-Object { "  - $_" })
    if ($missing -contains 'data/manifest.csv') {
        $details += @(
            '',
            'Create the manifest on the Alienware after verifying GRID audio/video and extracting mouth crops:',
            '  python -m scripts.verify_dataset --data-root <GRID_ROOT>',
            '  python -m scripts.extract_mouths --data-root <GRID_ROOT>',
            '  python -m scripts.prepare_audio --raw-dir data/raw/grid',
            '  python -m scripts.build_manifest',
            'The manifest builder expects reports/integrity_report.csv plus data/processed/mouths,',
            'data/processed/mouth_valid, data/processed/audio16k, and data/splits.csv.'
        )
    }
    if ($missing -contains 'data/stats/feature_norm.npz') {
        $details += @('', 'After data/manifest.csv is built, compute the training normalization statistics:',
            '  python -m scripts.compute_norm_stats --manifest data/manifest.csv --splits data/splits.csv')
    }
    if ($missing -contains 'checkpoints/b1_audio_only_best.pt' -or $missing -contains 'checkpoints/b2_visual_only_v2_best.pt') {
        $details += @('', 'Copy the trained B1 and B2 v2 checkpoint files into checkpoints/ on the Alienware.')
    }
    Stop-WithGuidance ($details -join "`n")
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runsRoot = Join-Path $repoRoot "runs/ablations_smoke/$stamp"
New-Item -ItemType Directory -Path $runsRoot -Force | Out-Null
$configs = @(
    'configs/x1_no_modality_dropout.yaml',
    'configs/x2_concat_fusion.yaml',
    'configs/x3_2d_frontend.yaml'
)
$failed = @()

foreach ($config in $configs) {
    $name = [IO.Path]::GetFileNameWithoutExtension($config)
    $log = Join-Path $runsRoot "$name.log"
    Write-Host "`n=== $name : $Steps optimizer updates ==="
    Write-Host "Log: $log"
    & python -u -m scripts.train --config $config --max-steps $Steps --no-results --runs-root $runsRoot 2>&1 |
        Tee-Object -FilePath $log
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $failed += $name
        Write-Warning "$name failed with exit code $code. See $log"
        break
    }
    if (-not (Select-String -LiteralPath $log -Pattern "Reached requested optimizer step limit: $Steps" -Quiet)) {
        $failed += $name
        Write-Warning "$name exited without confirming exactly $Steps optimizer updates. See $log"
        break
    }
    Write-Host "PASS: $name completed $Steps optimizer updates."
}

if ($failed.Count -gt 0) {
    Write-Error "Ablation smoke run did not complete: $($failed -join ', '). Logs: $runsRoot"
    exit 1
}
Write-Host "`nAll three ablations completed $Steps optimizer updates. Logs: $runsRoot"
