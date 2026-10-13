# Full reproduction of GLWM-2 on Windows + NVIDIA GPU.
#   .\scripts\reproduce_all.ps1                 # setup, tests, all sweeps, figures, tables
#   .\scripts\reproduce_all.ps1 -FiguresOnly    # rebuild figures/tables from existing results/
#   .\scripts\reproduce_all.ps1 -Workers 2      # two sweep processes share the GPU
param(
    [switch]$FiguresOnly,
    [int]$Workers = 1
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    uv venv --python 3.12 --managed-python .venv
    uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv -r requirements-lock.txt
    uv pip install --python .venv --no-deps -e .
    & $py -m ipykernel install --user --name glwm2 --display-name "GLWM-2 (py3.12, cu128)"
}

& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "tests failed" }

if (-not $FiguresOnly) {
    # keep Windows awake while the sweeps run
    powercfg /change standby-timeout-ac 0
    & $py scripts\build_notebooks.py --run 00 01 02
    $grids = @(
        "configs\pilot_addition.yaml",
        "configs\main_addition.yaml", "configs\main_parity.yaml", "configs\main_gates.yaml",
        "configs\headline_addition.yaml",
        "configs\main_mazes.yaml", "configs\ablations.yaml"
    )
    $procs = @()
    for ($w = 0; $w -lt $Workers; $w++) {
        $procs += Start-Process -FilePath $py -ArgumentList (@("scripts\run_sweep.py") + $grids) `
            -RedirectStandardOutput "results\sweep_worker$w.log" -RedirectStandardError "results\sweep_worker$w.err" `
            -NoNewWindow -PassThru
        Start-Sleep -Seconds 20
    }
    $procs | Wait-Process
}

& $py scripts\make_figures.py
& $py scripts\check_bib.py
Write-Host "Done. Figures: paper\figures, tables: paper\tables, numbers: results\paper_numbers.json"
