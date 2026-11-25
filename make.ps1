# Windows equivalent of the Makefile:  .\make.ps1 setup | run | test | lint | typecheck | fetch | status | quality | sources | research | validate | daily | check
param([Parameter(Position = 0)][string]$Target = "run")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Invoke-Py { & $py @args; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

switch ($Target) {
    "setup" {
        & py -3 -m venv .venv
        Invoke-Py -m pip install --upgrade pip
        Invoke-Py -m pip install -r requirements.txt
    }
    "run" { Invoke-Py -m tfx run }
    "fetch" { Invoke-Py -m tfx fetch }
    "status" { Invoke-Py -m tfx status }
    "sources" { Invoke-Py -m tfx sources }
    "quality" { Invoke-Py -m tfx quality }
    "research" { Invoke-Py -m tfx research }
    "validate" { Invoke-Py -m tfx validate }
    "daily" { Invoke-Py -m tfx daily }
    "replay" { Invoke-Py -m tfx replay @($args) }
    "test" { Invoke-Py -m pytest --cov=tfx --cov-report=term-missing }
    "lint" { Invoke-Py -m ruff check tfx tests }
    "typecheck" { Invoke-Py -m mypy }
    "check" { Invoke-Py -m ruff check tfx tests; Invoke-Py -m mypy; Invoke-Py -m pytest --cov=tfx }
    default { Write-Error "Unknown target '$Target'" }
}
