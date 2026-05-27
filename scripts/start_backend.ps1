param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

function Test-VenvPython {
    param([string]$PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        return $false
    }
    try {
        & $PythonPath --version *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Invoke-Checked {
    param(
        [string]$Exe,
        [string[]]$CmdArgs,
        [string]$ErrorMessage
    )
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw $ErrorMessage
    }
}

function Ensure-Pip {
    param([string]$PythonExe)
    try {
        & $PythonExe -m pip --version *> $null
    } catch {
    }
    if ($LASTEXITCODE -eq 0) {
        return
    }
    Write-Host "[INFO] Dang bootstrap pip cho .venv..."
    Invoke-Checked -Exe $PythonExe -CmdArgs @("-m", "ensurepip", "--upgrade") -ErrorMessage "Khong the bootstrap pip trong .venv."
    try {
        & $PythonExe -m pip --version *> $null
    } catch {
    }
    if ($LASTEXITCODE -ne 0) {
        throw "pip van khong kha dung sau khi bootstrap."
    }
}

function Test-BackendDeps {
    param([string]$PythonExe)
    try {
        & $PythonExe -c "import fastapi, uvicorn, torch, torchvision, pandas, PIL, motor, pymongo, ultralytics" *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Resolve-SystemPython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.10 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{ exe = "py"; args = @("-3.10") }
            }
        } catch {
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            & python --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{ exe = "python"; args = @() }
            }
        } catch {
        }
    }
    $fallbackPython = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $fallbackPython) {
        try {
            & $fallbackPython --version *> $null
            if ($LASTEXITCODE -eq 0) {
                return @{ exe = $fallbackPython; args = @() }
            }
        } catch {
        }
    }
    return $null
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$tmpDir = Join-Path $projectRoot ".tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$env:TEMP = $tmpDir
$env:TMP = $tmpDir
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot ".ultralytics_config"
$env:MPLCONFIGDIR = Join-Path $projectRoot ".matplotlib_config"

$existingListeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $existingListeners) {
    $ownerPid = [int]$listener.OwningProcess
    if ($ownerPid -eq $PID) {
        continue
    }
    $proc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    $procPath = $proc.Path
    if ($procPath -and $procPath.StartsWith($projectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        Write-Warning "[WARN] Phat hien backend cu dang nghe port $Port (PID=$ownerPid). Dang dung process cu de tranh treo API."
        Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    } else {
        throw "Port $Port dang duoc su dung boi process khac (PID=$ownerPid). Hay dong process do hoac chay backend tren port khac."
    }
}

$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$yoloVenvPython = Join-Path $projectRoot ".venv_yolo\Scripts\python.exe"
$systemPython = Resolve-SystemPython

if (Test-VenvPython -PythonPath $yoloVenvPython) {
    Write-Host "[INFO] Dung .venv_yolo de tranh loi native YOLO/OpenSSL tren Windows."
    $venvPython = $yoloVenvPython
} elseif (-not (Test-VenvPython -PythonPath $venvPython)) {
    Write-Host "[INFO] Phat hien .venv hong hoac thieu Python goc. Dang tao lai .venv..."
    if (-not $systemPython) {
        throw "Khong tim thay Python tren may. Hay cai Python 3.10+ roi chay lai script."
    }
    Invoke-Checked -Exe $systemPython.exe -CmdArgs ($systemPython.args + @("-m", "venv", ".venv", "--clear")) -ErrorMessage "Khong the tao .venv tu Python he thong."
    if (-not (Test-VenvPython -PythonPath $venvPython)) {
        throw "Tao lai .venv that bai. Vui long kiem tra cai dat Python."
    }
}

$runPythonExe = $venvPython
$runPythonArgs = @()

try {
    Ensure-Pip -PythonExe $venvPython
    if (Test-BackendDeps -PythonExe $venvPython) {
        Write-Host "[INFO] Dependency backend da san sang, bo qua pip install de khoi dong nhanh hon."
    } else {
        Invoke-Checked -Exe $venvPython -CmdArgs @("-m", "pip", "install", "--upgrade", "pip") -ErrorMessage "Cap nhat pip that bai."
        Invoke-Checked -Exe $venvPython -CmdArgs @("-m", "pip", "install", "-r", "requirements-react-demo.txt") -ErrorMessage "Cai dependency backend that bai."
    }
} catch {
    Write-Warning "[WARN] .venv khong cai duoc dependency, chuyen sang che do Python fallback + .pydeps."
    if (-not $systemPython) {
        throw "Khong tim thay Python fallback de tiep tuc."
    }
    $pyDepsReady = (Test-Path -LiteralPath ".pydeps\\fastapi") `
        -and (Test-Path -LiteralPath ".pydeps\\uvicorn") `
        -and (Test-Path -LiteralPath ".pydeps\\torch") `
        -and (Test-Path -LiteralPath ".pydeps\\motor") `
        -and (Test-Path -LiteralPath ".pydeps\\pymongo")
    if (-not $pyDepsReady) {
        Invoke-Checked -Exe $systemPython.exe -CmdArgs ($systemPython.args + @("-m", "pip", "install", "-r", "requirements-react-demo.txt", "--target", ".pydeps")) -ErrorMessage "Cai dependency vao .pydeps that bai."
    }
    $env:PYTHONPATH = "$projectRoot\.pydeps;$projectRoot"
    $runPythonExe = $systemPython.exe
    $runPythonArgs = $systemPython.args
}

if (-not $env:MONGO_URI) {
    $env:MONGO_URI = "mongodb://127.0.0.1:27017"
}
if (-not $env:MONGO_DB_NAME) {
    $env:MONGO_DB_NAME = "ungthuphoi_demo"
}
if (-not $env:MONGO_COLLECTION_NAME) {
    $env:MONGO_COLLECTION_NAME = "prediction_history"
}
if (-not $env:SERVE_FRONTEND) {
    $env:SERVE_FRONTEND = "1"
}

$env:PYTHONUNBUFFERED = "1"
& $runPythonExe @($runPythonArgs + @("-m", "uvicorn", "backend.api_server:app", "--host", "0.0.0.0", "--port", "$Port"))
