$projectRoot = "C:\Users\ASUS\Desktop\UngThuPhoi"
$pythonExe = "$projectRoot\.venv_yolo\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "$projectRoot\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = $projectRoot
$env:SERVE_FRONTEND = "1"
$env:YOLO_CONFIG_DIR = "$projectRoot\.ultralytics_config"
$env:MPLCONFIGDIR = "$projectRoot\.matplotlib_config"

Set-Location -LiteralPath $projectRoot
& $pythonExe -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
