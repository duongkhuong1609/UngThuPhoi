$projectRoot = "C:\Users\ASUS\Desktop\UngThuPhoi"
$pythonExe = "$projectRoot\.venv_yolo\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = "$projectRoot\.venv\Scripts\python.exe"
}
$env:PYTHONPATH = $projectRoot
if (-not $env:MONGO_URI) { $env:MONGO_URI = "mongodb://127.0.0.1:27017" }
if (-not $env:MONGO_DB_NAME) { $env:MONGO_DB_NAME = "ungthuphoi_demo" }
if (-not $env:MONGO_COLLECTION_NAME) { $env:MONGO_COLLECTION_NAME = "prediction_history" }
$env:YOLO_CONFIG_DIR = "$projectRoot\.ultralytics_config"
$env:MPLCONFIGDIR = "$projectRoot\.matplotlib_config"

Set-Location -LiteralPath $projectRoot
& $pythonExe -m uvicorn backend.api_server:app --host 0.0.0.0 --port 8000
