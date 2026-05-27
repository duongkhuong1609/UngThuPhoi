param(
    [string]$MongoExePath = "C:\Program Files\MongoDB\Server\8.2\bin\mongod.exe",
    [string]$DbPath = "C:\Users\ASUS\mongodb-data",
    [int]$Port = 27017,
    [string]$BindIp = "127.0.0.1",
    [string]$LogPath = "C:\Users\ASUS\mongodb-data\mongod-local.log"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $MongoExePath)) {
    throw "Khong tim thay mongod.exe tai: $MongoExePath"
}

New-Item -ItemType Directory -Force -Path $DbPath | Out-Null
$logDir = Split-Path -Parent $LogPath
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[INFO] MongoDB da dang listen tren port $Port."
    $existing | Select-Object LocalAddress, LocalPort, OwningProcess, State | Format-Table -AutoSize
    exit 0
}

$mongoArgs = @(
    "--dbpath", $DbPath,
    "--bind_ip", $BindIp,
    "--port", "$Port",
    "--logpath", $LogPath,
    "--logappend"
)

$proc = Start-Process -FilePath $MongoExePath -ArgumentList $mongoArgs -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 4

if ($proc.HasExited) {
    throw "mongod thoat som voi ma: $($proc.ExitCode). Kiem tra log: $LogPath"
}

$listen = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listen) {
    throw "mongod da khoi dong process nhung chua listen tren port $Port. Kiem tra log: $LogPath"
}

Write-Host "[OK] MongoDB user-mode da khoi dong."
Write-Host "  PID: $($proc.Id)"
Write-Host "  URI: mongodb://$BindIp`:$Port"
Write-Host "  dbPath: $DbPath"
Write-Host "  log: $LogPath"
Write-Host "  Stop nhanh: Stop-Process -Id $($proc.Id) -Force"
