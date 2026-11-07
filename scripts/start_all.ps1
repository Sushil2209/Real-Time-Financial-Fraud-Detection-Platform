# Starts Feast, API and dashboard, each in its own PowerShell window.
# Usage: .\scripts\start_all.ps1 [-WithKafka]
param([switch]$WithKafka)

# Project root = parent of the scripts folder (works from any location)
$Root = Split-Path -Parent $PSScriptRoot

# True if something is already listening on the port
function Test-PortInUse($Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

# Opens a new PowerShell window running $Command, after checking the port is free
function Start-Service($Name, $Port, $Command) {
    if (Test-PortInUse $Port) {
        Write-Host "[SKIP] $Name : port $Port is already in use. Run scripts\stop_all.ps1 first." -ForegroundColor Yellow
        return
    }
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $Command
    Write-Host "[OK]   $Name starting on port $Port" -ForegroundColor Green
}

if ($WithKafka) {
    Write-Host "Starting Kafka..."
    docker compose -f "$Root\docker\docker-compose.yml" up -d
    Write-Host "Waiting 10 seconds for Kafka to be ready..."
    Start-Sleep -Seconds 10

    # Create topic only if it is not in the list already
    $kt = "/opt/kafka/bin/kafka-topics.sh"
    $topics = docker exec kafka $kt --bootstrap-server localhost:9092 --list
    if ($topics -contains "transactions") {
        Write-Host "Topic 'transactions' already exists."
    } else {
        docker exec kafka $kt --bootstrap-server localhost:9092 --create --topic transactions
    }
}

Start-Service "Feast feature server" 6566 "Set-Location '$Root\feature_repo'; ..\.venv-feast\Scripts\feast.exe serve"
Start-Service "API"                  8000 "Set-Location '$Root'; .\.venv\Scripts\python.exe -m uvicorn src.api:app --port 8000"
Start-Service "Streamlit dashboard"  8501 "Set-Location '$Root'; .\.venv\Scripts\python.exe -m streamlit run src\dashboard.py"
