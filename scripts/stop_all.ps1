# Stops the services started by start_all.ps1 (found by port).
# Usage: .\scripts\stop_all.ps1 [-WithKafka]
param([switch]$WithKafka)

$Root = Split-Path -Parent $PSScriptRoot

# Feast, API, Streamlit
foreach ($Port in 6566, 8000, 8501) {
    $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "Port $Port : nothing running."
        continue
    }
    foreach ($procId in ($conns.OwningProcess | Sort-Object -Unique)) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Host "Port $Port : stopped process $procId" -ForegroundColor Green
    }
}

if ($WithKafka) {
    docker compose -f "$Root\docker\docker-compose.yml" down
}
