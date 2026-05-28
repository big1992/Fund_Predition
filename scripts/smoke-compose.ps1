param(
    [string]$FrontendUrl = "http://localhost:5173",
    [string]$BackendUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"

Write-Host "Running compose smoke checks..."

$health = Invoke-RestMethod -Uri "$BackendUrl/api/health" -Method Get
if ($health.status -ne "healthy") {
    throw "Backend health check failed: status=$($health.status)"
}

$front = Invoke-WebRequest -Uri $FrontendUrl -Method Get
if ($front.StatusCode -ne 200) {
    throw "Frontend not reachable: status=$($front.StatusCode)"
}

Write-Host "Smoke checks passed."
