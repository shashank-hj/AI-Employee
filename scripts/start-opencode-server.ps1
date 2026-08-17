<#
Start the opencode HTTP server that the AI Employee Platform uses as its LLM
engine. Keeps the server alive (restarts on crash) and sets the shared basic-auth
password so the server is not left unsecured.

Usage:
  powershell -ExecutionPolicy Bypass -File scripts/start-opencode-server.ps1

The script reads OPENCODE_BASE_URL / OPENCODE_PASSWORD / OPENCODE_USERNAME from
.env-opencode (falling back to defaults). It is meant to be registered as a
startup/keep-alive job (e.g. a Windows Scheduled Task) that launches this script
and restarts it if it exits.

Notes:
  - opencode is launched via its npm shim (opencode.ps1) on Windows.
  - The server is bound to 127.0.0.1 by default; Docker reaches it through
    host.docker.internal. Do NOT bind to 0.0.0.0 unless you have firewalled it.
#>

[CmdletBinding()]
param(
    [int]$Port = 4096,
    [string]$Hostname = "127.0.0.1",
    [string]$EnvFile = ".env-opencode",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

function Get-EnvValue {
    param([string]$Name, [string]$Default = "")
    if (Test-Path $EnvFile) {
        $line = Get-Content $EnvFile -ErrorAction SilentlyContinue |
            Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
            Select-Object -First 1
        if ($line) {
            $val = ($line -split "=", 2)[1].Trim()
            return $val.Trim('"', "'")
        }
    }
    return $Default
}

$password = Get-EnvValue "OPENCODE_PASSWORD" ""
$username = Get-EnvValue "OPENCODE_USERNAME" "opencode"

$logDir = Join-Path $env:TEMP "opencode-engine"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = Join-Path $logDir "serve.log"
$errFile = Join-Path $logDir "serve.err.log"

Write-Host "opencode serve keeper starting on ${Hostname}:${Port}" -ForegroundColor Cyan
Write-Host "log: $logFile" -ForegroundColor Cyan

$env:OPENCODE_SERVER_USERNAME = $username
if ($password) {
    $env:OPENCODE_SERVER_PASSWORD = $password
    Write-Host "server password set (basic auth)." -ForegroundColor Green
} else {
    Write-Host "WARNING: OPENCODE_PASSWORD is empty; server will be UNSECURED." -ForegroundColor Yellow
}

while ($true) {
    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] starting opencode serve..." -ForegroundColor Cyan
    $p = Start-Process -FilePath "opencode" `
        -ArgumentList "serve", "--port", "$Port", "--hostname", "$Hostname" `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError $errFile

    # Wait for the HTTP server to come up (or the process to exit).
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($p.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "http://${Hostname}:${Port}/global/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { Start-Sleep -Seconds 1 }
    }

    if ($ready) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] opencode serve is UP (pid $($p.Id))." -ForegroundColor Green
        $p.WaitForExit()
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] opencode serve EXITED with code $($p.ExitCode)." -ForegroundColor Yellow
    } else {
        if ($p.HasExited) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] opencode serve failed to start (exit $($p.ExitCode))." -ForegroundColor Red
        } else {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] server did not report healthy; killing pid $($p.Id)." -ForegroundColor Red
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    }

    if ($Once) { break }
    Start-Sleep -Seconds 5
}
