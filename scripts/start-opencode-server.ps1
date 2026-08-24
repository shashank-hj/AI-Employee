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
  - The server is bound to 0.0.0.0 by default so Docker containers reach it
    through host.docker.internal:4096. Security is provided by the basic-auth
    password (OPENCODE_SERVER_PASSWORD) that this script sets before launch,
    so exposing on all interfaces is acceptable on a trusted machine.
 #>

[CmdletBinding()]
param(
    [int]$Port = 4096,
    [string]$Hostname = "0.0.0.0",
    [string]$EnvFile = "",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

# Resolve the env file relative to this script's directory (not the caller's CWD),
# so it works whether launched manually, from a Scheduled Task, or at startup.
if (-not $EnvFile) {
    $EnvFile = Join-Path $PSScriptRoot ".env-opencode"
    if (-not (Test-Path $EnvFile)) {
        # Script lives in <root>/scripts; the env files live in <root>.
        $EnvFile = Join-Path (Split-Path $PSScriptRoot) ".env-opencode"
    }
    if (-not (Test-Path $EnvFile)) {
        $EnvFile = Join-Path (Split-Path $PSScriptRoot) ".env"
    }
}
Write-Host "using env file: $EnvFile" -ForegroundColor Cyan

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

# Resolve the real opencode executable. On Windows `opencode` is an npm .ps1
# shim, which Start-Process cannot launch directly, so point at the wrapped
# opencode.exe instead.
$ExePath = "opencode"
$opencodeCmd = Get-Command opencode -ErrorAction SilentlyContinue
if ($opencodeCmd -and $opencodeCmd.Source -match '\.ps1$') {
    $shimDir = Split-Path $opencodeCmd.Source
    $candidate = Join-Path $shimDir "node_modules/opencode-ai/bin/opencode.exe"
    if (Test-Path $candidate) { $ExePath = $candidate }
}

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

# Probe headers: once the server is secured it returns 401 without basic auth,
# so the health checks below must send the same credentials the platform uses.
$probeHeaders = @{}
if ($password) {
    $probeAuth = "Basic $([Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("$username`:$password")))"
    $probeHeaders["Authorization"] = $probeAuth
}

while ($true) {
    # If a server is already healthy on this port (started manually or by a
    # previous keeper instance), don't spawn a second one — just monitor it.
    try {
$probe = Invoke-WebRequest -Uri "http://127.0.0.1:${Port}/global/health" -Headers $probeHeaders -UseBasicParsing -TimeoutSec 2
        if ($probe.StatusCode -eq 200) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] opencode serve already UP on ${Hostname}:${Port}; monitoring..." -ForegroundColor Green
            Start-Sleep -Seconds 30
            continue
        }
    } catch {
        # Not up yet — fall through and start one.
    }

    Write-Host "`n[$(Get-Date -Format 'HH:mm:ss')] starting opencode serve..." -ForegroundColor Cyan
    $p = Start-Process -FilePath $ExePath `
        -ArgumentList "serve", "--port", "$Port", "--hostname", "$Hostname" `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput $logFile -RedirectStandardError $errFile

    # Wait for the HTTP server to come up (or the process to exit).
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        if ($p.HasExited) { break }
        try {
$r = Invoke-WebRequest -Uri "http://127.0.0.1:${Port}/global/health" -Headers $probeHeaders -UseBasicParsing -TimeoutSec 2
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

