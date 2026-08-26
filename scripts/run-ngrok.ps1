# Persistent ngrok tunnel for the AI Employee dashboard / Sarvam webhooks.
# Auto-restarts if the process dies, and logs to %TEMP%\opencode\ngrok.
# Uses the free-tier dev domain: comic-paragraph-peroxide.ngrok-free.dev
#
# First-time setup (one time, from your ngrok dashboard):
#   ngrok config add-authtoken <YOUR_AUTHTOKEN>
#
param(
    [string]$Url = "http://localhost:8001",
    [string]$Domain = "comic-paragraph-peroxide.ngrok-free.dev"
)

$ngrok = (Get-Command ngrok -ErrorAction SilentlyContinue).Source
if (-not $ngrok) {
    $candidates = @(
        (Join-Path $env:TEMP "opencode\ngrok\bin\ngrok.exe"),
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe\ngrok.exe",
        "$env:USERPROFILE\.local\bin\ngrok.exe"
    )
    $ngrok = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $ngrok) {
    Write-Error "ngrok not found. Install it with: winget install --id Ngrok.Ngrok"
    exit 1
}

# Keep the agent current: an outdated agent is rejected by the account
# (ERR_NGROK_121). Update quietly; ignore failures so the tunnel can still start.
& $ngrok update 2>&1 | Out-Null

$logDir = Join-Path $env:TEMP "opencode\ngrok"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$logFile = Join-Path $logDir "ngrok_persistent.log"
$urlFile = Join-Path $logDir "ngrok_url.txt"
$autoLog = Join-Path $logDir "ngrok_auto.log"

Write-Output "ngrok: $ngrok"
Write-Output "public base: https://$Domain"

while ($true) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logFile -Value "$stamp starting ngrok ($Url -> $Domain)"
    try {
        & $ngrok http $Url --url=$Domain --log=$autoLog --log-format=logfmt 2>&1 | Out-Null
    } catch {
        Add-Content -Path $logFile -Value "$stamp ngrok exited with error: $($_.Exception.Message)"
    }
    Add-Content -Path $logFile -Value "$stamp ngrok stopped; restarting in 5s"
    Start-Sleep -Seconds 5
}