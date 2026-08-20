# Elevated helper: adds Windows Defender exclusions for ngrok and re-downloads
# a fresh ngrok binary. Run via `Start-Process -Verb RunAs` (UAC prompt).
$ErrorActionPreference = "Continue"

$tempBase = "C:\Users\ShashankHJ\AppData\Local\Temp\opencode"
$bin = Join-Path $tempBase "ngrok\bin"
$wingetDir = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe"
$marker = Join-Path $tempBase "ngrok\exclusion_ok.txt"

New-Item -ItemType Directory -Path $bin -Force | Out-Null

foreach ($p in @($bin, $wingetDir, $tempBase)) {
    try {
        Add-MpPreference -ExclusionPath $p -ErrorAction Stop
        Add-Content $marker "excluded: $p"
    } catch {
        Add-Content $marker "exclude FAILED: $p -> $($_.Exception.Message)"
    }
}

$zip = Join-Path $bin "ngrok.zip"
try {
    Invoke-WebRequest -Uri "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-windows-amd64.zip" -OutFile $zip -UseBasicParsing -ErrorAction Stop
    Expand-Archive -Path $zip -DestinationPath $bin -Force -ErrorAction Stop
    Remove-Item $zip -Force
    Add-Content $marker "ngrok re-downloaded to $bin"
} catch {
    Add-Content $marker "download FAILED: $($_.Exception.Message)"
}

Add-Content $marker "done"