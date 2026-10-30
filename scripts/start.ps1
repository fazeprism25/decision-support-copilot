# One-click setup and launch for Windows (run by "START HERE - Windows.bat"). Every step is skipped when already done:
# AI key (asked once) -> uv (installs Python + libraries, no admin rights) -> Python 3.11 in .venv -> libraries -> app.
# -ChangeKey (run by "Change AI Key - Windows.bat") only asks for a new key and saves it.
param([switch]$ChangeKey)
$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($msg) { Write-Host "`n== $msg" -ForegroundColor Green }
function Fail($msg) { Write-Host "`n$msg" -ForegroundColor Red; exit 1 }

function Save-Key($key) {
    # keep any other settings already in .env; only the key line changes
    $src = if (Test-Path .env) { '.env' } else { '.env.example' }
    $lines = (Get-Content $src) -replace '^LLM_API_KEY=.*', "LLM_API_KEY=$key"
    [IO.File]::WriteAllLines((Join-Path (Get-Location) '.env'), [string[]]$lines, (New-Object Text.UTF8Encoding $false))
}

# 1. AI key: asked first (so the install can run unattended), checked with DeepSeek before saving
if ($ChangeKey -or -not (Test-Path .env)) {
    Step 'AI key'
    Write-Host 'Paste your DeepSeek API key and press Enter (get one at https://platform.deepseek.com/api_keys).'
    Write-Host 'Or just press Enter to skip: the app works without AI. You can add a key later with "Change AI Key - Windows.bat".'
    while ($true) {
        $key = "$(Read-Host 'API key')".Trim()
        if (-not $key) {
            if (-not (Test-Path .env)) { Save-Key '' }
            Write-Host 'Skipped. AI features stay off.'
            break
        }
        try {
            Invoke-RestMethod -Uri 'https://api.deepseek.com/models' -Headers @{ Authorization = "Bearer $key" } -TimeoutSec 20 | Out-Null
            Save-Key $key
            Write-Host 'Key works and is saved.' -ForegroundColor Green
            break
        } catch {
            $code = $_.Exception.Response.StatusCode.value__
            if ($code -eq 401 -or $code -eq 403) {
                Write-Host 'DeepSeek rejected this key. Copy it again and paste it, or press Enter to skip.' -ForegroundColor Yellow
            } else {
                Save-Key $key
                Write-Host 'Could not reach DeepSeek to check the key (no internet?). Saved it anyway.' -ForegroundColor Yellow
                break
            }
        }
    }
}
if ($ChangeKey) {
    Write-Host "`nDone. If the app is open, close its window and start it again with START HERE - Windows.bat."
    Read-Host 'Press Enter to close' | Out-Null
    exit 0
}

# 2. uv
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = Join-Path $env:USERPROFILE '.local\bin\uv.exe' }
if (-not (Test-Path $uv)) {
    Step 'Installing uv (it downloads Python and the libraries the app needs)'
    powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    if (-not (Test-Path $uv)) { Fail 'Could not install uv. Check the internet connection and double-click "START HERE - Windows.bat" again.' }
}

# 3. Python 3.11 in .venv
$py = '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
    Step 'Setting up Python 3.11 (first run only)'
    & $uv venv .venv --python 3.11 --seed
    if ($LASTEXITCODE -ne 0) { Fail 'Could not set up Python. Check the internet connection and double-click "START HERE - Windows.bat" again.' }
}

# 4. Libraries: reinstalled only when requirements.txt changes
$hash = (Get-FileHash requirements.txt).Hash
$stamp = '.venv\.installed'
if (-not (Test-Path $stamp) -or (Get-Content $stamp) -ne $hash) {
    Step 'Installing libraries (first run takes a few minutes)'
    & $uv pip install --python $py -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail 'Could not install the libraries. Check the internet connection and double-click "START HERE - Windows.bat" again.' }
    Set-Content $stamp $hash
}

# Streamlit asks for an e-mail on its very first start; answer it so the app starts straight away
$cred = Join-Path $env:USERPROFILE '.streamlit\credentials.toml'
if (-not (Test-Path $cred)) {
    New-Item -ItemType Directory -Force (Split-Path $cred) | Out-Null
    [IO.File]::WriteAllText($cred, "[general]`nemail = `"`"`n")
}

# 5. Run
Step 'Starting the app. It opens in your browser. Keep this window open while you use it; close it to stop.'
& $py -m streamlit run app.py
