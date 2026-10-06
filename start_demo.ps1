# Baseera live demo: LM Studio (Gemma on the GPU) + the app + a Cloudflare quick tunnel, supervised.
#   powershell -ExecutionPolicy Bypass -File start_demo.ps1            (add -SkipModelReload to leave LM Studio's model as it is)
# Needs: LM Studio with gemma-4-12b-it, cloudflared (winget install Cloudflare.cloudflared), .env.demo (copy of .env.demo.example).
# Privacy: no access log, no request bodies logged; logs\ holds only start/stop times and health OK/FAIL.
param([switch]$SkipModelReload, [int]$Port = 8000)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
function Log($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Add-Content logs\demo.log; Write-Host $m }

# 1. settings from .env.demo (they override .env, which may point at a paid provider)
if (-not (Test-Path .env.demo)) { Copy-Item .env.demo.example .env.demo; Log "created .env.demo from the example" }
Get-Content .env.demo | ForEach-Object {
  $l = ($_ -split '#')[0].Trim()
  if ($l -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process") }
}
$cf = if (Get-Command cloudflared -ErrorAction SilentlyContinue) { "cloudflared" } elseif (Test-Path "C:\Program Files (x86)\cloudflared\cloudflared.exe") { "& 'C:\Program Files (x86)\cloudflared\cloudflared.exe'" } else { Log "cloudflared not found: winget install Cloudflare.cloudflared"; exit 1 }
$py = if (Test-Path .venv\Scripts\python.exe) { (Resolve-Path .venv\Scripts\python.exe).Path } else { "python" }

# 2. keep Windows awake for as long as this window is open
Add-Type -Namespace Win -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Power]::SetThreadExecutionState(0x80000003)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
Log "sleep prevention on"

# 3. LM Studio server + model with 2 parallel slots (each slot gets 8K of the 16K context)
$model = $env:LOCAL_MODEL
try { $ok = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:1234/v1/models" -TimeoutSec 3).StatusCode -eq 200 } catch { $ok = $false }
if (-not $ok) { Log "starting LM Studio server"; lms server start --port 1234 | Out-Null; Start-Sleep 5 }
$ps = (lms ps) -join "`n"
$needsLoad = ($ps -notmatch [regex]::Escape($model)) -or (-not $SkipModelReload -and $ps -notmatch "\s2\s+Local")
if ($needsLoad) {
  Log "loading $model (16K context, 2 parallel, full GPU)"
  lms unload --all 2>&1 | Out-Null
  lms load $model --gpu max --context-length 16384 --parallel 2 --identifier $model -y
}
lms ps

# 4. supervised children: each restarts 5 s after it exits
function Start-Supervised($name, $cmd) {
  # one small script per service (quoting a long command through Start-Process breaks on spaces)
  $f = Join-Path $PSScriptRoot ("logs" + [char]92 + "run_$name.ps1")
  @"
`$host.UI.RawUI.WindowTitle = 'baseera-$name'
Set-Location '$PSScriptRoot'
while (`$true) {
  $cmd
  "`$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $name exited" | Add-Content logs\demo.log
  Start-Sleep 5
}
"@ | Set-Content -Encoding UTF8 $f
  Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$f`""
}
Start-Supervised "app" "& '$py' -m uvicorn api.main:app --host 127.0.0.1 --port $Port --workers 1 --no-access-log"
Remove-Item logs\cloudflared.log -ErrorAction SilentlyContinue
Start-Supervised "tunnel" "$cf tunnel --no-autoupdate --url http://127.0.0.1:$Port 2>&1 | Tee-Object -FilePath logs\cloudflared.log -Append"
Log "app and tunnel started"

# 5. watchdog: report the public URL (it changes if the tunnel restarts) and restart a hung app
$fails = 0; $lastUrl = ""
while ($true) {
  Start-Sleep 20
  $url = $null
  if (Test-Path logs\cloudflared.log) {
    $m = Select-String -Path logs\cloudflared.log -Pattern 'https://[a-z0-9-]+\.trycloudflare\.com' -AllMatches -ErrorAction SilentlyContinue
    if ($m) { $url = $m[-1].Matches[-1].Value }
  }
  if ($url -and $url -ne $lastUrl) {
    $lastUrl = $url; Set-Content demo_url.txt $url
    Write-Host ""; Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "  LIVE DEMO LINK:  $url" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Log "tunnel url changed (saved to demo_url.txt)"
  }
  try {
    $h = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 15
    $fails = 0
    Log ("health OK vectors=$($h.vectors_ready) llm_warm=$($h.llm_warm) queue=$($h.queue.active)/$($h.queue.waiting)")
  } catch {
    $fails++; Log "health FAIL ($fails)"
    if ($fails -ge 3) {
      Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'api.main:app' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
      $fails = 0; Log "app killed; its supervisor restarts it"
    }
  }
  Start-Sleep 40
}
