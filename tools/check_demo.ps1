# One health check of the live demo: app, model, tunnel, public link, usage, disk. Appends a summary to logs\checks.log and prints it.
# Exit code 0 = all good, 1 = a problem. Read-only: it never restarts anything.
param([int]$Port = 8000)
Set-Location (Split-Path $PSScriptRoot)
New-Item -ItemType Directory -Force logs | Out-Null
$p = @(); $info = @()
try {
  $h = Invoke-RestMethod "http://127.0.0.1:$Port/api/health" -TimeoutSec 20
  if (-not $h.ok) { $p += "database not ok" }
  if ($h.llm.reachable -ne $true) { $p += "model server NOT reachable" }
  if (-not $h.vectors_ready) { $p += "vectors not ready" }
  $info += "app ok (model reachable=$($h.llm.reachable), warm=$($h.llm_warm), queue=$($h.queue.active)/$($h.queue.waiting))"
} catch { $p += "app NOT answering on port $Port" }
$url = if (Test-Path demo_url.txt) { (Get-Content demo_url.txt -Raw).Trim() } else { "" }
if ($url) {
  try { $c = (Invoke-WebRequest "$url/api/health" -UseBasicParsing -TimeoutSec 25).StatusCode; if ($c -ne 200) { $p += "public link returned $c" } else { $info += "public link ok ($url)" } }
  catch { $p += "public link NOT reachable ($url)" }
} else { $p += "demo_url.txt missing" }
if (-not (Get-Process cloudflared -ErrorAction SilentlyContinue)) { $p += "cloudflared not running" }
try {
  $s = Invoke-RestMethod "http://127.0.0.1:$Port/api/stats" -TimeoutSec 10
  $info += "usage since $($s.since): $($s.questions) questions, $($s.verifications) verifications, $($s.errors) errors, last request $($s.last_request)"
  if ($s.errors -gt 0) { $p += "$($s.errors) request error(s) since $($s.since)" }
} catch { $info += "stats unavailable" }
if (Test-Path logs\cloudflared.log) {
  try {  # the tunnel keeps the file open: read it with sharing allowed
    $fs = [System.IO.File]::Open((Resolve-Path logs\cloudflared.log).Path, 'Open', 'Read', 'ReadWrite')
    $t = (New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::Unicode)).ReadToEnd(); $fs.Close()
    $info += "tunnel reconnect errors in log so far: $(([regex]::Matches($t, ' ERR ')).Count)"
  } catch { $info += "tunnel log unreadable" }
}
$free = [math]::Round((Get-PSDrive C).Free / 1GB, 1); $info += "C: free $free GB"; if ($free -lt 5) { $p += "low disk space ($free GB)" }
$status = if ($p.Count) { "PROBLEM: " + ($p -join "; ") } else { "OK" }
$line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm') $status | " + ($info -join " | ")
$line | Add-Content logs\checks.log; Write-Host $line
exit ([int]($p.Count -gt 0))
