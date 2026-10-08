# Starts the 2-hourly check loop outside the process tree of the app that launched it (same reason as launch_demo_detached.ps1).
$cmd = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSScriptRoot\check_loop.ps1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = (Split-Path $PSScriptRoot) }
if ($r.ReturnValue -eq 0) { Write-Host "check loop started detached (pid $($r.ProcessId)); results in logs\checks.log" } else { Write-Host "could not start (code $($r.ReturnValue))"; exit 1 }
