# Starts start_demo.ps1 OUTSIDE the process tree of whatever launched it (Claude app, terminal, IDE).
# Why: the Claude desktop app updates itself every few hours and kills every process it started, which took the live demo
# (watchdog, tunnel, LM Studio) down twice on 2026-10-07 (15:39 and 21:50, matching its package updates in the Windows log).
# Win32_Process.Create is executed by the WMI service, so the new process is not a child of the caller and not in its job object.
#   powershell -ExecutionPolicy Bypass -File launch_demo_detached.ps1
$dir = $PSScriptRoot
$cmd = "powershell.exe -NoExit -ExecutionPolicy Bypass -File `"$dir\start_demo.ps1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $dir }
if ($r.ReturnValue -eq 0) { Write-Host "demo started detached (pid $($r.ProcessId)); the link appears in demo_url.txt in about 30 seconds" }
else { Write-Host "could not start (code $($r.ReturnValue))"; exit 1 }
