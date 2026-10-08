# Runs tools\check_demo.ps1 every 2 hours (CHECK_EVERY_S to change). Launched detached by tools\launch_checks_detached.ps1.
$every = if ($env:CHECK_EVERY_S) { [int]$env:CHECK_EVERY_S } else { 7200 }
while ($true) { & "$PSScriptRoot\check_demo.ps1" | Out-Null; Start-Sleep $every }
