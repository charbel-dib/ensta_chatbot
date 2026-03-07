# install_daily_task.ps1

param(
    [Parameter(Mandatory=$true)][string]$ProjectRoot,
    [Parameter(Mandatory=$true)][string]$PythonExe
)

$ScriptPath = Join-Path $ProjectRoot "scripts\run_web_daily_update.py"
$TaskName = "ACAENSTA_WebDailyUpdate"

# Si la tâche existe déjà -> on la remplace
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
  Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$Action  = New-ScheduledTaskAction -Execute $PythonExe -Argument "`"$ScriptPath`""
$Trigger = New-ScheduledTaskTrigger -Daily -At 3:00am

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
  -Description "Daily rebuild web RAG index (crawl + clean + merge + chunk + embed)" -Force

Write-Host "✅ Task created: $TaskName"
Write-Host "   Python: $PythonExe"

Write-Host "   Script: $ScriptPath"
