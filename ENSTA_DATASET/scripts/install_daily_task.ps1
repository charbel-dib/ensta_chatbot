# install_daily_task.ps1

# 1) ENSTA_DATASET root (celui qui contient 01_raw/02_clean/03_chunks/04_index/scripts)
$ProjectRoot = "C:\Users\charb\OneDrive - ENSTA\Documents\Projet 3A\to github\ENSTA_DATASET"

# 2) python.exe de TON venv (celui que tu utilises pour lancer uvicorn / scripts)
$PythonExe   = "C:\Users\charb\Downloads\Final Exam\.venv\Scripts\python.exe"

$ScriptPath  = Join-Path $ProjectRoot "scripts\run_web_daily_update.py"
$TaskName    = "ACAENSTA_WebDailyUpdate"

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