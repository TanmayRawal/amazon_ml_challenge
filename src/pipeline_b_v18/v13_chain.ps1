[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:BER_DATA_DIR = "data/dataset"
$env:BER_WORK_DIR = "work"
$env:BER_UPLOAD_DIR = "submissions"
$env:BER_GPU_PYTHON = "python"
$env:BER_TMP = "work/tmp"
Set-Location "."
$py = "python"
$score = "work/pj/score_all.log"   # log of the pair-judge scoring run this chain waits for

# wait for pair-judge scoring (train + test band pairs)
while ($true) {
    $s = if (Test-Path $score) { Get-Content $score -Raw } else { "" }
    if ($s -match "PJ SCORE DONE") { break }
    if ($s -match "PJ SCORE FAILED|not scoring") { Write-Output "===== FAILED: pair-judge scoring did not finish"; exit 1 }
    Start-Sleep -Seconds 60
}

function Step($name, [string[]]$cmdArgs) {
    Write-Output "===== STEP $name ($(Get-Date -Format HH:mm:ss))"
    & $py -u src/run.py @cmdArgs 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { Write-Output "===== FAILED at $name (exit $LASTEXITCODE)"; exit 1 }
}

# keep v12's stage-2 models: train-c overwrites models/matcher_c
$bk = "work/models_v12_backup_matcher_c"
if (-not (Test-Path $bk)) { Copy-Item -Recurse "work/models/matcher_c" $bk }
Step "stack-features train" @("stack-features", "--split", "train")
Step "train-c" @("train-c")
Step "stack-features test" @("stack-features", "--split", "test")
Step "submit-c v13" @("submit-c", "--version", "v13")
Write-Output "===== ALL DONE ($(Get-Date -Format HH:mm:ss))"
