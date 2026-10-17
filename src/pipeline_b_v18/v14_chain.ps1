[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:BER_DATA_DIR = "data/dataset"
$env:BER_WORK_DIR = "work"
$env:BER_UPLOAD_DIR = "submissions"
$env:BER_GPU_PYTHON = "python"
$env:BER_TMP = "work/tmp"
Set-Location "."
$py = "python"
$W = "work"

function Step($name, [string[]]$cmdArgs) {
    Write-Output "===== STEP $name ($(Get-Date -Format HH:mm:ss))"
    & $py -u src/run.py @cmdArgs 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { Write-Output "===== FAILED at $name (exit $LASTEXITCODE)"; exit 1 }
}

# v14 = v12 candidates cut to 30 per S1 + stage 1 retrained + pair judge retrained on this run's
# stage-1 band + stage 2 (decision shift 0.9). Keep v13's models: these steps overwrite them.
$from = if ($args.Count -gt 0) { $args[0] } else { "features train c" }
if ($from -eq "features train c") {
    if (-not (Test-Path "$W\models_v13_backup_matcher_c")) { Copy-Item -Recurse "$W\models\matcher_c" "$W\models_v13_backup_matcher_c" }
    if (-not (Test-Path "$W\models_v12_backup_matcher_b")) { Copy-Item -Recurse "$W\models\matcher_b" "$W\models_v12_backup_matcher_b" }
    New-Item -ItemType Directory -Force "$W\pj\v13" | Out-Null
    foreach ($m in "model_A", "model_B") { if (Test-Path "$W\pj\$m") { Move-Item "$W\pj\$m" "$W\pj\v13\$m" } }
    if (Test-Path "$W\pj\train_pairs.parquet") { Move-Item -Force "$W\pj\train_pairs.parquet" "$W\pj\v13\train_pairs.parquet" }
}
$steps = @(
    @("features train c", @("features", "--split", "train", "--variant", "c")),
    @("features test c", @("features", "--split", "test", "--variant", "c")),
    @("train-b", @("train-b")),
    @("score-b test", @("score-b", "--split", "test")),
    @("pj-train", @("pj-train")),
    @("pj-score", @("pj-score", "--split", "all")),
    @("stack-features train", @("stack-features", "--split", "train")),
    @("train-c", @("train-c")),
    @("stack-features test", @("stack-features", "--split", "test")),
    @("submit-c v14", @("submit-c", "--version", "v14"))
)
$started = $false
foreach ($s in $steps) {
    if ($s[0] -eq $from) { $started = $true }
    if (-not $started) { continue }
    if ($s[0] -eq "score-b test") { Remove-Item -LiteralPath "$W\test\scores_b.parquet" -ErrorAction SilentlyContinue }
    Step $s[0] $s[1]
}
Write-Output "===== ALL DONE ($(Get-Date -Format HH:mm:ss))"
