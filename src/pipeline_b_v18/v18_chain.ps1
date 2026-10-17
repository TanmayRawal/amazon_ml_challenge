[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:BER_DATA_DIR = "data/dataset"
$env:BER_WORK_DIR = "work"
$env:BER_UPLOAD_DIR = "submissions"
$env:BER_GPU_PYTHON = "python"
Set-Location "."
$py = "python"
$W = "work"
$list = "fils,associes,developpement,services,cie,groupe,france"

function Step($name, [string[]]$cmdArgs) {
    Write-Output "===== STEP $name ($(Get-Date -Format HH:mm:ss))"
    & $py -u src/run.py @cmdArgs 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { Write-Output "===== FAILED at $name (exit $LASTEXITCODE)"; exit 1 }
}

$from = if ($args.Count -gt 0) { $args[0] } else { "pj-train 2" }
if (-not (Test-Path "$W\models_v14_backup_matcher_c")) { Copy-Item -Recurse "$W\models\matcher_c" "$W\models_v14_backup_matcher_c" }
$steps = @(
    @("pj-train 2", @("pj-train", "--pj-tag", "2")),
    @("pj-score 2", @("pj-score", "--split", "all", "--pj-tag", "2")),
    @("stack-features train", @("stack-features", "--split", "train", "--pj-sets", ",2", "--drop-noisy-hidden")),
    @("train-c", @("train-c", "--pj-sets", ",2", "--drop-noisy-hidden")),
    @("stack-features test", @("stack-features", "--split", "test", "--pj-sets", ",2")),
    @("submit-c v18", @("submit-c", "--version", "v18", "--pj-sets", ",2", "--fr-list-words", $list, "--fr-list-adds")),
    @("submit-c v18c", @("submit-c", "--version", "v18c", "--pj-sets", ",2", "--fr-list-words", $list, "--fr-list-adds", "--noisy-boost", "3"))
)
$started = $false
foreach ($s in $steps) {
    if ($s[0] -eq $from) { $started = $true }
    if (-not $started) { continue }
    Step $s[0] $s[1]
}
Write-Output "===== ALL DONE ($(Get-Date -Format HH:mm:ss))"
