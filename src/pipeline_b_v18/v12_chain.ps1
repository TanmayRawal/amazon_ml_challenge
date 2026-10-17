[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$env:BER_DATA_DIR = "data/dataset"
$env:BER_WORK_DIR = "work"
$env:BER_UPLOAD_DIR = "submissions"
$env:BER_GPU_PYTHON = "python"
$env:BER_TMP = "work/tmp"
Set-Location "."
$py = "python"

function Step($name, [string[]]$cmdArgs) {
    Write-Output "===== STEP $name ($(Get-Date -Format HH:mm:ss))"
    & $py -u src/run.py @cmdArgs 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { Write-Output "===== FAILED at $name (exit $LASTEXITCODE)"; exit 1 }
}

$from = if ($args.Count -gt 0) { $args[0] } else { "prepare" }
$steps = @(
    @("prepare", @("prepare", "--split", "all")),
    @("translit", @("translit", "--split", "all")),
    @("folds", @("folds")),
    @("indic-dict", @("indic-dict")),
    @("fuzzy train", @("fuzzy", "--split", "train")),
    @("fuzzy test", @("fuzzy", "--split", "test")),
    @("rank-train", @("rank-train")),
    @("rank train", @("rank", "--split", "train")),
    @("rank test", @("rank", "--split", "test")),
    @("features train c", @("features", "--split", "train", "--variant", "c")),
    @("features test c", @("features", "--split", "test", "--variant", "c")),
    @("train-b", @("train-b")),
    @("score-b test", @("score-b", "--split", "test")),
    @("stack-features train", @("stack-features", "--split", "train")),
    @("train-c", @("train-c")),
    @("stack-features test", @("stack-features", "--split", "test")),
    @("submit-c v12", @("submit-c", "--version", "v12"))
)
$started = $false
foreach ($s in $steps) {
    if ($s[0] -eq $from) { $started = $true }
    if (-not $started) { continue }
    if ($s[0] -eq "score-b test") { Remove-Item -LiteralPath "work/test/scores_b.parquet" -ErrorAction SilentlyContinue }
    Step $s[0] $s[1]
}
Write-Output "===== ALL DONE ($(Get-Date -Format HH:mm:ss))"
