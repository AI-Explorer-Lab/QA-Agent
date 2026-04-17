param(
    [int]$Port = 8010,
    [switch]$SkipIngest,
    [string]$BaseUrl = ""
)

$ErrorActionPreference = "Stop"
$env:CONDA_NO_PLUGINS = "true"

$cmd = @("run", "-n", "chat-llm", "python", "scripts/e2e_acceptance.py", "--port", "$Port")
if ($SkipIngest) {
    $cmd += "--skip-ingest"
}
if ($BaseUrl -ne "") {
    $cmd += "--base-url"
    $cmd += $BaseUrl
}

Write-Host "[RUN_E2E] conda $($cmd -join ' ')"
conda @cmd
