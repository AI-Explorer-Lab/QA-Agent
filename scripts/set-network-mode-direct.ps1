param(
    [switch]$PersistUser = $true
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Clear-ProxyEnv {
    param([string]$Name, [bool]$Persist)
    if (Test-Path "Env:$Name") {
        Remove-Item -Path "Env:$Name"
    }
    if ($Persist) {
        [Environment]::SetEnvironmentVariable($Name, $null, "User")
    }
}

Write-Host "[1/4] Clear proxy environment variables..."
$proxyVars = @(
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy"
)
foreach ($name in $proxyVars) {
    Clear-ProxyEnv -Name $name -Persist $PersistUser
}

Write-Host "[2/4] Normalize pip defaults..."
python -m pip config set global.index-url "https://mirrors.aliyun.com/pypi/simple/"
python -m pip config set global.timeout 60
python -m pip config set global.retries 5
try {
    python -m pip config unset global.proxy | Out-Null
}
catch {
    Write-Host "pip global.proxy not set, skip unset."
}

Write-Host "[3/4] Normalize conda defaults..."
$env:CONDA_NO_PLUGINS = "true"
conda config --set show_channel_urls true
conda config --set channel_priority flexible
conda config --set ssl_verify true
try {
    conda config --remove-key proxy_servers | Out-Null
}
catch {
    Write-Host "conda proxy_servers not found, skip remove."
}
try {
    conda config --remove-key channels | Out-Null
}
catch {
    Write-Host "conda channels not found, skip remove."
}
conda config --add channels defaults
conda config --add channels conda-forge

Write-Host "[4/4] Done. Proxy has been disabled for new terminals."
Write-Host "Tip: Run scripts\\set-network-mode-clash.ps1 to switch back."
