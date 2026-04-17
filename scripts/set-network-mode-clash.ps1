param(
    [string]$Proxy = "http://127.0.0.1:7897",
    [switch]$PersistUser = $true
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

function Set-ProxyEnv {
    param([string]$Name, [string]$Value, [bool]$Persist)
    Set-Item -Path "Env:$Name" -Value $Value
    if ($Persist) {
        [Environment]::SetEnvironmentVariable($Name, $Value, "User")
    }
}

Write-Host "[1/4] Set proxy environment variables..."
$proxyVars = @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
foreach ($name in $proxyVars) {
    Set-ProxyEnv -Name $name -Value $Proxy -Persist $PersistUser
}
Set-ProxyEnv -Name "NO_PROXY" -Value "localhost,127.0.0.1" -Persist $PersistUser
Set-ProxyEnv -Name "no_proxy" -Value "localhost,127.0.0.1" -Persist $PersistUser

Write-Host "[2/4] Normalize pip defaults..."
python -m pip config set global.index-url "https://mirrors.aliyun.com/pypi/simple/"
python -m pip config set global.timeout 60
python -m pip config set global.retries 5
python -m pip config set global.proxy $Proxy

Write-Host "[3/4] Normalize conda defaults..."
$env:CONDA_NO_PLUGINS = "true"
conda config --set show_channel_urls true
conda config --set channel_priority flexible
conda config --set ssl_verify true
try {
    conda config --remove-key channels | Out-Null
}
catch {
    Write-Host "conda channels not found, skip remove."
}
conda config --add channels defaults
conda config --add channels conda-forge
conda config --set proxy_servers.http $Proxy
conda config --set proxy_servers.https $Proxy

Write-Host "[4/4] Done. New terminals will use Clash proxy."
Write-Host "Tip: Run scripts\\set-network-mode-direct.ps1 when you want to disable proxy."
