# SPDX-License-Identifier: Apache-2.0
# AMUDA CUDA-like POC fleet runner (Windows PC -> Termux phones over the existing ssh aliases).
#
#   powershell -ExecutionPolicy Bypass -File scripts\fleet\run_poc_fleet.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\fleet\run_poc_fleet.ps1 -Hosts a35,s25 -Backends vulkan
#   add -InstallDeps to let each phone run 'pkg install -y' for missing build tools
#   add -WakeLock to hold termux-wake-lock during the runs (latency A/B), -NoClLdPath to run OpenCL without
#   LD_LIBRARY_PATH (default /system/lib64:/vendor/lib64:/vendor/lib64/egl, see scripts/termux/amuda_poc.sh),
#   -NoTrace to skip the AMUDA_TRACE phase-timing run
#
# For every host: copy sources, build on the phone, run the 5 samples, keep the full log in
# evidence\poc_cuda_like\<timestamp>\<host>.log and print a summary. Unreachable hosts are reported
# as UNREACHABLE; they are never skipped silently.
param(
    [string[]]$Hosts = @('s25', 's21', 's20', 'a35', 'a53'),
    [string]$Backends = 'vulkan,opencl',
    [switch]$InstallDeps,
    [switch]$WakeLock,
    [switch]$NoClLdPath,
    [switch]$NoTrace
)
$ErrorActionPreference = 'Continue'
# 'powershell -File ... -Hosts a35,s25' binds the single string "a35,s25"; split it so each host runs separately
$Hosts = @($Hosts | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($Hosts.Count -eq 0) { Write-Host "no hosts given"; exit 2 }
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$outDir = Join-Path $root "evidence\poc_cuda_like\$stamp"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$pkg = Join-Path $env:TEMP "amuda-poc-$stamp.tar.gz"
Push-Location $root
tar -czf $pkg include src/amuda_rt src/backend src/runtime src/amuda tools/amudacc samples/cuda_like scripts/termux
$tarExit = $LASTEXITCODE
Pop-Location
if ($tarExit -ne 0) { Write-Host "tar failed ($tarExit)"; exit 2 }

$remoteDir = 'amuda-poc-cuda-like'
$summary = @()
foreach ($h in $Hosts) {
    $log = Join-Path $outDir "$h.log"
    Write-Host "=== $h ==="
    scp -q -o ConnectTimeout=8 -o BatchMode=yes $pkg "${h}:amuda-poc.tar.gz" 2>&1 | Out-File -Encoding utf8 $log
    if ($LASTEXITCODE -ne 0) {
        "UNREACHABLE scp exit=$LASTEXITCODE" | Add-Content -Encoding utf8 $log
        $summary += "$h UNREACHABLE"
        continue
    }
    $cmd = "rm -rf ~/$remoteDir && mkdir -p ~/$remoteDir && tar -xzf ~/amuda-poc.tar.gz -C ~/$remoteDir && " +
           "bash ~/$remoteDir/scripts/termux/amuda_poc.sh --backends $Backends"
    if ($InstallDeps) { $cmd += " --install-deps" }
    if ($WakeLock) { $cmd += " --wake-lock" }
    if ($NoClLdPath) { $cmd += " --no-cl-ldpath" }
    if ($NoTrace) { $cmd += " --no-trace" }
    Write-Host "building and running on $h (a few minutes, output appears when it finishes)..."
    $out = ssh -o ConnectTimeout=8 -o BatchMode=yes $h $cmd 2>&1
    $rc = $LASTEXITCODE
    $out | ForEach-Object { "$_" } | Add-Content -Encoding utf8 $log
    $out | ForEach-Object { Write-Host "$_" }
    "REMOTE_EXIT=$rc" | Add-Content -Encoding utf8 $log
    $lines = Select-String -Path $log -Pattern '^AMU_SUMMARY' | ForEach-Object { $_.Line }
    $summary += "$h exit=$rc " + ($lines -join ' | ')
}

Write-Host ""
Write-Host "==== AMUDA CUDA-like POC fleet summary ($stamp) ===="
$summary | ForEach-Object { Write-Host $_ }
$summary | Out-File -Encoding utf8 (Join-Path $outDir 'SUMMARY.txt')
Write-Host "logs: $outDir"
Remove-Item -Force $pkg -ErrorAction SilentlyContinue
