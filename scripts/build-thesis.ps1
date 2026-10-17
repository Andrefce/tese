<#
.SYNOPSIS
    Builds main.tex with XeLaTeX + Biber without latexmk.

.DESCRIPTION
    MiKTeX's latexmk needs Perl, which is not installed here, so this script
    provides the same "only rerun what is needed" logic for LaTeX Workshop.

    fast  One XeLaTeX pass. Biber (plus one more pass) runs only when the
          citations or the .bib file changed. The PDF is written with light
          compression. Intended for build-on-save previews.
    full  Always runs Biber, reruns XeLaTeX until cross-references settle,
          and writes a well-compressed PDF. Use before sharing/submitting.

    In both modes, damaged auxiliary files (e.g. after a build that stopped
    on an error) are detected, deleted and regenerated automatically.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\build-thesis.ps1 -Mode full
#>
param(
    [ValidateSet('fast', 'full')]
    [string]$Mode = 'fast'
)

# Native tools print MiKTeX notices on stderr; never let them abort the build.
$ErrorActionPreference = 'Continue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

# Two builds writing main.aux/.bcf/.pdf at the same time corrupt them (NUL-
# filled files, vanished .bcf). Serialise every build of this folder: the
# editor's build-on-save, manual runs and coding agents. Windows releases the
# lock when this process exits, however it exits.
$lockName = 'Local\thesis-build-' + [BitConverter]::ToString(
    [Security.Cryptography.SHA1]::Create().ComputeHash(
        [Text.Encoding]::UTF8.GetBytes($root.ToLowerInvariant()))).Replace('-', '')
$buildLock = New-Object System.Threading.Mutex($false, $lockName)
try {
    if (-not $buildLock.WaitOne(0)) {
        Write-Host '[build] Another build of this thesis is running; waiting for it to finish.'
        [void]$buildLock.WaitOne()
    }
}
catch [System.Threading.AbandonedMutexException] {
    # The previous holder was killed mid-build; we now own the lock.
}

$job = 'main'
$maxPasses = 5
$auxExtensions = 'aux', 'toc', 'lof', 'lot', 'out'
$biberStamp = "$job.bcf.sha1"

function Resolve-Tool([string]$name) {
    $cmd = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    $fallback = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\$name.exe"
    if (Test-Path -LiteralPath $fallback) { return $fallback }
    Write-Host "[build] ERROR: cannot find '$name'. Is MiKTeX installed and on PATH?"
    exit 1
}

$xelatex = Resolve-Tool 'xelatex'
$biber = Resolve-Tool 'biber'
$xdvipdfmx = Resolve-Tool 'xdvipdfmx'

function Get-LogText {
    $path = Join-Path $root "$job.log"
    if (Test-Path -LiteralPath $path) { return [IO.File]::ReadAllText($path) }
    return ''
}

$script:passes = 0
function Invoke-XeLaTeX {
    $script:passes++
    Write-Host "[build] XeLaTeX pass $($script:passes)"
    # Out-Host keeps the console log out of the return value; LaTeX Workshop
    # parses it from stdout to fill the Problems panel.
    & $xelatex -synctex=1 -interaction=nonstopmode -file-line-error -no-pdf "$job.tex" | Out-Host
    return Get-LogText
}

function Remove-JobFiles([string[]]$extensions) {
    foreach ($ext in $extensions) {
        Remove-Item -LiteralPath "$job.$ext" -Force -ErrorAction SilentlyContinue
    }
}

# Errors reported *inside* an auxiliary file, or at \begin{document} /
# \end{document} (where LaTeX reads main.aux back), mean it was left
# half-written by an earlier failed build.
$docLines = (Select-String -LiteralPath "$job.tex" -Pattern '^\s*\\(begin|end)\{document\}' |
    ForEach-Object { $_.LineNumber }) -join '|'
function Test-AuxDamaged([string]$log) {
    if ($log -match "(?m)^(\./)?$job\.(aux|toc|lof|lot|out):\d+: ") { return $true }
    return ($docLines -and $log -match "(?m)^(\./)?$job\.tex:($docLines): ")
}

function Test-BblDamaged([string]$log) {
    return $log -match "(?m)^(\./)?$job\.bbl:\d+: "
}

function Test-RerunNeeded([string]$log) {
    return $log -match '(?m)(Rerun to get|Label\(s\) may have changed|Please rerun LaTeX|Rerun LaTeX|has changed\. Rerun)'
}

function Test-BiberNeeded([string]$log) {
    if (-not (Test-Path -LiteralPath "$job.bbl")) { return $true }
    if ($log -match 'Please \(re\)run Biber') { return $true }
    if (-not (Test-Path -LiteralPath "$job.bcf")) { return $false }
    if (-not (Test-Path -LiteralPath $biberStamp)) { return $true }
    # The .bcf lists every cited key in order, so any change means new numbering.
    $current = (Get-FileHash -LiteralPath "$job.bcf" -Algorithm SHA1).Hash
    if ($current -ne (Get-Content -LiteralPath $biberStamp -Raw).Trim()) { return $true }
    $bblTime = (Get-Item -LiteralPath "$job.bbl").LastWriteTime
    $newerBib = Get-ChildItem -Path 'bibliography' -Filter '*.bib' -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -gt $bblTime }
    return [bool]$newerBib
}

function Invoke-Biber {
    Write-Host '[build] Running Biber'
    $backup = "$job.bbl.prev"
    if (Test-Path -LiteralPath "$job.bbl") { Copy-Item -LiteralPath "$job.bbl" $backup -Force }
    & $biber $job | Out-Host
    $ok = ($LASTEXITCODE -eq 0)
    if ($ok) {
        (Get-FileHash -LiteralPath "$job.bcf" -Algorithm SHA1).Hash | Set-Content -LiteralPath $biberStamp
        Remove-Item -LiteralPath $backup -Force -ErrorAction SilentlyContinue
    }
    else {
        Write-Host '[build] WARNING: Biber failed; keeping the previous bibliography.'
        Remove-Item -LiteralPath $biberStamp -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $backup) { Move-Item -LiteralPath $backup "$job.bbl" -Force }
    }
    return $ok
}

$started = Get-Date
$log = Invoke-XeLaTeX
$settle = ($Mode -eq 'full')

if ((Test-AuxDamaged $log) -or (Test-BblDamaged $log)) {
    Write-Host '[build] Damaged auxiliary files detected; regenerating them.'
    Remove-JobFiles $auxExtensions
    if (Test-BblDamaged $log) {
        Remove-JobFiles @('bbl')
        Remove-Item -LiteralPath $biberStamp -Force -ErrorAction SilentlyContinue
    }
    $log = Invoke-XeLaTeX
    $settle = $true
}

if (($Mode -eq 'full') -or (Test-BiberNeeded $log)) {
    [void](Invoke-Biber)
    $log = Invoke-XeLaTeX
}

if ($settle) {
    while ($script:passes -lt $maxPasses -and (Test-RerunNeeded $log)) {
        $log = Invoke-XeLaTeX
    }
}

if ($log -notmatch "Output written on $job\.xdv") {
    Write-Host '[build] ERROR: XeLaTeX produced no output; see the errors above.'
    exit 1
}

# xdvipdfmx's default (-z 9) spends ~35 s recompressing the PNG figures for
# barely smaller output; -z 6 is ~4x faster and -z 1 faster still for previews.
$compression = if ($Mode -eq 'fast') { 1 } else { 6 }
$tmpPdf = "$job.building.pdf"
Write-Host "[build] Writing PDF (compression level $compression)"
& $xdvipdfmx -q -E -z $compression -o $tmpPdf "$job.xdv" | Out-Host
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $tmpPdf)) {
    Remove-Item -LiteralPath $tmpPdf -Force -ErrorAction SilentlyContinue
    Write-Host '[build] ERROR: xdvipdfmx failed to write the PDF.'
    exit 1
}

# Replace the PDF in one step so the viewer never reads a half-written file.
$moved = $false
for ($i = 0; $i -lt 10 -and -not $moved; $i++) {
    try {
        Move-Item -LiteralPath $tmpPdf "$job.pdf" -Force -ErrorAction Stop
        $moved = $true
    }
    catch { Start-Sleep -Milliseconds 300 }
}
if (-not $moved) {
    Remove-Item -LiteralPath $tmpPdf -Force -ErrorAction SilentlyContinue
    Write-Host "[build] ERROR: $job.pdf is locked by another program (close external PDF viewers)."
    exit 1
}

$errors = ([regex]::Matches($log, '(?m)^(!|\S+\.\w+:\d+: )')).Count
$elapsed = ((Get-Date) - $started).TotalSeconds
Write-Host ("[build] Done in {0:N1}s ({1} XeLaTeX pass(es), {2} LaTeX error(s))." -f $elapsed, $script:passes, $errors)
exit 0
