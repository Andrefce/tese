<#
.SYNOPSIS
    Extracts all embedded images from the official ISTA cover .docx into images/.

.DESCRIPTION
    A .docx file is a ZIP archive. Images live under word/media/. This script
    copies the docx to a temp .zip, expands it, and copies every image into the
    workspace images/ folder with a "docx-" prefix so existing files are not
    overwritten. After running, identify which file is the department logo
    (Iscte Tecnologias e Arquitetura) and rename it to images/logo-dept.png.

.USAGE
    From the workspace root:
        powershell -ExecutionPolicy Bypass -File scripts/extract-docx-images.ps1
#>

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root 'templates/ista-capas-mestrado.docx'
$dst  = Join-Path $root 'images'

if (-not (Test-Path $src)) { throw "Source docx not found: $src" }
if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Path $dst | Out-Null }

$tmp = Join-Path $env:TEMP ("docx-extract-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null

try {
    $zip = Join-Path $tmp 'file.zip'
    Copy-Item $src $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    $media = Join-Path $tmp 'word/media'
    if (-not (Test-Path $media)) { throw "No word/media folder in archive." }

    Get-ChildItem $media | ForEach-Object {
        $target = Join-Path $dst ("docx-" + $_.Name)
        Copy-Item $_.FullName $target -Force
        Write-Host "Extracted: images/docx-$($_.Name)"
    }
}
finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Next step: identify the department logo (Iscte Tecnologias e Arquitetura)"
Write-Host "           in images/ and rename it to images/logo-dept.png"
