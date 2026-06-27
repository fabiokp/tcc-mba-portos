param(
    [string]$Version = "v1.0.0",
    [string]$OutputRoot = "dist/zenodo",
    [string]$DatasetTitle = "processed-data-portos-brasil"
)

$ErrorActionPreference = "Stop"

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$FullPath
    )

    $base = [System.IO.Path]::GetFullPath($BasePath)
    if (-not $base.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $base += [System.IO.Path]::DirectorySeparatorChar
    }
    $full = [System.IO.Path]::GetFullPath($FullPath)
    $baseUri = [System.Uri]$base
    $fullUri = [System.Uri]$full
    $rel = $baseUri.MakeRelativeUri($fullUri).ToString()
    return [System.Uri]::UnescapeDataString($rel).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRoot = $repoRoot.Path

$stageDir = Join-Path $repoRoot (Join-Path $OutputRoot ("$DatasetTitle-$Version"))
$zipPath = "$stageDir.zip"

if (Test-Path $stageDir) {
    Remove-Item -Recurse -Force $stageDir
}
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

$includeDirs = @(
    "data/processed",
    "data/tabelas_auxiliares"
)

$includeOutputExt = @(".csv", ".parquet")

$files = New-Object System.Collections.Generic.List[string]

foreach ($dir in $includeDirs) {
    $absDir = Join-Path $repoRoot $dir
    if (Test-Path $absDir) {
        Get-ChildItem -Path $absDir -File -Recurse | ForEach-Object {
            $files.Add($_.FullName)
        }
    }
}

$absOutput = Join-Path $repoRoot "data/output"
if (Test-Path $absOutput) {
    Get-ChildItem -Path $absOutput -File -Recurse | Where-Object {
        $includeOutputExt -contains $_.Extension.ToLowerInvariant()
    } | ForEach-Object {
        $files.Add($_.FullName)
    }
}

$files = $files | Sort-Object -Unique

if ($files.Count -eq 0) {
    throw "Nenhum arquivo encontrado para o pacote Zenodo."
}

foreach ($src in $files) {
    $rel = Get-RelativePath -BasePath $repoRoot -FullPath $src
    $dst = Join-Path $stageDir $rel
    $dstDir = Split-Path -Parent $dst
    New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
    Copy-Item -Path $src -Destination $dst -Force
}

$manifestPath = Join-Path $stageDir "manifest.csv"
"path,size_bytes,sha256" | Out-File -FilePath $manifestPath -Encoding utf8

$totalBytes = 0L
foreach ($f in (Get-ChildItem -Path $stageDir -File -Recurse | Sort-Object FullName)) {
    if ($f.Name -eq "manifest.csv") {
        continue
    }
    $hash = (Get-FileHash -Path $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $rel = Get-RelativePath -BasePath $stageDir -FullPath $f.FullName
    $line = '"{0}",{1},"{2}"' -f $rel.Replace('"', '""'), $f.Length, $hash
    Add-Content -Path $manifestPath -Value $line -Encoding utf8
    $totalBytes += $f.Length
}

$readmePath = Join-Path $stageDir "DATASET_README.txt"
$readme = @(
    "Processed dataset package for reproducibility",
    "",
    "Repository: https://github.com/fabiokp/tcc-mba-portos",
    "Version: $Version",
    "Generated at (UTC): $([DateTime]::UtcNow.ToString('yyyy-MM-dd HH:mm:ss'))",
    "",
    "Included data:",
    "- data/processed/**",
    "- data/tabelas_auxiliares/**",
    "- data/output/**/*.csv",
    "- data/output/**/*.parquet",
    "",
    "Excluded data:",
    "- raw data sources (download from original providers)",
    "- rendered images/html and other non-tabular artifacts",
    "",
    "Integrity:",
    "- Use manifest.csv (SHA-256) to verify each file.",
    "",
    "Suggested citation:",
    "- Paim, Fabio Kouri. Processed data package for 'Monitoramento e Diagnostico de Eventos Disruptivos nos Portos Brasileiros'. Version $Version.",
    "",
    "License:",
    "- Define in Zenodo deposit (recommended CC-BY-4.0 for data)."
)
$readme -join [Environment]::NewLine | Out-File -FilePath $readmePath -Encoding utf8

$metadataPath = Join-Path $stageDir "zenodo_metadata_template.json"
$metadata = @"
{
  "title": "Processed data package - Monitoramento e Diagnostico de Eventos Disruptivos nos Portos Brasileiros ($Version)",
  "upload_type": "dataset",
  "description": "Processed tabular data used in the TCC article and dashboard. Includes data/processed, data/tabelas_auxiliares, and selected tabular outputs from data/output.",
  "creators": [
    {
      "name": "Paim, Fabio Kouri",
      "affiliation": "ENAP"
    }
  ],
  "keywords": [
    "portos",
    "anomalias",
    "supply chain",
    "comercio exterior",
    "brasil"
  ],
  "access_right": "open",
  "license": "cc-by-4.0",
  "version": "$Version",
  "related_identifiers": [
    {
      "identifier": "https://github.com/fabiokp/tcc-mba-portos",
      "relation": "isSupplementTo",
      "resource_type": "software"
    }
  ]
}
"@
$metadata | Out-File -FilePath $metadataPath -Encoding utf8

Compress-Archive -Path (Join-Path $stageDir "*") -DestinationPath $zipPath -CompressionLevel Optimal -Force

$zipInfo = Get-Item $zipPath

Write-Output "Pacote Zenodo preparado com sucesso."
Write-Output "Stage: $stageDir"
Write-Output "ZIP:   $zipPath"
Write-Output ("Arquivos incluídos: {0}" -f $files.Count)
Write-Output ("Dados tabulares (bytes): {0}" -f $totalBytes)
Write-Output ("ZIP size (MB): {0}" -f [math]::Round($zipInfo.Length / 1MB, 2))
