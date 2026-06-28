param(
    [ValidateSet("all", "html", "pdf")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$artigoQmd = Join-Path $repoRoot "artigo/artigo_tcc.qmd"
$beforeBodyTex = Join-Path $repoRoot "artigo/before-body.tex"

if (-not (Test-Path $artigoQmd)) {
    throw "Arquivo não encontrado: $artigoQmd"
}

if (-not (Test-Path $beforeBodyTex)) {
    throw "Arquivo não encontrado: $beforeBodyTex"
}

$quartoCmd = Get-Command quarto -ErrorAction SilentlyContinue
if (-not $quartoCmd) {
    throw "Quarto não encontrado no PATH. Instale em https://quarto.org/docs/get-started/."
}

# Guarda de consistência: o resumo da folha de rosto deve existir no before-body.
$beforeBodyRaw = Get-Content -Raw -Path $beforeBodyTex
if ($beforeBodyRaw -notmatch '\\noindent\\textbf\{Resumo\.\}') {
    throw "before-body.tex inconsistente: bloco de Resumo não encontrado."
}

Push-Location $repoRoot
try {
    switch ($Target) {
        "html" {
            quarto render $artigoQmd --to html
        }
        "pdf" {
            quarto render $artigoQmd --to pdf
        }
        default {
            quarto render $artigoQmd --to html
            quarto render $artigoQmd --to pdf
        }
    }

    Write-Host "Build do artigo concluído: $Target"
}
finally {
    Pop-Location
}
