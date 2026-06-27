param(
    [ValidateSet("all", "html", "pdf")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$artigoQmd = Join-Path $repoRoot "artigo/artigo_tcc.qmd"

if (-not (Test-Path $artigoQmd)) {
    throw "Arquivo não encontrado: $artigoQmd"
}

$quartoCmd = Get-Command quarto -ErrorAction SilentlyContinue
if (-not $quartoCmd) {
    throw "Quarto não encontrado no PATH. Instale em https://quarto.org/docs/get-started/."
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
