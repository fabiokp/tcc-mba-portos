param(
    [Parameter(Mandatory = $true)]
    [string]$AccessToken,

    [Parameter(Mandatory = $true)]
    [string]$ZipPath,

    [Parameter(Mandatory = $true)]
    [string]$MetadataJsonPath,

    [switch]$Sandbox,
    [switch]$Publish
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath {
    param([string]$Path)
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    return $item.FullName
}

$zipAbs = Resolve-AbsolutePath -Path $ZipPath
$metadataAbs = Resolve-AbsolutePath -Path $MetadataJsonPath

$baseUrl = if ($Sandbox) { "https://sandbox.zenodo.org" } else { "https://zenodo.org" }
$apiUrl = "$baseUrl/api/deposit/depositions"

$headers = @{
    "Authorization" = "Bearer $AccessToken"
}

Write-Output "1) Criando novo depósito em $baseUrl ..."
$deposition = Invoke-RestMethod -Method POST -Uri $apiUrl -Headers $headers -ContentType "application/json" -Body "{}"

$depositionId = $deposition.id
$bucketUrl = $deposition.links.bucket

if (-not $depositionId -or -not $bucketUrl) {
    throw "Falha ao criar depósito: id/bucket não retornados."
}

Write-Output "Deposition ID: $depositionId"
Write-Output "2) Fazendo upload do ZIP..."
$fileName = [System.IO.Path]::GetFileName($zipAbs)
$uploadUrl = "$bucketUrl/$fileName"

Invoke-RestMethod -Method PUT -Uri $uploadUrl -Headers $headers -InFile $zipAbs -ContentType "application/octet-stream"

Write-Output "3) Aplicando metadados do arquivo JSON..."
$metadataObj = Get-Content -LiteralPath $metadataAbs -Raw | ConvertFrom-Json
$payload = @{ metadata = $metadataObj } | ConvertTo-Json -Depth 100

Invoke-RestMethod -Method PUT -Uri "$apiUrl/$depositionId" -Headers $headers -ContentType "application/json" -Body $payload | Out-Null

$draftUrl = "$baseUrl/deposit/$depositionId"
Write-Output "Rascunho pronto: $draftUrl"

if ($Publish) {
    Write-Output "4) Publicando depósito..."
    $publishUrl = "$apiUrl/$depositionId/actions/publish"
    $published = Invoke-RestMethod -Method POST -Uri $publishUrl -Headers $headers

    $doi = $published.doi
    $recordUrl = $published.links.record_html

    Write-Output "Publicado com sucesso."
    if ($doi) { Write-Output "DOI: $doi" }
    if ($recordUrl) { Write-Output "Registro: $recordUrl" }
} else {
    Write-Output "Publicação NÃO executada (modo rascunho)."
    Write-Output "Revise no Zenodo e publique manualmente, ou rode novamente com -Publish."
}
