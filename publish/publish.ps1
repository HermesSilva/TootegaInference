# publish.ps1 -- monta a pasta de upload do HuggingFace e, so com -Upload, envia.
#
# O default e MONTAR E CONFERIR. Publicar e acao externa e irreversivel na
# pratica -- um arquivo enviado pode ficar em cache e em indice mesmo depois de
# apagado --, entao o envio exige `-Upload` explicito.
#
#   .\publish\publish.ps1 -Container models\glm47flash-v4\model.tootega
#   .\publish\publish.ps1 -Container models\glm47flash-v4\model.tootega -Upload
#
# Requer o CLI `hf` (do pacote huggingface_hub) so no passo de envio. A montagem
# nao depende de nada.

param(
    [Parameter(Mandatory = $true)] [string]$Container,
    [string]$Base   = 'models\glm47flash-bf16',
    [string]$Stage  = 'publish\stage',
    [string]$Repo   = 'HermesSilva/GLM-4.7-Flash-tootega-int4',
    [switch]$Upload
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

function Need($path, $what) {
    if (-not (Test-Path $path)) { throw "$what nao encontrado: $path" }
    return (Get-Item $path)
}

# --- o que vai junto ---------------------------------------------------------

# Nomes distintos dos parametros de proposito: PowerShell nao distingue
# maiusculas em nome de variavel, entao `$container` E `$Container`, e como o
# parametro e `[string]`, guardar um FileInfo nele o converte de volta para
# texto -- `.FullName` passa a ser $null e o argumento some da linha de comando.
$containerFile = Need $Container 'container'
$cardFile      = Need (Join-Path $PSScriptRoot 'model-card.md') 'model card'
$licenseFile   = Need (Join-Path $Root 'LICENSE') 'LICENSE'

$fromBase = @('tokenizer.json', 'chat_template.jinja', 'config.json')
foreach ($f in $fromBase) { Need (Join-Path $Base $f) $f | Out-Null }

# --- o container tem de abrir ------------------------------------------------
#
# Terminar de escrever nao prova nada; abrir prova. `--verify` passa o arquivo
# pelo leitor do proprio engine, que confere magic, versao, o alinhamento de 64
# de todo blob e o tamanho de cada um contra shape e group.

Write-Host "`n--- conferindo o container ---" -ForegroundColor Cyan
& (Join-Path $Root 'tools\convert.exe') --verify $containerFile.FullName
if ($LASTEXITCODE -ne 0) { throw 'o container nao passou no --verify; nada foi montado' }

# --- a busca de clipping esta dentro? ----------------------------------------
#
# Um container escrito com absmax puro nao tem a chave `clip_grid` no `meta`, e
# o card afirma 9,67% de erro de saida. Publicar um sem o outro faz o card
# mentir -- entao o aviso e alto e a decisao e de quem roda.

$hasClip = & (Join-Path $Root 'tootega.exe') inspect $containerFile.FullName |
           Select-String -Pattern 'clip-grid\s+\d+\s+pontos' -Quiet
if (-not $hasClip) {
    Write-Host "`nAVISO: este container parece ter sido escrito SEM a busca de clipping." -ForegroundColor Yellow
    Write-Host "       O card afirma erro de saida 9,67%; sem clip-grid o numero e 10,81%." -ForegroundColor Yellow
    Write-Host "       Reconverta com --clip-grid 13 ou ajuste o card antes de publicar.`n" -ForegroundColor Yellow
}

# --- montagem ----------------------------------------------------------------

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

Copy-Item $cardFile.FullName (Join-Path $Stage 'README.md')
Copy-Item $licenseFile.FullName (Join-Path $Stage 'LICENSE')
foreach ($f in $fromBase) { Copy-Item (Join-Path $Base $f) (Join-Path $Stage $f) }

# O container e o unico arquivo grande. Copiar 19,8 GB so para depois enviar e
# desperdicio: link duro quando o volume permite, copia quando nao.
$dest = Join-Path $Stage 'model.tootega'
try {
    New-Item -ItemType HardLink -Path $dest -Target $containerFile.FullName -ErrorAction Stop | Out-Null
    $how = 'link duro'
} catch {
    Copy-Item $containerFile.FullName $dest
    $how = 'copia'
}

Write-Host "--- montado em $Stage ($how para o container) ---`n" -ForegroundColor Cyan
Get-ChildItem $Stage | ForEach-Object {
    '{0,-24} {1,12:N1} MB' -f $_.Name, ($_.Length / 1MB)
}

$total = (Get-ChildItem $Stage | Measure-Object -Property Length -Sum).Sum
Write-Host ('{0,-24} {1,12:N1} MB' -f 'TOTAL', ($total / 1MB))

# --- envio -------------------------------------------------------------------

if (-not $Upload) {
    Write-Host "`nNada enviado. Destino seria huggingface.co/$Repo" -ForegroundColor Green
    Write-Host "Para publicar:"
    Write-Host "  hf auth login"
    Write-Host "  .\publish\publish.ps1 -Container $Container -Upload"
    exit 0
}

if (-not $Repo) { throw '-Upload exige -Repo <usuario>/<nome>' }

Write-Host "`n--- enviando para huggingface.co/$Repo ---" -ForegroundColor Cyan
& hf upload $Repo $Stage --repo-type model --commit-message 'GLM-4.7-Flash int4 g64, container .tootega'
if ($LASTEXITCODE -ne 0) { throw "upload falhou ($LASTEXITCODE)" }

Write-Host "`n  https://huggingface.co/$Repo" -ForegroundColor Green
