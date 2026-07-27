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
#
# O `hf` 1.13 vinha quebrado nesta maquina -- `typer` 0.16 nao tem o
# `suggest_commands` que ele passa --, e a falha so apareceria no envio, depois
# de montar 19 GB. Por isso o script confere a ferramenta ANTES de montar.

param(
    [Parameter(Mandatory = $true)] [string]$Container,
    [string]$Base   = 'models\glm47flash-bf16',
    [string]$Stage  = 'publish\stage',
    # Namespace do HuggingFace, que NAO e o do GitHub: la e HermesSilva, aqui
    # HermesX21. Confira com `hf auth whoami` antes de duvidar do nome.
    [string]$Repo   = 'HermesX21/GLM-4.7-Flash-tootega-int4',
    [switch]$Upload,
    # Cria o repo PRIVADO antes de enviar. Publicar nao se desfaz de verdade --
    # arquivo enviado fica em cache e em indice mesmo depois de apagado --, entao
    # da para ver o card renderizado e o repo montado antes de abrir. A troca
    # para publico e um clique em Settings, e ai sim e definitiva.
    [switch]$Private
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

# O `hf` e Python, e imprime marca de visto (U+2713) e barra de progresso. Com a
# saida redirecionada, o Python assume a codificacao da localidade -- cp1252
# nesta maquina --, o `charmap` nao codifica o caractere e o processo MORRE com
# UnicodeEncodeError. Nao e aviso: e falha, e aconteceria no meio do upload.
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

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

# --- a ferramenta de envio funciona? -----------------------------------------
#
# Conferida ANTES de montar, e nao no fim: montar 19 GB para descobrir que o
# `hf` esta quebrado e desperdicio, e a quebra que aconteceu aqui -- `typer`
# velho demais para o `huggingface_hub` -- so aparece ao rodar.

if ($Upload) {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if (-not $hf) { throw 'CLI `hf` nao encontrado: python -m pip install -U huggingface_hub' }

    & hf version *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'o CLI `hf` existe mas nao roda. Tente: python -m pip install -U huggingface_hub typer'
    }

    $who = & hf auth whoami 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'sem sessao no HuggingFace: rode `hf auth login`' }
    Write-Host "`nautenticado: $who" -ForegroundColor Cyan
}

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
#
# O CONTAINER NAO ENTRA NO STAGE. A primeira versao deste script punha um link
# duro de 19,8 GB aqui dentro, e a limpeza da execucao seguinte batia em
# "Access to the path is denied" -- `Remove-Item` nao apaga link duro nesta
# maquina. Nao ha motivo para o arquivo passar por aqui: `hf upload` aceita um
# arquivo solto com o nome que ele tera no repo, entao ele vai direto do lugar
# onde o conversor o escreveu.

if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null

Copy-Item $cardFile.FullName (Join-Path $Stage 'README.md')
Copy-Item $licenseFile.FullName (Join-Path $Stage 'LICENSE')
foreach ($f in $fromBase) { Copy-Item (Join-Path $Base $f) (Join-Path $Stage $f) }

Write-Host "--- montado em $Stage ---`n" -ForegroundColor Cyan
Get-ChildItem $Stage | ForEach-Object {
    '{0,-24} {1,12:N1} MB' -f $_.Name, ($_.Length / 1MB)
}
'{0,-24} {1,12:N1} MB  (enviado direto, fora do stage)' -f
    'model.tootega', ($containerFile.Length / 1MB)

$small = (Get-ChildItem $Stage | Measure-Object -Property Length -Sum).Sum
Write-Host ('{0,-24} {1,12:N1} MB' -f 'TOTAL', (($small + $containerFile.Length) / 1MB))

# --- envio -------------------------------------------------------------------

if (-not $Upload) {
    Write-Host "`nNada enviado. Destino seria huggingface.co/$Repo" -ForegroundColor Green
    Write-Host "Para publicar:"
    Write-Host "  hf auth login"
    Write-Host "  .\publish\publish.ps1 -Container $Container -Upload"
    exit 0
}

if (-not $Repo) { throw '-Upload exige -Repo <usuario>/<nome>' }

if ($Private) {
    Write-Host "`n--- criando $Repo como PRIVADO ---" -ForegroundColor Cyan
    & hf repo create $Repo --repo-type model --private --exist-ok
    if ($LASTEXITCODE -ne 0) { throw "nao consegui criar o repo ($LASTEXITCODE)" }
}

# Os arquivos pequenos primeiro: se o card ou a licenca estiverem errados, o
# erro aparece em segundos e nao depois de subir 19 GB.
Write-Host "`n--- enviando card, licenca e tokenizer para huggingface.co/$Repo ---" -ForegroundColor Cyan
& hf upload $Repo $Stage --repo-type model `
    --commit-message 'Model card, licence and tokenizer'
if ($LASTEXITCODE -ne 0) { throw "upload dos arquivos pequenos falhou ($LASTEXITCODE)" }

Write-Host "`n--- enviando o container ($('{0:N1}' -f ($containerFile.Length / 1GB)) GB) ---" -ForegroundColor Cyan
& hf upload $Repo $containerFile.FullName model.tootega --repo-type model `
    --commit-message 'GLM-4.7-Flash int4 g64, container .tootega'
if ($LASTEXITCODE -ne 0) { throw "upload do container falhou ($LASTEXITCODE)" }

Write-Host "`n  https://huggingface.co/$Repo" -ForegroundColor Green
