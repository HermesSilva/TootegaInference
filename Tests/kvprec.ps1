# kvprec.ps1 -- decide se o KV em int8 pode virar default do servidor.
#
# A pergunta e de CONTEXTO LONGO: int8 no KV liberaria 202.752 posicoes em
# 5,8 GB contra 22,0 GB do f32. A triagem de 384 posicoes ja passou; o que
# falta e o regime em que o KV realmente pesa.
#
# Roda SEQUENCIAL de proposito: cada medicao carrega o container e os pesos
# BF16, e duas ao mesmo tempo disputariam banda de memoria -- que e exatamente
# a grandeza medida.
#
# Uso:  .\Tests\kvprec.ps1 [-Tokens 4096] [-Chunks 4]

param(
    [int]$Tokens = 4096,
    [int]$Chunks = 4,
    [string]$Corpus = 'corpus.txt'
)

$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$model = 'models\glm47flash-v3\model.tootega'
$bf16  = 'models\glm47flash-bf16'
$tok   = 'models\glm47flash-bf16\tokenizer.json'
$out   = Join-Path $root 'Tests\kvprec'
New-Item -ItemType Directory -Force $out | Out-Null

function Run($nome, $exe, $argumentos) {
    $log = Join-Path $out "$nome.txt"
    Write-Host ""
    Write-Host ("=== {0} " -f $nome).PadRight(64, '=')
    Write-Host ("  inicio {0}" -f (Get-Date -Format 'HH:mm:ss'))
    $t0 = Get-Date
    & $exe @argumentos 2>&1 | Tee-Object $log | Out-Null
    Write-Host ("  fim    {0}  ({1:N1} min)" -f (Get-Date -Format 'HH:mm:ss'),
                ((Get-Date) - $t0).TotalMinutes)
    Get-Content $log | Select-String -Pattern 'nll|match@ref|rep4|parou|kv |INCONCL|delta de'
}

foreach ($kv in 'f32', 'int8') {
    Run "ablate-$kv" '.\Tests\ablate.exe' @(
        $model, '--bf16', $bf16, '--tokenizer', $tok, '--corpus', $Corpus,
        '--tokens', $Tokens, '--chunks', $Chunks, '--no-ablate', '--kv-prec', $kv)
    # O raw e sobrescrito a cada rodada: guarda um por configuracao.
    if (Test-Path 'ablate_raw.json') {
        Move-Item 'ablate_raw.json' (Join-Path $out "ablate_raw-$kv.json") -Force
    }
}

foreach ($kv in 'f32', 'int8') {
    Run "gentest-$kv" '.\Tests\gentest.exe' @(
        $model, '--tokenizer', $tok, '--bf16', $bf16, '--kv-prec', $kv)
}

Write-Host ''
Write-Host 'todas as medicoes terminaram; saidas em Tests\kvprec\'
