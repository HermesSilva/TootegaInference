# smoke.ps1 -- prova que o agente fala o contrato que a UI espera.
#
# Nao valida a resposta do modelo: valida o ENVELOPE. A UI (Cockpit) so
# renderiza o que reconhece, e um campo com nome errado nao da erro -- da tela
# vazia. Entao a checagem e por tipo de evento, contra `shared/events.ts`.
#
# Uso:  .\tools\agent\smoke.ps1 [-Port 8099] [-Prompt "..."]

param(
    [int]$Port = 8099,
    [string]$Prompt = 'Em uma frase: o que e um mutex?'
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot '..\agent.exe'
if (-not (Test-Path $exe)) { throw "agent.exe nao existe. Rode .\tools\agent\build.ps1" }

# --version tem de sair com 0: `CliProcessManager.detect()` reprova o
# executavel inteiro se este comando falhar.
$ver = & $exe --version
if ($LASTEXITCODE -ne 0) { throw "--version saiu com $LASTEXITCODE" }
Write-Host "versao: $ver"

$in = @(
    (@{ type='control_request'; request_id='init'; request=@{ subtype='initialize' } } | ConvertTo-Json -Depth 6 -Compress),
    (@{ type='user'; message=@{ role='user'; content=@(@{ type='text'; text=$Prompt }) } } | ConvertTo-Json -Depth 6 -Compress)
) -join "`n"

$tmpIn  = Join-Path $env:TEMP 'agent-smoke-in.ndjson'
$tmpOut = Join-Path $env:TEMP 'agent-smoke-out.ndjson'
Set-Content $tmpIn -Value $in -Encoding utf8 -NoNewline

$args = @('-p', '--output-format', 'stream-json', '--input-format', 'stream-json',
          '--include-partial-messages', '--permission-prompt-tool', 'stdio', '--verbose',
          '--server', "127.0.0.1:$Port")

$t0 = Get-Date
Get-Content $tmpIn -Raw | & $exe @args > $tmpOut
$secs = ((Get-Date) - $t0).TotalSeconds

$lines = Get-Content $tmpOut | Where-Object { $_.Trim() }
Write-Host ("{0} eventos em {1:N1} s" -f $lines.Count, $secs)

$kinds = @{}
$deltaKinds = @{}
$assistant = $null
$result = $null
foreach ($l in $lines) {
    $e = $null
    try { $e = $l | ConvertFrom-Json } catch { Write-Host "  LINHA INVALIDA: $l"; continue }
    $kinds[$e.type] = 1 + ($kinds[$e.type] ?? 0)
    if ($e.type -eq 'stream_event') {
        $t = $e.event.type
        if ($t -eq 'content_block_delta') { $t = "delta:$($e.event.delta.type)" }
        $deltaKinds[$t] = 1 + ($deltaKinds[$t] ?? 0)
    }
    if ($e.type -eq 'assistant') { $assistant = $e }
    if ($e.type -eq 'result')    { $result = $e }
}

Write-Host ''
Write-Host 'eventos por tipo:'
$kinds.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Host ("  {0,-16} {1}" -f $_.Key, $_.Value) }
Write-Host 'stream_event por subtipo:'
$deltaKinds.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Host ("  {0,-24} {1}" -f $_.Key, $_.Value) }

# O que a UI EXIGE para nao ficar em branco.
$checks = @(
    @{ nome='system/init com session_id'; ok = ($lines | Where-Object { $_ -match '"subtype":"init"' -and $_ -match '"session_id"' }).Count -gt 0 },
    @{ nome='pelo menos um text_delta';   ok = ($deltaKinds['delta:text_delta'] ?? 0) -gt 0 },
    @{ nome='message_start e message_stop'; ok = ($deltaKinds['message_start'] ?? 0) -gt 0 -and ($deltaKinds['message_stop'] ?? 0) -gt 0 },
    @{ nome='assistant com bloco text';   ok = $null -ne $assistant -and ($assistant.message.content | Where-Object { $_.type -eq 'text' }).Count -gt 0 },
    @{ nome='result subtype success';     ok = $null -ne $result -and $result.subtype -eq 'success' },
    @{ nome='result traz usage';          ok = $null -ne $result -and $null -ne $result.usage.output_tokens }
)

Write-Host ''
$bad = 0
foreach ($c in $checks) {
    Write-Host ("  {0,-32} {1}" -f $c.nome, $(if ($c.ok) { 'ok' } else { 'FALTOU'; }))
    if (-not $c.ok) { $bad++ }
}

if ($assistant) {
    $txt = ($assistant.message.content | Where-Object { $_.type -eq 'text' } | Select-Object -First 1).text
    Write-Host ''
    Write-Host 'resposta:'
    Write-Host ("  " + ($txt -replace "`n", "`n  "))
}

Write-Host ''
if ($bad) { Write-Host "$bad checagem(ns) falharam"; exit 1 }
Write-Host 'contrato ok'
