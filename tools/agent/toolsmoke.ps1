# toolsmoke.ps1 -- prova o laco de ferramentas ponta a ponta.
#
# Faz o papel da UI: manda o pedido, recebe `tool_use`, responde a aprovacao
# pelo protocolo de controle e confere que o `tool_result` voltou.
#
# A resposta a aprovacao vai PRE-ENFILEIRADA no stdin: o `request_id` e
# deterministico (`perm_1`) e o agente le a linha quando bloqueia esperando.
# Assim o teste roda sem precisar de dois processos conversando.
#
# Uso:  .\tools\agent\toolsmoke.ps1 [-Port 8099]

param(
    [int]$Port = 8099,
    [string]$Prompt = 'Quantas linhas tem o arquivo docs/03-architecture.md? Use as ferramentas e responda so o numero.'
)

$ErrorActionPreference = 'Stop'
$exe  = Join-Path $PSScriptRoot '..\agent.exe'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

$lines = @(
    (@{ type='control_request'; request_id='init'; request=@{ subtype='initialize' } } | ConvertTo-Json -Depth 6 -Compress),
    (@{ type='user'; message=@{ role='user'; content=@(@{ type='text'; text=$Prompt }) } } | ConvertTo-Json -Depth 6 -Compress),
    # Aprovacoes pre-autorizadas para as tres primeiras ferramentas que mudem estado.
    (@{ type='control_response'; response=@{ subtype='success'; request_id='perm_1'; response=@{ behavior='allow'; updatedInput=@{} } } } | ConvertTo-Json -Depth 8 -Compress),
    (@{ type='control_response'; response=@{ subtype='success'; request_id='perm_2'; response=@{ behavior='allow'; updatedInput=@{} } } } | ConvertTo-Json -Depth 8 -Compress),
    (@{ type='control_response'; response=@{ subtype='success'; request_id='perm_3'; response=@{ behavior='allow'; updatedInput=@{} } } } | ConvertTo-Json -Depth 8 -Compress)
) -join "`n"

$tmpIn  = Join-Path $env:TEMP 'agent-tools-in.ndjson'
$tmpOut = Join-Path $env:TEMP 'agent-tools-out.ndjson'
Set-Content $tmpIn -Value $lines -Encoding utf8 -NoNewline

$argv = @('-p', '--output-format', 'stream-json', '--input-format', 'stream-json',
          '--include-partial-messages', '--permission-prompt-tool', 'stdio', '--verbose',
          '--server', "127.0.0.1:$Port", '--cwd', $root, '--max-steps', '8')

$t0 = Get-Date
Get-Content $tmpIn -Raw | & $exe @argv > $tmpOut
$secs = ((Get-Date) - $t0).TotalSeconds

$events = @()
foreach ($l in (Get-Content $tmpOut | Where-Object { $_.Trim() })) {
    try { $events += ($l | ConvertFrom-Json) } catch { Write-Host "  LINHA INVALIDA: $l" }
}

$toolUses = @()
$toolResults = @()
$perms = @()
foreach ($e in $events) {
    if ($e.type -eq 'assistant') {
        foreach ($b in $e.message.content) { if ($b.type -eq 'tool_use') { $toolUses += $b } }
    }
    if ($e.type -eq 'user') {
        foreach ($b in $e.message.content) { if ($b.type -eq 'tool_result') { $toolResults += $b } }
    }
    if ($e.type -eq 'control_request' -and $e.request.subtype -eq 'can_use_tool') { $perms += $e }
}
$result = $events | Where-Object { $_.type -eq 'result' } | Select-Object -Last 1

Write-Host ("{0} eventos em {1:N1} s" -f $events.Count, $secs)
Write-Host ''
Write-Host 'ferramentas chamadas:'
foreach ($t in $toolUses) {
    Write-Host ("  {0,-6} {1}" -f $t.name, ($t.input | ConvertTo-Json -Depth 6 -Compress))
}
Write-Host 'aprovacoes pedidas:'
if ($perms.Count -eq 0) { Write-Host '  (nenhuma -- so ferramentas de leitura)' }
foreach ($p in $perms) { Write-Host ("  {0} {1}" -f $p.request.tool_name, $p.request_id) }

Write-Host ''
Write-Host 'resultados:'
foreach ($r in $toolResults) {
    $txt = "$($r.content)"
    if ($txt.Length -gt 160) { $txt = $txt.Substring(0,160) + '...' }
    Write-Host ("  erro={0}  {1}" -f $r.is_error, ($txt -replace "`n", ' | '))
}

Write-Host ''
if ($result) {
    Write-Host ("result: {0}  passos={1}  in={2} out={3}" -f $result.subtype, $result.steps,
                $result.usage.input_tokens, $result.usage.output_tokens)
    Write-Host ("resposta: {0}" -f $result.result)
}

$bad = 0
if ($toolUses.Count -eq 0)    { Write-Host 'FALTOU: nenhuma ferramenta foi chamada'; $bad++ }
if ($toolResults.Count -eq 0) { Write-Host 'FALTOU: nenhum tool_result voltou'; $bad++ }
if (-not $result -or $result.subtype -ne 'success') { Write-Host 'FALTOU: result nao foi success'; $bad++ }
if ($bad) { exit 1 }
Write-Host 'laco de ferramentas ok'
