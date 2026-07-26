# pipetest.ps1 -- reproduz o padrao de escrita da UI.
#
# O Cockpit escreve o `initialize` logo apos o spawn e a mensagem do usuario
# depois, no MESMO pipe, sem fechar o stdin. Alimentar o agente com um arquivo
# pronto nao reproduz isso: o arquivo chega inteiro de uma vez e o EOF encerra
# o processo -- os dois casos que escondiam o bug do buffer.
#
# Uso:  .\tools\agent\pipetest.ps1 [-Port 8080] [-Espera 3]

param(
    [int]$Port = 8080,
    [int]$Espera = 3,
    [string]$Prompt = 'Responda apenas OK.'
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot '..\agent.exe'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = (Resolve-Path $exe).Path
foreach ($a in @('-p', '--output-format', 'stream-json', '--input-format', 'stream-json',
                 '--include-partial-messages', '--permission-prompt-tool', 'stdio', '--verbose',
                 '--no-tools', '--cwd', $root, '--server', "127.0.0.1:$Port",
                 '--resume', 'pipetest', '--idle-exit', '20')) {
    $psi.ArgumentList.Add($a)
}
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false

$p = [System.Diagnostics.Process]::Start($psi)
# Leitura por Task e nao por evento: um handler de evento do .NET roda numa
# thread sem runspace do PowerShell e estoura na primeira linha recebida.
$outTask = $p.StandardOutput.ReadToEndAsync()
$errTask = $p.StandardError.ReadToEndAsync()

# 1) handshake, como o CliProcessManager faz no spawn
$p.StandardInput.WriteLine('{"type":"control_request","request_id":"init","request":{"subtype":"initialize"}}')
$p.StandardInput.Flush()

# 2) a mensagem do usuario, depois -- e este intervalo e o teste
Start-Sleep -Seconds $Espera
$msg = @{ type = 'user'; message = @{ role = 'user'; content = @(@{ type = 'text'; text = $Prompt }) } } |
       ConvertTo-Json -Depth 6 -Compress
$p.StandardInput.WriteLine($msg)
$p.StandardInput.Flush()

if (-not $p.WaitForExit(300000)) { $p.Kill($true); throw 'o agente nao terminou em 5 min' }
$p.WaitForExit()
$outText = $outTask.Result
$errText = $errTask.Result

Write-Host '--- stderr (rastro do agente) ---'
Write-Host $errText.TrimEnd()

$result = $null
foreach ($l in $outText -split "`n") {
    if ($l -match '"type":"result"') { $result = $l | ConvertFrom-Json }
}
Write-Host ''
if ($result) {
    Write-Host ("resposta: {0}" -f $result.result)
    Write-Host 'ok: a mensagem enviada DEPOIS do handshake foi processada'
} else {
    Write-Host 'FALTOU: nenhum result -- a mensagem tardia nao foi lida'
    exit 1
}
