# toolgate.ps1 -- portao do M4.
#
# Pergunta unica: com ferramentas declaradas, o modelo emite chamada BEM
# FORMADA na primeira tentativa, e escolhe a ferramenta certa?
#
# Criterio, fixado ANTES de medir (docs/17: faixa esperada declarada antes):
#   >= 8/10 chamadas bem formadas e com a ferramenta certa  -> M4 segue
#   5..7                                                    -> segue com ressalva
#   <= 4                                                    -> o agente e fachada
#
# Uso:  .\Tests\toolgate.ps1 [-Port 8099] [-Think]

param([int]$Port = 8099, [switch]$Think)

$ErrorActionPreference = 'Stop'
$base = "http://127.0.0.1:$Port"

$tools = @(
  @{ type='function'; function=@{ name='read_file'; description='Le um arquivo de texto do disco'
     parameters=@{ type='object'; properties=@{
        path=@{ type='string'; description='Caminho do arquivo' }
        max_lines=@{ type='integer'; description='Maximo de linhas' } }; required=@('path') } } },
  @{ type='function'; function=@{ name='search_code'; description='Busca um padrao nos arquivos do projeto'
     parameters=@{ type='object'; properties=@{
        pattern=@{ type='string' }; glob=@{ type='string' }
        case_sensitive=@{ type='boolean' } }; required=@('pattern') } } },
  @{ type='function'; function=@{ name='run_command'; description='Executa um comando no shell e devolve a saida'
     parameters=@{ type='object'; properties=@{
        command=@{ type='string' }; timeout_ms=@{ type='integer' } }; required=@('command') } } }
)

# Cada caso: pedido + ferramenta esperada. `$null` = nao deve chamar nada.
$cases = @(
  @{ q='Leia o arquivo src/pool.c e me diga o que ele faz.';              want='read_file' },
  @{ q='Mostre as primeiras 40 linhas de README.md.';                     want='read_file' },
  @{ q='Onde no projeto aparece a palavra tt_futex_wait?';                want='search_code' },
  @{ q='Procure por TODO nos arquivos .c, ignorando maiusculas.';         want='search_code' },
  @{ q='Rode o comando git status e me diga o resultado.';                want='run_command' },
  @{ q='Compile o projeto com build.ps1 e me diga se passou.';            want='run_command' },
  @{ q='Quantas linhas tem o arquivo docs/03-arquitetura.md?';            want='run_command' },
  @{ q='Preciso ver o conteudo de CLAUDE.md.';                            want='read_file' },
  @{ q='Ache todas as chamadas de malloc no codigo.';                     want='search_code' },
  @{ q='Quem foi Alan Turing? Responda em uma frase.';                    want=$null }
)

Write-Host ("portao do M4 -- raciocinio {0}" -f ($(if ($Think) {'LIGADO'} else {'desligado'})))
Write-Host ''

$ok = 0; $n = 0; $rows = @()
foreach ($c in $cases) {
  $n++
  $body = @{ messages=@(@{ role='user'; content=$c.q }); tools=$tools
             stream=$false; temperature=0.0; think=[bool]$Think } |
          ConvertTo-Json -Depth 8 -Compress
  # Sem historico entre casos: cada um mede a PRIMEIRA tentativa.
  Invoke-RestMethod "$base/reset" -Method Post -TimeoutSec 30 | Out-Null

  $t0 = Get-Date
  $r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -Body $body `
        -ContentType 'application/json' -TimeoutSec 900
  $secs = ((Get-Date) - $t0).TotalSeconds

  $m = $r.choices[0].message
  $called = if ($m.tool_calls) { $m.tool_calls[0].function.name } else { $null }
  $args = if ($m.tool_calls) { $m.tool_calls[0].function.arguments } else { '' }

  # Bem formado = o JSON de `arguments` parseia e traz o argumento obrigatorio.
  $wellFormed = $false
  if ($called) {
    try {
      $a = $args | ConvertFrom-Json
      $req = switch ($called) { 'read_file' {'path'} 'search_code' {'pattern'} 'run_command' {'command'} default {$null} }
      $wellFormed = ($null -eq $req) -or ($null -ne $a.$req -and "$($a.$req)".Length -gt 0)
    } catch { $wellFormed = $false }
  }

  $pass = if ($null -eq $c.want) { $null -eq $called } else { $called -eq $c.want -and $wellFormed }
  if ($pass) { $ok++ }

  $rows += [pscustomobject]@{
    '#' = $n; esperado = ($c.want ?? '(nenhuma)'); chamou = ($called ?? '(nenhuma)')
    ok = $(if ($pass) {'sim'} else {'NAO'}); tokens = $r.usage.completion_tokens
    seg = [math]::Round($secs,1); argumentos = $args
  }
  Write-Host ("  {0,2}. {1,-12} -> {2,-12} {3}" -f $n, ($c.want ?? '-'), ($called ?? '-'), $(if($pass){'ok'}else{'NAO'}))
}

Write-Host ''
$rows | Format-Table -AutoSize -Wrap
$pct = [math]::Round(100.0 * $ok / $n, 0)
Write-Host ("resultado: {0}/{1} ({2}%)" -f $ok, $n, $pct)
if     ($ok -ge 8) { Write-Host 'PASSOU -- o M4 segue como planejado' }
elseif ($ok -ge 5) { Write-Host 'RESSALVA -- segue, mas o laco precisa de retentativa' }
else               { Write-Host 'REPROVOU -- o agente seria fachada; mudar o escopo do M4' }
