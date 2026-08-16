# bench-serve.ps1 -- decode throughput of the serving path, measured the same
# way every time.
#
#   .\Tests\benchserve\bench-serve.ps1 -Label baseline
#   .\Tests\benchserve\bench-serve.ps1 -Label spin-2000
#
# Why this exists: the profile of 2026-08-16 (Profile/Report20260816-0844)
# showed ~31% of the process CPU inside NtYieldExecution/tt_cpu_relax -- the
# pool spinning on the barrier, not computing.  Any change aimed at that has to
# be compared against a number taken BEFORE it, under identical conditions.
#
# What is held fixed, and why each one matters:
#
#   - the prompt, word for word (PROMPT below).  A different prompt is a
#     different number of prefill tokens and a different decode length.
#   - temperature 0.  `sample.c:36` returns the argmax before touching the RNG,
#     so the token sequence does not depend on the clock-seeded sampler.  Two
#     runs generate the SAME tokens; only the time differs.  Without this the
#     runs generate different text and tok/s compares nothing.
#   - think:false.  Reasoning is on by default (`server.c:830`) and its length
#     varies run to run, which would move max_tokens' meaning.
#   - max_tokens fixed, so every run decodes the same amount of work.
#   - the server is restarted for every repetition, so no run inherits the
#     prefix cache or the warm KV of the one before it.
#   - --threads 20, the measured optimum on this box (docs/23).  It is a
#     parameter like any other: changing it produces numbers that do not compare
#     with the ones already in results.csv, which is why every row records it.
#
# The result is one JSON per run plus a summary line in results.csv, appended
# never overwritten -- the history of the measurements IS the artefact.

param(
    # Names this measurement in results.csv.  Use the state of the tree:
    # `baseline` before the change, something descriptive after it.
    [Parameter(Mandatory = $true)]
    [string]$Label,

    [int]$Repeats    = 3,
    [int]$MaxTokens  = 256,

    # 20, measured, not guessed: the sweep of docs/23 puts it at 21,55 tok/s
    # against 15,84 at 24 (+36%) and 20,43 at 22.  24 pins two threads onto
    # cores that already have one -- `cpus[(i+1) % 22]` wraps -- and the barrier
    # waits for them, which also makes it erratic (sd 1,27 against 0,15).
    # 22 loses too: it is the physical core count, but two of those CPUs are
    # P-core first logicals whose SMT siblings idle, so the last two threads buy
    # contention rather than throughput.
    [int]$Threads    = 20,

    [int]$Ctx        = 150000,
    [int]$Port       = 8080,

    [string]$Model     = 'models\glm47flash-v4\model.tootega',
    [string]$Tokenizer = 'models\glm47flash-bf16\tokenizer.json',

    # Seconds to wait for the model to be resident and answering.
    [int]$LoadTimeout = 900
)

$ErrorActionPreference = 'Stop'

$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$OutDir  = Join-Path $Root 'Tests\benchserve\results'
$Csv     = Join-Path $OutDir 'results.csv'
$Exe     = Join-Path $Root 'tootega.exe'

# 30 words.  Fixed forever: changing it invalidates every comparison already in
# results.csv.  It asks for prose, not a list, so the decode does not stop early.
$PROMPT = 'Explain in a single continuous paragraph how a mixture of experts ' +
          'language model decides which experts to activate for each token, ' +
          'and why that routing matters for overall inference speed'

# ------------------------------------------------------------------ preflight

if (-not (Test-Path $Exe)) {
    throw "tootega.exe nao encontrado em '$Exe' -- rode .\build.ps1 primeiro."
}
foreach ($p in @($Model, $Tokenizer)) {
    if (-not (Test-Path (Join-Path $Root $p))) { throw "nao encontrado: $p" }
}

# An optimized binary is the only one worth measuring: a -Debug build is /Od
# and its number describes nothing.  `version` does not say which it is -- the
# build id is a hash of src/, not of the flags -- so we check the two things
# that a debug build leaves behind: the .pdb next to the .exe and the /Zi size.
# Neither is proof on its own, so this warns and asks rather than throwing.
$dbgSigns = @()
if ((Get-Item $Exe).Length -gt 2MB) { $dbgSigns += 'o .exe passa de 2 MB' }
$ilk = Join-Path $Root 'build\vs\Debug\tootega.ilk'
if ((Test-Path $ilk) -and (Get-Item $ilk).LastWriteTime -gt (Get-Item $Exe).LastWriteTime) {
    $dbgSigns += 'ha um build de Debug mais novo que o .exe'
}
if ($dbgSigns) {
    Write-Host ''
    Write-Host "ATENCAO: $($dbgSigns -join '; ')." -ForegroundColor Yellow
    Write-Host "Medir um build de debug da um numero que nao descreve nada." -ForegroundColor Yellow
    Write-Host "Rode .\build.ps1 (SEM -Debug) se houver duvida." -ForegroundColor Yellow
    Write-Host ''
}

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "porta $Port ja esta em uso -- feche o servidor anterior."
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# A label that already exists under DIFFERENT parameters is the one way this CSV
# can quietly lie: two groups of rows, same name, not comparable.  Say so now,
# while there is still time to pick another label.
if (Test-Path $Csv) {
    $clash = @(Import-Csv $Csv | Where-Object {
        $_.label -eq $Label -and
        ($_.threads -ne "$Threads" -or $_.ctx -ne "$Ctx" -or
         $_.max_tokens -ne "$MaxTokens")
    })
    if ($clash) {
        $was = $clash[0]
        Write-Host ''
        Write-Host "ATENCAO: ja existem $($clash.Count) linha(s) com o rotulo '$Label'" -ForegroundColor Yellow
        Write-Host "em OUTRA configuracao (threads=$($was.threads) ctx=$($was.ctx) max_tokens=$($was.max_tokens))." -ForegroundColor Yellow
        Write-Host "Esta execucao usa threads=$Threads ctx=$Ctx max_tokens=$MaxTokens." -ForegroundColor Yellow
        Write-Host "Os dois grupos vao dividir o mesmo nome e NAO sao comparaveis." -ForegroundColor Yellow
        Write-Host "Use outro -Label se quiser distingui-los depois." -ForegroundColor Yellow
        Write-Host ''
    }
}

# The exact binary under test.  A result without this is a result you cannot
# come back to: `baseline` means nothing if you cannot tell which .exe it was.
$exeItem = Get-Item $Exe
$exeHash = (Get-FileHash $Exe -Algorithm SHA256).Hash.Substring(0, 16)
$gitRev  = (& git -C $Root rev-parse --short HEAD 2>$null)
$gitDirty = if (& git -C $Root status --porcelain 2>$null) { 'dirty' } else { 'clean' }

Write-Host ''
Write-Host "bench-serve  label=$Label" -ForegroundColor Cyan
Write-Host "  exe        $($exeItem.Name)  $exeHash  $($exeItem.LastWriteTime)"
Write-Host "  git        $gitRev ($gitDirty)"
Write-Host "  params     threads=$Threads ctx=$Ctx max_tokens=$MaxTokens repeats=$Repeats"
Write-Host ''

# ------------------------------------------------------------------- helpers

function Wait-Ready {
    param([int]$Port, [int]$TimeoutSec, [System.Diagnostics.Process]$Proc)

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($Proc.HasExited) {
            throw "o servidor morreu durante o carregamento (exit $($Proc.ExitCode))."
        }
        try {
            $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/models" `
                                   -TimeoutSec 5 -ErrorAction Stop
            if ($r.data) { return }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "servidor nao respondeu em $TimeoutSec s."
}

function Stop-Server {
    param([System.Diagnostics.Process]$Proc)

    if (-not $Proc -or $Proc.HasExited) { return }
    # CloseMainWindow does not reach a console app started without a window;
    # Kill is the honest way, and the server holds nothing that needs flushing.
    $Proc.Kill()
    $Proc.WaitForExit(30000) | Out-Null
}

# ---------------------------------------------------------------------- runs

$rows = @()

for ($i = 1; $i -le $Repeats; $i++) {
    Write-Host "run $i/$Repeats" -ForegroundColor Yellow

    $stamp   = Get-Date -Format 'yyyyMMdd-HHmmss'
    $logPath = Join-Path $OutDir "$Label-run$i-$stamp.server.log"

    # Not `$args`: that is an automatic variable and assigning to it is asking
    # for a silent surprise.
    $srvArgs = @(
        'serve', (Join-Path $Root $Model),
        '--threads', $Threads,
        '--tokenizer', (Join-Path $Root $Tokenizer),
        '--port', $Port,
        '--ctx', $Ctx,
        '--verbose'
    )

    $proc = Start-Process -FilePath $Exe -ArgumentList $srvArgs `
                          -WorkingDirectory $Root -PassThru -NoNewWindow `
                          -RedirectStandardOutput $logPath `
                          -RedirectStandardError "$logPath.err"

    try {
        $tLoad0 = Get-Date
        Wait-Ready -Port $Port -TimeoutSec $LoadTimeout -Proc $proc
        $loadSec = ((Get-Date) - $tLoad0).TotalSeconds
        Write-Host ("  carregado em {0:N1} s" -f $loadSec)

        $body = @{
            model       = 'tootega'
            messages    = @(@{ role = 'user'; content = $PROMPT })
            temperature = 0          # greedy: mesma saida em toda execucao
            think       = $false
            max_tokens  = $MaxTokens
            stream      = $false
        } | ConvertTo-Json -Depth 5 -Compress

        $t0 = Get-Date
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" `
                                  -Method Post -Body $body `
                                  -ContentType 'application/json' `
                                  -TimeoutSec 3600
        $wallSec = ((Get-Date) - $t0).TotalSeconds

        $u    = $resp.usage
        $text = $resp.choices[0].message.content

        # The server's own tok/s covers decode only; the wall clock here also
        # carries prefill and HTTP.  Both are recorded: they answer different
        # questions and disagreeing is informative, not an error.
        $row = [pscustomobject]@{
            timestamp        = (Get-Date -Format 'o')
            label            = $Label
            run              = $i
            tokens_per_second = [double]$u.tokens_per_second
            completion_tokens = [int]$u.completion_tokens
            prompt_tokens     = [int]$u.prompt_tokens
            wall_seconds      = [math]::Round($wallSec, 3)
            load_seconds      = [math]::Round($loadSec, 3)
            threads           = $Threads
            ctx               = $Ctx
            max_tokens        = $MaxTokens
            exe_sha256_16     = $exeHash
            exe_mtime         = $exeItem.LastWriteTime.ToString('o')
            git_rev           = $gitRev
            git_state         = $gitDirty
            output_sha256_16  = (Get-FileHash -InputStream (
                                    [IO.MemoryStream]::new(
                                        [Text.Encoding]::UTF8.GetBytes($text))
                                 ) -Algorithm SHA256).Hash.Substring(0, 16)
        }
        $rows += $row

        Write-Host ("  {0,7:N2} tok/s   {1} tokens   wall {2:N1} s" -f `
                    $row.tokens_per_second, $row.completion_tokens, $wallSec)

        # The full reply, kept so a later run can be diffed token for token.
        # With temperature 0 the text must be identical across runs; if it is
        # not, something changed the arithmetic and the tok/s comparison is void.
        $resp | ConvertTo-Json -Depth 8 |
            Set-Content -Path (Join-Path $OutDir "$Label-run$i-$stamp.json") `
                        -Encoding utf8
    }
    finally {
        Stop-Server -Proc $proc
        # The port lingers in TIME_WAIT for a moment after the process dies.
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            if (-not (Get-NetTCPConnection -LocalPort $Port -State Listen `
                                           -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 500
        }
    }
}

# ------------------------------------------------------------------ summary

$rows | Export-Csv -Path $Csv -NoTypeInformation -Append -Encoding utf8

$tps    = $rows.tokens_per_second | Sort-Object
# [math]::Floor, not [int]: PowerShell's [int] cast rounds half to EVEN, so
# [int](3/2) is 2 and not 1 -- with three samples the "median" would come out as
# the largest one.
$mid    = [int][math]::Floor($tps.Count / 2)
$median = if ($tps.Count % 2) { $tps[$mid] }
          else { ($tps[$mid - 1] + $tps[$mid]) / 2 }
$mean   = ($tps | Measure-Object -Average).Average
$sd     = if ($tps.Count -gt 1) {
              [math]::Sqrt((($tps | ForEach-Object { [math]::Pow($_ - $mean, 2) }) |
                            Measure-Object -Sum).Sum / ($tps.Count - 1))
          } else { 0 }

# @() so that a single distinct hash still indexes as an array: without it
# `$hashes[0]` hands the whole string to -f and prints one character.
$hashes = @($rows.output_sha256_16 | Sort-Object -Unique)

Write-Host ''
Write-Host "resumo  $Label" -ForegroundColor Cyan
Write-Host ("  mediana    {0:N2} tok/s" -f $median)
Write-Host ("  media      {0:N2} tok/s  (desvio {1:N2})" -f $mean, $sd)
Write-Host ("  amostras   {0}" -f (($tps | ForEach-Object { '{0:N2}' -f $_ }) -join ', '))

if ($hashes.Count -gt 1) {
    Write-Host ''
    Write-Host "ATENCAO: as execucoes geraram textos DIFERENTES ($($hashes.Count) hashes)." -ForegroundColor Red
    Write-Host "Com temperature 0 isso nao deveria acontecer -- o tok/s nao esta" -ForegroundColor Red
    Write-Host "comparando o mesmo trabalho. Investigue antes de usar estes numeros." -ForegroundColor Red
} else {
    Write-Host ("  saida      identica nas {0} execucoes ({1})" -f $rows.Count, $hashes[0])
}

Write-Host ''
Write-Host "  -> $Csv"
