# sweep-threads.ps1 -- how many threads this box actually wants.
#
#   .\Tests\benchserve\sweep-threads.ps1
#   .\Tests\benchserve\sweep-threads.ps1 -Counts 6,12,20,24 -Repeats 3
#
# Runs `bench-serve.ps1` once per thread count, with everything else identical,
# and prints the comparison. Each configuration lands in results.csv under the
# label `threads-<n>`, so the history survives the terminal.
#
# Why this is worth measuring instead of reasoning about:
#
# `tt_cmd_serve` builds the pool with TT_AFFINITY_ALL, which is the FIRST
# logical processor of every physical core -- never the SMT sibling, because the
# sibling competes for the same execution unit. On this i9-14900K that is 22
# CPUs: 6 P-cores (efficiency class 1, SMT, logical 0/2/4/6/8/10) and 16 E-cores
# (class 0, no SMT, logical 12..27).
#
# The pool assigns `cpus[(i + 1) % ncpu]`, so:
#
#   --threads 20  ->  20 of 22 CPUs, none doubled up
#   --threads 24  ->  22 of 22 CPUs, and 2 of them carry TWO pinned threads
#   --threads 6   ->  the 6 P-cores only
#
# So 24 is not "more parallelism than 20": it is 20 plus two cores doing double
# duty while the barrier waits for them. Whether that costs anything is exactly
# the kind of question a measurement answers and an argument does not.
#
# The deeper issue this cannot fix is that the blocks are equal and the cores
# are not: a P-core and an E-core get the same `grain`, and `tt_pool_run`'s
# barrier waits for the slowest. Picking the thread count is choosing where on
# that trade-off to sit -- it does not remove it. See docs/23.

param(
    # Thread counts to try. The defaults bracket the interesting region: P-cores
    # only, half, the no-overlap point, and the current setting.
    [int[]]$Counts   = @(6, 12, 20, 24),

    [int]$Repeats    = 3,
    [int]$MaxTokens  = 256,
    [int]$Ctx        = 150000,
    [int]$Port       = 8080,

    # Prefix for the label in results.csv, so a sweep can be told apart from
    # another one run later against a different binary.
    [string]$Tag     = ''
)

$ErrorActionPreference = 'Stop'

$Bench = Join-Path $PSScriptRoot 'bench-serve.ps1'
$Csv   = Join-Path $PSScriptRoot 'results\results.csv'

if (-not (Test-Path $Bench)) { throw "nao encontrei bench-serve.ps1 em '$Bench'." }

# Read the CSV before, so the summary reports only what THIS sweep produced.
# Without this a rerun would average over every past measurement.
$before = if (Test-Path $Csv) { @(Import-Csv $Csv).Count } else { 0 }

Write-Host ''
Write-Host "sweep-threads  $($Counts -join ', ')  x$Repeats" -ForegroundColor Cyan
Write-Host "  cada configuracao recarrega o modelo $Repeats vezes -- isto demora." -ForegroundColor DarkGray
Write-Host ''

foreach ($n in $Counts) {
    $label = if ($Tag) { "$Tag-threads-$n" } else { "threads-$n" }
    Write-Host ("=" * 60) -ForegroundColor DarkGray
    Write-Host "--threads $n" -ForegroundColor Cyan

    & $Bench -Label $label -Threads $n -Repeats $Repeats `
             -MaxTokens $MaxTokens -Ctx $Ctx -Port $Port
    if ($LASTEXITCODE -ne 0) { throw "bench-serve falhou com --threads $n." }
}

# ------------------------------------------------------------------ summary

$rows = @(Import-Csv $Csv) | Select-Object -Skip $before
if (-not $rows) { Write-Host 'nenhuma linha nova em results.csv.'; exit 0 }

Write-Host ''
Write-Host ('=' * 60) -ForegroundColor DarkGray
Write-Host 'resumo do sweep' -ForegroundColor Cyan
Write-Host ''
Write-Host '  threads   mediana     media    desvio   amostras'
Write-Host '  -------   -------   -------   -------   --------'

$best = $null
foreach ($g in ($rows | Group-Object threads | Sort-Object { [int]$_.Name })) {
    # The CSV is written with the current culture, so a decimal comma is
    # possible; parse invariantly rather than trusting the separator.
    $t = @($g.Group | ForEach-Object {
        [double]::Parse(($_.tokens_per_second -replace ',', '.'),
                        [Globalization.CultureInfo]::InvariantCulture)
    }) | Sort-Object

    # [math]::Floor, not [int]: PowerShell's [int] cast rounds half to EVEN, so
    # [int](3/2) is 2 and not 1 -- with three samples the "median" would be the
    # largest one.
    $mid = [int][math]::Floor($t.Count / 2)
    $median = if ($t.Count % 2) { $t[$mid] } else { ($t[$mid - 1] + $t[$mid]) / 2 }
    $mean = ($t | Measure-Object -Average).Average
    $sd   = if ($t.Count -gt 1) {
                [math]::Sqrt((($t | ForEach-Object { [math]::Pow($_ - $mean, 2) }) |
                              Measure-Object -Sum).Sum / ($t.Count - 1))
            } else { 0 }

    Write-Host ("  {0,7}   {1,7:N2}   {2,7:N2}   {3,7:N2}   {4}" -f `
                $g.Name, $median, $mean, $sd,
                (($t | ForEach-Object { '{0:N2}' -f $_ }) -join ' '))

    if (-not $best -or $median -gt $best.Median) {
        $best = [pscustomobject]@{ Threads = $g.Name; Median = $median; Sd = $sd }
    }
}

# Did the output stay the same across every configuration?  It has to: the
# partition is by output row and the reduction lives inside one thread, so
# --threads must not change a single token (docs/15).  If this trips, the
# throughput numbers are the least of the problems.
$hashes = @($rows.output_sha256_16 | Sort-Object -Unique)
Write-Host ''
if ($hashes.Count -gt 1) {
    Write-Host "ATENCAO: --threads MUDOU A SAIDA ($($hashes.Count) textos distintos)." -ForegroundColor Red
    Write-Host "Isto quebra o determinismo de docs/15 e vale mais que qualquer" -ForegroundColor Red
    Write-Host "numero acima. Investigue antes de escolher configuracao." -ForegroundColor Red
} else {
    Write-Host "  saida identica em todas as configuracoes ($($hashes[0]))" -ForegroundColor DarkGray
}

Write-Host ''
Write-Host ("  melhor: --threads {0} com {1:N2} tok/s de mediana" -f `
            $best.Threads, $best.Median) -ForegroundColor Green
Write-Host ''
Write-Host "  Antes de trocar o padrao: veja se a diferenca supera o desvio." -ForegroundColor DarkGray
Write-Host "  Duas configuracoes cujos espalhamentos se sobrepoem nao foram" -ForegroundColor DarkGray
Write-Host "  distinguidas -- e o erro que docs/04 registra." -ForegroundColor DarkGray
Write-Host ''
